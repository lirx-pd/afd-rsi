"""Frozen real Ascend build/rank-result adapter for the campaign contract."""

import os
import socket
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.native import analyze_profile, artifact_identity, build, fixture_result, runtime_env, verify_source
from afd_rsi.runner import execute
from afd_rsi.workspace import copy_source, inventory

plan = read_json(Path(os.environ["AFD_RSI_PLAN"]))


def _free_port() -> int:
    """A loopback port the rendezvous can bind without colliding with the runtime."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


build_dir = Path(os.environ["AFD_RSI_BUILD_DIR"])
source = Path.cwd()
if sys.argv[1] == "build":
    # A verified cache hit avoids a 20-minute rebuild of an unchanged checkout.
    cache = os.environ.get("AFD_RSI_BUILD_CACHE") or None
    build(source, build_dir / "native", float(os.environ["AFD_RSI_TIMEOUT"]),
          cache_root=Path(cache) if cache else None)
    sys.exit(0)

repo = build_dir / "native/source"
lock = read_json(build_dir / "native/source.lock.json")
assert lock["hash"] == os.environ["AFD_RSI_SOURCE_HASH"]
verify_source(repo, lock)
expected = read_json(build_dir / "native/artifacts.json")
assert expected == artifact_identity(repo)
settings = plan["native"]
output = Path(os.environ["AFD_RSI_RESULT"]).parent
reference = Path(os.environ["AFD_RSI_ATTEMPT"]) / "reference"
# One node, one rendezvous, on loopback. torchrun resolves the container
# hostname by default, and inside an iTASK Pod that name is the node FQDN, which
# is not always resolvable from the container; binding the store to loopback
# removes that dependency without changing rank layout or device usage.
args = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        "--master-addr", "127.0.0.1", "--master-port", str(_free_port()),
        f"--nproc-per-node={plan['ranks']}", str(Path(__file__).with_name("eager_fixture.py")),
        "--boundary", "--tile-boundary", "--random-weights", "--routed-scale", "2.5",
        "--hidden-size", str(settings["hidden"]), "--intermediate-size", str(settings["intermediate"]),
        "--local-experts", str(settings["experts"]), "--attn-ranks", str(settings["attn_ranks"])]
if settings.get("lengths"):
    # Deployed lengths, so the transaction reflects the real matrix.
    args += ["--lengths", settings["lengths"] if isinstance(settings["lengths"], str)
             else ",".join(map(str, settings["lengths"]))]
if settings["per_group"]:
    args.append("--per-group")
env = runtime_env(repo, settings["devices"])


def launch(name, argv, *, mega, repeats, reference_run=False):
    process = execute(argv, repo, output / name, float(os.environ["AFD_RSI_TIMEOUT"]), env)
    if process["status"] != "completed" or not process["cleanup_complete"]:
        raise RuntimeError(f"native {name} failed: {process['status']}")
    result = fixture_result(output / name / "stdout.log", plan["ranks"], expected, mega,
                            settings["attn_ranks"], repeats, reference_run,
                            full_graph=settings.get("execution", "eager") == "graph" and not reference_run,
                            lengths=settings.get("lengths"))
    assert artifact_identity(repo) == expected
    verify_source(repo, lock)
    return result


if os.environ["AFD_RSI_SIDE"] == "baseline" and sys.argv[1] == "correctness":
    launch("reference", args + ["--save-outputs", str(reference)], mega=False, repeats=0, reference_run=True)
    reference_files = {str(p): digest(p) for p in reference.parent.glob(reference.name + ".rank*.pt")}
    assert len(reference_files) == settings["attn_ranks"]
    write_json(reference.with_suffix(".lock.json"), reference_files)
reference_files = read_json(reference.with_suffix(".lock.json"))
assert all(digest(Path(name)) == value for name, value in reference_files.items())
measured = sys.argv[1] == "benchmark"
argv = args + ["--mega-moe", "--reference-outputs", str(reference)]
if measured:
    argv += ["--measure", "--warmups", str(settings["warmups"]), "--repeats", str(settings["repeats"])]
if settings.get("execution", "eager") == "graph":
    argv.append("--full-graph")
if sys.argv[1] == "profile":
    argv += ["--profile-dir", str(output / "traces")]
    # Diagnostic overlay is built and tested separately; timed artifacts stay immutable.
    # It anchors on the kernel layout this checkout actually ships, so a pull-native
    # build now yields per-stage rounds instead of a skip.
    from pipeline_trace import eligible, instrument
    mode = eligible(source)
    if mode is None:
        write_json(output / 'traces/pipeline-unsupported.json', {
            'reason': 'no per-stage anchor exists for this FFN kernel layout',
            'production_profile': 'unaffected; kernel_details.csv evidence still applies'})
    else:
        diagnostic_input = output / 'diagnostic-input'
        copy_source(source, diagnostic_input, inventory(source))
        instrument(diagnostic_input)
        diagnostic_repo = build(diagnostic_input, output / 'diagnostic-build', float(os.environ['AFD_RSI_TIMEOUT']))
        diagnostic_expected = artifact_identity(diagnostic_repo)
        diagnostic_argv = args + ['--mega-moe', '--reference-outputs', str(reference),
                                 '--pipeline-trace-dir', str(output / 'traces/pipeline')]
        process = execute(diagnostic_argv, diagnostic_repo, output / 'pipeline-process',
                          float(os.environ['AFD_RSI_TIMEOUT']), runtime_env(diagnostic_repo, settings['devices']))
        if process['status'] != 'completed' or not process['cleanup_complete']:
            raise RuntimeError('pipeline diagnostic failed; inspect pipeline-process')
        fixture_result(output / 'pipeline-process/stdout.log', plan['ranks'], diagnostic_expected,
                       True, settings['attn_ranks'], 0, lengths=settings.get('lengths'))
        assert artifact_identity(diagnostic_repo) == diagnostic_expected
        verify_source(diagnostic_repo, read_json(output / 'diagnostic-build/source.lock.json'))
        write_json(output / 'traces/pipeline/identity.json', {
            'production_source_hash': lock['hash'], 'diagnostic_artifacts': diagnostic_expected,
            'diagnostic_source_hash': read_json(output / 'diagnostic-build/source.lock.json')['hash'],
            'overlay': mode,
            'mode': 'eager diagnostic only; production timing artifacts unchanged'})

def case_spread(result):
    """Worst per-case relative spread across repetitions, in percent.

    A paired median gain smaller than the sampling noise cannot be judged, so the
    measurement itself decides whether more repetitions are needed.
    """
    worst = 0.0
    reports = result.get("reports") if isinstance(result, dict) else None
    if not isinstance(reports, list):
        raise ValueError(f'fixture result carries no reports list: {sorted(result or ())}')
    for report in reports:
        by_case = {}
        for sample in report["timing"]:
            by_case.setdefault(sample["case"], []).append(sample["transaction_us"])
        for values in by_case.values():
            if len(values) < 2:
                continue
            middle = statistics.median(values)
            if middle <= 0:
                continue
            worst = max(worst, 100 * (max(values) - min(values)) / middle)
    return worst


def densify(native, argv, settings):
    """Re-measure with more repetitions when the first pass is too noisy."""
    spread = case_spread(native)
    limit = float(os.environ.get("AFD_RSI_SPREAD_LIMIT") or 5.0)
    factor = int(os.environ.get("AFD_RSI_REPEAT_FACTOR") or 3)
    record = {"configured_repeats": settings["repeats"], "worst_case_spread_pct": spread,
              "limit_pct": limit, "used": "configured"}
    if spread > limit and os.environ.get("AFD_RSI_NO_ADAPTIVE_REPEATS") != "1":
        dense = settings["repeats"] * factor
        dense_argv = list(argv)
        dense_argv[dense_argv.index("--repeats") + 1] = str(dense)
        denser = launch("transaction-dense", dense_argv, mega=True, repeats=dense)
        dense_spread = case_spread(denser)
        # Use the declared denser rerun, never choose the quieter result after seeing both.
        record.update(dense_repeats=dense, dense_spread_pct=dense_spread,
                      why=f"first pass spread {spread:.2f}% exceeds {limit:.2f}%",
                      resolved=dense_spread <= limit)
        native = denser
        record["used"] = "dense"
    else:
        record["resolved"] = spread <= limit
    write_json(output / "adaptive-repeats.json", record)
    if not record["resolved"]:
        raise RuntimeError("native timing spread exceeds the configured limit; preserve both trials")
    return native


native = launch("transaction", argv, mega=True, repeats=settings["repeats"] if measured else 0)
if measured:
    native = densify(native, argv, settings)
assert all(digest(Path(name)) == value for name, value in reference_files.items())
artifacts = {name: digest(build_dir / name) for name in plan["artifacts"]}
result = {key: plan[key] for key in ("backend", "workload_id", "scope", "unit", "metric")}
result["ranks"] = []
if measured:
    primary = next(report for report in native["reports"] if report["rank"] == plan["primary_rank"])
    cases = {}
    for sample in primary["timing"]:
        key = (sample["case"], sample["layer"], sample["sparse"], sample["tokens"])
        cases.setdefault(key, []).append(sample["transaction_us"])
    result["case_metrics"] = [
        {"case": case, "layer": layer, "sparse": sparse, "tokens": tokens,
         "median_us": statistics.median(values), "samples_us": values}
        for (case, layer, sparse, tokens), values in sorted(cases.items())
    ]
if sys.argv[1] == "profile":
    result["profiling"] = analyze_profile(output / "traces", settings["attn_ranks"], plan["ranks"])
for report in native["reports"]:
    samples = [s["transaction_us"] for s in report["timing"]]
    # The primary metric is the configured serialized transaction suite mean.
    value = statistics.mean(samples) if samples else None
    result["ranks"].append({"rank": report["rank"], "source_hash": lock["hash"], "artifacts": artifacts,
                            "checks": {"native_reference": True, "actual_loaded_identity": True,
                                       "finite_completion": report["cleanup_complete"]},
                            "value": value, "native_identity": report["identity"],
                            "metric_observed": bool(samples)})
write_json(Path(os.environ["AFD_RSI_RESULT"]), result)
