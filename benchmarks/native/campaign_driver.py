"""Frozen real Ascend build/rank-result adapter for the campaign contract."""

import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.native import analyze_profile, artifact_identity, build, fixture_result, runtime_env, verify_source
from afd_rsi.runner import execute
from afd_rsi.workspace import copy_source, inventory

plan = read_json(Path(os.environ["AFD_RSI_PLAN"]))
build_dir = Path(os.environ["AFD_RSI_BUILD_DIR"])
source = Path.cwd()
if sys.argv[1] == "build":
    build(source, build_dir / "native", float(os.environ["AFD_RSI_TIMEOUT"]))
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
args = [sys.executable, "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={plan['ranks']}", str(Path(__file__).with_name("eager_fixture.py")),
        "--boundary", "--tile-boundary", "--random-weights", "--routed-scale", "2.5",
        "--hidden-size", str(settings["hidden"]), "--intermediate-size", str(settings["intermediate"]),
        "--local-experts", str(settings["experts"]), "--attn-ranks", str(settings["attn_ranks"])]
if settings["per_group"]:
    args.append("--per-group")
env = runtime_env(repo, settings["devices"])


def launch(name, argv, *, mega, repeats, reference_run=False):
    process = execute(argv, repo, output / name, float(os.environ["AFD_RSI_TIMEOUT"]), env)
    if process["status"] != "completed" or not process["cleanup_complete"]:
        raise RuntimeError(f"native {name} failed: {process['status']}")
    result = fixture_result(output / name / "stdout.log", plan["ranks"], expected, mega,
                            settings["attn_ranks"], repeats, reference_run,
                            full_graph=settings.get("execution", "eager") == "graph" and not reference_run)
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
    from pipeline_trace import instrument
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
                   True, settings['attn_ranks'], 0)
    assert artifact_identity(diagnostic_repo) == diagnostic_expected
    verify_source(diagnostic_repo, read_json(output / 'diagnostic-build/source.lock.json'))
    write_json(output / 'traces/pipeline/identity.json', {
        'production_source_hash': lock['hash'], 'diagnostic_artifacts': diagnostic_expected,
        'diagnostic_source_hash': read_json(output / 'diagnostic-build/source.lock.json')['hash'],
        'mode': 'eager diagnostic only; production timing artifacts unchanged'})

native = launch("transaction", argv, mega=True, repeats=settings["repeats"] if measured else 0)
assert all(digest(Path(name)) == value for name, value in reference_files.items())
artifacts = {name: digest(build_dir / name) for name in plan["artifacts"]}
result = {key: plan[key] for key in ("backend", "workload_id", "scope", "unit", "metric")}
result["ranks"] = []
if sys.argv[1] == "profile":
    result["profiling"] = analyze_profile(output / "traces", settings["attn_ranks"], plan["ranks"])
for report in native["reports"]:
    samples = [s["transaction_us"] for s in report["timing"]]
    # The primary metric is the serialized seven-case transaction suite mean.
    value = statistics.mean(samples) if samples else None
    result["ranks"].append({"rank": report["rank"], "source_hash": lock["hash"], "artifacts": artifacts,
                            "checks": {"native_reference": True, "actual_loaded_identity": True,
                                       "finite_completion": report["cleanup_complete"]},
                            "value": value, "native_identity": report["identity"],
                            "metric_observed": bool(samples)})
write_json(Path(os.environ["AFD_RSI_RESULT"]), result)
