"""Native eager/FULL graph experiments on a prepared local Ascend host."""

import argparse
import configparser
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

# The FFN-side pull kernel plus the comm kernel it pairs with. The profiler may
# label either one, and older builds fused both into AfdMegaMoeFFN.
MEGA_KERNELS = ("afd_mega_moe_pull_native", "afd_mega_moe_comm")
# Mirrors benchmarks/native/pipeline_trace.py:PULL_STAGES. The overlay ships inside
# the frozen foundation, so the controller keeps its own copy of the code names.
PULL_STAGE_NAMES = {40: 'round_prepare', 41: 'pull_rows', 42: 'gmm1', 43: 'activation',
                    44: 'gmm2', 45: 'combine', 46: 'round_complete', 47: 'sync_in',
                    48: 'sync_out', 49: 'rx_decode'}
PULL_SUBSTAGE_NAMES = {
    60: ('round_prepare', 'rx_select'), 61: ('rx_select', 'rx_header_validation'),
    62: ('round_prepare', 'rx_metadata'), 63: ('round_prepare', 'rx_commit_check'),
    64: ('round_prepare', 'rx_group_plan'), 65: ('round_prepare', 'rx_publish'),
    66: ('round_prepare', 'rx_layer_map'), 70: ('combine', 'output_local'),
    71: ('combine', 'tx_remote'), 72: ('combine', 'output_sync'),
    73: ('combine', 'output_wait_gmm2'), 74: ('gmm1', 'gmm1_input_wait'),
    75: ('gmm2', 'gmm2_activation_wait')}
MEGA_TASK_STEMS = ("afdmegamoepullnative", "afdmegamoecomm", "afdmegamoeffn")

from . import skills
from .evidence import digest, read_json, write_json
from .runner import execute
from .workspace import copy_source, inventory


def verify_source(repo: Path, snapshot: dict):
    for name, info in snapshot["files"].items():
        if digest(repo / name) != info["sha256"] or bool((repo / name).stat().st_mode & 0o111) != info["executable"]:
            raise ValueError(f"source differs from build input: {name}")


def artifact_identity(repo: Path) -> dict:
    bindings = list((repo / "afd_plugin").glob("_C_ascend*.so"))
    if len(bindings) != 1:
        raise ValueError("need exactly one in-place AFD native binding")
    vendor = repo / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    entrypoints = repo / "vllm_afd_plugin.egg-info/entry_points.txt"
    config = configparser.ConfigParser()
    if (not config.read(entrypoints)
            or config.get("vllm.general_plugins", "afd", fallback=None)
            != "afd_plugin:register_afd"):
        raise ValueError("isolated build does not expose the AFD vLLM plugin entry point")
    files = [repo / "afd_plugin/__init__.py", *bindings, vendor / "op_api/lib/libcust_opapi.so"]
    kernel_root = vendor / "op_impl/ai_core/tbe/kernel/ascend910_93"
    # One Mega transaction is the pull/FFN kernel plus the comm kernel.
    kernels = [p for name in MEGA_KERNELS for p in sorted((kernel_root / name).glob("*"))
               if p.suffix in (".o", ".json")]
    if not any(p.suffix == ".o" for p in kernels):
        raise ValueError("MegaMoe device binary missing")
    return {"files": {str(p.relative_to(repo)): digest(p) for p in files},
            "kernel_artifacts": {str(p.relative_to(repo)): digest(p) for p in kernels}}


def runtime_env(repo: Path, devices: str) -> dict:
    vendor = repo / "afd_plugin/_cann_ops_custom/vendors/afd-plugin"
    return {"PYTHONPATH": os.pathsep.join([str(repo), os.environ.get("PYTHONPATH", "")]),
            "VLLM_PLUGINS": "ascend,afd", "SOC_VERSION": "910c", "AFD_RSI_TARGET": str(repo),
            "ASCEND_RT_VISIBLE_DEVICES": devices,
            "ASCEND_CUSTOM_OPP_PATH": str(vendor),
            "AFD_CUST_OPAPI_LIB_PATH": str(vendor / "op_api/lib/libcust_opapi.so"),
            "LD_LIBRARY_PATH": str(vendor / "op_api/lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", ""),
            "PYTHONDONTWRITEBYTECODE": "1"}


def toolchain_fingerprint() -> dict:
    """Everything outside the source tree that a rebuild would pick up."""
    from importlib import metadata
    versions = {}
    for name in ("torch", "torch-npu", "setuptools", "numpy"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version.split()[0], "abiflags": getattr(sys, "abiflags", ""),
            "packages": versions, "soc_version": "910c",
            "cann": os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_OPP_PATH")}


def build_key(snapshot: dict) -> str:
    return hashlib.sha256(json.dumps(
        {"source": snapshot["hash"], "toolchain": toolchain_fingerprint(),
         "build_recipe": "vllm-entrypoint-1"},
        sort_keys=True).encode()).hexdigest()


CACHE_SKIP = ("source/build", "source/csrc/npu/build")


def cache_copy(src: Path, dst: Path) -> None:
    """Copy a build tree without the compiler's intermediate directories.

    Those trees hold temporary objects and dangling symlinks (a removed
    `liboptiling.so` is a real case), so dereferencing them both wastes space and
    fails the copy; nothing in `artifact_identity` lives there.
    """
    base = Path(src)

    def ignore(directory, names):
        relative = Path(directory).relative_to(base)
        return {name for name in names if str(relative / name) in CACHE_SKIP}

    shutil.copytree(base, dst, symlinks=True, ignore=ignore)


def build(source: Path, output: Path, timeout: float, cache_root: Path | None = None):
    """Build in an isolated copy, reusing a verified identical build when allowed.

    A cache hit is never trusted: the cached tree is copied and its artifacts are
    re-hashed against a fresh identity calculation, so a stale or corrupted entry
    fails loudly instead of becoming evidence. Failing to *populate* the cache is
    not fatal — a completed build stays usable and only the note is recorded.
    """
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("build output must be separate from input source")
    snapshot = inventory(source)
    key = build_key(snapshot) if cache_root else None
    cached = cache_root / key if key else None
    if cached is not None and (cached / "artifacts.json").is_file():
        cache_copy(cached, output)
        record = read_json(output / "cache.json")
        if record.get("key") != key or record.get("source") != snapshot["hash"]:
            raise ValueError("cached build does not match this source")
        target = output / "source"
        verify_source(target, snapshot)
        if artifact_identity(target) != read_json(output / "artifacts.json"):
            raise ValueError("cached build artifacts changed")
        write_json(output / "build.json", {"status": "reused", "cache_key": key,
                                          "reused_from": str(cached)})
        return target
    output.mkdir(parents=True, exist_ok=False)
    target = output / "source"
    copy_source(source, target, snapshot)
    write_json(output / "source.lock.json", snapshot)
    result = execute([sys.executable, "setup.py", "egg_info", "build_ext", "--inplace"], target,
                     output / "build", timeout,
                     {"SOC_VERSION": "910c", "MAX_JOBS": "8", "AFD_BUILD_JOBS": "8",
                      "AFD_BUILD_ASCEND_OPS": "1", "AFD_SKIP_ACLNN_BUILD": "0",
                      "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN": f"0.0.0+rsi.{snapshot['hash'][:12]}"})
    if result["status"] != "completed" or not result["cleanup_complete"]:
        write_json(output / "build.json", result)
        raise RuntimeError(f"isolated build failed; inspect {output / 'build/stderr.log'}")
    verify_source(target, snapshot)
    artifacts = artifact_identity(target)
    write_json(output / "artifacts.json", artifacts)
    write_json(output / "build.json", result)
    if cached is not None:
        # Populate only after the hashes above are known to be good. A cache that
        # cannot be written must not discard an otherwise valid build.
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            staging = cached.with_name(cached.name + ".tmp")
            if staging.exists():
                shutil.rmtree(staging)
            cache_copy(output, staging)
            write_json(staging / "cache.json", {"key": key, "source": snapshot["hash"],
                                                "toolchain": toolchain_fingerprint()})
            if cached.exists():
                shutil.rmtree(cached)
            staging.rename(cached)
        except (OSError, shutil.Error) as error:
            write_json(output / "cache.json", {"key": key, "populated": False,
                                               "error": str(error)[:2000]})
    return target


def fixture_result(path: Path, ranks: int, expected: dict, mega: bool, attn_ranks: int,
                   repeats: int, reference: bool = False, *, full_graph=False, lengths=None) -> dict:
    records = []
    for line in path.read_text().splitlines():
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if isinstance(data, dict) and data.get("result") == "PASS":
            records.append(data)
    if len(records) != 1:
        raise ValueError("missing or duplicate complete native result")
    reports = records[0]["reports"]
    if (not isinstance(reports, list) or len(reports) != ranks
            or any(not isinstance(r, dict) or type(r.get("rank")) is not int for r in reports)
            or {r["rank"] for r in reports} != set(range(ranks))):
        raise ValueError("native rank evidence incomplete")
    if lengths:
        sizes = lengths.split(",") if isinstance(lengths, str) else lengths
        cases = [(layer, slot == 1, int(tokens)) for tokens in sizes
                 for slot, layer in enumerate((2, 5))]
    else:
        cases = [(2, False, 7), (5, True, 7), (2, False, 7), (5, False, 16),
                 (2, True, 16), (5, True, 129), (2, False, 129)]
    for report in reports:
        if report.get("cleanup_complete") is not True or report.get("mega_moe") is not mega:
            raise ValueError("native mode/cleanup evidence incomplete")
        if {k: report.get("identity", {}).get(k) for k in expected} != expected:
            raise ValueError("actual loaded native identity mismatch")
        if full_graph:
            graph = report.get("graph", {})
            if report.get("execution") != "graph":
                raise ValueError("FULL graph execution evidence missing")
            if report["rank"] >= attn_ranks and (
                    type(graph.get("capture_count")) is not int or graph["capture_count"] not in ((1, 2) if mega else (1,))
                    or type(graph.get("replay_count")) is not int or graph["replay_count"] <= 0
                    or graph.get("python_transaction_disabled") is not True
                    or type(graph.get("stable_exposed_addresses")) is not int or graph["stable_exposed_addresses"] <= 0):
                raise ValueError("FULL capture/replay evidence incomplete")
        attention = report["rank"] < attn_ranks
        if attention and not reference and report.get("layered_reference_close" if mega else "eager_exact_match") is not True:
            raise ValueError("native numerical checks missing")
        samples = report.get("timing", [])
        if len(samples) != (len(cases) * repeats if attention else 0):
            raise ValueError("native timing sample count mismatch")
        for index, sample in enumerate(samples):
            case = index % len(cases)
            if (sample["case"], sample["layer"], sample["sparse"], sample["tokens"]) != (case, *cases[case]):
                raise ValueError("native timing workload mismatch")
            value = sample["transaction_us"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("native timing must be finite and positive")
    return records[0]


def timing_summary(observations: list, pairs: int, attn_ranks: int,
                   sides=("layered", "mega")) -> list:
    trials = {entry["phase"]: entry["result"] for entry in observations}
    comparisons = []
    for rank in range(attn_ranks):
        for case in range(7):
            values = {side: [] for side in sides}
            for pair in range(pairs):
                for side in values:
                    report = next(r for r in trials[f"pair-{pair}-{side}"]["reports"] if r["rank"] == rank)
                    values[side].append(statistics.median(s["transaction_us"] for s in report["timing"] if s["case"] == case))
            gains = [(a - b) / a * 100 for a, b in zip(values[sides[0]], values[sides[1]], strict=True)]
            comparisons.append({"attention_rank": rank, "case": case,
                                "baseline_us_by_trial": values[sides[0]], "candidate_us_by_trial": values[sides[1]],
                                "gain_pct_by_trial": gains, "median_gain_pct": statistics.median(gains)})
    return comparisons


def mega_task(name: str) -> bool:
    """Whether a profiler task name belongs to the Mega transaction."""
    normalized = name.lower().replace("_", "")
    return any(stem in normalized for stem in MEGA_TASK_STEMS)


def profile_evidence(directory: Path, rank_ids) -> list:
    """One device trace per FFN rank, each holding a Mega task.

    Scoping by rank matters: the comm kernel also runs on Attention, so a global
    search would count that rank as FFN evidence and hide a rank that never
    produced a trace.
    """
    evidence = []
    for rank in rank_ids:
        files = sorted((directory / f"rank-{rank}").rglob("kernel_details.csv"))
        if len(files) != 1:
            raise ValueError(f"need one exported device trace for FFN rank {rank}")
        with files[0].open(newline="") as handle:
            names = [row.get("Name", "") for row in csv.DictReader(handle)]
        matches = [name for name in names if mega_task(name)]
        if not matches:
            raise ValueError(f"profiler must contain actual MegaMoe device kernels for FFN rank {rank}")
        evidence.append({"file": str(files[0].relative_to(directory)), "sha256": digest(files[0]),
                         "mega_kernel_count": len(matches), "names": sorted(set(matches))})
    return evidence


def analyze_profile(directory: Path, attn_ranks: int, ranks: int) -> dict:
    """Summarize device tasks without inventing a fused kernel's internal bound."""
    evidence = profile_evidence(directory, range(attn_ranks, ranks))
    summaries = []
    for rank in range(attn_ranks, ranks):
        files = list((directory / f'rank-{rank}').rglob('kernel_details.csv'))
        if len(files) != 1:
            raise ValueError(f'need one exported device trace for FFN rank {rank}')
        tasks = {}
        with files[0].open(newline='') as handle:
            for row in csv.DictReader(handle):
                duration = float(row['Duration(us)'])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError('invalid profiler device task duration')
                item = tasks.setdefault(row['Name'], {'name': row['Name'], 'count': 0, 'total_us': 0})
                item['count'] += 1
                item['total_us'] += duration
        if not any(mega_task(name) for name in tasks):
            raise ValueError(f'MegaMoe device tasks missing for FFN rank {rank}')
        total = sum(t['total_us'] for t in tasks.values())
        top = sorted(tasks.values(), key=lambda t: t['total_us'], reverse=True)[:10]
        for item in top:
            item.update(mean_us=item['total_us'] / item['count'], task_duration_share_pct=100 * item['total_us'] / total)
        summaries.append({'rank': rank, 'device_task_total_us': total, 'top_kernels': top})
    pipeline = directory / 'pipeline'
    internal = analyze_pipeline(pipeline, attn_ranks, ranks) if pipeline.exists() else None
    return {'pipeline': internal, 'scope': 'native synthetic seven-case transactions; separate profiled execution',
            'evidence': evidence, 'ffn_ranks': summaries,
            'analysis': 'Rank tasks by summed device duration. Investigate the largest tasks first. '
                'The all-Mega transaction is one fused pull-native kernel plus its comm kernel, so the '
                'device trace alone cannot separate Cube, Vector, MTE or communication time; the '
                'instrumented diagnostic names those stages when a pipeline trace is present. '
                'active_share_pct excludes only sync_in; rx_select, GMM and output intervals can '
                'still contain waits. Child intervals are nested diagnostics, not additional '
                'kernel time. Compare each child with its parent on the same core rather than '
                'treating summed spans as critical-path latency. '
                'Unequal FFN work is intentional in sparse cases; rank totals do not prove imbalance.',
            'next_step': 'Use source tiling, tails, buffers and synchronization to propose a minimal falsifiable hypothesis. '
                'Preserve producers/consumers and validate numerics, GSM8K and unprofiled timings. '
                'HTTP prefill TTFT is a separate service metric; do not attribute it to this trace without evidence.'}


def analyze_pipeline(directory: Path, attn_ranks: int, ranks: int) -> dict:
    identities = read_json(directory / 'identity.json')
    summaries = []
    for rank in range(attn_ranks, ranks):
        transactions = read_json(directory / f'rank-{rank}.json')
        if not transactions:
            raise ValueError('empty pipeline trace')
        totals = {}
        for transaction in transactions:
            cores = transaction['cores']
            if len(cores) != 72 or {c['core'] for c in cores} != set(range(72)):
                raise ValueError('incomplete pipeline cores')
            for core in cores:
                for start, end, code, sequence in core['events']:
                    if not core['start'] <= start <= end <= core['end']:
                        raise ValueError('pipeline event outside core lifetime')
                    key = str(code)
                    totals[key] = totals.get(key, 0) + end-start
        selected = []
        for core in transactions[0]['cores']:
            if core['core'] in (0, 4, 8, 28, 47, 48, 60, 71):
                selected.append(dict(core=core['core'], lifetime_cycles=core['end']-core['start'],
                    events=[[start-core['start'], end-core['start'], code, sequence]
                            for start, end, code, sequence in core['events'] if sequence in (0, 1, 2, 3, 4)]))
        summaries.append(dict(rank=rank, transactions=len(transactions), summed_core_cycles=totals,
                              named_stages=named_stage_cycles(totals),
                              substage_breakdown=named_substage_cycles(totals),
                              first_transaction=transactions[0]['work'], selected_cores=selected))
    return dict(identity=identities, ffn_ranks=summaries,
                scope='instrumented eager diagnostic of seven-case fixture; not promotion timing',
                warning='Per-core cycles include waits. Selected events are [start,end,code,sequence], '
                    'relative to each core entry; cross-core clocks are not aligned. '
                    'Summed spans are neither critical-path latency nor PMU activity.')


def named_stage_cycles(totals: dict) -> list:
    """Name the pull-native stage codes so a proposal can address a stage.

    Two shares are reported. ``share_pct`` covers every instrumented cycle, which
    is dominated by ``sync_in`` because the FFN idles until Attention dispatches.
    ``active_share_pct`` drops only ``sync_in``. It still includes source polling,
    GMM dependency waits and output drains; neither share is critical-path latency.
    """
    known = {name: int(totals.get(str(code), 0)) for code, name in PULL_STAGE_NAMES.items()}
    total = sum(known.values())
    if not total:
        return []
    waiting = known.pop('sync_in', 0)
    active = total - waiting
    rows = []
    for name, cycles in sorted(known.items(), key=lambda item: -item[1]):
        if not cycles:
            continue
        rows.append({'stage': name, 'core_cycles': cycles,
                     'share_pct': round(100 * cycles / total, 2),
                     'active_share_pct': round(100 * cycles / active, 2) if active else 0.0})
    if waiting:
        rows.append({'stage': 'sync_in', 'core_cycles': waiting,
                     'share_pct': round(100 * waiting / total, 2),
                     'active_share_pct': None,
                     'note': 'Attention dispatch wait; excluded from active shares'})
    return rows


def named_substage_cycles(totals: dict) -> dict | None:
    """Report nested intervals separately so parent and child time is never added twice."""
    if not any(str(code) in totals for code in PULL_SUBSTAGE_NAMES):
        return None
    known = {name: int(totals.get(str(code), 0)) for code, name in PULL_STAGE_NAMES.items()}
    known.update({name: int(totals.get(str(code), 0))
                  for code, (_, name) in PULL_SUBSTAGE_NAMES.items()})
    rows = []
    for code, (parent, name) in PULL_SUBSTAGE_NAMES.items():
        cycles = known[name]
        if cycles > known[parent]:
            raise ValueError(f'pipeline substage {name} exceeds {parent}')
        if cycles:
            rows.append({'stage': name, 'parent': parent, 'core_cycles': cycles,
                         'parent_share_pct': round(100 * cycles / known[parent], 2)})
    residual = {}
    for parent in dict.fromkeys(parent for parent, _ in PULL_SUBSTAGE_NAMES.values()):
        children = sum(known[name] for p, name in PULL_SUBSTAGE_NAMES.values() if p == parent)
        if children > known[parent]:
            raise ValueError(f'pipeline children exceed {parent}')
        residual[parent] = known[parent] - children
    return {'stages': rows, 'residual_cycles': residual,
            'note': 'Residual includes unmarked work and trace overhead; rx_select residual also includes polling.'}


def write_report(output: Path, summary: dict):
    lines = ["# Native comparison", "", summary["claim"], "",
             f"Numerical consistency: **{summary['correctness']}**", "",
             "Timing is diagnostic; correctness does not establish a performance improvement.", "",
             f"Baseline: {summary.get('baseline_mode', 'layered')}; candidate: MegaMoe.", "",
             "| Attention rank | Case | Baseline median us | Candidate median us | Paired median gain |",
             "| --- | --- | --- | --- | --- |"]
    for case in summary["timing"]:
        baseline = case.get("baseline_us_by_trial", case.get("layered_us_by_trial"))
        candidate = case.get("candidate_us_by_trial", case.get("mega_us_by_trial"))
        lines.append(f"| {case['attention_rank']} | {case['case']} | "
                     f"{statistics.median(baseline):.2f} | "
                     f"{statistics.median(candidate):.2f} | {case['median_gain_pct']:.2f}% |")
    lines.extend(["", "Negative gain means higher latency. No automatic promotion is performed.", "",
                  f"Mega device traces verified: {len(summary['profiler'])} FFN processes.",
                  ("Per-stage pipeline trace present; stage shares are per-core sums, not latency."
                   if any((rank.get('named_stages') for rank in (summary.get('pipeline') or {}).get('ffn_ranks', [])))
                   else "No per-stage pipeline trace in this run; device tasks are the finest evidence."), "",
                  "Evidence: [plan](plan.lock.json), [observations](observations.json), [summary](summary.json).", ""])
    (output / "report.md").write_text("\n".join(lines))


def compare(repo: Path, output: Path, devices: str, root: Path, args):
    execution = getattr(args, "execution", "eager")
    selected = devices.split(",")
    if not all(s.isascii() and s.isdecimal() for s in selected):
        raise ValueError("device IDs must be decimal integers")
    selected = [str(int(s)) for s in selected]
    if len(selected) != args.attn_ranks + 2 or len(set(selected)) != len(selected):
        raise ValueError("native fixture needs attn_ranks+2 distinct local device IDs")
    devices = ",".join(selected)
    baseline = args.baseline_repo.resolve() if args.baseline_repo else repo
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (repo, baseline)):
        raise ValueError("experiment output must be separate from built source")
    skills.validate(root)
    output.mkdir(parents=True, exist_ok=False)
    frozen = output / "foundation"
    for name in ("skills", "knowledge", "benchmarks", "src"):
        shutil.copytree(root / name, frozen / name, ignore=shutil.ignore_patterns("__pycache__"))
    controller = {str(p.relative_to(root)): digest(p) for p in sorted((root / "src").rglob("*.py"))}
    driver = frozen / "benchmarks/native/eager_fixture.py"
    # Ignore generated build directories by locking the original source-file set.
    identities = {}
    for path in dict.fromkeys((repo, baseline)):
        source_lock = path.parent / "source.lock.json"
        snapshot = read_json(source_lock) if source_lock.is_file() else inventory(path)
        verify_source(path, snapshot)
        identities[path] = {"source": snapshot, "artifacts": artifact_identity(path)}
    sides = ("baseline", "candidate") if args.baseline_repo else ("layered", "mega")
    repositories = {sides[0]: baseline, sides[1]: repo}
    write_json(output / "plan.lock.json", {"mode": execution, "scope": "synthetic-weight native A/F transaction",
               "devices": selected, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "sides": {side: {"repo": str(path), **identities[path]} for side, path in repositories.items()},
               "foundation_hash": digest(frozen / "skills/manifest.json"),
               "controller": controller})
    reference = output / "reference"
    observations = []
    for phase, target, mega, measured in [("reference", baseline, False, False)] + (
            [("baseline-correctness", baseline, True, False)] if args.baseline_repo else []) + [
            ("correctness", repo, True, False)] + [
            (f"pair-{i}-{side}", repositories[side], bool(args.baseline_repo) or side == "mega", True)
            for i in range(args.pairs)
            for side in (sides if i % 2 == 0 else sides[::-1])] + (
            ([("profile-baseline", baseline, True, False)] if args.baseline_repo else [])
            + [("profile", repo, True, False)] if args.profile else []):
        argv = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={len(selected)}", str(driver), "--boundary", "--tile-boundary",
                "--hidden-size", str(args.hidden), "--local-experts", str(args.experts),
                "--intermediate-size", str(args.intermediate), "--random-weights", "--routed-scale", "2.5",
                "--attn-ranks", str(args.attn_ranks)]
        if args.per_group:
            argv.append("--per-group")
        if execution == "graph" and phase != "reference":
            argv.append("--full-graph")
        if mega:
            argv.append("--mega-moe")
        if phase == "reference":
            argv.extend(["--save-outputs", str(reference)])
        else:
            argv.extend(["--reference-outputs", str(reference)])
        if measured:
            argv.extend(["--measure", "--warmups", str(args.warmups), "--repeats", str(args.repeats)])
        if phase.startswith("profile"):
            argv.extend(["--profile-dir", str(output / phase)])
        snapshot, artifacts = identities[target]["source"], identities[target]["artifacts"]
        verify_source(target, snapshot)
        if any(digest(root / name) != value for name, value in controller.items()):
            raise ValueError("controller changed during experiment")
        if artifact_identity(target) != artifacts:
            raise ValueError("native artifacts changed between phases")
        process = execute(argv, target, output / phase, args.timeout, runtime_env(target, devices))
        entry = {"phase": phase, "process": process}
        try:
            if process["status"] != "completed" or not process["cleanup_complete"]:
                raise RuntimeError(f"native {phase} failed; preserve logs and inspect device health")
            entry["result"] = fixture_result(output / phase / "stdout.log", len(selected), artifacts, mega,
                                             args.attn_ranks, args.repeats if measured else 0, phase == "reference",
                                             full_graph=execution == "graph" and phase != "reference")
            verify_source(target, snapshot)
            if artifact_identity(target) != artifacts:
                raise ValueError("native artifacts changed during phase")
        except (ValueError, RuntimeError, KeyError, TypeError) as error:
            entry["evidence_error"] = str(error)
            observations.append(entry)
            write_json(output / "observations.json", observations)
            raise ValueError(f"invalid {phase} evidence: {error}") from error
        observations.append(entry)
        write_json(output / "observations.json", observations)
    comparisons = timing_summary(observations, args.pairs, args.attn_ranks, sides)
    profiler = profile_evidence(output / "profile", 2) if args.profile else []
    if args.profile and args.baseline_repo:
        profiler.extend({**item, "phase": "profile-baseline"}
                        for item in profile_evidence(output / "profile-baseline", 2))
    for target, identity in identities.items():
        verify_source(target, identity["source"])
        if artifact_identity(target) != identity["artifacts"]:
            raise ValueError("native artifacts changed before reporting")
    summary = {"correctness": "PASS", "timing": comparisons, "profiler": profiler,
               "baseline_mode": "MegaMoe" if args.baseline_repo else "layered",
               "promotion": "not evaluated; paired diagnostic comparison only",
               "claim": f"native synthetic weights, {execution}, serialized Attention sources; not model/service performance",
               "observations": observations}
    write_json(output / "summary.json", summary)
    write_report(output, summary)
    return output


def main(argv, root):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    builder = sub.add_parser("build")
    builder.add_argument("--source", type=Path, required=True)
    builder.add_argument("--output", type=Path, required=True)
    builder.add_argument("--timeout", type=float, default=3600)
    run = sub.add_parser("compare")
    run.add_argument("--repo", type=Path, required=True)
    run.add_argument("--baseline-repo", type=Path, help="Built MegaMoe baseline; omit for layered vs MegaMoe")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", required=True)
    for name, default in (("hidden", 256), ("intermediate", 512), ("experts", 4),
                          ("attn-ranks", 1), ("pairs", 2), ("warmups", 3), ("repeats", 10)):
        run.add_argument(f"--{name}", type=int, default=default)
    run.add_argument("--execution", choices=("eager", "graph"), default="eager")
    run.add_argument("--per-group", action="store_true")
    run.add_argument("--profile", action="store_true")
    run.add_argument("--timeout", type=float, default=360)
    args = parser.parse_args(argv)
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.action == "build":
        print(build(args.source.resolve(), args.output.resolve(), args.timeout))
    else:
        if args.attn_ranks not in (1, 2) or min(args.pairs, args.repeats) <= 0 or args.warmups < 0:
            raise ValueError("invalid native repetition/topology settings")
        if any(v % 256 or not 256 <= v <= 8192 for v in (args.hidden, args.intermediate)) or not 4 <= args.experts <= 256:
            raise ValueError("invalid native shape settings")
        print(compare(args.repo.resolve(), args.output.resolve(), args.device, root, args))
    return 0
