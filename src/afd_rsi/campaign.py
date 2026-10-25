"""Frozen manual-candidate campaigns, using only local commands."""

import fcntl
import fnmatch
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path

from . import skills
from .contracts import load_recipe, validate_result
from .evaluate import evaluate
from .evidence import digest, read_json, write_json
from .runner import execute, host_identity, stop_owned
from .workspace import changed_files, copy_source, inventory


def prepare(recipe: Path, run: Path, root: Path) -> Path:
    plan = load_recipe(recipe)
    source, run = Path(plan["source"]), run.resolve()
    if run.is_relative_to(source) or source.is_relative_to(run):
        raise ValueError("campaign and original source must not contain one another")
    if any(run.is_relative_to(root / name) for name in ("skills", "knowledge", "benchmarks", "src")):
        raise ValueError("campaign output must be outside foundation directories")
    skills.validate(root)
    snapshot = inventory(source)
    if not snapshot["files"]:
        raise ValueError("empty target source")
    run.mkdir(parents=True, exist_ok=False)
    for directory in ("skills", "knowledge", "benchmarks", "src"):
        shutil.copytree(root / directory, run / "foundation" / directory,
                        ignore=shutil.ignore_patterns("__pycache__"))
    for side in ("baseline", "candidate"):
        copy_source(source, run / "workspaces" / side, snapshot)
    plan.update(schema_version=1, source_snapshot=snapshot,
                foundation_hash=digest(run / "foundation/skills/manifest.json"),
                controller_hash=inventory(run / "foundation/src")["hash"],
                created_at=time.time(), expires_at=time.time() + plan["budget_seconds"])
    write_json(run / "plan.lock.json", plan)
    (run / "plan.sha256").write_text(digest(run / "plan.lock.json") + "\n")
    write_json(run / "environment.json", {"platform": platform.platform(), "python": sys.version,
               "npu_smi": shutil.which("npu-smi"), "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
               "hardware_qualified": False})
    return run


def load(run: Path, root: Path) -> dict:
    if digest(run / "plan.lock.json") != (run / "plan.sha256").read_text().strip():
        raise ValueError("frozen plan changed")
    plan = read_json(run / "plan.lock.json")
    skills.validate(run / "foundation")
    if digest(run / "foundation/skills/manifest.json") != plan["foundation_hash"]:
        raise ValueError("foundation manifest changed")
    if inventory(root / "src")["hash"] != plan["controller_hash"]:
        raise ValueError("controller differs from frozen campaign; use the matching checkout")
    if inventory(run / "foundation/src")["hash"] != plan["controller_hash"]:
        raise ValueError("frozen controller snapshot changed")
    return plan


@contextmanager
def exclusive(run: Path, *, recovery=False):
    with (run / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("campaign already in use") from error
        if not recovery and (run / "recovery.json").is_file() and read_json(run / "recovery.json").get("health") in ("required", "failed"):
            raise ValueError("unresolved device health barrier; run resume before scheduling")
        for receipt in process_receipts(run) if not recovery else []:
            record = effective_receipt(receipt)
            if record["status"] == "running" or record.get("cleanup_complete") is not True:
                raise ValueError("unfinished process receipt; inspect and reconcile before continuing")
        yield


def process_receipts(run):
    # Only executor receipts have this ownership schema; source trees may contain namesakes.
    for directory in (run / "attempts", run / "search", run / "health"):
        for path in directory.rglob("process.json"):
            if {"source", "staging"}.intersection(path.relative_to(directory).parts):
                continue
            record = read_json(path)
            if isinstance(record, dict) and {"status", "cleanup_complete"} <= record.keys():
                yield path


def effective_receipt(path):
    record = read_json(path)
    recovery = path.with_name("process-recovery.json")
    if recovery.is_file():
        recovered = read_json(recovery)
        if recovered["receipt_hash"] != digest(path):
            raise ValueError("recovered process receipt changed")
        if recovered.get("cleanup_complete") is True:
            return {**record, "status": "recovered_interruption", "cleanup_complete": True}
    return record


def check_candidate(run: Path, plan: dict, source: Path | None = None) -> dict:
    base = inventory(run / "workspaces/baseline")
    candidate = inventory(source or run / "workspaces/candidate")
    if base != plan["source_snapshot"]:
        raise ValueError("frozen baseline source changed")
    changes = changed_files(base, candidate)
    if any(not any(fnmatch.fnmatchcase(name, pattern) for pattern in plan["allowed_files"])
           for name in changes):
        raise ValueError("candidate changed a file outside allowed_files")
    return candidate


def apply_patch(run: Path, patch: Path, root: Path, *, source=None, locked=False) -> None:
    with exclusive(run) if not locked else nullcontext():
        plan = load(run, root)
        source = source or run / "workspaces/candidate"
        check_candidate(run, plan, source)
        with tempfile.TemporaryDirectory(dir=run) as temporary:
            staged = Path(temporary) / "candidate"
            shutil.copytree(source, staged)
            before = inventory(source)
            subprocess.run(["git", f"--git-dir={staged / '.git'}", f"--work-tree={staged}",
                            "apply", str(patch)], cwd=staged, check=True)
            staged_identity = inventory(staged)
            if staged_identity == before:
                raise ValueError("patch made no source changes")
            changes = changed_files(plan["source_snapshot"], staged_identity)
            if any(not any(fnmatch.fnmatchcase(name, pattern) for pattern in plan["allowed_files"])
                   for name in changes):
                raise ValueError("patch changes a file outside allowed_files")
            subprocess.run(["git", f"--git-dir={source / '.git'}", f"--work-tree={source}",
                            "apply", str(patch)], cwd=source, check=True)
            if inventory(source) != staged_identity:
                raise ValueError("applied patch differs from validated patch")


def artifacts(build: Path, names: list[str], *, native=False) -> dict:
    values = {}
    for name in names:
        path = build / name
        if path.is_symlink() or not path.resolve().is_relative_to(build.resolve()):
            raise ValueError("artifact escapes build directory")
        values[name] = digest(path)
    if native:
        from .native import artifact_identity
        expected = read_json(build / "native/artifacts.json")
        if artifact_identity(build / "native/source") != expected:
            raise ValueError("actual native artifacts changed")
    return values


def command(plan: dict, run: Path, attempt: Path, side: str, phase: str,
            identity: dict, build: Path, root: Path, *, source=None, output=None) -> dict:
    source = source or run / "workspaces" / side
    output = output or attempt / side / phase / uuid.uuid4().hex[:12]
    result_path = output / "result.json"
    substitutions = {"{python}": sys.executable, "{repo}": str(source),
                     "{foundation}": str(run / "foundation"), "{build}": str(build)}
    argv = []
    for argument in plan["commands"].get(phase, []):
        for marker, value in substitutions.items():
            argument = argument.replace(marker, value)
        argv.append(argument)
    if phase == "profile":
        argv = [sys.executable, str(run / "foundation/benchmarks/native/campaign_driver.py"), "profile"]
    observation = {"side": side, "phase": phase, "status": "invalid", "trial": str(output.relative_to(run))}
    try:
        load(run, root)
        if inventory(source) != identity:
            raise ValueError("source changed before command")
        expected = read_json(build / "artifacts.lock.json") if phase != "build" else {}
        if phase != "build" and artifacts(build, plan["artifacts"], native="native" in plan) != expected:
            raise ValueError("built artifacts changed before command")
        remaining = plan["expires_at"] - time.time()
        if remaining <= 0:
            raise ValueError("campaign budget exhausted")
        timeout = min(plan["timeout_seconds"], remaining)
        if phase == "service":
            from .service_phase import run_service
            service = run_service(plan, run, output, build, identity,
                                  min(plan["service"]["timeout_seconds"], remaining))
            process = service["process"]
            observation.update(service)
        else:
            process = execute(argv, source, output, timeout, {
            "AFD_RSI_RESULT": str(result_path), "AFD_RSI_BUILD_DIR": str(build),
            "AFD_RSI_SOURCE_HASH": identity["hash"], "AFD_RSI_SIDE": side,
            "AFD_RSI_WORKLOAD_ID": plan["workload_id"],
            "AFD_RSI_PLAN": str(run / "plan.lock.json"), "AFD_RSI_RUN": str(run),
            "AFD_RSI_ATTEMPT": str(attempt), "AFD_RSI_TIMEOUT": str(timeout),
            "PYTHONDONTWRITEBYTECODE": "1",
            })
        observation["process"] = process
        load(run, root)
        if inventory(source) != identity:
            raise ValueError("source changed during command")
        if not process["cleanup_complete"]:
            raise ValueError("process group cleanup incomplete")
        if process["status"] != "completed" and not (phase == "service" and process["status"] == "failed"):
            raise ValueError(f"command {process['status']}")
        actual = artifacts(build, plan["artifacts"], native="native" in plan)
        if phase != "build" and actual != expected:
            raise ValueError("artifacts changed during command")
        if phase == "build":
            write_json(build / "artifacts.lock.json", actual)
            observation.update(status="valid", artifacts=actual)
        elif phase != "service":
            data = read_json(result_path)
            value = validate_result(data, plan, identity["hash"], actual, measured=phase == "benchmark")
            checks = all(all(rank["checks"].values()) for rank in data["ranks"])
            observation.update(status="valid", value=value, checks_passed=checks,
                               result_hash=digest(result_path), artifacts=actual)
            if phase == "profile":
                from .native import analyze_profile
                profiling = analyze_profile(output / "traces", plan["native"]["attn_ranks"], plan["ranks"])
                if data["profiling"] != profiling:
                    raise ValueError("profile analysis differs from device evidence")
                observation.update(profiling=profiling,
                                   profile_files={str(p.relative_to(output)): digest(p)
                                                  for p in (output / "traces").rglob("*") if p.is_file()})
        else:
            observation.update(status="valid", artifacts=actual)
    except (ValueError, OSError, KeyError, TypeError) as error:
        observation.update(status="invalid", reason=str(error))
    if (plan["backend"] == "ascend" and phase in ("correctness", "benchmark", "service", "profile")
            and (observation["status"] != "valid" or observation.get("checks_passed") is False)):
        write_json(run / "recovery.json", {"health": "required", "trial": observation["trial"]})
    observation["receipts"] = {str(p.relative_to(output)): digest(p) for p in output.rglob("process.json")}
    write_json(output / "observation.json", observation)
    return observation


def verify_observation(run, observation, source, identity, build, plan, *, check_runtime=True):
    if check_runtime and observation["status"] == "valid" and inventory(source) != identity:
        raise ValueError("source changed after completed phase")
    if check_runtime and observation["status"] == "valid" and artifacts(build, plan["artifacts"], native="native" in plan) != observation["artifacts"]:
        raise ValueError("completed phase artifacts changed")
    trial = run / observation["trial"]
    for name, value in observation.get("receipts", {}).items():
        if digest(trial / name) != value:
            raise ValueError("completed process receipt changed")
    if "result_hash" in observation and digest(trial / "result.json") != observation["result_hash"]:
        raise ValueError("raw result changed")
    if observation["phase"] == "service" and observation["status"] == "valid":
        from .service_phase import evidence_files, service_metrics
        if (digest(trial / "service/summary.json") != observation["summary_hash"]
                or digest(trial / "loaded-maps.json") != observation["loaded_maps_hash"]
                or evidence_files(trial / "service") != observation["evidence"]):
            raise ValueError("service evidence changed")
        if "service_metrics" in observation:
            summary = read_json(trial / "service/summary.json")
            metrics = service_metrics(summary, read_json(trial / "service/plan.json")["configuration"],
                                      run_dir=trial / 'service', foundation=run / 'foundation') if all(
                value == "passed" for value in summary["test_results"].values()) else {}
            if metrics != observation["service_metrics"]:
                raise ValueError("service metrics differ from evidence")
    if observation["phase"] == "profile" and observation["status"] == "valid":
        from .native import analyze_profile
        actual = {str(p.relative_to(trial)): digest(p) for p in (trial / "traces").rglob("*") if p.is_file()}
        if (actual != observation["profile_files"] or analyze_profile(
                trial / "traces", plan["native"]["attn_ranks"], plan["ranks"]) != observation["profiling"]):
            raise ValueError("profile evidence changed")


def _new_attempt(run, root, kind, *, candidate_source=None, baseline_source=None, name=None, level="full"):
    if level not in ("full", "operator"):
        raise ValueError("unknown validation level")
    plan = load(run, root)
    candidate_source = candidate_source or run / "workspaces/candidate"
    baseline_source = baseline_source or run / "workspaces/baseline"
    identities = {"baseline": check_candidate(run, plan, baseline_source)}
    paths = {"baseline": str(baseline_source)}
    if kind == "comparison":
        identities["candidate"] = check_candidate(run, plan, candidate_source)
        paths["candidate"] = str(candidate_source)
    if any(not Path(path).resolve().is_relative_to(run) for path in paths.values()):
        raise ValueError("attempt source must be owned by the campaign")
    attempt = run / "attempts" / (name or uuid.uuid4().hex)
    staging = run / "preparing" / uuid.uuid4().hex
    staging.mkdir(parents=True)
    metadata = {"kind": kind, "sources": identities, "paths": paths, "validation_level": level}
    write_json(staging / "attempt.lock.json", metadata)
    (staging / "attempt.sha256").write_text(digest(staging / "attempt.lock.json") + "\n")
    write_json(staging / "sources.json", identities)
    write_json(staging / "progress.json", {"status": "running", "steps": []})
    attempt.parent.mkdir(exist_ok=True)
    staging.rename(attempt)
    return attempt


def _continue(run, attempt, root):
    plan = load(run, root)
    if digest(attempt / "attempt.lock.json") != (attempt / "attempt.sha256").read_text().strip():
        raise ValueError("frozen attempt changed")
    metadata = read_json(attempt / "attempt.lock.json")
    if read_json(attempt / "sources.json") != metadata["sources"]:
        raise ValueError("attempt source identities changed")
    state = read_json(attempt / "progress.json")
    sides = list(metadata["sources"])
    level = metadata.get("validation_level", "full")
    jobs = phase_jobs(plan, sides, level=level)
    for index, (side, phase, pair) in enumerate(jobs):
        if (plan.get("validation") == "operator-first" and len(sides) == 2
                and level == "full" and phase in ("service", "profile")):
            gate = evaluate(plan, [step["result"] for step in state["steps"]
                                  if "result" in step and step["phase"] != "build"], level="operator")
            gate_path = attempt / "operator-gate.json"
            if gate_path.exists() and read_json(gate_path) != gate:
                raise ValueError("operator gate changed")
            write_json(gate_path, gate)
            if gate["verdict"] != "PASS":
                break
        source = Path(metadata["paths"][side])
        identity = metadata["sources"][side]
        check_candidate(run, plan, source)
        if inventory(source) != identity:
            raise ValueError("attempt source changed")
        build = attempt / "build" / side
        build.mkdir(parents=True, exist_ok=True)
        if index == len(state["steps"]):
            state["steps"].append({"side": side, "phase": phase,
                                   "trial": str((attempt / side / phase / uuid.uuid4().hex[:12]).relative_to(run))})
            write_json(attempt / "progress.json", state)
        step = state["steps"][index]
        if (step["side"], step["phase"]) != (side, phase):
            raise ValueError("attempt phase order changed")
        trial = run / step["trial"]
        if "result" not in step and (trial / "observation.json").is_file():
            step["result"] = read_json(trial / "observation.json")
        if "result" in step:
            result = step["result"]
            verify_observation(run, result, source, identity, build, plan)
        else:
            # An interrupted, uncommitted phase is retried in a new preserved trial.
            if trial.exists():
                step.setdefault("interrupted_trials", []).append(step["trial"])
                trial = attempt / side / phase / uuid.uuid4().hex[:12]
                step["trial"] = str(trial.relative_to(run))
                if phase == "build" and any(build.iterdir()):
                    preserved = attempt / "abandoned-builds" / f"{side}-{uuid.uuid4().hex}"
                    preserved.parent.mkdir(parents=True, exist_ok=True)
                    build.rename(preserved)
                    build.mkdir()
            write_json(attempt / "progress.json", state)
            result = command(plan, run, attempt, side, phase, identity, build, root, source=source, output=trial)
            if pair is not None:
                result["pair"] = pair
            step["result"] = result
            write_json(attempt / "progress.json", state)
        if result["status"] != "valid" or result.get("checks_passed") is False:
            break
    observations = [step["result"] for step in state["steps"] if "result" in step
                    and (step["phase"] != "build" or step["result"]["status"] != "valid")]
    if metadata["kind"] == "baseline":
        complete = len(state["steps"]) == len(jobs) and all(
            step.get("result", {}).get("status") == "valid" and step["result"].get("checks_passed", True)
            for step in state["steps"])
        write_json(attempt / "baseline.json", {"complete": complete, "observations": observations,
                                               "hardware_qualified": False})
    else:
        verdict = evaluate(plan, observations, level=level)
        write_json(attempt / "observations.json", observations)
        write_json(attempt / "verdict.json", verdict)
        write_json(run / "latest.json", {"attempt": attempt.name, "verdict_hash": digest(attempt / "verdict.json")})
        report(run, root)
    state["status"] = "completed"
    write_json(attempt / "progress.json", state)
    return attempt


def phase_jobs(plan, sides, *, level="full"):
    if level == "operator" or plan.get("validation") == "operator-first":
        jobs = [(side, phase, None) for side in sides for phase in ("build", "correctness")]
        for pair in range(plan["repetitions"]):
            jobs.extend((side, "benchmark", pair) for side in (sides if pair % 2 == 0 else sides[::-1]))
        if level == "full":
            if len(sides) == 2 and "service" in plan:
                jobs.extend((side, "service", None) for side in sides)
            if plan.get("native", {}).get("profile", False):
                jobs.append((sides[-1], "profile", None))
        return jobs
    phases = ["build", "correctness"] + (["service"] if "service" in plan else [])
    jobs = [(side, phase, None) for side in sides for phase in phases]
    if plan.get("native", {}).get("profile", False):
        # Qualify first; profile the incumbent once, then each qualified candidate.
        jobs.append((sides[-1], "profile", None))
    for pair in range(plan["repetitions"]):
        jobs.extend((side, "benchmark", pair) for side in (sides if pair % 2 == 0 else sides[::-1]))
    return jobs


def run_baseline(run: Path, root: Path) -> Path:
    run = run.resolve()
    with exclusive(run):
        return _continue(run, _new_attempt(run, root, "baseline"), root)


def run_comparison(run: Path, root: Path, *, candidate_source=None, baseline_source=None, locked=False, level="full") -> Path:
    run = run.resolve()
    with exclusive(run) if not locked else nullcontext():
        attempt = _new_attempt(run, root, "comparison", candidate_source=candidate_source,
                               baseline_source=baseline_source, level=level)
        return _continue(run, attempt, root)


def recover(run: Path, root: Path):
    plan = load(run, root)
    changed = []
    for path in process_receipts(run):
        record = effective_receipt(path)
        if record["status"] != "running" and record.get("cleanup_complete") is True:
            continue
        if record.get("host_identity") != host_identity():
            raise ValueError("stale receipt belongs to a different host/boot; cannot establish resource ownership")
        cleanup = stop_owned(record, record.get("grace_seconds", 0.5))
        write_json(path.with_name("process-recovery.json"), {
            **cleanup, "receipt_hash": digest(path), "host_identity": host_identity(), "recovered_at": time.time()})
        changed.append(str(path.relative_to(run)))
        if not cleanup["cleanup_complete"]:
            raise ValueError("recovery could not clean owned processes")
    barrier = (run / "recovery.json").is_file() and read_json(run / "recovery.json").get("health") in ("required", "failed")
    pending = any(read_json(path)["status"] != "completed" for path in (run / "attempts").glob("*/progress.json"))
    if plan["backend"] == "ascend" and (changed or barrier or pending):
        health = plan.get("health")
        if not health:
            write_json(run / "recovery.json", {"receipts": changed, "health": "required"})
            raise ValueError("Ascend recovery requires a frozen health command before rescheduling")
        argv = [part.replace("{python}", sys.executable).replace("{foundation}", str(run / "foundation"))
                for part in health["command"]]
        probe = execute(argv, run, run / "health" / uuid.uuid4().hex, health["timeout_seconds"], {})
        if probe["status"] != "completed" or not probe["cleanup_complete"]:
            write_json(run / "recovery.json", {"receipts": changed, "health": "failed", "process": probe})
            raise ValueError("device health check failed; no new trial launched")
    write_json(run / "recovery.json", {"receipts": changed, "health": "passed" if changed or barrier or pending else "not_needed"})
    return changed


def resume(run: Path, root: Path, *, locked=False):
    run = run.resolve()
    with exclusive(run, recovery=True) if not locked else nullcontext():
        recover(run, root)
        pending = [p.parent for p in (run / "attempts").glob("*/progress.json")
                   if read_json(p)["status"] != "completed"]
        if len(pending) > 1:
            raise ValueError("multiple unfinished attempts; reconcile before resuming")
        return _continue(run, pending[0], root) if pending else run


def report(run: Path, root: Path) -> Path:
    plan = load(run, root)
    latest = read_json(run / "latest.json")
    attempt = run / "attempts" / latest["attempt"]
    verdict = read_json(attempt / "verdict.json")
    if digest(attempt / "verdict.json") != latest["verdict_hash"]:
        raise ValueError("verdict changed")
    observations = audit_attempt(run, attempt, root)
    if evaluate_attempt(plan, attempt, observations) != verdict:
        raise ValueError("verdict inconsistent with observations")
    path = run / "report.md"
    shutdown_policy = plan.get("service", {}).get("require_clean_shutdown", True)
    path.write_text(f"# {plan['name']}\n\nVerdict: {verdict['verdict']}\n\n"
                    f"Backend: {plan['backend']}; scope: {plan['scope']}; unit: {plan['unit']}\n\n"
                    f"Require clean service shutdown: {shutdown_policy}; raw lifecycle results are retained.\n\n"
                    f"Claim: {verdict['claim']}\n\n"
                    f"Attempt: {attempt.name}\n\n```json\n{json.dumps(verdict, indent=2)}\n```\n")
    return path


def evaluate_attempt(plan, attempt, observations):
    level = read_json(attempt / "attempt.lock.json").get("validation_level", "full")
    return evaluate(plan, observations, level=level)


def audit_attempt(run, attempt, root):
    plan = load(run, root)
    if digest(attempt / "attempt.lock.json") != (attempt / "attempt.sha256").read_text().strip():
        raise ValueError("frozen attempt changed")
    metadata = read_json(attempt / "attempt.lock.json")
    state = read_json(attempt / "progress.json")
    observations = []
    strict = all(step.get("result", {}).get("status") == "valid" for step in state["steps"])
    jobs = phase_jobs(plan, list(metadata["sources"]), level=metadata.get("validation_level", "full"))
    if len(state["steps"]) > len(jobs):
        raise ValueError("unexpected attempt phases")
    for index, step in enumerate(state["steps"]):
        if (step["side"], step["phase"]) != jobs[index][:2]:
            raise ValueError("committed attempt phase order changed")
        if "result" not in step:
            continue
        result, side = step["result"], step["side"]
        if (result["side"], result["phase"], result["trial"]) != (side, step["phase"], step["trial"]):
            raise ValueError("committed phase identity changed")
        saved = read_json(run / result["trial"] / "observation.json")
        if {key: value for key, value in result.items() if key != "pair"} != saved:
            raise ValueError("committed observation changed")
        verify_observation(run, result, Path(metadata["paths"][side]), metadata["sources"][side],
                           attempt / "build" / side, plan, check_runtime=strict)
        if step["phase"] != "build" or result["status"] != "valid":
            observations.append(result)
    gate = None
    if (attempt / "operator-gate.json").exists():
        gate = evaluate(plan, observations, level="operator")
        if (plan.get("validation") != "operator-first" or metadata.get("validation_level", "full") != "full"
                or read_json(attempt / "operator-gate.json") != gate):
            raise ValueError("operator gate differs from audited evidence")
    if (state["status"] == "completed" and len(state["steps"]) != len(jobs)
            and all(step.get("result", {}).get("status") == "valid"
                    and step["result"].get("checks_passed", True) for step in state["steps"])
            and not (gate is not None and gate["verdict"] != "PASS"
                     and len(state["steps"]) == next((i for i, job in enumerate(jobs)
                         if job[1] in ("service", "profile")), len(jobs)))):
        raise ValueError("completed attempt phases missing")
    if metadata["kind"] == "comparison" and read_json(attempt / "observations.json") != observations:
        raise ValueError("comparison observations changed")
    return observations


def export(run: Path, destination: Path, root: Path, *, best=False) -> Path:
    plan = load(run, root)
    if destination.is_relative_to(run) or destination.is_relative_to(Path(plan["source"])):
        raise ValueError("export destination must be outside campaign and original source")
    source = run / "workspaces/candidate"
    if best:
        from .search import best_candidate, search_report
        source = run / best_candidate(run, root)["source"]
        report_path = search_report(run, root)
    else:
        check_candidate(run, plan)
        latest = read_json(run / "latest.json")
        identity = read_json(run / "attempts" / latest["attempt"] / "sources.json")
        if inventory(source) != identity["candidate"]:
            raise ValueError("candidate changed after evaluation")
        report_path = report(run, root)
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(dir=run) as temporary:
        staged = Path(temporary)
        for name, path in (("baseline", run / "workspaces/baseline"), ("candidate", source)):
            copy_source(path, staged / name, inventory(path))
        diff = subprocess.run(["git", "diff", "--no-index", "--binary", "--", "baseline", "candidate"],
                              cwd=staged, capture_output=True)
    if diff.returncode not in (0, 1):
        raise ValueError(diff.stderr.decode())
    (destination / "change.patch").write_bytes(diff.stdout)
    shutil.copy2(report_path, destination / "report.md")
    if best:
        shutil.copy2(run / "search/best.json", destination / "best.json")
    shutil.copy2(run / "plan.lock.json", destination / "plan.lock.json")
    (destination / "APPLY.md").write_text("From the exact frozen source snapshot root, apply with "
                                          "`git --git-dir=.git --work-tree=. apply -p2 /path/to/change.patch`.\n"
                                          "Export retains rejected/inconclusive status; it does not imply acceptance.\n")
    return destination
