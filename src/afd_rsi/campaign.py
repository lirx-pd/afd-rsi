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
from contextlib import contextmanager
from pathlib import Path

from . import skills
from .contracts import load_recipe, validate_result
from .evaluate import evaluate
from .evidence import digest, read_json, write_json
from .runner import execute
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
def exclusive(run: Path):
    with (run / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("campaign already in use") from error
        for receipt in (run / "attempts").rglob("process.json"):
            record = read_json(receipt)
            if record["status"] == "running" or record.get("cleanup_complete") is not True:
                raise ValueError("unfinished process receipt; inspect and reconcile before continuing")
        yield


def check_candidate(run: Path, plan: dict) -> dict:
    base = inventory(run / "workspaces/baseline")
    candidate = inventory(run / "workspaces/candidate")
    if base != plan["source_snapshot"]:
        raise ValueError("frozen baseline source changed")
    changes = changed_files(base, candidate)
    if any(not any(fnmatch.fnmatchcase(name, pattern) for pattern in plan["allowed_files"])
           for name in changes):
        raise ValueError("candidate changed a file outside allowed_files")
    return candidate


def apply_patch(run: Path, patch: Path, root: Path) -> None:
    with exclusive(run):
        plan = load(run, root)
        check_candidate(run, plan)
        source = run / "workspaces/candidate"
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


def artifacts(build: Path, names: list[str]) -> dict:
    values = {}
    for name in names:
        path = build / name
        if path.is_symlink() or not path.resolve().is_relative_to(build.resolve()):
            raise ValueError("artifact escapes build directory")
        values[name] = digest(path)
    return values


def command(plan: dict, run: Path, attempt: Path, side: str, phase: str,
            identity: dict, build: Path, root: Path) -> dict:
    source = run / "workspaces" / side
    output = attempt / side / phase / uuid.uuid4().hex[:12]
    result_path = output / "result.json"
    substitutions = {"{python}": sys.executable, "{repo}": str(source),
                     "{foundation}": str(run / "foundation"), "{build}": str(build)}
    argv = []
    for argument in plan["commands"][phase]:
        for marker, value in substitutions.items():
            argument = argument.replace(marker, value)
        argv.append(argument)
    observation = {"side": side, "phase": phase, "status": "invalid", "trial": str(output.relative_to(run))}
    try:
        load(run, root)
        if inventory(source) != identity:
            raise ValueError("source changed before command")
        expected = read_json(build / "artifacts.lock.json") if phase != "build" else {}
        if phase != "build" and artifacts(build, plan["artifacts"]) != expected:
            raise ValueError("built artifacts changed before command")
        remaining = plan["expires_at"] - time.time()
        if remaining <= 0:
            raise ValueError("campaign budget exhausted")
        process = execute(argv, source, output, min(plan["timeout_seconds"], remaining), {
            "AFD_RSI_RESULT": str(result_path), "AFD_RSI_BUILD_DIR": str(build),
            "AFD_RSI_SOURCE_HASH": identity["hash"], "AFD_RSI_SIDE": side,
            "AFD_RSI_WORKLOAD_ID": plan["workload_id"],
            "PYTHONDONTWRITEBYTECODE": "1",
        })
        observation["process"] = process
        load(run, root)
        if inventory(source) != identity:
            raise ValueError("source changed during command")
        if not process["cleanup_complete"]:
            raise ValueError("process group cleanup incomplete")
        if process["status"] != "completed":
            raise ValueError(f"command {process['status']}")
        actual = artifacts(build, plan["artifacts"])
        if phase != "build" and actual != expected:
            raise ValueError("artifacts changed during command")
        if phase == "build":
            write_json(build / "artifacts.lock.json", actual)
            observation.update(status="valid", artifacts=actual)
        else:
            data = read_json(result_path)
            value = validate_result(data, plan, identity["hash"], actual, measured=phase == "benchmark")
            checks = all(all(rank["checks"].values()) for rank in data["ranks"])
            observation.update(status="valid", value=value, checks_passed=checks,
                               result_hash=digest(result_path), artifacts=actual)
    except (ValueError, OSError, KeyError, TypeError) as error:
        observation["reason"] = str(error)
    write_json(output / "observation.json", observation)
    return observation


def run_baseline(run: Path, root: Path) -> Path:
    with exclusive(run):
        plan = load(run, root)
        check_candidate(run, plan)
        attempt = run / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        build = attempt / "build/baseline"
        build.mkdir(parents=True)
        observations = []
        for phase in ["build", "correctness"] + ["benchmark"] * plan["repetitions"]:
            result = command(plan, run, attempt, "baseline", phase, plan["source_snapshot"], build, root)
            observations.append(result)
            if result["status"] != "valid" or result.get("checks_passed") is False:
                break
        complete = len(observations) == plan["repetitions"] + 2
        complete = complete and all(x["status"] == "valid" and x.get("checks_passed", True) for x in observations)
        write_json(attempt / "baseline.json", {"complete": complete, "observations": observations,
                                              "hardware_qualified": False})
        return attempt


def run_comparison(run: Path, root: Path) -> Path:
    run = run.resolve()
    with exclusive(run):
        plan = load(run, root)
        candidate = check_candidate(run, plan)
        identities = {"baseline": plan["source_snapshot"], "candidate": candidate}
        attempt = run / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        write_json(attempt / "sources.json", identities)
        observations = []
        for side in identities:
            build = attempt / "build" / side
            build.mkdir(parents=True)
            result = command(plan, run, attempt, side, "build", identities[side], build, root)
            if result["status"] != "valid":
                observations.append(result)
                break
            result = command(plan, run, attempt, side, "correctness", identities[side], build, root)
            observations.append(result)
            if result["status"] != "valid" or not result["checks_passed"]:
                break
        else:
            for repeat in range(plan["repetitions"]):
                sides = ("baseline", "candidate") if repeat % 2 == 0 else ("candidate", "baseline")
                for side in sides:
                    result = command(plan, run, attempt, side, "benchmark", identities[side],
                                     attempt / "build" / side, root)
                    result["pair"] = repeat
                    observations.append(result)
                    if result["status"] != "valid" or not result["checks_passed"]:
                        break
                else:
                    continue
                break
        verdict = evaluate(plan, observations)
        write_json(attempt / "observations.json", observations)
        write_json(attempt / "verdict.json", verdict)
        write_json(run / "latest.json", {"attempt": attempt.name, "verdict_hash": digest(attempt / "verdict.json")})
        report(run, root)
        return attempt


def report(run: Path, root: Path) -> Path:
    plan = load(run, root)
    latest = read_json(run / "latest.json")
    attempt = run / "attempts" / latest["attempt"]
    verdict = read_json(attempt / "verdict.json")
    if digest(attempt / "verdict.json") != latest["verdict_hash"]:
        raise ValueError("verdict changed")
    observations = read_json(attempt / "observations.json")
    for observation in observations:
        if observation["status"] == "valid" and "result_hash" in observation:
            if digest(run / observation["trial"] / "result.json") != observation["result_hash"]:
                raise ValueError("raw result changed")
    if evaluate(plan, observations) != verdict:
        raise ValueError("verdict inconsistent with observations")
    path = run / "report.md"
    path.write_text(f"# {plan['name']}\n\nVerdict: {verdict['verdict']}\n\n"
                    f"Backend: {plan['backend']}; scope: {plan['scope']}; unit: {plan['unit']}\n\n"
                    f"Claim: {verdict['claim']}\n\n"
                    f"Attempt: {attempt.name}\n\n```json\n{json.dumps(verdict, indent=2)}\n```\n")
    return path


def export(run: Path, destination: Path, root: Path) -> Path:
    plan = load(run, root)
    if destination.is_relative_to(run) or destination.is_relative_to(Path(plan["source"])):
        raise ValueError("export destination must be outside campaign and original source")
    check_candidate(run, plan)
    report(run, root)
    latest = read_json(run / "latest.json")
    identity = read_json(run / "attempts" / latest["attempt"] / "sources.json")
    if inventory(run / "workspaces/candidate") != identity["candidate"]:
        raise ValueError("candidate changed after evaluation")
    destination.mkdir(parents=True, exist_ok=False)
    diff = subprocess.run(["git", "diff", "--no-index", "--binary", "--", "baseline", "candidate"],
                          cwd=run / "workspaces", capture_output=True)
    if diff.returncode not in (0, 1):
        raise ValueError(diff.stderr.decode())
    (destination / "change.patch").write_bytes(diff.stdout)
    shutil.copy2(run / "report.md", destination / "report.md")
    shutil.copy2(run / "plan.lock.json", destination / "plan.lock.json")
    (destination / "APPLY.md").write_text("From the exact frozen source snapshot root, apply with "
                                          "`git --git-dir=.git --work-tree=. apply -p2 /path/to/change.patch`.\n"
                                          "Export retains rejected/inconclusive status; it does not imply acceptance.\n")
    return destination
