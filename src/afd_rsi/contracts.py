"""Small, strict local recipe contract."""

import math
import tomllib
from pathlib import Path


def positive(value, label, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    if (integer and not isinstance(value, int)) or value < 0 or (not zero and value == 0):
        raise ValueError(f"invalid {label}")
    return value


def load_recipe(path: Path) -> dict:
    config = tomllib.loads(path.read_text())
    allowed = {"name", "backend", "source", "workload_id", "scope", "unit", "metric",
               "ranks", "primary_rank", "repetitions", "timeout_seconds", "budget_seconds",
               "min_gain_pct", "allowed_files", "artifacts", "commands"}
    if set(config) != allowed:
        raise ValueError(f"recipe keys missing/unknown: {sorted(set(config) ^ allowed)}")
    for key in ("name", "workload_id", "scope", "unit", "metric"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"invalid {key}")
    if config["backend"] not in ("cpu-synthetic", "ascend"):
        raise ValueError("backend must be cpu-synthetic or ascend")
    source = (path.resolve().parent / config["source"]).resolve(strict=True)
    if not source.is_dir():
        raise ValueError("source must be a directory")
    config["source"] = str(source)
    for key in ("ranks", "repetitions"):
        positive(config[key], key, integer=True)
    positive(config["primary_rank"], "primary_rank", integer=True, zero=True)
    if config["primary_rank"] >= config["ranks"]:
        raise ValueError("primary_rank must be in ranks")
    for key in ("timeout_seconds", "budget_seconds"):
        positive(config[key], key)
    positive(config["min_gain_pct"], "min_gain_pct", zero=True)
    for key in ("allowed_files", "artifacts"):
        values = config[key]
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be nonempty and unique")
        for name in values:
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"{key} must contain relative paths")
    if set(config["commands"]) != {"build", "correctness", "benchmark"}:
        raise ValueError("build, correctness and benchmark commands are required")
    for command in config["commands"].values():
        if not isinstance(command, list) or not command or not all(isinstance(x, str) and x for x in command):
            raise ValueError("commands must be nonempty argv lists")
    for phase in ("correctness", "benchmark"):
        command = config["commands"][phase]
        if len(command) < 2 or command[0] != "{python}" or not command[1].startswith("{foundation}/benchmarks/"):
            raise ValueError("validation must use a frozen repository benchmark Python driver")
        if ".." in Path(command[1]).parts:
            raise ValueError("validation driver escapes foundation")
    return config


def validate_result(data: dict, plan: dict, source_hash: str, artifacts: dict, *, measured: bool) -> float | None:
    if not isinstance(data, dict):
        raise ValueError("result must be an object")
    for key in ("backend", "workload_id", "scope", "unit", "metric"):
        if data.get(key) != plan[key]:
            raise ValueError(f"result {key} mismatch")
    ranks = data.get("ranks")
    if not isinstance(ranks, list) or len(ranks) != plan["ranks"]:
        raise ValueError("missing rank results")
    seen, metric = set(), None
    for rank in ranks:
        if not isinstance(rank, dict):
            raise ValueError("rank result must be an object")
        index = rank.get("rank")
        if type(index) is not int or index not in range(plan["ranks"]) or index in seen:
            raise ValueError("invalid or duplicate rank")
        seen.add(index)
        if rank.get("source_hash") != source_hash or rank.get("artifacts") != artifacts:
            raise ValueError("rank source/artifact mismatch")
        checks = rank.get("checks")
        if not isinstance(checks, dict) or not checks or not all(type(v) is bool for v in checks.values()):
            raise ValueError("checks must be nonempty boolean assertions")
        if measured:
            value = positive(rank.get("value"), "rank metric")
            if index == plan["primary_rank"]:
                metric = value
    return metric
