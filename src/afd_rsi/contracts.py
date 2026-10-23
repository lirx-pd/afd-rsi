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
    optional = {"service", "search", "native", "health"}
    if not allowed <= set(config) or set(config) - allowed - optional:
        raise ValueError(f"recipe keys missing/unknown: {sorted(allowed - set(config) | (set(config) - allowed - optional))}")
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
    if "native" in config:
        native = config["native"]
        if set(native) != {"devices", "hidden", "intermediate", "experts", "attn_ranks", "warmups", "repeats", "per_group"}:
            raise ValueError("invalid native settings")
        for key in ("hidden", "intermediate"):
            positive(native[key], key, integer=True)
            if native[key] % 256 or not 256 <= native[key] <= 8192:
                raise ValueError("invalid native matrix width")
        for key in ("experts", "attn_ranks", "repeats"):
            positive(native[key], key, integer=True)
        positive(native["warmups"], "warmups", integer=True, zero=True)
        if (native["attn_ranks"] not in (1, 2) or not 4 <= native["experts"] <= 256
                or type(native["per_group"]) is not bool or config["backend"] != "ascend"):
            raise ValueError("invalid native topology/quantization")
        devices = native["devices"].split(",")
        if (len(devices) != config["ranks"] or config["ranks"] != native["attn_ranks"] + 2
                or config["primary_rank"] >= native["attn_ranks"]
                or not all(s.isascii() and s.isdecimal() for s in devices)
                or len({int(s) for s in devices}) != len(devices)):
            raise ValueError("invalid native devices")
    if "service" in config:
        service = config["service"]
        required = {"repo", "args", "timeout_seconds", "cleanup_seconds"}
        if not required <= set(service) or set(service) - required - {"require_clean_shutdown"}:
            raise ValueError("invalid service settings")
        if type(service.get("require_clean_shutdown", True)) is not bool:
            raise ValueError("require_clean_shutdown must be boolean")
        relative_path(service["repo"], "service repo")
        argv_list(service["args"], "service args")
        reserved = {"--repo", "--output", "--baseline", "--python", "--cleanup-timeout"}
        if any(arg.split("=", 1)[0] in reserved or "AFD_RSI_PROCESS_TAG" in arg or "AFD_RSI_ANCESTOR_TAGS" in arg for arg in service["args"]):
            raise ValueError("service repo/output/python/cleanup and process ownership are framework owned")
        if "graph" in service["args"] or any(arg.startswith("--execution=") and arg != "--execution=eager" for arg in service["args"]):
            raise ValueError("campaign service currently supports eager only")
        for key in ("timeout_seconds", "cleanup_seconds"):
            positive(service[key], f"service {key}")
    if "search" in config:
        search = config["search"]
        if set(search) != {"agent_command", "max_rounds", "max_calls", "agent_timeout_seconds", "max_context_bytes", "experience_dirs"}:
            raise ValueError("invalid search settings")
        argv_list(search["agent_command"], "agent command")
        for key in ("max_rounds", "max_calls", "max_context_bytes"):
            positive(search[key], key, integer=True)
        positive(search["agent_timeout_seconds"], "agent timeout")
        if not isinstance(search["experience_dirs"], list) or not all(isinstance(item, str) and item for item in search["experience_dirs"]):
            raise ValueError("experience_dirs must be a list")
        search["experience_dirs"] = [str((path.parent / item).resolve(strict=True)) for item in search["experience_dirs"]]
        if not all(Path(item).is_dir() for item in search["experience_dirs"]):
            raise ValueError("experience_dirs must contain directories")
        if search["max_context_bytes"] < 4096:
            raise ValueError("search context budget must be at least 4096 bytes")
        if config["backend"] == "ascend" and "health" not in config:
            raise ValueError("Ascend search requires a device health command")
    if "health" in config:
        if set(config["health"]) != {"command", "timeout_seconds"}:
            raise ValueError("invalid health settings")
        argv_list(config["health"]["command"], "health command")
        positive(config["health"]["timeout_seconds"], "health timeout")
        command = config["health"]["command"]
        if len(command) < 2 or command[0] != "{python}" or not command[1].startswith("{foundation}/benchmarks/") or ".." in Path(command[1]).parts:
            raise ValueError("health must use a frozen repository benchmark Python driver")
    return config


def argv_list(value, label):
    if not isinstance(value, list) or not value or not all(isinstance(s, str) and s for s in value):
        raise ValueError(f"{label} must be a nonempty argv list")


def relative_path(value, label):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"{label} must be a relative path")


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
        if measured and (index == plan["primary_rank"] or rank.get("value") is not None):
            value = positive(rank.get("value"), "rank metric")
            if index == plan["primary_rank"]:
                metric = value
    return metric
