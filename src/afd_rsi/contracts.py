"""Small, strict local recipe contract."""

import math
import tomllib
from pathlib import Path

from .router import DEFAULT_WEIGHTS


def positive(value, label, *, integer=False, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric data")
    if (integer and not isinstance(value, int)) or value < 0 or (not zero and value == 0):
        raise ValueError(f"invalid {label}")
    return value


PROVIDERS = {"codex": "codex_agent.py", "deepseek": "deepseek_agent.py"}
# Providers whose adapter takes the per-stage models from the frozen recipe.
# The Codex adapter keeps this repository's fixed model policy instead.
MODEL_POLICY_IN_ADAPTER = {"deepseek"}
DEFAULT_STAGE_MODELS = {"analysis": "DeepSeek-V4.1-Flash-1017",
                        "code": "DeepSeek-V4.1-Flash-1017"}
STAGES = ("analysis", "code")
AGENT_PREFIX = "{foundation}/benchmarks/agents/"


def agent_driver(command) -> str:
    """Relative path of the frozen proposer adapter a search command names."""
    drivers = [part for part in command if part.startswith(AGENT_PREFIX)]
    if len(drivers) != 1:
        raise ValueError(
            "agent command must name exactly one frozen benchmark/agents driver; "
            "declare provider or point at {foundation}/benchmarks/agents/<name>_agent.py")
    relative = drivers[0][len("{foundation}/"):]
    if ".." in Path(relative).parts:
        raise ValueError("agent driver escapes foundation")
    return relative


def dbo_enabled(args) -> bool:
    """Whether service args ask for native DBO, in either spelling."""
    for index, argument in enumerate(args):
        if argument == "--dbo":
            return index + 1 < len(args) and args[index + 1].strip().lower() == "on"
        if argument.startswith("--dbo="):
            return argument.split("=", 1)[1].strip().lower() == "on"
    return False


def stage_models(search: dict):
    """Validate and default the per-stage proposer models."""
    if "models" not in search:
        return
    models = search["models"]
    if not isinstance(models, dict) or set(models) - set(STAGES) or not models:
        raise ValueError(f"search models must name any of {list(STAGES)}")
    for stage, name in models.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"search models.{stage} must be a nonempty string")
    provider_name = search.get("provider")
    if provider_name not in MODEL_POLICY_IN_ADAPTER:
        raise ValueError(
            "search models is only accepted for a provider whose recipe owns the model choice: "
            + ", ".join(sorted(MODEL_POLICY_IN_ADAPTER)))
    search["models"] = {stage: models.get(stage, DEFAULT_STAGE_MODELS[stage])
                        for stage in STAGES}


def provider(search: dict):
    """Resolve and cross-check the proposer provider, defaulting the adapter argv."""
    if "provider" in search and search["provider"] not in PROVIDERS:
        raise ValueError(f"search provider must be one of {sorted(PROVIDERS)}")
    driver = agent_driver(search["agent_command"]) if "agent_command" in search else None
    if "provider" in search:
        expected = f"{AGENT_PREFIX}{PROVIDERS[search['provider']]}"
        if driver is None:
            search["agent_command"] = ["{python}", expected, "--request", "{request}",
                                       "--output", "{response}"]
        elif driver != expected[len("{foundation}/"):]:
            raise ValueError(f"agent command and provider={search['provider']} disagree")
    elif driver is None:
        raise ValueError("search requires agent_command or provider")
    stage_models(search)


def load_recipe(path: Path) -> dict:
    config = tomllib.loads(path.read_text())
    allowed = {"name", "backend", "source", "workload_id", "scope", "unit", "metric",
               "ranks", "primary_rank", "repetitions", "timeout_seconds", "budget_seconds",
               "min_gain_pct", "allowed_files", "artifacts", "commands"}
    optional = {"service", "search", "native", "health", "validation"}
    if not allowed <= set(config) or set(config) - allowed - optional:
        raise ValueError(f"recipe keys missing/unknown: {sorted(allowed - set(config) | (set(config) - allowed - optional))}")
    for key in ("name", "workload_id", "scope", "unit", "metric"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"invalid {key}")
    if config.get("validation", "full") not in ("full", "operator-first"):
        raise ValueError("validation must be full or operator-first")
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
        optional = {"execution", "profile", "lengths", "max_case_regression_pct"}
        if (set(native) - optional != {"devices", "hidden", "intermediate", "experts",
                                       "attn_ranks", "warmups", "repeats", "per_group"}):
            raise ValueError("invalid native settings")
        if native.get("execution", "eager") not in ("eager", "graph"):
            raise ValueError("native execution must be eager or graph")
        if type(native.get("profile", False)) is not bool:
            raise ValueError("native profile must be boolean")
        if "max_case_regression_pct" in native:
            positive(native["max_case_regression_pct"], "max_case_regression_pct", zero=True)
            if native["repeats"] < 2:
                raise ValueError("case regression screening needs at least two repeats")
        for key in ("hidden", "intermediate"):
            positive(native[key], key, integer=True)
            if native[key] % 256 or not 256 <= native[key] <= 8192:
                raise ValueError("invalid native matrix width")
        for key in ("experts", "attn_ranks", "repeats"):
            positive(native[key], key, integer=True)
        positive(native["warmups"], "warmups", integer=True, zero=True)
        if "lengths" in native:
            # Deployed sequence coverage, bounded by what one op can receive.
            values = native["lengths"]
            if isinstance(values, str):
                values = [item for item in values.split(",") if item.strip()]
            if (not isinstance(values, list) or not values
                    or not all(isinstance(item, (int, str)) and str(item).strip().isdigit()
                               for item in values)
                    or not all(1 <= int(item) <= 16384 for item in values)):
                raise ValueError("native lengths must be integers within [1, 16384]")
        if (native["attn_ranks"] not in (1, 2, 4) or not 4 <= native["experts"] <= 256
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
        if not required <= set(service) or set(service) - required - {"require_clean_shutdown", "max_regression_pct", "primary", "screen_scenarios"}:
            raise ValueError("invalid service settings")
        relative_path(service["repo"], "service repo")
        argv_list(service["args"], "service args")
        if "primary" in service:
            primary = service["primary"]
            components = primary if isinstance(primary, list) else [primary]
            if (not components or config.get("validation") != "operator-first"
                    or any(not isinstance(item, dict) or set(item) != {"id", "metric"}
                           or not isinstance(item["id"], str) or not item["id"]
                           or not isinstance(item["metric"], str)
                           or item["metric"] not in {"ttft_p50_ms", "common_window_tpot_ms_p50",
                                                      "input_tokens_per_s", "common_window_output_tokens_per_s"}
                           for item in components)
                    or len({(item["id"], item["metric"]) for item in components}) != len(components)):
                raise ValueError("service primary requires an operator-first performance metric")
            if config["backend"] == "ascend":
                args = service["args"]
                tests = (args[args.index("--test") + 1].split(",")
                         if "--test" in args and args.index("--test") + 1 < len(args) else [])
                if not {"smoke", "gsm8k", "perf"} <= set(tests):
                    raise ValueError("service primary needs smoke, gsm8k and perf in final validation")
        if "screen_scenarios" in service:
            scenarios = service["screen_scenarios"]
            args = service["args"]
            configured = (args[args.index("--perf-scenarios") + 1].split(",")
                          if "--perf-scenarios" in args and args.index("--perf-scenarios") + 1 < len(args) else [])
            if ("primary" not in service or not isinstance(scenarios, list) or not scenarios
                    or not all(isinstance(name, str) for name in scenarios)
                    or len(scenarios) != len(set(scenarios)) or not set(scenarios) <= set(configured)
                    or not all(item["id"] in scenarios for item in components)):
                raise ValueError("screen scenarios must include the configured primary scenario")
        if type(service.get("require_clean_shutdown", True)) is not bool:
            raise ValueError("require_clean_shutdown must be boolean")
        if "max_regression_pct" in service:
            positive(service["max_regression_pct"], "service max_regression_pct", zero=True)
        reserved = {"--repo", "--output", "--baseline", "--python", "--cleanup-timeout"}
        if any(arg.split("=", 1)[0] in reserved or "AFD_RSI_PROCESS_TAG" in arg or "AFD_RSI_ANCESTOR_TAGS" in arg for arg in service["args"]):
            raise ValueError("service repo/output/python/cleanup and process ownership are framework owned")
        execution = "eager"
        for index, argument in enumerate(service["args"]):
            if argument == "--execution":
                execution = service["args"][index + 1] if index + 1 < len(service["args"]) else ""
            elif argument.startswith("--execution="):
                execution = argument.split("=", 1)[1]
        if execution not in ("eager", "graph"):
            raise ValueError("invalid service execution")
        # A campaign here decides whether MegaMoe actually earns its keep. DBO
        # overlaps Attention with FFN work, so an FFN-side improvement is absorbed
        # by the overlap instead of appearing in the measured transaction; it is
        # refused rather than silently flattening every result.
        if dbo_enabled(service["args"]):
            raise ValueError(
                "service args must keep --dbo off: DBO overlaps Attention with FFN and "
                "masks the measured gain; validate benefit without it")
        if "native" in config and execution != config["native"].get("execution", "eager"):
            raise ValueError("native and service execution must match")
        for key in ("timeout_seconds", "cleanup_seconds"):
            positive(service[key], f"service {key}")
    if "search" in config:
        search = config["search"]
        required = {"max_rounds", "max_calls", "agent_timeout_seconds", "max_context_bytes", "experience_dirs"}
        optional = {'goal', 'router', 'explore_max_fraction', 'info_floor', 'margin', 'features',
                    'provider', 'agent_command', 'models', 'global_redesign_round'}
        if not required <= set(search) or set(search) - required - optional:
            raise ValueError("invalid search settings")
        if 'goal' in search and (not isinstance(search['goal'], str) or not search['goal'].strip()):
            raise ValueError('search goal must be a nonempty string')
        provider(search)
        argv_list(search["agent_command"], "agent command")
        for key in ("max_rounds", "max_calls", "max_context_bytes"):
            positive(search[key], key, integer=True)
        if 'global_redesign_round' in search:
            positive(search['global_redesign_round'], 'global_redesign_round', integer=True)
            if search['global_redesign_round'] > search['max_rounds']:
                raise ValueError('global_redesign_round exceeds max_rounds')
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
        routing(search)
    if "health" in config:
        if set(config["health"]) != {"command", "timeout_seconds"}:
            raise ValueError("invalid health settings")
        argv_list(config["health"]["command"], "health command")
        positive(config["health"]["timeout_seconds"], "health timeout")
        command = config["health"]["command"]
        if len(command) < 2 or command[0] != "{python}" or not command[1].startswith("{foundation}/benchmarks/") or ".." in Path(command[1]).parts:
            raise ValueError("health must use a frozen repository benchmark Python driver")
    return config


def routing(search: dict):
    """Validate and default the direction-router settings of a search recipe."""
    if search.get("router", "auto") not in ("auto", "off"):
        raise ValueError("router must be auto or off")
    search["router"] = search.get("router", "auto")
    for key, default in (("explore_max_fraction", 0.25), ("info_floor", 0.0), ("margin", 0.02)):
        value = search.setdefault(key, default)
        positive(value, f"search {key}", zero=True)
        if value > 1:
            raise ValueError(f"search {key} must stay within [0, 1]")
    features = search.setdefault("features", {})
    if not isinstance(features, dict) or set(features) - {"weights"}:
        raise ValueError("search features must only define weights")
    weights = features.setdefault("weights", dict(DEFAULT_WEIGHTS))
    if not isinstance(weights, dict) or set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError(f"search feature weights must name {sorted(DEFAULT_WEIGHTS)}")
    for name, value in weights.items():
        positive(value, f"feature weight {name}")
    # The frozen copy keeps tomllib's insertion order out of the plan digest.
    features["weights"] = {name: float(weights[name]) for name in sorted(DEFAULT_WEIGHTS)}


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
