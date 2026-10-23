"""Only complete and compatible evidence may satisfy an acceptance gate."""

import math
from statistics import median

from .contracts import positive


def evaluate(plan: dict, observations: list[dict]) -> dict:
    reasons, incomplete, failed = [], False, False
    samples = {"baseline": [], "candidate": []}
    quality, service_quality = set(), set()
    for observation in observations:
        if observation["status"] != "valid":
            incomplete = True
            reasons.append(observation.get("reason", observation["status"]))
            continue
        side, phase = observation["side"], observation["phase"]
        if not observation["checks_passed"]:
            failed = True
            reasons.append(f"{side} {phase} checks failed")
        if phase == "correctness":
            quality.add(side)
        elif phase == "service":
            service_quality.add(side)
        elif phase == "benchmark":
            samples[side].append(positive(observation["value"], "measurement"))
    if quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both correctness results required")
    if "service" in plan and service_quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both configured service gates required")
    if any(len(values) != plan["repetitions"] for values in samples.values()):
        incomplete = True
        reasons.append("paired measurements incomplete")
    gain = None
    if all(samples.values()):
        before, after = median(samples["baseline"]), median(samples["candidate"])
        gain = (1 - after / before) * 100
        if not math.isfinite(gain):
            gain = None
            incomplete = True
            reasons.append("derived gain is not finite")
        elif gain <= 0 or gain < plan["min_gain_pct"]:
            failed = True
            reasons.append("positive gain meeting the frozen threshold required")
    return {"verdict": "FAIL" if failed else "INCONCLUSIVE" if incomplete else "PASS",
            "backend": plan["backend"], "scope": plan["scope"], "unit": plan["unit"],
            "gain_pct": gain, "samples": samples, "reasons": reasons,
            "claim": "synthetic controller fixture only" if plan["backend"] == "cpu-synthetic"
            else "limited to configured native and service gates; not production qualification" if "service" in plan
            else "limited to the supplied benchmark and validators; not service qualification"}
