"""Only complete and compatible evidence may satisfy an acceptance gate."""

import math
from statistics import median

from .contracts import positive


def evaluate(plan: dict, observations: list[dict]) -> dict:
    reasons, incomplete, failed = [], False, False
    samples = {"baseline": [], "candidate": []}
    quality, service_quality = set(), set()
    service_metrics, profiles = {}, set()
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
            if observation.get("service_metrics"):
                service_metrics[side] = observation["service_metrics"]
        elif phase == "profile":
            profiles.add(side)
        elif phase == "benchmark":
            samples[side].append(positive(observation["value"], "measurement"))
    if quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both correctness results required")
    if "service" in plan and service_quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both configured service gates required")
    if plan.get("native", {}).get("profile", False) and "candidate" not in profiles:
        incomplete = True
        reasons.append("qualified candidate profile required")
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
    result = {"verdict": "FAIL" if failed else "INCONCLUSIVE" if incomplete else "PASS",
            "backend": plan["backend"], "scope": plan["scope"], "unit": plan["unit"],
            "gain_pct": gain, "samples": samples, "reasons": reasons,
            "claim": "synthetic controller fixture only" if plan["backend"] == "cpu-synthetic"
            else "limited to configured native and service gates; not production qualification" if "service" in plan
            else "limited to the supplied benchmark and validators; not service qualification"}
    if service_metrics:
        result["service_metrics"] = service_metrics
        result["service_metric_policy"] = "GSM8K is an accuracy gate; prefill TTFT is an auxiliary performance metric. Promotion uses the frozen primary metric."
        if all(side in service_metrics for side in samples):
            before = service_metrics['baseline'].get('performance', [])
            after = service_metrics['candidate'].get('performance', [])
            result['service_performance_comparison'] = [
                {'point': a['id'], 'metric': 'ttft_p50_ms', 'baseline': a['ttft_ms']['p50'],
                 'candidate': b['ttft_ms']['p50'], 'gain_pct': (1 - b['ttft_ms']['p50'] / a['ttft_ms']['p50']) * 100,
                 'claim': 'auxiliary client HTTP measurement; not a promotion gate'}
                for a, b in zip(before, after, strict=True)]
    return result
