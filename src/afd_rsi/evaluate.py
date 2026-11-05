"""Only complete and compatible evidence may satisfy an acceptance gate."""

import math
from statistics import mean, median

from .contracts import positive


def evaluate(plan: dict, observations: list[dict], *, level="full") -> dict:
    if level not in ("full", "operator"):
        raise ValueError("unknown validation level")
    service_primary = plan.get("service", {}).get("primary")
    reasons, incomplete, failed = [], False, False
    samples = {"baseline": [], "candidate": []}
    case_trials = {"baseline": [], "candidate": []}
    quality, service_quality = set(), set()
    service_metrics, profiles = {}, set()
    for observation in observations:
        if level == "operator" and (observation["phase"] == "profile" or
                                    (observation["phase"] == "service" and not service_primary)):
            continue
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
            if "case_metrics" in observation:
                case_trials[side].append(observation["case_metrics"])
    if quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both correctness results required")
    if (level == "full" or service_primary) and "service" in plan and service_quality != {"baseline", "candidate"}:
        incomplete = True
        reasons.append("both configured service gates required")
    if level == "full" and plan.get("native", {}).get("profile", False) and "candidate" not in profiles:
        incomplete = True
        reasons.append("qualified candidate profile required")
    if any(len(values) != plan["repetitions"] for values in samples.values()):
        incomplete = True
        reasons.append("paired measurements incomplete")
    native_gain = None
    if all(samples.values()):
        before, after = median(samples["baseline"]), median(samples["candidate"])
        native_gain = (1 - after / before) * 100
        if not math.isfinite(native_gain):
            native_gain = None
            incomplete = True
            reasons.append("derived native gain is not finite")
        elif not service_primary and (native_gain <= 0 or native_gain < plan["min_gain_pct"]):
            failed = True
            reasons.append("positive gain meeting the frozen threshold required")
    case_comparison = []
    case_limit = plan.get("native", {}).get("max_case_regression_pct")
    if case_limit is not None:
        if any(len(case_trials[side]) != len(samples[side]) for side in samples):
            incomplete = True
            reasons.append("operator case timing evidence incomplete")
        elif all(case_trials.values()):
            reference = case_trials["baseline"][0]
            if not reference or any(len(trial) != len(reference) for trials in case_trials.values() for trial in trials):
                incomplete = True
                reasons.append("operator case timing shapes differ")
            else:
                for index, base_case in enumerate(reference):
                    identity = {key: base_case[key] for key in ("case", "layer", "sparse", "tokens")}
                    if any(any(trial[index][key] != value for key, value in identity.items())
                           for trials in case_trials.values() for trial in trials):
                        incomplete = True
                        reasons.append("operator case workload differs")
                        break
                    before = [positive(trial[index]["median_us"], "operator case") for trial in case_trials["baseline"]]
                    after = [positive(trial[index]["median_us"], "operator case") for trial in case_trials["candidate"]]
                    gains = [(1 - new / old) * 100 for old, new in zip(before, after, strict=True)]
                    case_comparison.append({**identity, "baseline_us": before, "candidate_us": after,
                                            "gain_pct_by_trial": gains, "median_gain_pct": median(gains)})
                if not service_primary and any(row["median_gain_pct"] < -case_limit for row in case_comparison):
                    failed = True
                    reasons.append("operator case regression exceeds the frozen limit")
    result = {"verdict": "FAIL" if failed else "INCONCLUSIVE" if incomplete else "PASS",
            "validation_level": level,
            "backend": plan["backend"], "scope": plan["scope"], "unit": plan["unit"],
            "gain_pct": None if service_primary else native_gain, "samples": samples, "reasons": reasons,
            "claim": "synthetic controller fixture only" if plan["backend"] == "cpu-synthetic"
            else "limited to configured native and service gates; not production qualification" if "service" in plan
            else "limited to the supplied benchmark and validators; not service qualification"}
    if level == "operator":
        result["claim"] = ("synthetic controller fixture only" if plan["backend"] == "cpu-synthetic" else
                           "real model service performance screen; fresh confirmation required; not promotable"
                           if service_primary else
                           "native operator numerical and timing screen only; service deferred; not promotable")
    if service_primary:
        components = service_primary if isinstance(service_primary, list) else [service_primary]
        result["primary_source"] = "service"
        result["native_gain_pct"] = native_gain
        result["native_samples"] = samples
        result["scope"] = components[0]["id"] if len(components) == 1 else "service-scenario-mean"
        result["unit"] = ("tokens/s" if components[0]["metric"].endswith("_per_s") else "ms") if len(components) == 1 else "percent gain"
        result["metric"] = components[0]["metric"] if len(components) == 1 else "mean_gain_pct"
        result["samples"] = {"baseline": [], "candidate": []}
    if case_comparison:
        result["operator_case_comparison"] = case_comparison
    if service_metrics:
        result["service_metrics"] = service_metrics
        result["service_metric_policy"] = ("GSM8K is a final accuracy gate; the configured service metric is primary; native timing is diagnostic."
                                           if service_primary else "GSM8K is an accuracy gate; service latency/throughput are a regression screen when configured, separate from native timing.")
        if all(side in service_metrics for side in samples):
            before = service_metrics['baseline'].get('performance', [])
            after = service_metrics['candidate'].get('performance', [])
            if len(before) != len(after) or any(a['id'] != b['id'] or
                    any(a[key] != b[key] for key in ('phase', 'input_lens', 'output_lens', 'concurrency', 'requests'))
                    for a, b in zip(before, after, strict=True)):
                raise ValueError('service matrix comparison contract mismatch')
            result['service_performance_comparison'] = [
                {'point': a['id'], 'metric': 'ttft_p50_ms', 'baseline': a['ttft_ms']['p50'],
                 'candidate': b['ttft_ms']['p50'], 'gain_pct': (1 - positive(b['ttft_ms']['p50'], 'service TTFT') / positive(a['ttft_ms']['p50'], 'service TTFT')) * 100,
                 'claim': 'client HTTP TTFT; regression screen when configured'}
                for a, b in zip(before, after, strict=True)]
            before = service_metrics['baseline'].get('scenarios', [])
            after = service_metrics['candidate'].get('scenarios', [])
            if len(before) != len(after):
                raise ValueError('service scenario comparison count mismatch')
            comparisons = []
            for a, b in zip(before, after, strict=True):
                if a['id'] != b['id'] or a['configuration'] != b['configuration']:
                    raise ValueError('service scenario comparison contract mismatch')
                rows = [(f"dp{r['dp_rank']}_ttft_p50_ms", r['ttft_ms']['p50'], s['ttft_ms']['p50'])
                        for r, s in zip(a.get('per_dp', []), b.get('per_dp', []), strict=True)]
                if a['configuration'].get('routing') == 'native':
                    rows.extend((f'ttft_{stat}_ms', a['ttft_ms'][stat], b['ttft_ms'][stat]) for stat in ('mean', 'p50', 'p90'))
                    rows.extend((f'group{r["workload_group"]}_ttft_p50_ms', r['ttft_ms']['p50'], s['ttft_ms']['p50'])
                                for r, s in zip(a['per_workload_group'], b['per_workload_group'], strict=True))
                    rows.extend((key, a[key], b[key]) for key in ('requests_per_s', 'input_tokens_per_s',
                                'common_window_output_tokens_per_s') if key in a)
                if 'long-decode' in a['id']:
                    rows.extend((key + '_p50', a[key]['p50'], b[key]['p50'])
                                for key in ('common_window_tpot_ms', 'request_tpot_ms'))
                for metric, old, new in rows:
                    old, new = positive(old, metric), positive(new, metric)
                    comparisons.append(dict(scenario=a['id'], metric=metric, baseline=old, candidate=new,
                        gain_pct=((new / old - 1) if metric.endswith('_per_s') else (1 - new / old)) * 100,
                        claim='auxiliary service HTTP/SSE metric; not native kernel time'))
            if comparisons:
                result['service_scenario_comparison'] = comparisons
    if service_primary:
        rows = result.get('service_performance_comparison', []) + result.get('service_scenario_comparison', [])
        selected = [[row for row in rows if row.get('point', row.get('scenario')) == item['id']
                     and row['metric'] == item['metric']] for item in components]
        if any(len(matches) != 1 for matches in selected):
            reasons.append('configured primary service metric missing or ambiguous')
            if result['verdict'] == 'PASS':
                result['verdict'] = 'INCONCLUSIVE'
        else:
            selected = [{**matches[0], 'claim': 'primary model-service HTTP/SSE metric; not native kernel time'}
                        for matches in selected]
            result['primary_metric'] = selected[0] if len(selected) == 1 else selected
            result['gain_pct'] = mean(row['gain_pct'] for row in selected)
            result['samples'] = ({side: [selected[0][side]] for side in ('baseline', 'candidate')}
                                 if len(selected) == 1 else
                                 {side: [{'id': row.get('point', row.get('scenario')), 'metric': row['metric'],
                                          'value': row[side]} for row in selected]
                                  for side in ('baseline', 'candidate')})
            if result['gain_pct'] <= 0 or result['gain_pct'] < plan['min_gain_pct']:
                reasons.append('positive model service gain meeting the frozen threshold required')
                result['verdict'] = 'FAIL'
    service_limit = plan.get('service', {}).get('max_regression_pct') if level == 'full' or service_primary else None
    if service_limit is not None:
        if level == 'operator' and service_primary:
            primary_rows = result.get('primary_metric', [])
            gated = primary_rows if isinstance(primary_rows, list) else [primary_rows]
        else:
            gated = result.get('service_performance_comparison', []) + [
                row for row in result.get('service_scenario_comparison', [])
                if row['metric'] in {'ttft_p50_ms', 'common_window_tpot_ms_p50',
                                     'input_tokens_per_s', 'common_window_output_tokens_per_s'}
                or row['metric'].endswith('_ttft_p50_ms')]
        if not gated:
            reasons.append('service performance comparison missing')
            if result['verdict'] == 'PASS':
                result['verdict'] = 'INCONCLUSIVE'
        elif any(row['gain_pct'] < -service_limit for row in gated):
            reasons.append('service performance regression exceeds the frozen limit; repeat on model')
            if result['verdict'] == 'PASS':
                result['verdict'] = 'INCONCLUSIVE'
        result['service_regression_limit_pct'] = service_limit
    return result
