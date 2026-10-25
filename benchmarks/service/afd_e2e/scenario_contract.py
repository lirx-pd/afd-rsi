# SPDX-License-Identifier: Apache-2.0
"""Pure workload contracts and reductions, shared by execution and evidence audit."""
import math
import re
import statistics

NAMES = {'matrix', 'fixed-dp-prefill', 'long-decode'}
COUNTERS = ('prompt_tokens_total', 'generation_tokens_total', 'num_preemptions_total')


def scenario_settings(args, selected, dp):
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= NAMES:
        raise ValueError('perf-scenarios requires unique matrix,fixed-dp-prefill,long-decode names')
    if set(selected) == {'matrix'}:
        if args.perf_dp_input_lens:
            raise ValueError('perf-dp-input-lens requires fixed-dp-prefill')
        return {}
    if dp not in (2, 3):
        raise ValueError('DP scenarios require async V4 AFD with two or three Attention DPs')
    result = {}
    if 'fixed-dp-prefill' in selected:
        inputs = list(map(int, args.perf_dp_input_lens.split(','))) if args.perf_dp_input_lens else [16384] + [512] * (dp - 1)
        if len(inputs) != dp or min(inputs) <= 0 or len(set(inputs)) < 2:
            raise ValueError('fixed-dp-prefill needs one positive input length per DP and an unequal load')
        result['fixed-dp-prefill'] = dict(input_lens=inputs, output_tokens=1, requests_per_dp=8,
            warmup_per_dp=1, warmup_output_tokens=1, rps=3, dp=dp)
    if 'long-decode' in selected:
        if args.perf_decode_context <= 1 or args.perf_decode_tokens <= 128:
            raise ValueError('long-decode needs context > 1 and output tokens > 128')
        result['long-decode'] = dict(input_lens=[args.perf_decode_context] * dp,
            output_tokens=args.perf_decode_tokens, requests_per_dp=2, warmup_per_dp=2,
            warmup_output_tokens=32, rps=None, dp=dp, window_trim_seconds=1.0,
            min_window_tokens=128)
    return result


def distribution(values):
    values = sorted(values)
    result = {'count': len(values), 'mean': statistics.mean(values) if values else None,
              'min': values[0] if values else None, 'max': values[-1] if values else None}
    for p in (50, 90, 95, 99):
        pos = (len(values) - 1) * p / 100
        result[f'p{p}'] = ((values[math.floor(pos)] * (1 - pos % 1) + values[math.ceil(pos)] * (pos % 1))
                            if values else None)
    return result


def counters(text, dp):
    values = {key: {} for key in COUNTERS}
    for line in text.splitlines():
        match = re.fullmatch(r'vllm:(\w+)\{([^{}]*)\}\s+(\S+)(?:\s+\S+)?', line)
        if not match or match[1] not in values:
            continue
        engine = re.search(r'(?:^|,)\s*engine="([0-9]+)"(?:,|$)', match[2])
        if engine is None:
            raise ValueError('scenario counter lacks a numeric engine label')
        rank, value = int(engine[1]), float(match[3])
        if rank in values[match[1]] or not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError('duplicate or invalid per-engine scenario counter')
        values[match[1]][rank] = value
    if any(set(rows) != set(range(dp)) for rows in values.values()):
        raise ValueError('scenario counters missing expected DP engines')
    return values


def counter_delta(before, after, contract, *, warmup=False):
    old, new = counters(before, contract['dp']), counters(after, contract['dp'])
    count = contract['warmup_per_dp' if warmup else 'requests_per_dp']
    output = contract['warmup_output_tokens' if warmup else 'output_tokens']
    result = []
    for dp, length in enumerate(contract['input_lens']):
        delta = {key: new[key][dp] - old[key][dp] for key in COUNTERS}
        if delta != dict(prompt_tokens_total=count * length, generation_tokens_total=count * output,
                         num_preemptions_total=0):
            raise ValueError(f'DP {dp} counters differ from pinned workload or preemptions occurred: {delta}')
        result.append({'dp_rank': dp, **delta})
    return result


def verify_requests(records, contract, *, warmup=False):
    count = contract['warmup_per_dp' if warmup else 'requests_per_dp']
    output = contract['warmup_output_tokens' if warmup else 'output_tokens']
    if len(records) != contract['dp'] * count:
        raise ValueError('scenario request count mismatch')
    per_dp = []
    for dp, length in enumerate(contract['input_lens']):
        group = [r for r in records if type(r.get('dp_rank')) is int and r['dp_rank'] == dp]
        if len(group) != count or {r.get('index') for r in group} != set(range(count)):
            raise ValueError('scenario DP request identities are missing or duplicated')
        for record in group:
            if (any(type(record.get(key)) is not int for key in ('index', 'input_tokens', 'output_tokens', 'input_target', 'output_target'))
                    or record.get('status') != 'passed' or record.get('input_tokens') != length
                    or record.get('output_tokens') != output or record.get('input_target') != length
                    or record.get('output_target') != output
                    or record.get('headers') != {'X-data-parallel-rank': str(dp)}):
                raise ValueError('scenario request failed or differs from fixed DP contract')
            value = record.get('ttft_ms')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('scenario TTFT must be finite and positive')
        per_dp.append(dict(dp_rank=dp, requests=count, input_tokens=length, output_tokens=output,
                           ttft_ms=distribution([r['ttft_ms'] for r in group])))
    return per_dp


def common_window(records, contract):
    times = [r.get('token_times', []) for r in records]
    for row in times:
        if (len(row) != contract['output_tokens'] or any(isinstance(t, bool) or not isinstance(t, (int, float))
                or not math.isfinite(t) for t in row) or any(b <= a for a, b in zip(row, row[1:]))):
            raise ValueError('decode needs strictly ordered single-token SSE arrival timestamps')
    start = max(row[0] for row in times) + contract['window_trim_seconds']
    end = min(row[-1] for row in times) - contract['window_trim_seconds']
    if end <= start:
        raise ValueError('no common decode-only active window')
    values, counts = [], []
    for row in times:
        kept = [t for t in row if start <= t <= end]
        if len(kept) < contract['min_window_tokens']:
            raise ValueError('insufficient tokens in the common decode window')
        counts.append(len(kept))
        values.append((kept[-1] - kept[0]) * 1000 / (len(kept) - 1))
    return dict(start=start, end=end, tokens_per_request=counts), distribution(values)


def summarize(name, contract, records, before, after):
    records = sorted(records, key=lambda r: (r['dp_rank'], r['index']))
    per_dp = verify_requests(records, contract)
    delta = counter_delta(before, after, contract)
    result = dict(id=name, status='passed', configuration=contract, requests=len(records),
                  per_dp=per_dp, counters=delta,
                  scope='client HTTP/SSE timings on fixed Attention DPs; not native kernel time')
    if name == 'long-decode':
        window, tpot = common_window(records, contract)
        request_values = []
        for record in records:
            value = record.get('tpot_ms')
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError('decode request TPOT must be finite and positive')
            expected = (record['token_times'][-1] - record['token_times'][0]) * 1000 / (contract['output_tokens'] - 1)
            if record.get('stream_chunks') != contract['output_tokens'] or not math.isclose(value, expected, rel_tol=1e-9):
                raise ValueError('decode request TPOT or event count differs from token timestamps')
            request_values.append(value)
        result.update(window=window, common_window_tpot_ms=tpot, request_tpot_ms=distribution(request_values))
    return result
