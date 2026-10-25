"""Synthetic protocol fixtures; no model/NPU performance claims."""
import asyncio
import copy
import json
import math
import sys
from pathlib import Path

import pytest

from afd_rsi.evidence import write_json
from afd_rsi.service_phase import service_metrics

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e import config, scenario_contract, scenarios, performance, planner
    from afd_e2e.cli import parser
    return config, scenario_contract, scenarios, performance, planner, parser


def case(modules, *extra):
    config, _, _, _, _, parser = modules
    return config.from_args(parser().parse_args(['plan', '--model', 'v4-flash', '--model-path', '/prepared/model',
        '--repo', '/prepared/source', '--comm', 'async', '--topology', 'dp3tp4-ep4', '--mrv', 'mrv2',
        '--execution', 'graph', '--dbo', 'on', '--test', 'perf', '--nic', 'lo', '--afd-host', '127.0.0.1',
        '--max-model-len', '65537', '--perf-scenarios', 'fixed-dp-prefill,long-decode', *extra]))


def prom(contract, warm=0, measured=0):
    lines = []
    for dp, length in enumerate(contract['input_lens']):
        values = dict(prompt_tokens_total=length * (warm * contract['warmup_per_dp'] + measured * contract['requests_per_dp']),
                      generation_tokens_total=warm * contract['warmup_per_dp'] * contract['warmup_output_tokens']
                                              + measured * contract['requests_per_dp'] * contract['output_tokens'],
                      num_preemptions_total=0)
        for key, value in values.items():
            lines.append(f'vllm:{key}{{engine="{dp}",model_name="fixture"}} {value}.0')
    return '\n'.join(lines) + '\n'


def records(contract, warmup=False):
    count = contract['warmup_per_dp' if warmup else 'requests_per_dp']
    output = contract['warmup_output_tokens' if warmup else 'output_tokens']
    result = []
    for dp, length in enumerate(contract['input_lens']):
        for index in range(count):
            item = dict(dp_rank=dp, index=index, phase='warmup' if warmup else 'measured',
                        status='passed', input_target=length, input_tokens=length,
                        output_target=output, output_tokens=output,
                        headers={'X-data-parallel-rank': str(dp)}, ttft_ms=100 + dp,
                        stream_chunks=output)
            if not warmup and output > 1:
                # Earlier requests suffer transient stalls before the last request finishes prefill.
                late = dp == contract['dp'] - 1 and index == count - 1
                first = 100 if late else dp + index * 0.1
                times = [first]
                for n in range(1, output):
                    times.append(times[-1] + (1 if n < 21 and not late else 0.25))
                item.update(token_times=times, tpot_ms=(times[-1] - times[0]) * 1000 / (output - 1))
            result.append(item)
    return result


def evidence(modules, directory):
    _, contract_module, _, _, _, _ = modules
    configuration = json.loads(json.dumps(case(modules).to_dict()))
    performance = dict(status='passed', configuration=configuration['performance'], points=[], scenarios=[])
    for name, contract in configuration['performance']['scenario_settings'].items():
        target = directory / 'performance/scenarios' / name
        target.mkdir(parents=True)
        warm, measured = records(contract, True), records(contract)
        (target / 'requests.jsonl').write_text('\n'.join(json.dumps(r) for r in warm + measured[::-1]) + '\n')
        for file, value in [('before-warmup.prom', prom(contract)), ('before.prom', prom(contract, 1)),
                            ('after.prom', prom(contract, 1, 1))]:
            (target / file).write_text(value)
        performance['scenarios'].append(contract_module.summarize(name, contract, measured,
                                                                 prom(contract, 1), prom(contract, 1, 1)))
    return configuration, {'performance': performance}


def test_topology_pinning_plan_and_stream_options(modules):
    plan = modules[4].describe(case(modules))
    assert plan['attention'] == dict(ranks=12, dp=3, tp=4)
    assert plan['ffn'] == dict(ranks=4, dp=4, tp=1, ep=4)
    assert plan['performance_points'] == []
    attention, ffn = plan['services']
    assert len(attention['devices']) == 12 and len(ffn['devices']) == 4
    assert attention['command'][attention['command'].index('--stream-interval') + 1] == '1'
    assert '--enable-dbo' in attention['command'] and '--enable-dbo' not in ffn['command']
    assert '--enforce-eager' in attention['command'] and '--enforce-eager' not in ffn['command']
    assert plan['performance_scenarios']['fixed-dp-prefill']['input_lens'] == [16384, 512, 512]


@pytest.mark.parametrize('extra', [
    ['--perf-scenarios', 'long-decode,long-decode'], ['--perf-scenarios', 'unknown'],
    ['--perf-dp-input-lens', '16384,512'], ['--perf-dp-input-lens', '512,512,512'],
    ['--perf-decode-tokens', '128'], ['--perf-decode-context', '1'], ['--max-model-len', '32768'],
    ['--comm', 'sync'], ['--baseline'],
])
def test_invalid_scenario_contract_rejected_before_launch(modules, extra):
    with pytest.raises(ValueError):
        case(modules, *extra)


def test_exact_dp_counts_and_common_window_tpot(modules):
    _, api, *_ = modules
    contracts = case(modules).performance.scenario_settings
    for name, contract in contracts.items():
        result = api.summarize(name, contract, records(contract), prom(contract, 1), prom(contract, 1, 1))
        assert len(result['per_dp']) == 3
        assert result['counters'][0]['prompt_tokens_total'] == contract['input_lens'][0] * contract['requests_per_dp']
        if name == 'long-decode':
            assert result['common_window_tpot_ms']['p50'] == pytest.approx(250)
            assert result['request_tpot_ms']['p50'] > 250
            assert min(result['window']['tokens_per_request']) >= 128


@pytest.mark.parametrize('bad', ['wrong_dp', 'failed', 'missing', 'preempted', 'counter_mismatch', 'merged', 'window', 'nonfinite'])
def test_invalid_measurements_never_become_metrics(modules, bad):
    api = modules[1]
    contract = case(modules).performance.scenario_settings['long-decode']
    measured = records(contract)
    after = prom(contract, 1, 1)
    if bad == 'wrong_dp': measured[0]['dp_rank'] = 2
    if bad == 'failed': measured[0]['status'] = 'failed'
    if bad == 'missing': measured.pop()
    if bad == 'preempted': after = after.replace('num_preemptions_total{engine="0",model_name="fixture"} 0.0', 'num_preemptions_total{engine="0",model_name="fixture"} 1.0')
    if bad == 'counter_mismatch': after = prom(contract, 1)
    if bad == 'merged': measured[0]['stream_chunks'] = 1
    if bad == 'window': measured[0]['token_times'] = [n * 0.001 for n in range(512)]
    if bad == 'nonfinite': measured[0]['token_times'][5] = math.nan
    with pytest.raises(ValueError):
        api.summarize('long-decode', contract, measured, prom(contract, 1), after)


def test_raw_evidence_reduction_audited_and_compared(modules, tmp_path):
    from afd_rsi.evaluate import evaluate
    config, summary = evidence(modules, tmp_path)
    metrics = service_metrics(summary, config, run_dir=tmp_path)
    assert metrics['scenarios'][1]['common_window_tpot_ms']['p50'] == pytest.approx(250)
    observations = [dict(side=side, phase=phase, status='valid', checks_passed=True, value=value,
                         service_metrics=copy.deepcopy(metrics))
                    for side, value in [('baseline', 10), ('candidate', 8)]
                    for phase in ('correctness', 'service', 'benchmark')]
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=1, min_gain_pct=2, service={})
    result = evaluate(plan, observations)
    assert result['verdict'] == 'PASS'
    assert any(r['metric'] == 'common_window_tpot_ms_p50' for r in result['service_scenario_comparison'])
    summary['performance']['scenarios'][1]['common_window_tpot_ms']['p50'] = 1
    with pytest.raises(ValueError, match='raw requests'):
        service_metrics(summary, config, run_dir=tmp_path)


def test_missing_and_changed_raw_scenario_evidence_rejected(modules, tmp_path):
    config, summary = evidence(modules, tmp_path)
    with pytest.raises(ValueError, match='raw request'):
        service_metrics(summary, config)
    summary['performance']['scenarios'].pop()
    with pytest.raises(ValueError, match='missing'):
        service_metrics(summary, config, run_dir=tmp_path)


def test_multitoken_sse_rejected_for_common_window(modules):
    import time
    performance = modules[3]
    class Stream:
        def raise_for_status(self): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def aiter_text(self):
            yield 'data: ' + json.dumps({'choices': [{'text': 'tokens', 'token_ids': [10, 20]}]}) + '\n\n'
    class Client:
        def stream(self, *args, **kwargs): return Stream()
    with pytest.raises(RuntimeError, match='one token ID'):
        asyncio.run(performance.stream_request(Client(), 'fixture', {'prompt': [1], 'max_tokens': 2},
                    {'sent_at': time.perf_counter(), 'capture_token_times': True}, {}))


def test_real_http_pinning_stream_and_counter_roundtrip(modules, tmp_path):
    httpx = pytest.importorskip('httpx')
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    api, executor = modules[1:3]
    contract = dict(input_lens=[16, 1, 1], output_tokens=1, requests_per_dp=2, warmup_per_dp=1,
                    warmup_output_tokens=1, rps=None, dp=3)
    totals = [[0, 0] for _ in range(3)]
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args): pass
        def send(self, content, content_type):
            self.send_response(200); self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content))); self.end_headers(); self.wfile.write(content)
        def do_GET(self):
            assert self.path == '/metrics'
            with lock:
                text = '\n'.join(f'vllm:{key}{{engine="{dp}"}} {value}' for dp, row in enumerate(totals)
                    for key, value in zip(api.COUNTERS, [*row, 0]))
            self.send(text.encode(), 'text/plain')
        def do_POST(self):
            assert self.path == '/v1/completions'
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            dp = int(self.headers['X-data-parallel-rank'])
            assert len(body['prompt']) == contract['input_lens'][dp] and body['return_token_ids']
            with lock:
                totals[dp][0] += len(body['prompt']); totals[dp][1] += 1
            chunks = [{'choices': [{'text': 'x', 'token_ids': [10], 'finish_reason': None}]},
                      {'choices': [{'text': '', 'finish_reason': 'length'}],
                       'usage': {'prompt_tokens': len(body['prompt']), 'completion_tokens': 1}}]
            self.send((''.join('data: ' + json.dumps(row) + '\n\n' for row in chunks) + 'data: [DONE]\n\n').encode(), 'text/event-stream')
    try:
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    except PermissionError:
        pytest.skip('local sockets unavailable')
    thread = threading.Thread(target=server.serve_forever); thread.start()
    async def run():
        async with httpx.AsyncClient(trust_env=False) as client:
            return await executor.evaluate_scenario(client, f'http://127.0.0.1:{server.server_port}',
                'fixture', [1, 2, 3], 'fixed-dp-prefill', contract, tmp_path, 10)
    try:
        result = asyncio.run(run())
        assert result['status'] == 'passed', result
        assert result['counters'][0]['prompt_tokens_total'] == 32
        assert result['counters'][1]['prompt_tokens_total'] == 2
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=3)


@pytest.mark.parametrize('fail_first', [False, True])
def test_both_scenarios_share_client_and_failures_remain_failed(modules, tmp_path, monkeypatch, fail_first):
    from types import SimpleNamespace
    performance, executor = modules[3], modules[2]
    seen, clients, pauses = [], [], []
    configuration = json.loads(json.dumps(case(modules, '--perf-point-interval', '3').to_dict()))
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['limits'] >= 24
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, *args, **kwargs):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'tokens': [1, 2, 3]})
    async def scenario(client, base, model, vocabulary, name, contract, directory, timeout):
        seen.append(name); clients.append(client)
        saved = json.loads((tmp_path / 'performance.json').read_text())
        assert saved['scenarios'][-1]['id'] == name and saved['scenarios'][-1]['status'] == 'running'
        return dict(id=name, status='failed' if fail_first and len(seen) == 1 else 'passed', error='fixture only')
    async def sleep(seconds): pauses.append(seconds)
    monkeypatch.setitem(sys.modules, 'httpx', SimpleNamespace(AsyncClient=Client, Limits=lambda **kw: kw['max_connections']))
    monkeypatch.setattr(executor, 'evaluate_scenario', scenario)
    monkeypatch.setattr(performance.asyncio, 'sleep', sleep)
    result = asyncio.run(performance.evaluate(configuration, tmp_path))
    assert result['status'] == ('failed' if fail_first else 'passed')
    assert seen == ['fixed-dp-prefill', 'long-decode'] and clients[0] is clients[1]
    assert pauses == [3, 3] and result['points'] == []


def test_audit_uses_the_frozen_scenario_validator(modules, tmp_path):
    import shutil
    config, summary = evidence(modules, tmp_path)
    frozen = tmp_path / 'foundation'
    target = frozen / 'benchmarks/service/afd_e2e/scenario_contract.py'
    target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / 'benchmarks/service/afd_e2e/scenario_contract.py', target)
    with target.open('a') as handle:
        handle.write('\ndef summarize(*args):\n    raise ValueError("selected frozen validator")\n')
    with pytest.raises(ValueError, match='selected frozen validator'):
        service_metrics(summary, config, run_dir=tmp_path, foundation=frozen)
    assert service_metrics(summary, config, run_dir=tmp_path)['scenarios']
