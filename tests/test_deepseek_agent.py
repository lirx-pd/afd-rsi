"""DeepSeek proposer adapter: contract, truncation, repair and credential hygiene.

The adapter is stdlib-only and imports nothing that needs a POSIX controller, so
these tests run on every host; the transport is substituted, never the network.
"""

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    spec = importlib.util.spec_from_file_location(
        'deepseek_adapter', ROOT / 'benchmarks/agents/deepseek_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATEWAY = 'http://gateway.test/v1'
MODEL = 'DeepSeek-V4.1-Flash-1017'


class FakeTransport:
    """Records requests and replays queued completions or a model listing."""

    def __init__(self, *contents, models=(MODEL,)):
        self.contents = list(contents)
        self.models = list(models)
        self.payloads = []
        self.headers = []
        self.urls = []

    def __call__(self, request, timeout=None):
        self.urls.append(request.full_url)
        self.headers.append(dict(request.header_items()))
        if request.data is None:                       # GET /models
            body = json.dumps({'object': 'list',
                               'data': [{'id': name} for name in self.models]}).encode()
        else:
            self.payloads.append(json.loads(request.data.decode()))
            content = self.contents.pop(0)
            body = json.dumps({'choices': [{'message': {'content': content}}],
                               'usage': {'prompt_tokens': 7, 'completion_tokens': 3}}).encode()

        class Response:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return body

        return Response()


def analysis_json(**overrides):
    value = {'stop': False, 'reason': 'direction found',
             'proposals': [{'hypothesis': 'hidden copy dominates',
                            'category': 'data-movement',
                            'evidence': ['baseline'], 'priority': 2}]}
    value.update(overrides)
    return json.dumps(value)


def code_json(patch='diff --git a/a b/a\n', **overrides):
    value = {'stop': False, 'reason': 'patch ready',
             'proposals': [{'hypothesis': 'hidden copy dominates',
                            'category': 'data-movement', 'evidence': ['baseline'],
                            'priority': 2, 'patch': patch}]}
    value.update(overrides)
    return json.dumps(value)


def request_document(tmp_path, **overrides):
    document = {'schema_version': 1, 'round': 1, 'fingerprint': 'f', 'provider_hash': 'h',
                'deadline': 9e12, 'agent_timeout_seconds': 60, 'max_response_bytes': 4096,
                'strategy': {'mode': 'deep'}, 'evidence': [{'id': 'baseline', 'facts': {}}],
                'allowed_files': ['kernel.cpp'], 'knowledge': 'k' * 50,
                'logs': [{'file': 'a.log', 'tail': 'x' * 50}],
                'source_files': {'kernel.cpp': 'y' * 50}}
    document.update(overrides)
    path = tmp_path / 'request.json'
    path.write_text(json.dumps(document))
    return path


def prepare(tmp_path, monkeypatch, *contents, **env):
    module = adapter()
    transport = FakeTransport(*contents)
    monkeypatch.setattr(module, 'urlopen', transport)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    request = request_document(tmp_path)
    output = tmp_path / 'out/proposal.json'
    args = type('Args', (), {'request': request, 'output': output,
                             'session_dir': tmp_path / 'session'})()
    return module, transport, args, output


def test_two_stage_proposal_and_usage_without_leaking_the_key(tmp_path, monkeypatch):
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              analysis_json(), code_json())
    module.propose(args)
    proposal = json.loads(output.read_text())
    assert proposal['proposals'][0]['patch'].startswith('diff --git')
    turns = json.loads((output.parent / 'usage.json').read_text())['turns']
    assert [turn['stage'] for turn in turns] == ['analysis', 'code']
    # One model serves both stages by default; the split is about what is asked.
    assert {turn['model'] for turn in turns} == {MODEL}
    # Direction analysis reasons at the top level; patch writing runs one below.
    assert [turn['effort'] for turn in turns] == ['max', 'high']
    assert transport.payloads[0]['reasoning_effort'] == 'max'
    assert transport.payloads[1]['reasoning_effort'] == 'high'
    assert turns[0]['prompt_tokens'] == 7 and turns[1]['completion_tokens'] == 3
    assert transport.payloads[0]['response_format'] == {'type': 'json_object'}
    assert transport.payloads[1]['temperature'] == 0
    assert transport.headers[0]['Authorization'] == 'Bearer sk-test-secret'
    for path in tmp_path.rglob('*'):
        if path.is_file():
            assert 'sk-test-secret' not in path.read_text(errors='replace'), path


def test_model_selection_honours_shared_and_per_stage_overrides(tmp_path, monkeypatch):
    module = adapter()
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_MODEL', raising=False)
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_ANALYSIS_MODEL', raising=False)
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_CODE_MODEL', raising=False)
    assert module.model_for('analysis') == module.model_for('code') == MODEL
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_MODEL', 'gateway/other-flash')
    assert module.model_for('code') == 'gateway/other-flash'
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_CODE_MODEL', 'other-code')
    assert module.model_for('code') == 'gateway/other-flash', \
        'the shared override outranks a stage override'
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_MODEL')
    assert module.model_for('code') == 'other-code'
    assert module.model_for('analysis') == MODEL


def test_json_mode_can_be_disabled_for_a_restrictive_gateway(tmp_path, monkeypatch):
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              analysis_json(), code_json(),
                                              **{'AFD_RSI_DEEPSEEK_JSON_MODE': '0'})
    module.propose(args)
    assert 'response_format' not in transport.payloads[0]
    assert 'temperature' not in transport.payloads[0]
    assert output.exists(), 'proposals are still parsed and validated locally'


def test_unsupported_parameter_is_dropped_once_and_recorded(tmp_path, monkeypatch):
    module = adapter()
    transport = FakeTransport(analysis_json(), code_json())
    calls = []

    def failing_then_ok(request, timeout=None):
        payload = json.loads(request.data.decode())
        calls.append(payload)
        if len(calls) == 1:
            raise RuntimeError('gateway HTTP 400: {"error": "response_format is not supported"}')
        return transport(request, timeout)

    monkeypatch.setattr(module, 'urlopen', failing_then_ok)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    args = type('Args', (), {'request': request_document(tmp_path),
                             'output': tmp_path / 'out/proposal.json',
                             'session_dir': tmp_path / 'session'})()
    module.propose(args)
    assert 'response_format' in calls[0] and 'response_format' not in calls[1]
    assert args.output.exists()
    record = json.loads((args.output.parent / 'analysis-events.unsupported-parameters.json').read_text())
    assert 'response_format' in record['dropped']


def test_thinking_level_defaults_and_overrides(monkeypatch):
    module = adapter()
    assert module.effort_for('analysis') == 'max'
    assert module.effort_for('code') == 'high'
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_EFFORT_CODE', 'medium')
    assert module.effort_for('code') == 'medium'
    assert module.effort_for('analysis') == 'max', 'stages stay independent'
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_EFFORT_ANALYSIS', 'low')
    assert module.effort_for('analysis') == 'low'


def test_thinking_level_is_dropped_if_the_gateway_rejects_it(tmp_path, monkeypatch):
    module = adapter()
    transport = FakeTransport(analysis_json(), code_json())
    seen = []

    def wrapped(request, timeout=None):
        payload = json.loads(request.data.decode())
        seen.append(payload)
        if 'reasoning_effort' in payload:
            raise urllib.error.HTTPError(request.full_url, 400, 'error', {},
                                         io.BytesIO(b'{"error":"reasoning_effort unsupported"}'))
        return transport(request, timeout)

    monkeypatch.setattr(module, 'urlopen', wrapped)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    args = type('Args', (), {'request': request_document(tmp_path),
                             'output': tmp_path / 'out/proposal.json',
                             'session_dir': tmp_path / 'session'})()
    module.propose(args)
    assert 'reasoning_effort' in seen[0] and 'reasoning_effort' not in seen[1]
    assert 'response_format' in seen[1], 'only the rejected parameter is dropped'
    record = json.loads((args.output.parent / 'analysis-events.unsupported-parameters.json').read_text())
    assert record['dropped'] == ['reasoning_effort']
    assert args.output.exists()


def test_transient_gateway_errors_are_retried_then_succeed(tmp_path, monkeypatch):
    module = adapter()
    transport = FakeTransport(analysis_json(), code_json())
    calls = []
    monkeypatch.setattr(module.time, 'sleep', lambda _seconds: None)

    def flaky(request, timeout=None):
        calls.append(request.full_url)
        if len(calls) <= 2:
            raise urllib.error.HTTPError(request.full_url, 503, 'unavailable', {}, io.BytesIO(b''))
        return transport(request, timeout)

    monkeypatch.setattr(module, 'urlopen', flaky)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    args = type('Args', (), {'request': request_document(tmp_path),
                             'output': tmp_path / 'out/proposal.json',
                             'session_dir': tmp_path / 'session'})()
    module.propose(args)
    assert args.output.exists()
    assert len(calls) == 4, 'two 503s are retried, then analysis and code succeed'


def test_client_errors_are_not_retried(tmp_path, monkeypatch):
    module = adapter()
    calls = []
    monkeypatch.setattr(module.time, 'sleep', lambda _seconds: None)

    def unauthorized(request, timeout=None):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 401, 'unauthorized', {},
                                     io.BytesIO(b'{"error":"bad key"}'))

    monkeypatch.setattr(module, 'urlopen', unauthorized)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    with pytest.raises(RuntimeError, match='gateway HTTP 401'):
        module.request_json(GATEWAY + '/chat/completions', {'model': 'm'}, 5)
    assert len(calls) == 1, 'a bad key cannot be fixed by retrying'


def test_retry_backoff_is_bounded():
    module = adapter()
    delays = [module.retry_delay(n) for n in range(1, 7)]
    assert delays == sorted(delays)
    assert delays[-1] == module.RETRY_SLEEPS[-1]


def test_truncated_reasoning_escalates_the_budget_instead_of_repeating(tmp_path, monkeypatch):
    """A reasoning model that spends the budget thinking gets a bigger one."""
    module = adapter()
    payloads = []
    truncated = json.dumps({'choices': [{'finish_reason': 'length',
                                        'message': {'content': '', 'reasoning_content': 'thinking'}}],
                            'usage': {'prompt_tokens': 5, 'completion_tokens': 8192}}).encode()

    def body(content):
        return json.dumps({'choices': [{'finish_reason': 'stop',
                                        'message': {'content': content}}],
                           'usage': {'prompt_tokens': 5, 'completion_tokens': 100}}).encode()

    # Call 1 truncates, call 2 answers the analysis, call 3 answers the coding stage.
    scripted = [truncated, body(analysis_json()), body(code_json())]

    def wrapped(request, timeout=None):
        payloads.append(json.loads(request.data.decode()))

        class Response:
            def __init__(inner, data):
                inner._data = data

            def __enter__(inner):
                return inner

            def __exit__(inner, *exc):
                return False

            def read(inner):
                return inner._data

        return Response(scripted[min(len(payloads), len(scripted)) - 1])

    monkeypatch.setattr(module, 'urlopen', wrapped)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    args = type('Args', (), {'request': request_document(tmp_path),
                             'output': tmp_path / 'out/proposal.json',
                             'session_dir': tmp_path / 'session'})()
    module.propose(args)
    assert payloads[0]['max_tokens'] == module.DEFAULT_MAX_TOKENS['analysis']
    assert payloads[1]['max_tokens'] > payloads[0]['max_tokens'], 'the repair raises the budget'
    # The truncated reply is kept as evidence, and the accepted turn records the
    # budget it actually ran with.
    raw = json.loads((args.output.parent / 'analysis-events.response.json').read_text())
    assert raw['choices'][0]['finish_reason'] == 'length'
    turns = json.loads((args.output.parent / 'usage.json').read_text())['turns']
    assert turns[0]['stage'] == 'analysis' and turns[0]['max_tokens'] == payloads[1]['max_tokens']


def test_token_budget_can_be_overridden_and_is_capped():
    module = adapter()
    assert module.token_budget('analysis') == module.DEFAULT_MAX_TOKENS['analysis']
    os_environ = __import__('os').environ
    previous = {key: os_environ.get(key) for key in
                ('AFD_RSI_DEEPSEEK_MAX_TOKENS', 'AFD_RSI_DEEPSEEK_TOKEN_CAP')}
    try:
        os_environ['AFD_RSI_DEEPSEEK_TOKEN_CAP'] = '40000'
        os_environ['AFD_RSI_DEEPSEEK_MAX_TOKENS'] = '999999'
        assert min(module.token_budget('code'), module.token_cap()) == 40000
    finally:
        for key, value in previous.items():
            if value is None:
                os_environ.pop(key, None)
            else:
                os_environ[key] = value

def test_gateway_access_is_direct_and_ignores_environment_proxies(tmp_path, monkeypatch):
    """The proposer talks to the gateway directly, never through a proxy."""
    import http.server
    import threading

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            body = b'{"choices": [{"message": {"content": "ok"}}]}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        # A closed proxy port plus an empty no_proxy: any proxied request fails.
        for name in ('HTTP_PROXY', 'http_proxy', 'HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
            monkeypatch.setenv(name, 'http://127.0.0.1:1')
        for name in ('NO_PROXY', 'no_proxy'):
            monkeypatch.delenv(name, raising=False)
        module = adapter()
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
        value = module.request_json(
            f'http://127.0.0.1:{server.server_port}/chat/completions', {'model': 'm'}, 5)
        assert value['choices'][0]['message']['content'] == 'ok'
        assert module.PROXY_HANDLER.proxies == {}, 'no proxy is configured for the opener'
    finally:
        server.shutdown()
        server.server_close()



def test_analysis_stop_skips_the_coding_call(tmp_path, monkeypatch):
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              analysis_json(stop=True, proposals=[]))
    module.propose(args)
    assert len(transport.payloads) == 1, 'a stop must not spend a second model call'
    assert json.loads(output.read_text())['stop'] is True
    assert len(json.loads((output.parent / 'usage.json').read_text())['turns']) == 1


def test_fenced_json_is_accepted(tmp_path, monkeypatch):
    module, _, args, output = prepare(tmp_path, monkeypatch,
                                      '```json\n' + analysis_json() + '\n```', code_json())
    module.propose(args)
    assert json.loads(output.read_text())['reason'] == 'patch ready'


def test_one_repair_attempt_then_success(tmp_path, monkeypatch):
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              'not json at all', analysis_json(), code_json())
    module.propose(args)
    assert len(transport.payloads) == 3
    repair = transport.payloads[1]['messages']
    assert repair[-1]['role'] == 'user' and 'rejected' in repair[-1]['content']
    turns = json.loads((output.parent / 'usage.json').read_text())['turns']
    assert turns[0]['attempt'] == 1
    assert (output.parent / 'analysis-events.response-repair.json').exists()


def test_invalid_evidence_is_rejected_and_never_written(tmp_path, monkeypatch):
    bad = json.loads(code_json())
    bad['proposals'][0]['evidence'] = ['invented-evidence']
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              analysis_json(), json.dumps(bad), json.dumps(bad))
    with pytest.raises(ValueError, match='exact IDs'):
        module.propose(args)
    assert not output.exists(), 'a rejected proposal must never reach the controller'
    assert len(transport.payloads) == 3, 'one repair attempt, then a visible failure'


def test_missing_credential_fails_before_any_request(tmp_path, monkeypatch):
    module, transport, args, output = prepare(tmp_path, monkeypatch, analysis_json())
    monkeypatch.delenv('DEEPSEEK_API_KEY')
    with pytest.raises(ValueError, match='DEEPSEEK_API_KEY'):
        module.propose(args)
    assert transport.payloads == []
    assert not output.exists()


def test_session_directory_belongs_to_one_campaign(tmp_path, monkeypatch):
    module, _, args, _ = prepare(tmp_path, monkeypatch, analysis_json(), code_json())
    module.propose(args)
    session = json.loads((tmp_path / 'session/session.json').read_text())
    assert session['provider'] == 'deepseek'
    assert session['campaign']['fingerprint'] == 'f'
    args.request = request_document(tmp_path, fingerprint='other')
    with pytest.raises(ValueError, match='another campaign'):
        module.propose(args)


EXPERIMENT_FIELDS = ['bottleneck', 'mechanism', 'prediction', 'falsification', 'risk']


def requirements_request(args):
    document = json.loads(args.request.read_text())
    document['proposal_requirements'] = {'experiment_fields': EXPERIMENT_FIELDS,
                                         'required_evidence_ids': ['baseline']}
    args.request.write_text(json.dumps(document))


def test_analysis_without_a_falsifiable_experiment_never_reaches_coding(tmp_path, monkeypatch):
    # The analysis stage is shape-checked against the same requirement the
    # controller enforces, so an incomplete experiment costs no coding call.
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              analysis_json(), analysis_json())
    requirements_request(args)
    with pytest.raises(ValueError, match='rejected'):
        module.propose(args)
    assert len(transport.payloads) == 2, 'analysis plus one repair, then no coding call'
    assert not output.exists()


def test_complete_experiment_unlocks_coding(tmp_path, monkeypatch):
    experiment = {name: 'controlled fixture' for name in EXPERIMENT_FIELDS}
    analysis = json.loads(analysis_json())
    analysis['proposals'][0]['experiment'] = experiment
    code = json.loads(code_json())
    code['proposals'][0]['experiment'] = experiment
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              json.dumps(analysis), json.dumps(code))
    requirements_request(args)
    module.propose(args)
    assert len(transport.payloads) == 2
    assert json.loads(output.read_text())['proposals'][0]['experiment'] == experiment


def test_empty_experiment_field_is_rejected(tmp_path, monkeypatch):
    experiment = {name: 'controlled fixture' for name in EXPERIMENT_FIELDS}
    experiment['falsification'] = ''
    analysis = json.loads(analysis_json())
    analysis['proposals'][0]['experiment'] = experiment
    module, transport, args, output = prepare(tmp_path, monkeypatch,
                                              json.dumps(analysis), json.dumps(analysis))
    requirements_request(args)
    with pytest.raises(ValueError, match='rejected'):
        module.propose(args)
    assert not output.exists()


def test_oversized_request_is_truncated_in_a_documented_order(tmp_path, monkeypatch):
    module = adapter()
    request = {'evidence': [{'id': 'baseline'}, {'id': 'history-0'}, {'id': 'history-1'}],
               'logs': [{'tail': 'x' * 400}], 'knowledge': 'k' * 400,
               'source_files': {'a.cpp': 'a' * 400, 'b.cpp': 'b' * 400},
               'proposal_requirements': {'required_evidence_ids': ['baseline']}}
    view, dropped = module.truncate(request, 500)
    assert dropped[0] == 'logs' and 'knowledge' in dropped
    assert view['logs'] == [] and view['knowledge'] == ''
    assert 'baseline' in [item['id'] for item in view['evidence']]
    assert view['context_truncation']['dropped'] == dropped
    assert len(json.dumps(view, ensure_ascii=False)) <= 500 or 'source_files' in dropped
    assert set(view['source_files']) | set(view.get('omitted_source_files', [])) == {'a.cpp', 'b.cpp'}


def test_small_request_is_not_touched():
    module = adapter()
    request = {'evidence': [{'id': 'baseline'}], 'logs': []}
    view, dropped = module.truncate(request, 100_000)
    assert dropped == [] and 'context_truncation' not in view


def test_extract_json_handles_prose_and_fences():
    module = adapter()
    assert module.extract_json('{"stop": true}') == {'stop': True}
    assert module.extract_json('here you go: {"stop": true} thanks') == {'stop': True}
    assert module.extract_json('```\n{"stop": true}\n```') == {'stop': True}
    with pytest.raises(ValueError):
        module.extract_json('no object here')


def probed(tmp_path, monkeypatch, *, key='sk-test-secret', base=GATEWAY, models=(MODEL,),
           http_errors=(), persistent=False, contents=('ok', '{"ok": true}')):
    """Run the preflight with a substituted transport; no network is touched."""
    module = adapter()
    transport = FakeTransport(*contents, models=models)
    remaining = list(http_errors)
    monkeypatch.setattr(module.time, 'sleep', lambda _seconds: None)

    def wrapped(request, timeout=None):
        if remaining and (persistent or len(remaining) > 1):
            code, body = remaining[0]
        elif remaining:
            code, body = remaining.pop(0)
        else:
            return transport(request, timeout)
        raise urllib.error.HTTPError(request.full_url, code, 'error', {},
                                     io.BytesIO(body.encode()))

    monkeypatch.setattr(module, 'urlopen', wrapped)
    if key is None:
        monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    else:
        monkeypatch.setenv('DEEPSEEK_API_KEY', key)
    if base is None:
        monkeypatch.delenv('AFD_RSI_DEEPSEEK_BASE_URL', raising=False)
    else:
        monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', base)
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_KEY_ENV', raising=False)
    return module.probe()


def test_preflight_passes_when_the_gateway_serves_the_model(tmp_path, monkeypatch):
    steps, ok = probed(tmp_path, monkeypatch)
    assert ok, steps
    assert [item['step'] for item in steps] == ['configuration', 'endpoint', 'model',
                                                'completion', 'json mode']
    assert all(item['ok'] for item in steps)


def test_preflight_names_the_missing_configuration_first(tmp_path, monkeypatch):
    steps, ok = probed(tmp_path, monkeypatch, key=None)
    assert not ok
    assert steps[0]['step'] == 'configuration' and 'DEEPSEEK_API_KEY' in steps[0]['detail']
    assert len(steps) == 1, 'no request is attempted without a credential'

    steps, ok = probed(tmp_path, monkeypatch, base=None)
    assert not ok and 'AFD_RSI_DEEPSEEK_BASE_URL' in steps[0]['detail']


def test_preflight_accepts_a_renamed_credential_variable(tmp_path, monkeypatch):
    module = adapter()
    monkeypatch.setattr(module, 'urlopen', FakeTransport('ok', '{"ok": true}', models=(MODEL,)))
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setenv('CSI_API_KEY', 'sk-gateway-key')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_KEY_ENV', 'CSI_API_KEY')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    steps, ok = module.probe()
    assert ok, steps
    assert 'CSI_API_KEY' in steps[0]['detail'], 'the report says which variable was used'


def test_preflight_reports_a_model_the_gateway_does_not_serve(tmp_path, monkeypatch):
    steps, ok = probed(tmp_path, monkeypatch, models=('some-other-model',))
    assert not ok
    model = [item for item in steps if item['step'] == 'model'][0]
    assert MODEL in model['detail'] and 'some-other-model' in model['detail']
    assert 'AFD_RSI_DEEPSEEK_MODEL' in model['remedy']


@pytest.mark.parametrize('code, hint', [
    (401, 'refresh the credential'),
    (404, 'AFD_RSI_DEEPSEEK_BASE_URL'),
    (503, 'availability problem'),
])
def test_preflight_translates_gateway_status_into_one_action(tmp_path, monkeypatch, code, hint):
    steps, ok = probed(tmp_path, monkeypatch, http_errors=[(code, '{"error":"nope"}')],
                       persistent=True)
    assert not ok
    endpoint = [item for item in steps if item['step'] == 'endpoint'][0]
    assert str(code) in endpoint['detail'] and hint in endpoint['remedy']


def test_preflight_keeps_the_credential_out_of_its_report(tmp_path, monkeypatch):
    # Gateways echo the presented token in auth errors; the report must not carry it.
    steps, ok = probed(tmp_path, monkeypatch,
                       http_errors=[(401, '{"error":"bad key sk-test-secret"}')])
    assert not ok
    text = json.dumps(steps)
    assert 'sk-test-secret' not in text
    assert 'sk-<redacted>' in text


def test_preflight_reports_json_mode_separately_from_generation(tmp_path, monkeypatch):
    module = adapter()
    listing = FakeTransport('ok', '{"ok": true}', models=(MODEL,))

    def wrapped(request, timeout=None):
        if request.data is None:
            return listing(request, timeout)
        payload = json.loads(request.data.decode())
        if 'response_format' in payload:
            raise urllib.error.HTTPError(request.full_url, 400, 'error', {},
                                         io.BytesIO(b'{"error":"response_format unsupported"}'))
        return listing(request, timeout)

    monkeypatch.setattr(module, 'urlopen', wrapped)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', GATEWAY)
    steps, ok = module.probe()
    json_step = [item for item in steps if item['step'] == 'json mode'][0]
    assert not json_step['ok'] and 'AFD_RSI_DEEPSEEK_JSON_MODE=0' in json_step['remedy']
    assert [item['step'] for item in steps if item['ok']][-1] == 'completion', \
        'generation must be proven before JSON mode is blamed'
    assert not ok, 'a partially working endpoint is still a failure'
