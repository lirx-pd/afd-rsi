"""DeepSeek proposer adapter: contract, truncation, repair and credential hygiene.

The adapter is stdlib-only and imports nothing that needs a POSIX controller, so
these tests run on every host; the transport is substituted, never the network.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def adapter():
    spec = importlib.util.spec_from_file_location(
        'deepseek_adapter', ROOT / 'benchmarks/agents/deepseek_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTransport:
    """Records requests and replays queued completions."""

    def __init__(self, *contents):
        self.contents = list(contents)
        self.payloads = []
        self.headers = []

    def __call__(self, request, timeout=None):
        self.payloads.append(json.loads(request.data.decode()))
        self.headers.append(dict(request.header_items()))
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
    assert {turn['model'] for turn in turns} == {'deepseek-v4.1-flash'}
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
    assert module.model_for('analysis') == module.model_for('code') == 'deepseek-v4.1-flash'
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_MODEL', 'deepseek-v4.1-flash (1017)')
    assert module.model_for('code') == 'deepseek-v4.1-flash (1017)'
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_CODE_MODEL', 'other-code')
    assert module.model_for('code') == 'deepseek-v4.1-flash (1017)', \
        'the shared override outranks a stage override'
    monkeypatch.delenv('AFD_RSI_DEEPSEEK_MODEL')
    assert module.model_for('code') == 'other-code'
    assert module.model_for('analysis') == 'deepseek-v4.1-flash'


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
            raise RuntimeError('DeepSeek HTTP 400: {"error": "response_format is not supported"}')
        return transport(request, timeout)

    monkeypatch.setattr(module, 'urlopen', failing_then_ok)
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-test-secret')
    args = type('Args', (), {'request': request_document(tmp_path),
                             'output': tmp_path / 'out/proposal.json',
                             'session_dir': tmp_path / 'session'})()
    module.propose(args)
    assert 'response_format' in calls[0] and 'response_format' not in calls[1]
    assert args.output.exists()
    record = json.loads((args.output.parent / 'analysis-events.unsupported-parameters.json').read_text())
    assert 'response_format' in record['dropped']


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
