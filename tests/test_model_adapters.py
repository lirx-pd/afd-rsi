import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from afd_rsi.evidence import read_json, write_json
from afd_rsi.model_protocol import ModelContext
from afd_rsi.proposal import validate_patches
from afd_rsi.search import validate_proposal

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['bottleneck', 'mechanism', 'prediction', 'falsification', 'risk']


def setup(tmp_path, provider):
    spec = importlib.util.spec_from_file_location('compact_' + provider, ROOT / f'benchmarks/agents/{provider}_agent.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    request = dict(model_protocol=2, fingerprint='campaign', provider_hash='adapter', source_hash='parent',
        round=1, deadline=9e12, agent_timeout_seconds=60, max_context_bytes=131072,
        source_files={'kernel.cpp': 'int x = 1;\n'}, allowed_files=['*.cpp'], knowledge='',
        dependency_context={'files': {
            'catlass/large.hpp': dict(content='// REQUIRED_HEADER_TOKEN\n' + 'unused_large_body\n' * 1000),
            'mc2/mm.hpp': dict(content='void Finalize();\n')}},
        evidence=[dict(id='baseline', facts={}), dict(id='profile-incumbent', facts={})],
        proposal_requirements=dict(experiment_fields=FIELDS, required_evidence_ids=['baseline', 'profile-incumbent']))
    args = SimpleNamespace(request=tmp_path / 'request.json', output=tmp_path / 'out/proposal.json',
                           session_dir=tmp_path / 'session')
    args.session_dir.mkdir()
    write_json(args.request, request)
    analysis = dict(stop=False, reason='one controlled mechanism', needs_context=[], proposals=[dict(
        hypothesis='remove redundant copy', category='memory', evidence=['baseline', 'profile-incumbent'],
        priority=3, experiment={key: 'bounded fixture ' + key for key in FIELDS}, files=['s0'])])
    return module, args, request, analysis


def coding(request):
    return dict(base=ModelContext(request).base, stop=False, reason='implemented', needs_context=[],
                proposals=[dict(id=0, edits=[dict(file='s0', old='int x = 1;', new='int x = 2;')])])


def install_http(module, monkeypatch, *responses):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-only')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', 'http://gateway.test/v1')
    queued, calls = list(responses), []
    def send(payload, timeout, prefix):
        calls.append(payload)
        return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(queued.pop(0))))],
                    usage=dict(prompt_tokens=10, completion_tokens=5))
    monkeypatch.setattr(module, 'send', send)
    return calls


def test_http_reads_once_then_codes_from_selected_context(tmp_path, monkeypatch):
    module, args, request, analysis = setup(tmp_path, 'deepseek')
    calls = install_http(module, monkeypatch, {'read': [['d0', 1, 1]]}, analysis, coding(request))
    module.propose(args)
    assert len(calls) == 3
    first = str(calls[0]['messages'])
    assert 'REQUIRED_HEADER_TOKEN' not in first and 'unused_large_body' not in first
    assert 'REQUIRED_HEADER_TOKEN' in str(calls[1]['messages'])
    assert 'REQUIRED_HEADER_TOKEN' in str(calls[2]['messages'])
    assert 'unused_large_body' not in str(calls[2]['messages'])
    value = read_json(args.output)
    validate_proposal(value, {'baseline', 'profile-incumbent'}, request['proposal_requirements'])
    validate_patches(value, request)
    assert value['proposals'][0]['experiment'] == analysis['proposals'][0]['experiment']
    assert read_json(args.output.parent / 'usage.json')['model_calls'] == 3
    assert read_json(args.session_dir / 'sdk-reads.json')['files']['d0']['ranges'] == [[1, 1]]


def test_http_second_read_stops_without_unbounded_calls(tmp_path, monkeypatch):
    module, args, request, _ = setup(tmp_path, 'deepseek')
    calls = install_http(module, monkeypatch, {'read': [['d0', 1, 1]]}, {'read': [['d0', 2, 1]]})
    module.propose(args)
    assert len(calls) == 2
    assert read_json(args.output)['stop'] and read_json(args.output)['needs_context']


def test_http_compact_edit_failure_has_only_one_code_repair(tmp_path, monkeypatch):
    module, args, request, analysis = setup(tmp_path, 'deepseek')
    bad = coding(request)
    bad['proposals'][0]['edits'][0]['old'] = 'not present'
    calls = install_http(module, monkeypatch, analysis, bad, coding(request))
    module.propose(args)
    assert len(calls) == 3
    assert 'exactly once' in calls[-1]['messages'][-1]['content']
    validate_patches(read_json(args.output), request)


@pytest.mark.parametrize('provider', ['deepseek', 'codex'])
@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('adaptive', [False, True])
def test_compact_adapters_restore_controller_contract_and_keep_debug_code_only(tmp_path, monkeypatch, provider, debug, adaptive):
    module, args, request, analysis = setup(tmp_path, provider)
    if adaptive:
        ref = 'skills/ascendc-perf-optimize/SKILL.md'
        request['proposal_requirements']['knowledge_refs'] = [ref]
        request['knowledge_files'] = {ref: '# Procedure\nApply tiling, inter-device, inter-core, intra-core analysis.'}
        analysis['proposals'][0]['direction'] = dict(id='novel-method', mechanism='source-backed alternative',
            step=3, refs=[ref], use='extend: new mechanism using the same evidence discipline', next='')
        write_json(args.request, request)
    if debug:
        original = ModelContext(request).decode(coding(request), analysis)['proposals'][0]
        request['debug'] = dict(original_proposal=original, failure=dict(phase='build', reason='compiler error'))
        write_json(args.request, request)
    if provider == 'deepseek':
        calls = install_http(module, monkeypatch, *([coding(request)] if debug else [analysis, coding(request)]))
    else:
        calls = []
        monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
        def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
            calls.append(model)
            manifest = read_json(workspace / 'request.json')
            assert manifest['file_paths']['s0'] == 'source/kernel.cpp'
            assert 'unused_large_body' not in json.dumps(manifest)
            if adaptive:
                path = next(path for key, path in manifest['file_paths'].items() if key.startswith('k'))
                assert (workspace / path).read_text().startswith('# Procedure')
            if model == 'gpt-6-astra':
                value = analysis
            else:
                assert 'code.json' in prompt and 'edits' in read_json(schema)['properties']['proposals']['items']['properties']
                value = coding(request)
            write_json(output, value)
            return value, [dict(model=model, input_tokens=1)]
        monkeypatch.setattr(module, 'invoke', invoke)
    module.propose(args)
    assert len(calls) == (1 if debug else 2)
    value = read_json(args.output)
    validate_proposal(value, {'baseline', 'profile-incumbent'}, request['proposal_requirements'])
    validate_patches(value, request)
    assert 'files' not in value['proposals'][0]
    assert value['proposals'][0]['hypothesis'] == analysis['proposals'][0]['hypothesis']


def test_unknown_knowledge_reference_is_rejected_before_coding(tmp_path):
    _, _, request, analysis = setup(tmp_path, 'codex')
    request['proposal_requirements']['knowledge_refs'] = ['known']
    analysis['proposals'][0]['direction'] = dict(id='new', mechanism='test', step=1, refs=['unfrozen'], use='new', next='')
    with pytest.raises(ValueError, match='knowledge'):
        ModelContext(request).normalize_analysis(analysis)
