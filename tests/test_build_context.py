import copy
import importlib.util
from pathlib import Path

import pytest

from afd_rsi.build_context import PULL_KERNEL, capture_context, load_context, verify_context
from afd_rsi.evidence import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def test_capture_uses_resolved_build_headers_and_checks_cache_drift(tmp_path):
    repo, sdk = tmp_path / 'source', tmp_path / 'sdk'
    (repo / PULL_KERNEL).mkdir(parents=True)
    (sdk / 'utils').mkdir(parents=True)
    (sdk / 'utils/block_mmad_w4a4.hpp').write_text('#include "policy.hpp"\n')
    (sdk / 'utils/policy.hpp').write_text('// Finalize and idle-core completion\n')
    (repo / 'CMakeCache.txt').write_text(f'AFD_MC2_92_DIR:PATH={sdk}\n')
    (repo / PULL_KERNEL / 'afd_mega_moe_activation.h').write_text('#include <kernel_operator.h>\n')
    output = tmp_path / 'dependencies.json'
    capture_context(repo, output, 'source-id', {'kernel': 'hash'})
    bundle = load_context(output, 'source-id', {'kernel': 'hash'}, origins=True)
    assert set(bundle['files']) == {'mc2/utils/block_mmad_w4a4.hpp', 'mc2/utils/policy.hpp',
                                    'generated/afd_mega_moe_activation.h'}
    assert bundle['unresolved_includes'] == ['kernel_operator.h']
    assert bundle['read_only'] is True
    corrupted = copy.deepcopy(bundle)
    corrupted['files']['mc2/utils/policy.hpp']['content'] += 'changed'
    with pytest.raises(ValueError, match='context changed'):
        verify_context(corrupted, 'source-id', {'kernel': 'hash'})
    with pytest.raises(ValueError, match='another build'):
        verify_context(bundle, 'other-source', {'kernel': 'hash'})
    (sdk / 'utils/policy.hpp').write_text('SDK changed in place')
    with pytest.raises(ValueError, match='installed build dependency changed'):
        load_context(output, 'source-id', {'kernel': 'hash'}, origins=True)


def test_missing_compiled_context_fails_explicitly(tmp_path):
    (tmp_path / PULL_KERNEL).mkdir(parents=True)
    with pytest.raises(ValueError, match='resolved MC2'):
        capture_context(tmp_path, tmp_path / 'dependencies.json', 'id', {})


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proposer_reads_files_individually_and_clears_stale_sources(tmp_path):
    agent = load_module('codex_context_test', 'benchmarks/agents/codex_agent.py')
    request = dict(source_files={'kernel.cpp': '// source', 'protocol.h': '// protocol'},
                   read_only_source_files=['protocol.h'], knowledge='local knowledge',
                   evidence=[dict(id='baseline', facts={'value': 10})],
                   dependency_context={'files': {'mc2/mm.hpp': {'content': '// SDK', 'sha256': 'hash'}}})
    agent.write_workspace(tmp_path, request)
    manifest = read_json(tmp_path / 'request.json')
    assert manifest['source_files']['protocol.h']['read_only'] is True
    assert (tmp_path / manifest['dependency_context']['files']['mc2/mm.hpp']['path']).read_text() == '// SDK'
    assert read_json(tmp_path / manifest['evidence'][0]['facts_file']) == {'value': 10}
    assert request['source_files']['kernel.cpp'] == '// source'
    agent.write_workspace(tmp_path, {'source_files': {}})
    assert not (tmp_path / 'source/kernel.cpp').exists()
    with pytest.raises(ValueError, match='escapes'):
        agent.write_workspace(tmp_path, {'source_files': {'../../outside': 'bad'}})


def test_model_shape_routes_cover_exact_cohort_boundaries_and_hot_rank():
    routing = load_module('workload_test', 'benchmarks/native/workload.py')
    for tokens in (7, 128, 511, 512, 513, 1025, 2050, 4100, 8125):
        dense = routing.routing_ids(tokens, 32, False, balanced=True)
        sparse = routing.routing_ids(tokens, 32, True, balanced=True)
        assert all(len(set(pair)) == 2 for pair in dense + sparse)
        assert [sum(low <= e < low + 32 for pair in dense for e in pair) for low in (0, 32)] == [tokens, tokens]
        assert [sum(low <= e < low + 32 for pair in sparse for e in pair) for low in (0, 32)] == [2 * tokens, 0]


def test_missing_contract_stops_before_the_coding_call(tmp_path, monkeypatch):
    agent = load_module('codex_readiness_test', 'benchmarks/agents/codex_agent.py')
    monkeypatch.setattr(agent.shutil, 'which', lambda _: 'codex')
    calls = []
    response = dict(stop=True, reason='Missing completion contract', proposals=[], needs_context=['Finalize'])
    def invoke(*args):
        calls.append(args)
        return response, [{'model': 'analysis'}]
    monkeypatch.setattr(agent, 'invoke', invoke)
    request = tmp_path / 'request.json'
    write_json(request, {'evidence': [{'id': 'baseline'}]})
    args = type('Args', (), dict(request=request, output=tmp_path / 'response.json', session_dir=tmp_path / 'session'))()
    args.session_dir.mkdir()
    agent.propose(args)
    assert len(calls) == 1 and read_json(args.output)['needs_context'] == ['Finalize']


def test_request_seals_build_contracts_and_preserves_allowed_protocol(tmp_path):
    import hashlib
    from afd_rsi import campaign, search
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text((ROOT / 'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source')))
    run = campaign.prepare(recipe, tmp_path / 'run', ROOT)
    search.search(run, ROOT, external=True)
    plan, best = campaign.load(run, ROOT), read_json(run / 'search/best.json')
    trial = run / 'context-trial'
    artifacts = {'kernel': 'compiled-object'}
    write_json(trial / 'artifacts.json', artifacts)
    content = '// SDK completion contract\n' * 2800
    bundle = dict(source_hash=best['source_hash'], artifacts=artifacts,
                  files={'mc2/mm.hpp': dict(content=content, sha256=hashlib.sha256(content.encode()).hexdigest())})
    write_json(trial / 'dependencies.json', bundle)
    attempt = run / 'attempts' / best['baseline']
    progress = read_json(attempt / 'progress.json')
    progress['steps'].append({'result': dict(side='baseline', trial='context-trial',
        dependency_context_hash=digest(trial / 'dependencies.json'),
        artifacts={'native/artifacts.json': digest(trial / 'artifacts.json')})})
    write_json(attempt / 'progress.json', progress)
    source = run / best['source']
    (source / 'kernel.cpp').write_text('#include "protocol.h"\n')
    (source / 'protocol.h').write_text('// protocol\n' * 2000)
    plan['allowed_files'] = ['kernel.cpp', 'protocol.h']
    records = [dict(category='fusion', decision='invalid_patch', reason='hunk does not apply',
                    candidate_hash=None, evidence={})]
    context = search.request_context(run, plan, best, records, 1)
    assert 'history-0' in context['protected_evidence_ids']
    assert next(item for item in context['evidence'] if item['id'] == 'history-0')['facts']['reason'] == 'hunk does not apply'
    assert context['dependency_context'] == bundle
    assert set(context['source_files']) == {'kernel.cpp', 'protocol.h'}
    import json
    required = {**context, 'knowledge': '', 'logs': []}
    plan['search']['max_context_bytes'] = len((json.dumps(required, indent=2, ensure_ascii=False) + '\n').encode()) + 1000
    bounded = search.request_context(run, plan, best, records, 1)
    assert set(bounded['source_files']) == {'kernel.cpp', 'protocol.h'}
    write_json(tmp_path / 'bounded.json', bounded)
    assert (tmp_path / 'bounded.json').stat().st_size <= plan['search']['max_context_bytes']
    (trial / 'dependencies.json').write_text('{}')
    with pytest.raises(ValueError, match='compiled dependency context changed'):
        search.request_context(run, plan, best, [], 1)
