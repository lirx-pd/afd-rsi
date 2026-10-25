import copy
import sys
from pathlib import Path

import pytest

from afd_rsi import campaign, search
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.runner import execute, host_identity, stop_owned
from afd_rsi.service_phase import evidence_files, validate_summary

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path):
    recipe = tmp_path / 'search.toml'
    text = (ROOT / 'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('repetitions = 4', 'repetitions = 2').replace('max_rounds = 3', 'max_rounds = 2').replace('max_calls = 3', 'max_calls = 2')
    recipe.write_text(text)
    return campaign.prepare(recipe, tmp_path / 'run', ROOT)


def proposal(patch=None, category='latency', priority=3):
    return {'stop': False, 'reason': 'A controlled synthetic fixture change', 'proposals': [{
        'hypothesis': 'The synthetic latency field is smaller while numeric output stays equal.',
        'category': category, 'priority': priority, 'evidence': ['baseline'],
        'patch': patch or (ROOT / 'examples/faster.patch').read_text()}]}


def respond(run, value):
    pending = read_json(run / 'search/pending.json')
    write_json(run / pending['response'], value)
    return run / pending['response']


def test_search_confirms_best_and_exports_it_after_later_rejection(run, tmp_path):
    search.search(run, ROOT, external=True)
    assert read_json(run / 'search/state.json')['calls'] == 1
    respond(run, proposal())
    search.search(run, ROOT)
    best = search.best_candidate(run, ROOT)
    assert best['screen']['attempt'] != best['confirmation']['attempt']
    assert read_json(run / best['source'] / 'kernel.json')['synthetic_us'] == 8
    # Generate a patch against the actual incumbent formatting.
    before = (run / best['source'] / 'kernel.json').read_text().rstrip('\n')
    after = before.replace('8', '12')
    slower = f'diff --git a/kernel.json b/kernel.json\n--- a/kernel.json\n+++ b/kernel.json\n@@ -1 +1 @@\n-{before}\n+{after}\n'
    respond(run, proposal(slower))
    search.search(run, ROOT)
    assert search.best_candidate(run, ROOT) == best
    assert read_json(run / 'search/state.json')['calls'] == 2
    assert read_json(run / 'search/state.json')['status'] == 'budget_exhausted'
    exported = campaign.export(run, tmp_path / 'export', ROOT, best=True)
    assert read_json(exported / 'best.json')['source_hash'] == best['source_hash']
    assert read_json(run / 'workspaces/candidate/kernel.json')['synthetic_us'] == 10
    assert len(list((run / 'search/experience').glob('*.json'))) == 2
    plan = campaign.load(run, ROOT)
    plan['search']['experience_dirs'] = [str(run / 'search/experience')]
    other = tmp_path / 'another-campaign'
    other.mkdir()
    assert len(search.load_experiences(other, plan)) == 2
    incompatible = {**plan, 'workload_id': 'another-workload'}
    assert not search.load_experiences(other, incompatible)


def test_history_changes_selection_and_deduplicates(run):
    first = proposal()['proposals'][0]
    other = {**first, 'category': 'other', 'priority': 2, 'patch': first['patch'] + '\n'}
    history = [{'parent_hash': 'old', 'patch_hash': 'unused', 'category': 'latency', 'decision': 'rejected'}] * 2
    assert search.select_proposal([first, other], history, 'new') == other
    seen = [{'parent_hash': 'same', 'patch_hash': search.identity(first['patch'].strip()), 'category': 'latency', 'decision': 'rejected'}]
    assert search.select_proposal([first], seen, 'same') is None
    assert search.select_proposal([first], seen, 'different') == first
    search.search(run, ROOT, external=True)
    wrong = proposal()
    wrong['proposals'][0]['patch'] = wrong['proposals'][0]['patch'].replace(
        '"synthetic_us": 8.0, "multiplier": 2, "bias": 1', '"synthetic_us": 8.0, "multiplier": 2, "bias": 2')
    respond(run, wrong)
    search.search(run, ROOT)
    context = read_json(run / read_json(run / 'search/pending.json')['request'])
    assert any(item['id'].startswith('history-') for item in context['evidence'])
    assert search.best_candidate(run, ROOT)['confirmation'] is None
    lessons = search.load_experiences(run, campaign.load(run, ROOT))
    assert lessons[0]['decision'] == 'rejected'
    path = next((run / 'search/experience').glob('*.json'))
    path.write_text('{}')
    with pytest.raises(ValueError, match='experience changed'):
        search.load_experiences(run, campaign.load(run, ROOT))


def test_confirmation_failure_never_promotes(run, monkeypatch):
    search.search(run, ROOT, external=True)
    respond(run, proposal())
    original = campaign.execute

    def changed_measurement(argv, cwd, output, timeout, env):
        result = original(argv, cwd, output, timeout, env)
        state = read_json(run / 'search/rounds/0001/state.json')
        if state['phase'] == 'confirming' and argv[-1] == 'benchmark' and env['AFD_RSI_SIDE'] == 'candidate':
            path = Path(env['AFD_RSI_RESULT'])
            data = read_json(path)
            for rank in data['ranks']:
                rank['value'] = 12
            write_json(path, data)
        return result

    monkeypatch.setattr(campaign, 'execute', changed_measurement)
    search.search(run, ROOT)
    assert search.best_candidate(run, ROOT)['confirmation'] is None
    decision = read_json(run / 'search/rounds/0001/state.json')
    assert decision['decision'] == 'rejected' and 'confirmation' in decision['reason']


def test_call_charge_and_pending_request_survive_crash(run, monkeypatch):
    original = search.write_json
    crashed = False

    def crash(path, value):
        nonlocal crashed
        if path.name == 'state.json' and value.get('phase') == 'awaiting_agent' and not crashed:
            crashed = True
            raise KeyboardInterrupt()
        original(path, value)

    monkeypatch.setattr(search, 'write_json', crash)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT, external=True)
    assert read_json(run / 'search/state.json')['calls'] == 1
    monkeypatch.setattr(search, 'write_json', original)
    search.search(run, ROOT)
    assert read_json(run / 'search/state.json')['calls'] == 1
    assert read_json(run / 'search/pending.json')['call'] == 1
    context = read_json(run / read_json(run / 'search/pending.json')['request'])
    assert context['execution']['ranks'] == 2 and context['execution']['service'] is None
    respond(run, {'stop': True, 'reason': 'No further justified experiment.', 'proposals': []})
    search.search(run, ROOT)
    assert read_json(run / 'search/state.json')['status'] == 'agent_stopped'


def test_completed_phase_is_adopted_after_crash_before_ack(run, monkeypatch):
    original = campaign.command
    calls = []
    crashed = False

    def crash(*args, **kwargs):
        nonlocal crashed
        calls.append(args[4])
        result = original(*args, **kwargs)
        if args[4] == 'correctness' and not crashed:
            crashed = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(campaign, 'command', crash)
    with pytest.raises(KeyboardInterrupt):
        campaign.run_baseline(run, ROOT)
    attempt = next((run / 'attempts').iterdir())
    build_receipt = next((attempt / 'baseline/build').rglob('process.json'))
    before = digest(build_receipt)
    campaign.resume(run, ROOT)
    assert read_json(attempt / 'baseline.json')['complete']
    assert calls.count('build') == 1 and calls.count('correctness') == 1
    assert digest(build_receipt) == before


def test_detached_child_is_cleaned_and_foreign_receipt_cannot_signal(tmp_path):
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)"
    result = execute([sys.executable, '-c', code], tmp_path, tmp_path / 'trial', 5, {})
    assert result['status'] == 'orphaned_children' and result['cleanup_complete']
    wrong = {**result, 'host_identity': {**host_identity(), 'boot': 'another-boot'}}
    with pytest.raises(ValueError, match='host/boot'):
        stop_owned(wrong)


def test_recovery_preserves_receipt_and_refuses_foreign_host(run):
    path = run / 'attempts/old/process.json'
    receipt = {'status': 'running', 'cleanup_complete': False, 'tag': 'unowned-test-tag',
               'host_identity': host_identity(), 'argv': ['test']}
    write_json(path, receipt)
    before = digest(path)
    campaign.resume(run, ROOT)
    assert digest(path) == before
    assert campaign.effective_receipt(path)['cleanup_complete']
    wrong = copy.deepcopy(receipt)
    wrong['host_identity']['boot'] = 'another-boot'
    write_json(run / 'attempts/foreign/process.json', wrong)
    with pytest.raises(ValueError, match='different host/boot'):
        campaign.resume(run, ROOT)


def test_service_shutdown_and_raw_evidence_are_acceptance_gates(run, tmp_path):
    summary = {'status': 'passed', 'inference_status': 'passed', 'shutdown_status': 'passed',
               'cleanup_status': 'passed', 'test_results': {'smoke': 'passed'}}
    assert validate_summary(summary, ['smoke'])
    for key in ('status', 'inference_status', 'shutdown_status', 'cleanup_status'):
        assert not validate_summary({**summary, key: 'failed'}, ['smoke'])
    assert not validate_summary(summary, ['smoke', 'perf'])
    trial = run / 'attempts/service'
    write_json(trial / 'service/summary.json', summary)
    write_json(trial / 'loaded-maps.json', {'processes': {}})
    observation = {'status': 'valid', 'phase': 'service', 'trial': str(trial.relative_to(run)),
                   'summary_hash': digest(trial / 'service/summary.json'),
                   'loaded_maps_hash': digest(trial / 'loaded-maps.json'), 'evidence': evidence_files(trial / 'service')}
    (trial / 'service/summary.json').write_text('{}')
    with pytest.raises(ValueError, match='service evidence changed'):
        campaign.verify_observation(run, observation, None, None, None, None, check_runtime=False)


def test_baseline_failure_prevents_model_dispatch(run, monkeypatch):
    original = campaign.execute

    def bad_baseline(argv, cwd, output, timeout, env):
        result = original(argv, cwd, output, timeout, env)
        if argv[-1] == 'correctness':
            path = Path(env['AFD_RSI_RESULT'])
            data = read_json(path)
            data['ranks'][0]['checks']['affine_output'] = False
            write_json(path, data)
        return result

    monkeypatch.setattr(campaign, 'execute', bad_baseline)
    search.search(run, ROOT, external=True)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'baseline_failed' and state['calls'] == 0
    assert not (run / 'search/best.json').exists()


def test_expired_pending_request_stops_without_new_charge(run, monkeypatch):
    search.search(run, ROOT, external=True)
    deadline = campaign.load(run, ROOT)['expires_at']
    monkeypatch.setattr(search.time, 'time', lambda: deadline + 1)
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'budget_exhausted' and state['calls'] == 1
    assert not (run / 'search/pending.json').exists()


def test_device_health_failure_blocks_scheduling_and_can_be_retried(run, monkeypatch):
    plan = campaign.load(run, ROOT)
    plan.update(backend='ascend', health={'command': ['{python}', '{foundation}/benchmarks/native/health.py', '--device', '0'], 'timeout_seconds': 5})
    monkeypatch.setattr(campaign, 'load', lambda *_: plan)
    calls = []

    def probe(argv, cwd, output, timeout, env):
        calls.append(argv)
        return {'status': 'failed' if len(calls) == 1 else 'completed', 'cleanup_complete': True}

    monkeypatch.setattr(campaign, 'execute', probe)
    write_json(run / 'recovery.json', {'health': 'required'})
    with pytest.raises(ValueError, match='health check failed'):
        campaign.resume(run, ROOT)
    with pytest.raises(ValueError, match='health barrier'):
        with campaign.exclusive(run):
            pytest.fail('must not schedule a device phase')
    campaign.resume(run, ROOT)
    assert read_json(run / 'recovery.json')['health'] == 'passed' and len(calls) == 2


@pytest.mark.parametrize('field,value', [('stop', 1), ('proposals', []), ('reason', ''), ('extra', 'unexpected')])
def test_agent_protocol_rejects_malformed_response(field, value):
    response = proposal()
    response[field] = value
    with pytest.raises(ValueError):
        search.validate_proposal(response, {'baseline'})


def test_stop_and_experience_commit_resume_after_crash(run, monkeypatch):
    search.search(run, ROOT, external=True)
    respond(run, {'stop': True, 'reason': 'No justified experiment remains.', 'proposals': []})
    original = search.write_json
    crashed = False

    def crash(path, value):
        nonlocal crashed
        if path.parent.name == 'experience' and not crashed:
            crashed = True
            raise KeyboardInterrupt()
        original(path, value)

    monkeypatch.setattr(search, 'write_json', crash)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    assert read_json(run / 'search/rounds/0001/state.json')['decision'] == 'stopped'
    monkeypatch.setattr(search, 'write_json', original)
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'agent_stopped' and state['calls'] == 1 and len(state['rounds']) == 1
    assert len(search.load_experiences(run, campaign.load(run, ROOT))) == 1


@pytest.mark.parametrize('scenario', ['stop', 'code', 'analysis_failure', 'code_failure'])
def test_codex_adapter_routes_models_and_constrains_evidence(tmp_path, monkeypatch, scenario):
    import os
    import subprocess

    executable = tmp_path / 'bin/codex'
    executable.parent.mkdir()
    executable.write_text(
        f'#!{sys.executable}\n'
        'import json,sys\n'
        'from pathlib import Path\n'
        f'scenario={scenario!r}\n'
        'schema=json.loads(Path(sys.argv[sys.argv.index("--output-schema")+1]).read_text())\n'
        'props=schema["properties"]["proposals"]["items"]["properties"]\n'
        'assert props["evidence"]["items"]["enum"] == ["baseline", "history-0"]\n'
        'assert sys.argv[sys.argv.index("--sandbox")+1] == "read-only"\n'
        'analysis="patch" not in props\n'
        'assert sys.argv[sys.argv.index("--model")+1] == ("gpt-6-astra" if analysis else "gpt-6.1-sol")\n'
        'assert sys.argv[sys.argv.index("-c")+1] == \'model_reasoning_effort="xhigh"\'\n'
        'if scenario == ("analysis_failure" if analysis else "code_failure"): sys.exit(7)\n'
        'if not analysis:\n'
        '    prior=json.loads((Path(sys.argv[sys.argv.index("--cd")+1])/"analysis.json").read_text())\n'
        '    assert not prior["stop"] and prior["proposals"][0]["evidence"] == ["baseline"]\n'
        'stop=scenario == "stop"\n'
        'proposal={"hypothesis":"test", "category":"latency", "priority":1,"evidence":["baseline"]}\n'
        'if not analysis: proposal["patch"]="test diff"\n'
        'Path(sys.argv[sys.argv.index("--output-last-message")+1]).write_text(json.dumps({"stop":stop,"reason":"done","proposals":[] if stop else [proposal]}))\n'
        'print(json.dumps({"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":3}}))\n')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(executable.parent) + os.pathsep + os.environ['PATH'])
    request = tmp_path / 'request.json'
    write_json(request, {'evidence': [{'id': 'baseline'}, {'id': 'history-0'}]})
    response = tmp_path / 'result/proposal.json'
    result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/agents/codex_agent.py'),
                             '--request', str(request), '--output', str(response)], capture_output=True)
    if scenario.endswith('failure'):
        assert result.returncode != 0 and not response.exists()
        assert b'Codex exited 7' in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert read_json(response)['stop'] == (scenario == 'stop')
    turns = read_json(response.parent / 'usage.json')['turns']
    assert [turn['model'] for turn in turns] == (['gpt-6-astra'] if scenario == 'stop' else ['gpt-6-astra', 'gpt-6.1-sol'])
    assert all(turn['effort'] == 'xhigh' and turn['input_tokens'] == 5 for turn in turns)
    assert (response.parent / 'codex-events.jsonl').exists() == (scenario == 'code')


def test_native_manifest_cannot_hide_changed_physical_artifacts(tmp_path):
    from afd_rsi.native import artifact_identity

    build = tmp_path / 'build'
    repo = build / 'native/source'
    vendor = repo / 'afd_plugin/_cann_ops_custom/vendors/afd-plugin'
    api = vendor / 'op_api/lib/libcust_opapi.so'
    kernel = vendor / 'op_impl/ai_core/tbe/kernel/ascend910_93/afd_mega_moe_ffn/kernel.o'
    for path in (repo / 'afd_plugin/__init__.py', repo / 'afd_plugin/_C_ascend.so', api, kernel):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'unit-test opaque artifact')
    manifest = build / 'native/artifacts.json'
    write_json(manifest, artifact_identity(repo))
    locked = campaign.artifacts(build, ['native/artifacts.json'], native=True)
    assert locked['native/artifacts.json'] == digest(manifest)
    api.write_bytes(b'changed after evaluation')
    assert locked['native/artifacts.json'] == digest(manifest)
    with pytest.raises(ValueError, match='actual native artifacts changed'):
        campaign.artifacts(build, ['native/artifacts.json'], native=True)


def test_pidfd_unavailable_fails_before_launch(tmp_path, monkeypatch):
    import os

    def unavailable(_):
        raise OSError('unsupported kernel')

    monkeypatch.setattr(os, 'pidfd_open', unavailable)
    marker = tmp_path / 'must-not-run'
    with pytest.raises(ValueError, match='working Linux pidfds'):
        execute([sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).touch()'],
                tmp_path, tmp_path / 'trial', 1, {})
    assert not marker.exists() and not (tmp_path / 'trial').exists()


@pytest.mark.parametrize('summary', [None, [], {'test_results': []}, {'test_results': True}])
def test_service_malformed_summary_is_not_accepted(summary):
    assert not validate_summary(summary, ['smoke'])


def test_shutdown_waiver_does_not_ignore_inference_cleanup_or_reporting_errors():
    summary = {'status': 'failed', 'inference_status': 'passed', 'shutdown_status': 'failed',
               'cleanup_status': 'passed', 'test_results': {'smoke': 'passed'},
               'errors': ['shutdown: EngineDeadError', 'shutdown: forced FFN exit']}
    assert not validate_summary(summary, ['smoke'])
    assert validate_summary(summary, ['smoke'], require_clean_shutdown=False)
    for key in ('inference_status', 'cleanup_status'):
        assert not validate_summary({**summary, key: 'failed'}, ['smoke'], require_clean_shutdown=False)
    assert not validate_summary({**summary, 'errors': ['Report write failed']}, ['smoke'], require_clean_shutdown=False)
    assert not validate_summary({**summary, 'test_results': {'smoke': 'failed'}}, ['smoke'], require_clean_shutdown=False)
    assert not validate_summary({**summary, 'status': 'cancelled'}, ['smoke'], require_clean_shutdown=False)
