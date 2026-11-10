"""Model faults are bounded, resumable and never performance evidence."""
import copy
import json
import sys
from pathlib import Path

import pytest

from afd_rsi import campaign, search, optimize
from afd_rsi.evidence import read_json, write_json, digest
from afd_rsi.model_recovery import Journal, ModelExhausted, failure_response, resumable
from test_model_adapters import setup, coding, install_http
from test_adaptive_search import prepare, proposal, respond, records, ROOT


def enable(args, request):
    request['model_recovery'] = 1
    write_json(args.request, request)


def test_journal_reserves_before_call_and_rejects_changed_request_or_result(tmp_path):
    request = dict(model_recovery=1, source_hash='a')
    journal = Journal(tmp_path, request)
    assert journal.reserve('analysis') == 0
    resumed = Journal(tmp_path, request)
    assert resumed.reserve('analysis') == 1
    resumed.complete('analysis', dict(answer=42))
    assert Journal(tmp_path, request).cached('analysis') == dict(answer=42)
    with pytest.raises(ValueError, match='another request'):
        Journal(tmp_path, dict(request, source_hash='b'))
    (tmp_path / 'model-analysis-complete.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        Journal(tmp_path, request).cached('analysis')


def test_http_recovers_partial_and_empty_length_with_focused_output(tmp_path, monkeypatch):
    module, args, request, analysis = setup(tmp_path, 'deepseek')
    enable(args, request)
    calls = []
    replies = [dict(finish_reason='stop', message=dict(content=json.dumps(analysis))),
               dict(finish_reason='length', message=dict(content='UNFINISHED_LARGE_ANSWER' * 1000)),
               dict(finish_reason='length', message=dict(content='')),
               dict(finish_reason='stop', message=dict(content=json.dumps(coding(request))))]
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', 'http://gateway.test/v1')
    def send(payload, timeout, prefix):
        calls.append(copy.deepcopy(payload))
        return dict(choices=[replies.pop(0)], usage=dict(prompt_tokens=1, completion_tokens=2))
    monkeypatch.setattr(module, 'send', send)
    module.propose(args)
    assert len(calls) == 4
    assert 'one highest-priority proposal' in calls[-1]['messages'][-1]['content']
    assert 'UNFINISHED_LARGE_ANSWER' not in str(calls[-1]['messages'])
    assert calls[-1]['reasoning_effort'] == 'low'
    state = read_json(args.output.parent / 'model-state.json')
    assert len(state['stages']['analysis']['attempts']) == 1
    assert len(state['stages']['code']['attempts']) == 3
    assert read_json(args.output.parent / 'usage.json')['model_calls'] == 4
    assert read_json(args.output)['proposals'][0]['patch']


@pytest.mark.parametrize('provider', ['deepseek', 'codex'])
def test_interrupted_code_reuses_completed_analysis_and_does_not_refund_attempt(tmp_path, monkeypatch, provider):
    module, args, request, analysis = setup(tmp_path, provider)
    enable(args, request)
    calls = []
    interrupted = False
    if provider == 'deepseek':
        monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
        monkeypatch.setenv('AFD_RSI_DEEPSEEK_BASE_URL', 'http://gateway.test/v1')
        def send(payload, timeout, prefix):
            nonlocal interrupted
            stage = 'analysis' if 'analysis-events' in str(prefix) else 'code'
            calls.append(stage)
            if stage == 'code' and not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            value = analysis if stage == 'analysis' else coding(request)
            return dict(choices=[dict(finish_reason='stop', message=dict(content=json.dumps(value)))], usage={})
        monkeypatch.setattr(module, 'send', send)
    else:
        monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
        def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
            nonlocal interrupted
            stage = 'analysis' if model == 'gpt-6-astra' else 'code'
            calls.append(stage)
            if stage == 'code' and not interrupted:
                interrupted = True
                raise KeyboardInterrupt()
            value = analysis if stage == 'analysis' else coding(request)
            write_json(output, value)
            return value, [dict(model=model)]
        monkeypatch.setattr(module, 'invoke', invoke)
    with pytest.raises(KeyboardInterrupt):
        module.propose(args)
    analysis_hash = digest(args.output.parent / 'model-analysis-complete.json')
    module.propose(args)
    assert calls == ['analysis', 'code', 'code']
    assert digest(args.output.parent / 'model-analysis-complete.json') == analysis_hash
    state = read_json(args.output.parent / 'model-state.json')
    assert len(state['stages']['code']['attempts']) == 2
    assert state['stages']['code']['attempts'][0]['status'] == 'reserved'


def test_exhausted_http_stage_does_not_loop_or_repeat_analysis(tmp_path, monkeypatch):
    module, args, request, analysis = setup(tmp_path, 'deepseek')
    enable(args, request)
    calls = install_http(module, monkeypatch, analysis, {}, {}, {})
    for _ in range(2):
        with pytest.raises(ModelExhausted, match='code'):
            module.propose(args)
    assert len(calls) == 4


def test_failed_model_drains_pool_then_resume_keeps_completed_device_rounds(tmp_path, monkeypatch):
    run = prepare(tmp_path, rounds=3, calls=3)
    search.search(run, ROOT, external=True)
    choices = [proposal(run, 11), proposal(run, 8, 'packing', 2)]
    respond(run, choices)
    original = campaign.execute
    def fail(argv, cwd, output, timeout, env):
        if env.get('AFD_RSI_SIDE') == 'candidate' and argv[-1] == 'build' and '0001' in Path(cwd).parts:
            argv = [sys.executable, '-c', 'raise RuntimeError("compile failure")']
        return original(argv, cwd, output, timeout, env)
    monkeypatch.setattr(campaign, 'execute', fail)
    search.search(run, ROOT)
    pending = read_json(run / 'search/pending.json')
    context = read_json(run / pending['request'])
    write_json(run / pending['response'], failure_response(context, ModelExhausted('code', 'model token limit')))
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'waiting_model' and state['calls'] == 2
    assert records(run)[0]['decision'] == 'model_failed'
    assert records(run)[1]['decision'] == 'promoted'
    before = {p: digest(p) for p in (run / 'attempts').rglob('process.json')}
    best = digest(run / 'search/best.json')
    assert search.resume_model(run, ROOT)
    assert not search.resume_model(run, ROOT)
    search.search(run, ROOT)
    assert digest(run / 'search/best.json') == best
    assert all(digest(path) == sha for path, sha in before.items())
    assert read_json(run / 'search/pending.json')['call'] == 3
    assert read_json(run / 'search/state.json')['model_calls_reserved'] == 15


def test_failure_receipt_cannot_be_replayed_for_another_request(tmp_path):
    value = failure_response(dict(source_hash='old'), ModelExhausted('analysis', 'bad JSON'))
    from afd_rsi.model_recovery import validate_failure
    with pytest.raises(ValueError, match='receipt'):
        validate_failure(value, dict(source_hash='new'))


def test_waiting_model_status_is_resumable_and_not_success_or_framework_failure():
    view = optimize.describe_status(dict(status='waiting_model', progress=dict(state=dict(status='waiting_model', reason='budget output'))))
    assert '模型恢复' in view['label'] and '失败' not in view['label']
    assert view['stop_follow']


@pytest.mark.parametrize('provider', ['deepseek', 'codex'])
def test_seeded_analysis_needs_only_a_coding_call(tmp_path, monkeypatch, provider):
    module, args, request, analysis = setup(tmp_path, provider)
    request['analysis_seed'] = dict(analysis=analysis, origin=dict(round='0001', analysis_sha256='sealed'))
    enable(args, request)
    if provider == 'deepseek':
        calls = install_http(module, monkeypatch, coding(request))
    else:
        calls = []
        monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
        def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
            assert model == 'gpt-6.1-sol'
            calls.append(model)
            value = coding(request)
            write_json(output, value)
            return value, [dict(model=model)]
        monkeypatch.setattr(module, 'invoke', invoke)
    module.propose(args)
    assert len(calls) == 1
    assert read_json(args.output)['proposals'][0]['patch']


def test_controller_seeds_only_unchanged_analysis_and_keeps_budget(tmp_path):
    run = prepare(tmp_path, rounds=2, calls=2, model_calls=9)
    search.search(run, ROOT, external=True)
    pending = read_json(run / 'search/pending.json')
    request = read_json(run / pending['request'])
    entry = proposal(run, 8)
    entry.pop('patch')
    entry['files'] = ['s0']
    analysis = dict(stop=False, reason='measured direction', proposals=[entry], needs_context=[])
    root = (run / pending['response']).parent
    journal = Journal(root, request)
    journal.reserve('analysis')
    journal.complete('analysis', analysis)
    write_json(run / pending['response'], failure_response(request, ModelExhausted('code', 'too long')))
    search.search(run, ROOT)
    assert read_json(run / 'search/state.json')['status'] == 'waiting_model'
    assert search.resume_model(run, ROOT)
    search.search(run, ROOT)
    next_pending = read_json(run / 'search/pending.json')
    next_request = read_json(run / next_pending['request'])
    assert next_request['analysis_seed']['analysis'] == analysis
    assert read_json(run / 'search/state.json')['model_calls_reserved'] == 9


def test_detached_worker_waits_then_resumes_completed_campaign(tmp_path, monkeypatch):
    import time
    from test_optimize import fake_codex
    from afd_rsi.runner import owned_alive
    fake_codex(tmp_path, monkeypatch)
    monkeypatch.setenv('AFD_RSI_TEST_BAD_CODE', '1')
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text((ROOT / 'recipes/cpu_adaptive_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source'))
        .replace('max_rounds = 9', 'max_rounds = 1').replace('max_calls = 9', 'max_calls = 2'))
    job = tmp_path / 'job'
    assert optimize.main([str(recipe), '--output', str(job), '--poll-seconds', '0.1'], ROOT) == 0
    def wait(target):
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            state = read_json(job / 'status.json')
            if state['status'] in optimize.TERMINAL:
                assert state['status'] == target, (state, (job / 'worker.log').read_text()[-4000:])
                return state
            time.sleep(.1)
        pytest.fail((job / 'worker.log').read_text()[-4000:])
    try:
        wait('waiting_model')
        run = job / 'execution/campaign'
        state = read_json(run / 'search/state.json')
        baseline = state['baseline']
        receipts = {p: digest(p) for p in (run / 'attempts').rglob('process.json')}
        first = read_json(job / 'proposals/1/agent/usage.json')
        assert first['model_calls'] == 4
        deadline = time.monotonic() + 10
        while owned_alive(read_json(job / 'worker.json')['tag']) and time.monotonic() < deadline:
            time.sleep(.1)
        monkeypatch.setenv('AFD_RSI_TEST_BAD_CODE', '0')
        assert optimize.main(['resume', str(job)], ROOT) == 0
        wait('finished')
        assert read_json(run / 'search/state.json')['baseline'] == baseline
        assert all(digest(path) == sha for path, sha in receipts.items())
        assert read_json(job / 'proposals/2/agent/usage.json')['model_calls'] == 1
        assert read_json(run / 'search/best.json')['confirmation'] is not None
    finally:
        from afd_rsi.runner import stop_owned
        record = read_json(job / 'worker.json')
        stop_owned(record)


def test_seeded_http_analysis_retains_exact_sdk_reads(tmp_path, monkeypatch):
    from afd_rsi.model_protocol import ModelContext
    module, args, request, analysis = setup(tmp_path, 'deepseek')
    old = ModelContext(request)
    old.read([['d0', 1, 1]])
    request['analysis_seed'] = dict(analysis=analysis, reads=old.reads(), origin=dict(round='0001'))
    enable(args, request)
    calls = install_http(module, monkeypatch, coding(request))
    module.propose(args)
    assert len(calls) == 1 and 'REQUIRED_HEADER_TOKEN' in str(calls[0]['messages'])


def test_context_full_reset_is_explicit_and_does_not_repeat_analysis(tmp_path, monkeypatch):
    from afd_rsi.model_recovery import ContextFull
    module, args, request, analysis = setup(tmp_path, 'codex')
    enable(args, request)
    monkeypatch.setattr(module.shutil, 'which', lambda _: 'codex')
    calls = []
    def invoke(executable, workspace, schema, output, prompt, model, effort, prefix, session):
        calls.append(model)
        if len(calls) == 2:
            old = read_json(session)
            write_json(session, {**old, 'thread_id': '12345678-1234-4234-8234-123456789abc'})
            raise ContextFull('context window exceeded')
        if len(calls) == 3:
            assert read_json(session)['thread_id'] is None
        value = analysis if model == 'gpt-6-astra' else coding(request)
        write_json(output, value)
        return value, [dict(model=model)]
    monkeypatch.setattr(module, 'invoke', invoke)
    module.propose(args)
    assert calls == ['gpt-6-astra', 'gpt-6.1-sol', 'gpt-6.1-sol']
    assert list(args.output.parent.glob('*.context-reset.json'))


def test_resumable_checkpoint_requires_clean_owned_processes(tmp_path):
    request = dict(model_recovery=1)
    journal = Journal(tmp_path, request)
    journal.complete('analysis', dict(stop=True, proposals=[], reason='complete'))
    write_json(tmp_path / 'launches/a/process.json', dict(cleanup_complete=False))
    assert not resumable(tmp_path, request)
    write_json(tmp_path / 'launches/a/process.json', dict(cleanup_complete=True))
    assert resumable(tmp_path, request)


def test_resume_exhausted_budget_is_a_normal_stop_without_refund(tmp_path):
    run = prepare(tmp_path, calls=1)
    search.search(run, ROOT, external=True)
    pending = read_json(run / 'search/pending.json')
    request = read_json(run / pending['request'])
    write_json(run / pending['response'], failure_response(request, ModelExhausted('analysis', 'bad response')))
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'budget_exhausted' and state['calls'] == 1
    assert not search.resume_model(run, ROOT)
    after = read_json(run / 'search/state.json')
    assert after['status'] == 'budget_exhausted' and after['calls'] == state['calls']
    assert after['model_calls_reserved'] == state['model_calls_reserved']


def test_model_resume_cannot_extend_stage_deadline(tmp_path, monkeypatch):
    import afd_rsi.model_recovery as recovery
    request = dict(model_recovery=1, agent_timeout_seconds=60)
    journal = Journal(tmp_path, request)
    journal.reserve('analysis')
    deadline = journal.state['deadline']
    monkeypatch.setattr(recovery.time, 'time', lambda: deadline + 1)
    with pytest.raises(ModelExhausted, match='time budget'):
        Journal(tmp_path, request).reserve('analysis')
    state = read_json(tmp_path / 'model-state.json')
    assert state['deadline'] == deadline and len(state['stages']['analysis']['attempts']) == 1


def test_clean_failed_preflight_can_resume_without_resetting_launch_identity(tmp_path):
    from afd_rsi.model_recovery import request_hash
    request = dict(model_recovery=1, source_hash='same')
    write_json(tmp_path / 'model-start.json', dict(request_hash=request_hash(request), deadline=9e12))
    write_json(tmp_path / 'launches/a/process.json', dict(status='failed', cleanup_complete=True))
    assert resumable(tmp_path, request)
    with pytest.raises(ValueError, match='another request'):
        resumable(tmp_path, dict(request, source_hash='changed'))
