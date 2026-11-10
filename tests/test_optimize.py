"""Real detached local orchestration with a fake model, plus transport boundaries."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from afd_rsi import optimize, search
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.runner import owned_alive, stop_owned

ROOT = Path(__file__).resolve().parents[1]


def test_gsm8k_dependencies_checked_before_campaign_launch(tmp_path):
    class Executor:
        base = '/remote'
        calls = []

        def python(self, *args):
            self.calls.append(args)

    executor = Executor()
    optimize.preflight_service_dependencies(
        tmp_path, {'recipe': '[service]\nargs = ["--test", "smoke,gsm8k", "--dataset-path", "/data"]'}, executor)
    assert len(executor.calls) == 1
    assert 'check_gsm8k_dependencies' in executor.calls[0][0]
    assert executor.calls[0][1:] == ('/remote/controller/benchmarks/service', '1')
    assert read_json(tmp_path / 'status.json')['activity'] == 'preflight'
    optimize.preflight_service_dependencies(
        tmp_path, {'recipe': '[service]\nargs = ["--test", "smoke"]'}, executor)
    assert len(executor.calls) == 1


def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / 'bin/codex'
    script.parent.mkdir()
    script.write_text(f'''#!{sys.executable}
import sys,json,os
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--cd')+1])
request=json.loads((root/'request.json').read_text())
assert request['strategy']['mode'] in ('deep','global-redesign')
assert '--ephemeral' not in sys.argv
if 'resume' in sys.argv: assert sys.argv[sys.argv.index('resume')+1]=='12345678-1234-4234-8234-123456789abc'
schema=json.loads(Path(sys.argv[sys.argv.index('--output-schema')+1]).read_text())
fields=schema['properties']['proposals']['items']['properties']
code='patch' in fields or 'edits' in fields
assert sys.argv[sys.argv.index('--model')+1]==('gpt-6.1-sol' if code else 'gpt-6-astra')
p={{'hypothesis':'synthetic fixture faster','category':'latency','priority':3,'evidence':['baseline']}}
if 'experiment' in fields: p['experiment']={{key:'fixture '+key for key in fields['experiment']['required']}}
if 'direction' in fields: p['direction']={{'id':'tiling','mechanism':'fixture latency','step':2,'refs':['skills/ascendc-perf-optimize/SKILL.md'],'use':'apply: CPU fixture only','next':''}}
if code: p['patch']={(ROOT/'examples/faster.patch').read_text()!r}
if 'files' in fields: p['files']=['s0']
result={{'stop':False,'reason':'fixture only','proposals':[p],'needs_context':[]}}
if 'edits' in fields:
    view=json.loads((root/'code.json').read_text())
    old=(root/view['file_paths']['s0']).read_text()
    result.update(base=view['base'],proposals=[{{'id':0,'edits':[{{'file':'s0','old':old,'new':old.replace('10.0','8.0')}}]}}])
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('{{' if code and os.environ.get('AFD_RSI_TEST_BAD_CODE') == '1' else json.dumps(result))
print(json.dumps({{'type':'thread.started','thread_id':'12345678-1234-4234-8234-123456789abc'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':1,'output_tokens':1}}}}))
''')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(script.parent) + os.pathsep + os.environ['PATH'])


@pytest.mark.parametrize('recipe_name', ['cpu_search', 'cpu_adaptive_search'])
def test_detached_local_round_confirms_and_downloads(tmp_path, monkeypatch, recipe_name):
    fake_codex(tmp_path, monkeypatch)
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text((ROOT/f'recipes/{recipe_name}.toml').read_text().replace('../examples/source', str(ROOT/'examples/source'))
        .replace('max_rounds = 9','max_rounds = 1').replace('max_calls = 9','max_calls = 1'))
    job = tmp_path/'job'
    assert optimize.main([str(recipe),'--output',str(job),'--poll-seconds','0.1'], ROOT)==0
    assert read_json(job/'worker.json')['pid'] != os.getpid()
    try:
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            state=read_json(job/'status.json')
            if state['status'] in optimize.TERMINAL:
                break
            time.sleep(0.2)
        assert state['status']=='finished', (state, (job/'worker.log').read_text()[-6000:])
        best=read_json(job/'results/best-export/best.json')
        assert best['screen']['attempt'] != best['confirmation']['attempt']
        assert read_json(job/'proposals/1/agent/process.json')['status']=='completed'
        assert read_json(job/'results/campaign/search/state.json')['calls']==1
        assert read_json(job/'results/campaign/search/rounds/0001/usage.json')['model_calls'] == 2
        # An unchanged, already consumed launch is idempotent.
        config=read_json(job/'job.json'); ex=optimize.Executor(config,job)
        old=ex.invoke('status',str(state['phase']))
        assert ex.invoke('launch',str(state['phase']))==old
        assert (job/'results.tar.gz').exists()
        assert read_json(job/'codex-session/session.json')['thread_id']=='12345678-1234-4234-8234-123456789abc'
        command=read_json(job/'proposals/1/agent/codex-events.command.json')
        assert command[command.index('resume')+1]=='12345678-1234-4234-8234-123456789abc'
        # Reconcile a terminal controller in a new phase without repeating a proposal.
        ex.invoke('launch', str(state['phase']+1))
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            recovered=ex.invoke('status', str(state['phase']+1))
            if recovered['exit'] is not None:
                break
            time.sleep(0.1)
        assert recovered['exit']['code']==0, recovered
        assert recovered['state']['calls']==1
        assert recovered['best']==best

    finally:
        # Never signal by an unverified bare PID, even on a failed test.
        for file in [job/'worker.json', *job.glob('execution/drivers/*/launch.json')]:
            if file.exists():
                record=read_json(file)
                if owned_alive(record['tag']):
                    stop_owned(record)


def test_strategy_changes_with_rounds(tmp_path):
    # Exercise actual request generation using the normal baseline fixture.
    from afd_rsi import campaign
    recipe=tmp_path/'recipe.toml'
    recipe.write_text((ROOT/'recipes/cpu_search.toml').read_text()
                      .replace('../examples/source',str(ROOT/'examples/source'))
                      .replace('max_rounds = 9', 'max_rounds = 9\nglobal_redesign_round = 1'))
    run=campaign.prepare(recipe,tmp_path/'run',ROOT)
    search.search(run,ROOT,external=True)
    plan=campaign.load(run,ROOT); best=read_json(run/'search/best.json')
    # The deep/incremental boundary follows the frozen round budget, so read it
    # from the plan instead of hardcoding a round number that a budget change
    # would silently invalidate.
    total=plan['search']['max_rounds']
    deep_rounds=1 if total==1 else min(total-1,(2*total+2)//3)
    early=search.request_context(run,plan,best,[],1)
    middle=search.request_context(run,plan,best,[],2)
    late=search.request_context(run,plan,best,[],deep_rounds+1)
    assert deep_rounds+1<=total, 'the fixture budget must reserve an incremental round'
    assert early['strategy']['mode']=='global-redesign'
    assert 'Compare architectural alternatives' in early['strategy']['guidance']
    assert middle['strategy']['mode']=='deep'
    assert late['strategy']['mode']=='incremental'
    assert early['allowed_files']==late['allowed_files']==plan['allowed_files']
    assert early['objective']==late['objective']


def test_itask_transport_quotes_paths_and_is_lazy(tmp_path, monkeypatch):
    ex=optimize.Executor(dict(executor='itask',task='task',workdir='/a path',id='abc',python='python3',
                             setup_script='/a path/setup.sh'),tmp_path)
    calls=[]
    monkeypatch.setattr(ex,'command',lambda argv: calls.append(argv) or '')
    ex.python('print("hello")','a;$(false)')
    cmd=calls[0]
    assert cmd[:7]==['itask-guian','exec','task','--tty=false','--stdin=false','--','bash']
    assert "'a;$(false)'" in cmd[-1] and "source '/a path/setup.sh'" in cmd[-1]


def test_status_checks_wrapper_not_finished_child(tmp_path, monkeypatch):
    directory=tmp_path/'drivers/0';directory.mkdir(parents=True)
    from afd_rsi.runner import host_identity
    write_json(directory/'launch.json',dict(tag='wrapper',host_identity=host_identity()))
    write_json(directory/'process.json',dict(tag='finished child',host_identity=host_identity()))
    monkeypatch.setattr(optimize,'owned_alive',lambda tag: tag=='wrapper')
    assert 'lost_driver' not in optimize.remote_status(tmp_path,'0')
    monkeypatch.setattr(optimize,'owned_alive',lambda tag: False)
    assert optimize.remote_status(tmp_path,'0')['lost_driver']


def test_interrupted_proposer_is_not_retried(tmp_path,monkeypatch):
    from afd_rsi.evidence import digest
    job=tmp_path/'job';job.mkdir()
    write_json(job/'job.json',{'executor':'local','recipe':''})
    write_json(job/'status.json',{'phase':0,'status':'running'})
    request=tmp_path/'request.json';write_json(request,{'finite':'fixture'})
    pending={'call':1,'request':'request.json','response':'proposal.json','request_hash':digest(request)}
    write_json(job/'proposals/1/agent/process.json',{'status':'interrupted','cleanup_complete':True})
    class FakeExecutor:
        base='remote'
        def __init__(self,*args):pass
        def invoke(self,*args):return {'exit':{'code':0},'state':{'status':'running'},'pending':pending}
        def fetch(self,source,destination):destination.write_bytes(request.read_bytes())
    monkeypatch.setattr(optimize,'Executor',FakeExecutor)
    monkeypatch.setattr(optimize,'stage',lambda *args:None)
    monkeypatch.setattr(search,'propose',lambda *args:pytest.fail('must not retry interrupted model dispatch'))
    with pytest.raises(ValueError,match='interrupted proposer'):
        optimize.worker(job)
    assert read_json(job/'status.json')['status']=='error'
    assert read_json(job/'proposals/1/agent/process.json')['status']=='interrupted'


def test_restart_refreshes_controller_and_keeps_measured_rounds(tmp_path, monkeypatch):
    """A restart replaces the proposer snapshot and archives the abandoned attempt."""
    root = tmp_path / 'root'
    for name in ('src', 'skills', 'knowledge', 'benchmarks', 'examples', 'recipes', 'tools'):
        (root / name).mkdir(parents=True)
    (root / 'benchmarks/agents').mkdir(parents=True)
    (root / 'benchmarks/agents/deepseek_agent.py').write_text('# fixed adapter\n')
    job = tmp_path / 'job'
    (job / 'controller/benchmarks/agents').mkdir(parents=True)
    (job / 'controller/benchmarks/agents/deepseek_agent.py').write_text('# defective adapter\n')
    write_json(job / 'job.json', {'executor': 'local', 'retry_remote': False})
    write_json(job / 'status.json', {'phase': 3, 'status': 'error'})
    write_json(job / 'proposals/5/request.json', {'provider_hash': 'frozen-digest'})
    write_json(job / 'proposals/5/agent/process.json', {'status': 'failed', 'cleanup_complete': True})
    write_json(job / 'proposals/4/agent/process.json', {'status': 'completed', 'cleanup_complete': True})
    spawned = []
    monkeypatch.setattr(optimize, 'spawn', lambda target: spawned.append(target))
    monkeypatch.setattr(optimize, 'skills', type('S', (), {'validate': staticmethod(lambda *_a, **_k: {})}))

    optimize.restart(job, root, 'proposer rejected a valid envelope')

    record = read_json(job / 'restart.json')
    assert record['reason'] == 'proposer rejected a valid envelope'
    assert record['expected_provider_hash'] == 'frozen-digest'
    # The fixed adapter is now staged, the defective one is preserved.
    assert 'fixed adapter' in (job / 'controller/benchmarks/agents/deepseek_agent.py').read_text()
    assert 'defective adapter' in (job / record['archived_controller'] /
                                   'benchmarks/agents/deepseek_agent.py').read_text()
    # Only the unfinished attempt is archived; measured rounds stay where they are.
    archived = list((job / 'proposals/5').glob('agent-abandoned-*'))
    assert len(archived) == 1 and read_json(archived[0] / 'abandoned.json')['reason']
    assert not (job / 'proposals/5/agent').exists()
    assert (job / 'proposals/4/agent').exists()
    assert read_json(job / 'job.json')['retry_remote'] is True
    assert read_json(job / 'status.json')['phase'] == 4
    assert spawned == [job]
    with pytest.raises(ValueError, match='already restarted'):
        optimize.restart(job, root, 'again')


def test_restart_refuses_while_the_worker_is_alive(tmp_path, monkeypatch):
    job = tmp_path / 'job'
    job.mkdir()
    write_json(job / 'job.json', {'executor': 'local'})
    write_json(job / 'status.json', {'phase': 0, 'status': 'running'})
    write_json(job / 'worker.json', {'tag': 'live', 'host_identity': 'host'})
    monkeypatch.setattr(optimize, 'owned_alive', lambda tag: True)
    with pytest.raises(ValueError, match='still running'):
        optimize.restart(job, tmp_path, 'nope')
    assert (job / 'status.json').exists() and not (job / 'restart.json').exists()


def test_changed_adapter_needs_a_recorded_override(tmp_path, monkeypatch):
    """Swapping the proposer is allowed only as an explicit, evidenced upgrade."""
    root = tmp_path / 'root'
    (root / 'benchmarks/agents').mkdir(parents=True)
    (root / 'benchmarks/agents/deepseek_agent.py').write_text('# fixed\n')
    request = tmp_path / 'request.json'
    write_json(request, {'provider': 'benchmarks/agents/deepseek_agent.py',
                         'provider_hash': 'frozen-digest', 'deadline': time.time() + 600,
                         'agent_timeout_seconds': 30, 'max_response_bytes': 1000,
                         'evidence': [], 'fingerprint': 'fp'})
    output = tmp_path / 'agent'
    output.mkdir()
    monkeypatch.setattr(search, 'skills', type('S', (), {'validate': staticmethod(lambda *_a, **_k: {})}))
    monkeypatch.setattr(search, 'execute',
                        lambda *a, **k: {'status': 'completed', 'cleanup_complete': True})
    write_json(output / 'proposal.json', {'stop': True, 'reason': 'no change', 'proposals': []})

    with pytest.raises(ValueError, match='differs from the frozen campaign adapter'):
        search.propose(request, output, root)

    override = {'replaces': 'frozen-digest', 'digest': digest(root / 'benchmarks/agents/deepseek_agent.py'),
                'reason': 'fixed envelope handling'}
    search.propose(request, output, root, provider_override=override)
    record = read_json(output / 'provider-change.json')
    assert record['replaces'] == 'frozen-digest' and record['digest'] == override['digest']
    assert record['campaign_fingerprint'] == 'fp'

    with pytest.raises(ValueError, match='differs from the frozen campaign adapter'):
        search.propose(request, output, root, provider_override={**override, 'digest': 'other'})

def test_partial_bootstrap_reuses_sealed_archive(tmp_path):
    from afd_rsi.evidence import digest
    job=tmp_path/'job';job.mkdir()
    (job/'controller').mkdir()
    (job/'controller/identity').write_text('frozen controller')
    config={'source':str(ROOT/'examples/source'),'id':'fixture',
            'recipe':'source = "unused"\n','experience_dirs':[]}
    class InterruptedUpload:
        remote=False
        base='/unused-fixture'
        fail=True
        def python(self,*args):pass
        def send(self,*args):
            if self.fail:
                raise RuntimeError('upload interrupted')
    executor=InterruptedUpload()
    with pytest.raises(RuntimeError,match='upload interrupted'):
        optimize.stage(job,config,executor)
    sealed=digest(job/'payload.tar.gz')
    assert not (job/'staged.json').exists()
    # A later gzip mtime must not change the bootstrap identity.
    executor.fail=False
    optimize.stage(job,config,executor)
    assert digest(job/'payload.tar.gz')==sealed
    assert read_json(job/'staged.json')['payload_sha256']==sealed
    (job/'staged.json').unlink()
    (job/'payload.tar.gz').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='sealed payload changed'):
        optimize.stage(job,config,executor)


def test_results_include_diagnostic_compiler_logs_without_build_sources(tmp_path):
    import tarfile
    build = tmp_path / 'campaign/attempts/a/baseline/profile/p/diagnostic-build'
    (build / 'build').mkdir(parents=True)
    (build / 'build/stderr.log').write_text("error: undeclared validation")
    (build / 'source').mkdir()
    (build / 'source/secret.o').write_bytes(b'large binary')
    optimize.pack_results(tmp_path)
    with tarfile.open(tmp_path / 'results.tar.gz') as tar:
        names = tar.getnames()
        assert str((build / 'build/stderr.log').relative_to(tmp_path)) in names
        assert not any(name.endswith('secret.o') for name in names)


def test_remote_failure_publishes_root_error_and_downloadable_evidence(tmp_path, monkeypatch):
    import tarfile
    from afd_rsi import campaign
    (tmp_path / 'campaign').mkdir()
    (tmp_path / 'drivers/0').mkdir(parents=True)
    monkeypatch.setattr(campaign, 'load', lambda *_: dict(expires_at=time.time() + 60, timeout_seconds=10))
    def failed(argv, cwd, output, *args):
        write_json(output / 'process.json', dict(status='failed', cleanup_complete=True, exit_code=2))
        (output / 'stderr.log').write_text('afd-rsi: request metadata exceeds the frozen context budget\n')
        return read_json(output / 'process.json')
    monkeypatch.setattr(optimize, 'execute', failed)
    with pytest.raises(RuntimeError, match='request metadata exceeds'):
        optimize.remote_run(tmp_path, '0')
    failure = read_json(tmp_path / 'drivers/0/exit.json')
    assert failure['code'] == 1 and failure['results_sha256'] == digest(tmp_path / 'results.tar.gz')
    class Executor:
        base = str(tmp_path)
        def fetch(self, source, target):
            import shutil
            shutil.copyfile(source, target)
    job = tmp_path / 'download'
    job.mkdir()
    optimize.collect_results(job, Executor(), {'exit': failure})
    assert 'request metadata exceeds' in (job / 'results/drivers/0/command/stderr.log').read_text()
    with tarfile.open(job / 'results.tar.gz') as tar:
        assert 'drivers/0/command/process.json' in tar.getnames()


def test_request_preparation_does_not_report_the_last_baseline_profile(tmp_path):
    write_json(tmp_path / 'campaign/search/state.json', {'active': '0001', 'baseline': 'base'})
    write_json(tmp_path / 'campaign/attempts/base/progress.json', {
        'steps': [{'side': 'baseline', 'phase': 'profile', 'trial': 'profile-trial'}]})
    status = optimize.remote_status(tmp_path, '0')
    assert status['round_phase'] == 'request' and status['step'] is None
    view = optimize.describe_status(dict(status='error', progress=status, error='request metadata exceeds'))
    assert view['stage'] == '准备 Agent 上下文'


@pytest.mark.parametrize('client_status', ['completed', 'timeout', 'failed'])
def test_task_start_preserves_receipt_and_reconciles_ready_pod(tmp_path, monkeypatch, client_status):
    calls = []
    ex = optimize.Executor(dict(executor='itask', task='task', workdir='/work', id='abc', python='python3',
                               setup_script=None), tmp_path)
    def execute(argv, cwd, output, timeout, env):
        assert argv == ['itask-guian', 'start', 'task'] and timeout == 1800
        write_json(output / 'process.json', dict(status=client_status, cleanup_complete=True))
        (output / 'stdout.log').write_text('migration completed after 180 seconds')
        return dict(status=client_status, cleanup_complete=True)
    monkeypatch.setattr(optimize, 'execute', execute)
    monkeypatch.setattr(ex, 'command', lambda argv, **kwargs: calls.append((argv, kwargs)) or '')
    ex.start_task()
    assert len(calls) == 1 and calls[0][0][-1] == 'true'
    assert calls[0][1]['timeout'] == 30
    receipt = next((tmp_path / 'task-start').glob('*/readiness.json'))
    assert read_json(receipt) == {'ready': True, 'start_status': client_status}


def test_task_start_does_not_retry_when_readiness_is_unknown(tmp_path, monkeypatch):
    ex = optimize.Executor(dict(executor='itask', task='task', workdir='/work', id='abc', python='python3',
                               setup_script=None), tmp_path)
    monkeypatch.setattr(optimize, 'execute', lambda *args: dict(status='timeout', cleanup_complete=True))
    def failure(*args, **kwargs):
        raise RuntimeError('Pod not Ready')
    monkeypatch.setattr(ex, 'command', failure)
    with pytest.raises(RuntimeError, match='不要重复提交启动'):
        ex.start_task()


def test_worker_lifetime_lock_detects_unrecorded_exit(tmp_path):
    import fcntl
    from afd_rsi.runner import host_identity
    write_json(tmp_path / 'worker.json', dict(host_identity=host_identity(), tag='fixture', pid=123))
    value = dict(status='running', updated_at=0, progress={'state': {'calls': 0}})
    with (tmp_path / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert optimize.worker_running(tmp_path, value) is True
        assert not optimize.describe_status(value, True)['stop_follow']
    assert optimize.worker_running(tmp_path, value) is False
    view = optimize.describe_status(value, False)
    assert view['stop_follow'] and view['exit_code'] == 2 and '异常退出' in view['label']


@pytest.mark.parametrize('status,search,best,code,label', [
    ('error', '', None, 1, '失败'),
    ('finished', 'baseline_failed', None, 1, '失败'),
    ('finished', 'budget_exhausted', None, 0, '未获得'),
    ('finished', 'agent_stopped', {'confirmation': {'attempt': 'fresh'}}, 0, '有独立确认'),
])
def test_human_status_distinguishes_terminal_results(status, search, best, code, label):
    value = dict(status=status, progress=dict(state=dict(status=search), best=best))
    view = optimize.describe_status(value, False)
    assert view['stop_follow'] and view['exit_code'] == code and label in view['label']


def test_status_json_and_human_error_have_nonzero_exit(tmp_path, capsys):
    write_json(tmp_path / 'status.json', dict(status='error', error="itask-guian start timed out after 180 seconds"))
    assert optimize.main(['status', str(tmp_path)]) == 1
    text = capsys.readouterr().out
    assert '失败' in text and 'iTASK 启动/迁移' in text and '180 seconds' in text
    assert optimize.main(['status', str(tmp_path), '--json']) == 1
    assert json.loads(capsys.readouterr().out)['monitor']['exit_code'] == 1


def test_follow_detaches_without_stopping_worker(tmp_path, monkeypatch, capsys):
    write_json(tmp_path / 'status.json', dict(status='running', updated_at=time.time(), heartbeat_at=time.time()))
    write_json(tmp_path / 'job.json', dict(poll_seconds=1))
    monkeypatch.setattr(optimize, 'worker_running', lambda *args: True)
    def interrupt(_):
        raise KeyboardInterrupt
    monkeypatch.setattr(optimize.time, 'sleep', interrupt)
    assert optimize.watch(tmp_path, follow=True) == 130
    assert '未向后台' in capsys.readouterr().out


def test_unknown_host_is_not_reported_as_dead_worker(tmp_path):
    write_json(tmp_path / 'worker.json', dict(host_identity={'host': 'foreign', 'boot': 'foreign'}, tag='x'))
    value = dict(status='running', updated_at=0)
    assert optimize.worker_running(tmp_path, value) is None
    assert '状态未知' in optimize.describe_status(value, None)['label']


def test_baseline_failure_waits_for_result_collection_and_old_activity_is_ignored():
    value = dict(status='running', activity='collect', progress={'state': {'status': 'baseline_failed'}})
    assert not optimize.describe_status(value)['stop_follow']
    assert optimize.describe_status(value)['stage'] == '下载并校验结果'
    value.update(activity='resume', progress={'step': {'side': 'baseline', 'phase': 'build'}})
    assert '构建算子' in optimize.describe_status(value)['stage']
