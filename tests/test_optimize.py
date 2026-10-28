"""Real detached local orchestration with a fake model, plus transport boundaries."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

from afd_rsi import optimize, search
from afd_rsi.evidence import read_json, write_json
from afd_rsi.runner import owned_alive, stop_owned

ROOT = Path(__file__).resolve().parents[1]


def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / 'bin/codex'
    script.parent.mkdir()
    script.write_text(f'''#!{sys.executable}
import sys,json
from pathlib import Path
root=Path(sys.argv[sys.argv.index('--cd')+1])
request=json.loads((root/'request.json').read_text())
assert request['strategy']['mode']=='deep'
assert '--ephemeral' not in sys.argv
if 'resume' in sys.argv: assert sys.argv[sys.argv.index('resume')+1]=='12345678-1234-4234-8234-123456789abc'
schema=json.loads(Path(sys.argv[sys.argv.index('--output-schema')+1]).read_text())
code='patch' in schema['properties']['proposals']['items']['properties']
assert sys.argv[sys.argv.index('--model')+1]==('gpt-6.1-sol' if code else 'gpt-6-astra')
p={{'hypothesis':'synthetic fixture faster','category':'latency','priority':3,'evidence':['baseline']}}
if code: p['patch']={(ROOT/'examples/faster.patch').read_text()!r}
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text(json.dumps({{'stop':False,'reason':'fixture only','proposals':[p]}}))
print(json.dumps({{'type':'thread.started','thread_id':'12345678-1234-4234-8234-123456789abc'}}))
print(json.dumps({{'type':'turn.completed','usage':{{'input_tokens':1,'output_tokens':1}}}}))
''')
    script.chmod(0o755)
    monkeypatch.setenv('PATH', str(script.parent) + os.pathsep + os.environ['PATH'])


def test_detached_local_round_confirms_and_downloads(tmp_path, monkeypatch):
    fake_codex(tmp_path, monkeypatch)
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text((ROOT/'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT/'examples/source'))
        .replace('max_rounds = 3','max_rounds = 1').replace('max_calls = 3','max_calls = 1'))
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
    recipe.write_text((ROOT/'recipes/cpu_search.toml').read_text().replace('../examples/source',str(ROOT/'examples/source')))
    run=campaign.prepare(recipe,tmp_path/'run',ROOT)
    search.search(run,ROOT,external=True)
    plan=campaign.load(run,ROOT); best=read_json(run/'search/best.json')
    early=search.request_context(run,plan,best,[],1)
    late=search.request_context(run,plan,best,[],3)
    assert early['strategy']['mode']=='deep'
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
    write_json(job/'job.json',{'executor':'local'})
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
