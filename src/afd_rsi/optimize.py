"""Detached orchestration; local or iTASK execution, authenticated local proposals."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import uuid
from pathlib import Path

from . import campaign, skills
from .contracts import load_recipe
from .evidence import digest, read_json, write_json
from .runner import execute, host_identity, owned_alive
from .workspace import copy_source, inventory
from .model_recovery import resumable as resumable_model, Journal, ModelExhausted, failure_response

MARKER = 'AFD_RSI_STATUS:'
TERMINAL = {'finished', 'error', 'waiting_model'}
ACTIVITIES = {'auth': '检查 iTASK 登录/权限', 'snapshot': '封存源码和框架', 'task_start': '等待 iTASK 启动/迁移（最多 30 分钟）',
              'upload': '上传并部署框架', 'preflight': '预检评测依赖',
              'collect': '下载并校验结果', 'resume': '恢复后台任务'}
PHASES = {'build': '构建算子', 'correctness': '算子精度验证', 'benchmark': '算子性能比较',
          'profile': '采集 profiling 和内部打点', 'service': '模型服务验收',
          'screening': '候选初筛', 'confirming': '独立复测', 'request': '准备 Agent 上下文',
          'awaiting_agent': '等待 Agent 提案'}


def describe_status(value, worker_alive=True):
    progress = value.get('progress') or {}
    state = progress.get('state', {})
    step = progress.get('step') or {}
    status = value['status']
    stage = (ACTIVITIES.get(value.get('activity'))
             if status in ('queued', 'preparing', 'error') or value.get('activity') == 'collect' else None)
    if not stage and status == 'error' and 'itask-guian' in value.get('error', '') and 'start' in value['error']:
        stage = 'iTASK 启动/迁移'
    if not stage:
        stage = ('Agent 分析证据并生成补丁' if status == 'proposing' else
                 '候选已提交，等待执行端' if status == 'submitted' else
                 '等待 worker 接管' if status == 'queued' else
                 PHASES.get(step.get('phase'), PHASES.get(progress.get('round_phase'), '等待执行端状态')))
        if step.get('side') and status not in ('proposing', 'submitted'):
            stage = ('基线 / ' if step['side'] == 'baseline' else '候选 / ') + stage
        service = step.get('service') or {}
        if service.get('stage'):
            stage += ' / ' + str(service['stage'])
        if progress.get('round_phase') in ('screening', 'confirming'):
            stage = PHASES[progress['round_phase']] + ' / ' + stage
    reason = value.get('error') or state.get('reason', '')
    code, done = 0, False
    if status == 'waiting_model':
        label, done = '等待模型恢复：已完成轮次已保留，可用 optimize resume 续跑', True
    elif status == 'error' or (status == 'finished' and state.get('status') == 'baseline_failed'):
        label, code, done = '失败：框架已停止推进', 1, True
    elif status == 'finished' and state.get('status') == 'context_required':
        label, code, done = '需要补充依赖上下文：未耗尽优化方向', 1, True
    elif status == 'finished':
        best = progress.get('best') or {}
        label = '已完成：有独立确认的优化候选' if best.get('confirmation') else '已结束：未获得独立确认的改进'
        done = True
    elif worker_alive is False:
        label, code, done = '异常退出：后台 worker 已消失', 2, True
        reason = '保存状态尚未结束，但 worker 已退出；远端实验是否仍在运行需要核对，勿重复提交。'
    elif worker_alive is None:
        label, code, done = '状态未知：无法核验本机 worker', 2, True
        reason = '请在原提案机和同一次系统启动中查看；当前不能确认后台是否存活。'
    else:
        label = '已提交：等待 worker 启动' if status == 'queued' else '运行中：后台 worker 存活'
    return dict(label=label, stage=stage, reason=reason, stop_follow=done, exit_code=code,
                round=state.get('active'), rounds=len(state.get('rounds', [])), calls=state.get('calls', 0))


def worker_running(job, value):
    receipt = job / 'worker.json'
    if not receipt.exists():
        return True if time.time() - value.get('updated_at', 0) < 5 else None
    record = read_json(receipt)
    if record.get('host_identity') != host_identity():
        return None
    # The worker holds this lock for its entire lifetime, even across PID namespaces.
    path = job / '.worker.lock'
    if not path.exists():
        return True if time.time() - value.get('updated_at', 0) < 5 else False
    try:
        with path.open('r') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return value['status'] == 'queued' and time.time() - value.get('updated_at', 0) < 5
    except OSError:
        return None


def status_text(value, job, view):
    stamp = max(value.get('heartbeat_at', 0), value.get('updated_at', 0)) or time.time()
    age = max(0, time.time() - stamp)
    lines = [f"[{view['label']}] {job.name}", f"阶段：{view['stage']}",
             f"轮次：{view['round'] or '-'}（已发起 {view['rounds']} 轮）；Agent 调用：{view['calls']} 次",
             f"最近状态/执行端回执：{age:.0f} 秒前（存活不等于设备健康或实验有进展）"]
    if view['reason']:
        lines.append('原因：' + str(view['reason']))
    lines.append(f'日志：{job / "worker.log"}')
    if value.get('startup_log') and value.get('activity') == 'task_start':
        lines.append(f'iTASK 启动日志：{value["startup_log"]}/stdout.log、stderr.log')
    if value['status'] == 'finished':
        lines.append(f'结果：{job / "results"}')
    return '\n'.join(lines)


def watch(job, *, follow=False, json_output=False):
    try:
        while True:
            value = read_json(job / 'status.json')
            view = describe_status(value, worker_running(job, value))
            print(json.dumps({**value, 'monitor': view}, ensure_ascii=False, indent=2) if json_output
                  else status_text(value, job, view), flush=True)
            if not follow or view['stop_follow']:
                return view['exit_code']
            time.sleep(read_json(job / 'job.json')['poll_seconds'])
    except KeyboardInterrupt:
        print('已停止查看；未向后台 worker 或远端实验发送停止信号。', flush=True)
        return 130


def event(job, status, **fields):
    previous = read_json(job / 'status.json') if (job / 'status.json').exists() else {}
    value = {**previous, 'activity': previous.get('activity', '') if status == 'error' else '',
             **fields, 'status': status, 'updated_at': time.time(), 'heartbeat_at': time.time()}
    write_json(job / 'status.json', value)
    line = json.dumps(value, ensure_ascii=False)
    with (job / 'events.jsonl').open('a') as handle:
        handle.write(line + '\n')
    view = describe_status(value)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {status} "
          f"[{view['label']}] {view['stage']} round={view['round'] or '-'} calls={view['calls']} "
          f"{view['reason']}", flush=True)


def snapshot(root, target):
    # Copy only repository content; no runs, login files, environment or Git internals.
    for name in ('src', 'skills', 'knowledge', 'benchmarks', 'examples', 'recipes', 'tools'):
        copy_source(root / name, target / name, inventory(root / name))
    skills.validate(target)


def start(args, root):
    plan = load_recipe(args.recipe)
    skills.validate(root)
    if 'search' not in plan:
        raise ValueError('optimization requires a search recipe')
    if args.executor == 'itask':
        if not args.task or not args.workdir or not Path(args.workdir).is_absolute() or ':' in args.workdir:
            raise ValueError('iTASK requires --task and an absolute --workdir without colon')
        if plan['search']['experience_dirs']:
            raise ValueError('iTASK recipes need execution-host experience paths; local experience_dirs cannot be uploaded implicitly')
    elif args.task or args.workdir:
        raise ValueError('--task/--workdir require --executor itask')
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        raise ValueError('--poll-seconds must be finite and positive')
    job = args.output.resolve()
    source = Path(plan['source'])
    if job.is_relative_to(source) or source.is_relative_to(job):
        raise ValueError('job and target source must not contain one another')
    if any(job.is_relative_to(root / p) for p in ('src', 'skills', 'knowledge', 'benchmarks')):
        raise ValueError('job output is inside framework code or foundation')
    job.mkdir(parents=True, exist_ok=False)
    snapshot(root, job / 'controller')
    config = dict(executor=args.executor, task=args.task, workdir=args.workdir,
                  python=args.python, setup_script=args.setup_script, poll_seconds=args.poll_seconds,
                  source=plan['source'], experience_dirs=plan['search']['experience_dirs'], recipe=args.recipe.read_text(),
                  id=uuid.uuid4().hex, retry_remote=False)
    write_json(job / 'job.json', config)
    event(job, 'queued', phase=0)
    spawn(job)
    return job


def spawn(job):
    with (job / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _spawn(job)


def _spawn(job):
    receipt = job / 'worker.json'
    if receipt.exists():
        old = read_json(receipt)
        if old['host_identity'] != host_identity():
            raise ValueError('resume requires the original proposer host and boot')
        if owned_alive(old['tag']):
            raise ValueError('optimization worker or its owned children are still running')
        event(job, 'queued', activity='resume', error='', progress={})
    tag = uuid.uuid4().hex
    with (job / 'worker.log').open('ab') as log:
        child = subprocess.Popen([sys.executable, '-m', 'afd_rsi.optimize', '_worker', str(job)],
            cwd=job, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
            env={**os.environ, 'PYTHONPATH': str(job / 'controller/src'),
                 'AFD_RSI_PROCESS_TAG': tag, 'PYTHONDONTWRITEBYTECODE': '1'})
    write_json(receipt, dict(pid=child.pid, tag=tag, host_identity=host_identity()))


class Executor:
    def __init__(self, config, job):
        self.config, self.job = config, job
        self.remote = config['executor'] == 'itask'
        self.base = (str(Path(config['workdir']) / ('rsi-' + config['id'])) if self.remote
                     else str(job / 'execution'))
        # Sibling of the job directories, so it outlives any single campaign.
        self.cache = (str(Path(config['workdir']) / '.rsi-cache') if self.remote
                      else str(job / 'cache'))
        self.env = {**os.environ, 'ITASK_DISABLE_AUTO_UPDATE': '1'}

    def command(self, argv, *, timeout=180, **kwargs):
        try:
            result = subprocess.run(argv, env=self.env, capture_output=True, text=True,
                                    timeout=timeout, **kwargs)
        except FileNotFoundError as error:
            # The detached worker inherits its launcher's PATH, which a non-login
            # shell may not include the CLI's install directory in.
            raise RuntimeError(
                f'{argv[0]} not found on PATH ({self.env.get("PATH", "")}); '
                f'run from a login shell or export the CLI location before starting') from error
        if result.returncode:
            raise RuntimeError(f'command failed: {shlex.join(argv[:4])}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
        return result.stdout

    def start_task(self):
        logs = self.job / 'task-start' / uuid.uuid4().hex[:12]
        event(self.job, 'preparing', activity='task_start', startup_log=str(logs))
        result = execute(['itask-guian', 'start', self.config['task']], self.job, logs, 1800, self.env)
        if not result['cleanup_complete']:
            raise RuntimeError(f'iTASK 启动客户端未完成清理；远端状态未知，请检查 {logs}')
        # A timed-out client does not prove that the remote migration failed.
        try:
            self.command(['itask-guian', 'exec', self.config['task'], '--tty=false', '--stdin=false', '--', 'true'], timeout=30)
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f'iTASK 启动状态 {result["status"]}，Pod 尚未确认可执行；不要重复提交启动。'
                               f'请核对任务状态及 {logs}/stdout.log、stderr.log：{error}') from error
        write_json(logs / 'readiness.json', {'ready': True, 'start_status': result['status']})

    def python(self, code, *args):
        setup = ('source ' + shlex.quote(self.config['setup_script']) + ' && ') if self.config['setup_script'] else ''
        # Content-addressed caches shared by every campaign on this host. The
        # build cache lets an unchanged checkout skip a full recompile; a hit is
        # re-verified against source and artifact hashes before it is used.
        exports = 'export AFD_RSI_BUILD_CACHE=' + shlex.quote(self.cache + '/build') + '; '
        command = (setup + exports + 'export PYTHONPATH=' + shlex.quote(self.base + '/controller/src')
                   + '${PYTHONPATH:+:$PYTHONPATH}; exec '
                   + shlex.join([self.config['python'], '-c', code, *map(str, args)]))
        argv = ['bash', '-c', command]
        if self.remote:
            argv = ['itask-guian', 'exec', self.config['task'], '--tty=false', '--stdin=false', '--', *argv]
        return self.command(argv)

    def stream(self, argv, *, source=None, sink=None, timeout=1800):
        """One transfer whose bytes travel on stdio instead of the CLI staging.

        Some installed itask variants drop the file with `mv --no-copy`, which
        this coreutils lacks; streaming keeps the transfer atomic enough for our
        own sha256 check without depending on that flag.
        """
        receipt = self.job / 'transfers' / uuid.uuid4().hex[:12]
        receipt.mkdir(parents=True, exist_ok=True)
        stdin = source.open('rb') if source else subprocess.DEVNULL
        stdout = sink.open('wb') if sink else subprocess.PIPE
        try:
            result = subprocess.run(argv, env=self.env, stdin=stdin,
                                    stdout=stdout, stderr=subprocess.PIPE, timeout=timeout)
        finally:
            if source:
                stdin.close()
            if sink:
                stdout.close()
        stderr = result.stderr.decode('utf-8', 'replace')[-2000:]
        write_json(receipt / 'receipt.json', {
            'argv': argv, 'exit_code': result.returncode,
            'bytes': Path(source).stat().st_size if source else None,
            'sha256': digest(source) if source else None, 'stderr': stderr})
        if result.returncode:
            raise RuntimeError(f'transfer failed: {shlex.join(argv[:4])}\n{stderr}')
        return receipt

    def send(self, source, destination):
        if not self.remote:
            shutil.copyfile(source, destination)
            return
        try:
            self.command(['itask-guian', 'cp', str(source), self.config['task'] + ':' + str(destination)])
        except RuntimeError as error:
            if '--no-copy' not in str(error):
                raise
            # Fall back to a stdio transfer, then prove the bytes arrived.
            command = f'mkdir -p {shlex.quote(str(Path(destination).parent))} && cat > {shlex.quote(str(destination))}'
            self.stream(['itask-guian', 'exec', self.config['task'], '--tty=false', '--stdin=true',
                         '--', 'bash', '-c', command], source=source)
            remote = self.command(['itask-guian', 'exec', self.config['task'], '--tty=false',
                                   '--stdin=false', '--', 'sha256sum', str(destination)]).split()[0]
            if remote != digest(source):
                raise RuntimeError('streamed upload hash mismatch')

    def fetch(self, source, destination):
        if not self.remote:
            shutil.copyfile(source, destination)
            return
        try:
            self.command(['itask-guian', 'cp', self.config['task'] + ':' + str(source), str(destination)])
        except RuntimeError as error:
            if '--no-copy' not in str(error):
                raise
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            self.stream(['itask-guian', 'exec', self.config['task'], '--tty=false', '--stdin=false',
                         '--', 'cat', str(source)], sink=Path(destination))
            if not Path(destination).is_file() or Path(destination).stat().st_size == 0:
                raise RuntimeError('streamed download produced no bytes')

    def invoke(self, action, *args):
        text = self.python('from afd_rsi.optimize import remote_action; import sys; remote_action(*sys.argv[1:])', action, self.base, *args)
        lines = [line[len(MARKER):] for line in text.splitlines() if line.startswith(MARKER)]
        if len(lines) != 1:
            raise ValueError('execution host did not return one framed status record')
        return json.loads(lines[0])


def stage(job, config, executor):
    state = job / 'staged.json'
    if state.exists():
        return
    event(job, 'preparing', phase=0, activity='snapshot')
    source = job / 'payload/source'
    source_lock = job / 'source.json'
    if source_lock.exists():
        if inventory(source) != read_json(source_lock):
            raise ValueError('staged source changed')
    else:
        if source.exists():
            shutil.rmtree(source)
        expected = inventory(Path(config['source']))
        copy_source(Path(config['source']), source, expected)
        write_json(source_lock, expected)
    archive = job / 'payload.tar.gz'
    archive_lock = job / 'payload.json'
    if archive_lock.exists():
        if digest(archive) != read_json(archive_lock)['sha256']:
            raise ValueError('sealed payload changed')
    else:
        with tarfile.open(archive, 'w:gz') as tar:
            tar.add(source, arcname='source')
            tar.add(job / 'controller', arcname='controller')
        # Reuse identical compressed bytes after a partial upload/bootstrap.
        write_json(archive_lock, {'sha256': digest(archive)})
    if archive.stat().st_size >= 95_000_000:
        raise ValueError('payload exceeds supported iTASK cp size; reduce target checkout before starting')
    if executor.remote:
        event(job, 'preparing', activity='auth')
        executor.command(['itask-guian', 'auth', 'status'])
        executor.start_task()
    event(job, 'preparing', phase=0, activity='upload')
    # mkdir is exclusive: interrupted/ambiguous bootstrap is not silently overwritten.
    executor.python('from pathlib import Path; import sys,json; b=Path(sys.argv[1]); b.mkdir(parents=True,exist_ok=True); marker=b/"bootstrap.json"; expected={"id":sys.argv[2],"sha256":sys.argv[3]}; assert (json.loads(marker.read_text())==expected if marker.exists() else not any(b.iterdir())), "foreign or changed bootstrap"; marker.write_text(json.dumps(expected))', executor.base, config['id'], digest(archive))
    if executor.remote:
        # Same source, same bytes: reuse a stored payload instead of re-sending it.
        stored = executor.cache + '/payloads/' + digest(archive)
        cache_hit = executor.python(
            'import hashlib,sys; from pathlib import Path; p=Path(sys.argv[1]); '
            'print("hit" if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2] else "miss")',
            stored, digest(archive)).strip().endswith('hit')
        if cache_hit:
            executor.python('from pathlib import Path; import sys,shutil; '
                            'shutil.copyfile(sys.argv[1], sys.argv[2])',
                            stored, executor.base + '/payload.tar.gz')
            event(job, 'preparing', phase=0, activity='upload', payload='reused')
        else:
            executor.send(archive, executor.base + '/payload.tar.gz')
            executor.python('from pathlib import Path; import sys,shutil; p=Path(sys.argv[1]); '
                            'p.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(sys.argv[2], p)',
                            stored, executor.base + '/payload.tar.gz')
    else:
        executor.send(archive, executor.base + '/payload.tar.gz')
    executor.python('from pathlib import Path; import tarfile,hashlib,sys; b=Path(sys.argv[1]); assert not (b/"campaign").exists(); a=b/"payload.tar.gz"; assert hashlib.sha256(a.read_bytes()).hexdigest()==sys.argv[2]; t=tarfile.open(a); t.extractall(b,filter="data"); t.close()', executor.base, digest(archive))
    # Replace only the top-level source field; retain the user's commands and gates verbatim.
    import re
    recipe, count = re.subn(r'(?m)^source\s*=.*$', 'source = "./source"', config['recipe'], count=1)
    if count != 1:
        raise ValueError('recipe requires a top-level source line')
    if not executor.remote:
        # Paths were resolved at submission, not relative to the relocated recipe.
        recipe = re.sub(r'(?m)^experience_dirs\s*=.*$', lambda _: 'experience_dirs = ' + json.dumps(config['experience_dirs']), recipe)
    (job / 'execution.toml').write_text(recipe)
    executor.send(job / 'execution.toml', executor.base + '/recipe.toml')
    write_json(state, {'base': executor.base, 'payload_sha256': digest(archive)})


def preflight_service_dependencies(job, config, executor):
    args = tomllib.loads(config['recipe']).get('service', {}).get('args', [])
    tests = [arg.split('=', 1)[1] for arg in args if arg.startswith('--test=')]
    tests += [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '--test']
    if not {'gsm8k', 'all'} & {name.strip() for value in tests for name in value.split(',')}:
        return
    local_dataset = any(arg == '--dataset-path' or arg.startswith('--dataset-path=') for arg in args)
    event(job, 'preparing', activity='preflight')
    executor.python(
        'import sys; sys.path.insert(0, sys.argv[1]); '
        'from afd_e2e.probe import check_gsm8k_dependencies; '
        'check_gsm8k_dependencies(sys.argv[2] == "1")',
        executor.base + '/controller/benchmarks/service', str(int(local_dataset)))


def remote_status(base, phase):
    run = base / 'campaign'
    state = read_json(run / 'search/state.json') if (run / 'search/state.json').exists() else {}
    pending = read_json(run / 'search/pending.json') if (run / 'search/pending.json').exists() else None
    result = dict(state=state, pending=pending, phase=int(phase), best=None, step=None, round_phase=None)
    if (run / 'search/best.json').exists():
        result['best'] = read_json(run / 'search/best.json')
    attempt = state.get('baseline')
    if state.get('active'):
        path = run / 'search/rounds' / state['active'] / 'state.json'
        if path.exists():
            record = read_json(path)
            result['round_phase'] = record.get('phase')
            attempt = record.get('confirmation') if record.get('phase') == 'confirming' else record.get('screen', attempt)
        else:
            result['round_phase'] = 'request'
    if attempt and (run / 'attempts' / attempt / 'progress.json').exists():
        progress = read_json(run / 'attempts' / attempt / 'progress.json')
        if progress['steps']:
            step = progress['steps'][-1]
            result['step'] = {key: step.get(key) for key in ('side', 'phase', 'trial')}
            summary = run / step['trial'] / 'service/summary.json'
            if summary.exists():
                service = read_json(summary)
                result['step']['service'] = {key: service.get(key) for key in ('stage', 'test_results', 'elapsed_seconds')}
    if result['round_phase'] in ('request', 'awaiting_agent'):
        result['step'] = None
    exit_path = base / 'drivers' / phase / 'exit.json'
    result['exit'] = read_json(exit_path) if exit_path.exists() else None
    if (run / 'plan.lock.json').exists():
        result['expires_at'] = read_json(run / 'plan.lock.json')['expires_at']
    receipt = base / 'drivers' / phase / 'launch.json'
    if receipt.exists() and result['exit'] is None:
        record = read_json(receipt)
        if record['host_identity'] != host_identity() or not owned_alive(record['tag']):
            result['lost_driver'] = True
    return result


def pack_results(base):
    files = []
    for directory in (base / 'campaign', base / 'best-export', base / 'drivers'):
        for current, dirs, names in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in {'build', 'source', 'staging', 'workspaces',
                       'diagnostic-build', 'diagnostic-input', '__pycache__', 'kernel_meta'}]
            files.extend(Path(current) / n for n in names if n != '.controller.lock')
    # Retain compiler failures without shipping source trees and build binaries.
    for pattern in ('attempts/*/build/*/native/build', 'attempts/*/*/profile/*/diagnostic-build/build'):
        for directory in (base / 'campaign').glob(pattern):
            files.extend(directory / name for name in ('stdout.log', 'stderr.log', 'process.json')
                         if (directory / name).is_file())
    manifest = {str(p.relative_to(base)): digest(p) for p in files}
    write_json(base / 'results.manifest.json', manifest)
    with tarfile.open(base / 'results.tar.gz', 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(base)), recursive=False)
        tar.add(base / 'results.manifest.json', arcname='results.manifest.json')
    if (base / 'results.tar.gz').stat().st_size >= 95_000_000:
        raise ValueError('results exceed cp limit; raw results remain on execution host')
    return digest(base / 'results.tar.gz')


def collect_results(job, executor, status):
    executor.fetch(executor.base + '/results.tar.gz', job / 'results.tar.gz')
    if digest(job / 'results.tar.gz') != status['exit']['results_sha256']:
        raise ValueError('downloaded result archive changed')
    with tarfile.open(job / 'results.tar.gz') as tar:
        manifest = json.load(tar.extractfile('results.manifest.json'))
        for name, value in manifest.items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != value:
                raise ValueError('result archive evidence changed')
        tar.extractall(job / 'results', filter='data')


def remote_action(action, directory, phase='0'):
    if action not in ('launch', 'status') or not phase.isascii() or not phase.isdecimal():
        raise ValueError('invalid execution action or phase')
    base = Path(directory)
    if action == 'launch':
        target = base / 'drivers' / phase
        target.mkdir(parents=True, exist_ok=True)
        with (base / 'launch.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not (target / 'intent.json').exists():
                write_json(target / 'intent.json', {'phase': phase})
                with (target / 'driver.log').open('ab') as log:
                    tag = uuid.uuid4().hex
                    child = subprocess.Popen([sys.executable, '-m', 'afd_rsi.optimize', '_remote', str(base), phase],
                        stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
                        env={**os.environ, 'AFD_RSI_PROCESS_TAG': tag})
                write_json(target / 'launch.json', {'pid': child.pid, 'tag': tag, 'host_identity': host_identity()})
            elif not (target / 'launch.json').exists():
                raise ValueError('incomplete launch intent; inspect before recovery')
    result = remote_status(base, phase)
    print(MARKER + json.dumps(result), flush=True)


def remote_run(base, phase):
    root = base / 'controller'
    directory = base / 'drivers' / phase
    with (base / 'driver.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run = base / 'campaign'
            if not run.exists():
                campaign.prepare(base / 'recipe.toml', run, root)
            plan = campaign.load(run, root)
            action = ['resume', str(run)] if (run / 'search/state.json').exists() else ['search', str(run), '--external-agent']
            result = execute([sys.executable, '-m', 'afd_rsi', *action], base, directory / 'command',
                             max(1, plan['expires_at'] - time.time() + plan['timeout_seconds']), {})
            # Publish the command receipt before status consumers inspect its liveness.
            write_json(directory / 'process.json', result)
            if result['status'] != 'completed' or not result['cleanup_complete']:
                stderr = directory / 'command/stderr.log'
                detail = stderr.read_text(errors='replace')[-4000:].strip() if stderr.exists() else ''
                raise RuntimeError(f'executor {result["status"]} (exit {result.get("exit_code")}); '
                                   f'inspect {directory / "command/process.json"}'
                                   + (f': {detail}' if detail else ''))
            state = read_json(run / 'search/state.json')
            archive_hash = None
            if state['status'] not in ('running', 'waiting_model'):
                if (run / 'search/best.json').exists():
                    staging = base / ('export-' + phase)
                    campaign.export(run, staging, root, best=True)
                    destination = base / 'best-export'
                    if destination.exists():
                        old = {p.name: digest(p) for p in destination.iterdir() if p.is_file()}
                        new = {p.name: digest(p) for p in staging.iterdir() if p.is_file()}
                        if old != new:
                            raise ValueError('existing export differs from freshly audited best')
                        shutil.rmtree(staging)
                    else:
                        staging.rename(destination)
                archive_hash = pack_results(base)
            elif state['status'] == 'waiting_model':
                archive_hash = pack_results(base)
            write_json(directory / 'exit.json', {'code': 0, 'results_sha256': archive_hash})
        except BaseException as error:
            failure = {'code': 1, 'error': str(error)}
            try:
                failure['results_sha256'] = pack_results(base)
            except Exception as archive_error:
                failure['archive_error'] = str(archive_error)
            write_json(directory / 'exit.json', failure)
            raise


PROPOSER_INFRA_MARKERS = (
    'gateway HTTP 429', 'gateway HTTP 500', 'gateway HTTP 502',
    'gateway HTTP 503', 'gateway HTTP 504',
    'gateway request failed', 'timed out', 'Connection reset',
    'Temporary failure in name resolution',
)


def proposer_unavailable(proposal_dir: Path) -> str:
    """Return the infrastructure error a failed proposal hit, if it was one.

    A shared model gateway being down says nothing about the candidate, so the
    same proposal is retried instead of burning the round. The classification is
    read from the proposer's own stderr, not guessed from the wrapper message.
    """
    stderr = proposal_dir / 'stderr.log'
    if not stderr.is_file():
        return ''
    text = stderr.read_text(errors='replace')
    for marker in PROPOSER_INFRA_MARKERS:
        if marker in text:
            return marker
    return ''


def restart(job, root, reason):
    """Refresh the controller and resume a stopped campaign without losing rounds.

    The measured rounds live in the campaign on the execution host and are left
    untouched. Only the local proposer snapshot is replaced, and any interrupted
    proposal attempt is archived rather than deleted. The adapter digest that the
    frozen campaign expects is recorded, so accepting a new adapter is an explicit,
    evidenced upgrade instead of a silent swap.
    """
    status = read_json(job / 'status.json')
    if (job / 'worker.json').exists() \
            and owned_alive(read_json(job / 'worker.json')['tag']):
        raise ValueError('campaign worker is still running')
    if (job / 'restart.json').exists():
        raise ValueError('campaign was already restarted; inspect restart.json before repeating it')
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    previous = job / f'controller-replaced-{stamp}'
    shutil.move(str(job / 'controller'), previous)
    snapshot(root, job / 'controller')
    abandoned = []
    for request in sorted((job / 'proposals').glob('*/request.json')):
        agent = request.parent / 'agent'
        receipt = agent / 'process.json'
        complete = receipt.exists() and read_json(receipt).get('status') == 'completed'
        if agent.exists() and not complete:
            target = request.parent / f'agent-abandoned-{stamp}'
            agent.rename(target)
            write_json(target / 'abandoned.json', {
                'reason': reason, 'stamp': stamp,
                'note': 'kept as evidence; the round was never measured'})
            abandoned.append(str(target.relative_to(job)))
    provider_hash = None
    for request in sorted((job / 'proposals').glob('*/request.json')):
        provider_hash = read_json(request).get('provider_hash')
    write_json(job / 'restart.json', {
        'at': stamp, 'reason': reason, 'archived_controller': previous.name,
        'abandoned_attempts': abandoned, 'expected_provider_hash': provider_hash})
    config = read_json(job / 'job.json')
    config['retry_remote'] = True
    write_json(job / 'job.json', config)
    phase = int(read_json(job / 'status.json').get('phase', 0)) + 1
    write_json(job / 'status.json', {**read_json(job / 'status.json'),
                                     'status': 'restarted', 'phase': phase,
                                     'activity': '', 'error': ''})
    event(job, 'restarted', phase=phase, reason=reason,
          archived_controller=previous.name, abandoned=abandoned)
    spawn(job)
    return job


def worker(job):
    with (job / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(job / 'job.json')
        executor = Executor(config, job)
        try:
            stage(job, config, executor)
            preflight_service_dependencies(job, config, executor)
            phase = read_json(job / 'status.json').get('phase', 0)
            previous = None
            while True:
                status = executor.invoke('launch', str(phase))
                signature = json.dumps(status, sort_keys=True)
                if signature != previous:
                    event(job, 'running', phase=phase, execution=executor.base, progress=status)
                    previous = signature
                else:
                    write_json(job / 'status.json', {**read_json(job / 'status.json'), 'heartbeat_at': time.time()})
                if status.get('lost_driver'):
                    if config['retry_remote']:
                        config['retry_remote'] = False
                        write_json(job / 'job.json', config)
                        phase += 1
                        event(job, 'recovering', phase=phase)
                        continue
                    raise RuntimeError('execution driver disappeared; inspect receipts before retrying')
                if status['exit'] is None:
                    time.sleep(config['poll_seconds'])
                    continue
                if status['exit']['code']:
                    if status['exit'].get('results_sha256'):
                        collect_results(job, executor, status)
                    if config['retry_remote']:
                        config['retry_remote'] = False
                        write_json(job / 'job.json', config)
                        phase += 1
                        event(job, 'recovering', phase=phase)
                        continue
                    raise RuntimeError(status['exit'].get('error', 'execution failed'))
                if status['state'].get('status') == 'waiting_model':
                    collect_results(job, executor, status)
                    event(job, 'waiting_model', phase=phase, progress=status)
                    return
                if status['state'].get('status') != 'running':
                    event(job, 'running', phase=phase, activity='collect', progress=status)
                    collect_results(job, executor, status)
                    event(job, 'finished', phase=phase, progress=status)
                    return
                pending = status['pending']
                if not pending:
                    raise ValueError('successful running search has no pending proposal')
                if type(pending['call']) is not int or pending['call'] <= 0:
                    raise ValueError('invalid pending call number')
                for name in (pending['request'], pending['response']):
                    if Path(name).is_absolute() or '..' in Path(name).parts:
                        raise ValueError('pending path escapes the campaign')
                round_dir = job / 'proposals' / str(pending['call'])
                round_dir.mkdir(parents=True, exist_ok=True)
                request = round_dir / 'request.json'
                executor.fetch(executor.base + '/campaign/' + pending['request'], request)
                if digest(request) != pending['request_hash']:
                    raise ValueError('pending request hash mismatch')
                event(job, 'proposing', phase=phase, progress=status)
                proposal_dir = round_dir / 'agent'
                receipt = proposal_dir / 'process.json'
                adaptive_request = 'knowledge_refs' in read_json(request).get('proposal_requirements', {})
                context = read_json(request)
                resume_stage = False
                retries = 0
                if proposal_dir.exists():
                    completed = (receipt.exists() and read_json(receipt)['status'] == 'completed'
                                 and read_json(receipt)['cleanup_complete'])
                    resume_stage = not completed and resumable_model(proposal_dir, context)
                    # A proposer killed by an unavailable gateway is retried; an
                    # ambiguous local interruption still requires inspection.
                    if not completed and not resume_stage and not proposer_unavailable(proposal_dir):
                        raise ValueError('interrupted proposer is not retried or refunded; inspect its owned process receipts')
                    if not completed and not resume_stage and adaptive_request and (proposal_dir / 'usage.json').exists() and read_json(proposal_dir / 'usage.json').get('model_calls', 0):
                        raise ValueError('adaptive proposer already consumed completions; inspect preserved receipts before any new dispatch')
                    if not completed and not resume_stage:
                        retries += 1
                        proposal_dir.rename(proposal_dir.with_name(f'agent-retry{retries:02d}'))
                if not proposal_dir.exists() or resume_stage:
                    # This finite process exits before any device experiment starts.
                    # A proposer outage is retried rather than failing the campaign:
                    # the dispatch is already charged and the work is untouched.
                    from .search import propose
                    override = None
                    if (job / 'restart.json').exists():
                        # A restarted campaign may run a fixed adapter; the swap is
                        # recorded against the digest the request was frozen with.
                        record = read_json(job / 'restart.json')
                        driver = job / 'controller' / read_json(request).get('provider', '')
                        override = {'replaces': read_json(request).get('provider_hash'),
                                    'digest': digest(driver) if driver.is_file() else None,
                                    'reason': record.get('reason', '')}
                    deadline = time.monotonic() + float(
                        os.environ.get('AFD_RSI_PROPOSER_RETRY_SECONDS') or 1800)
                    while True:
                        try:
                            propose(request, proposal_dir, job / 'controller',
                                    session_dir=job / 'codex-session',
                                    provider_override=override)
                            break
                        except (RuntimeError, ValueError) as error:
                            if any(marker in str(error) for marker in ('different or duplicate thread', 'checkpoint', 'another request')):
                                raise
                            if resumable_model(proposal_dir, context):
                                start = read_json(proposal_dir / 'model-start.json')
                                if len(list((proposal_dir / 'launches').iterdir())) < 3 and time.time() < min(start['deadline'], context['deadline']):
                                    event(job, 'proposing', phase=phase, progress=status,
                                          reason='Resume completed model stages; previous attempts remain charged')
                                    continue
                                journal = Journal(proposal_dir, context)
                                stage_name = 'code' if journal.stage('analysis').get('result') or context.get('debug') else 'analysis'
                                write_json(proposal_dir / 'proposal.json', failure_response(context,
                                    ModelExhausted(stage_name, f'bounded proposer process recovery exhausted: {error}')))
                                break
                            if adaptive_request and (proposal_dir / 'usage.json').exists() and read_json(proposal_dir / 'usage.json').get('model_calls', 0):
                                raise
                            reason = proposer_unavailable(proposal_dir)
                            if not reason or time.monotonic() >= deadline:
                                raise
                            retries += 1
                            # Preserve the failed attempt instead of overwriting it.
                            if proposal_dir.exists():
                                proposal_dir.rename(
                                    proposal_dir.with_name(f'agent-retry{retries:02d}'))
                            delay = min(30 * retries, 300)
                            event(job, 'proposing', phase=phase, progress=status,
                                  proposer_retry=retries, reason=reason, delay=delay)
                            time.sleep(delay)
                usage = proposal_dir / 'usage.json'
                if usage.is_file():
                    executor.send(usage, executor.base + '/campaign/' + str(Path(pending['response']).with_name('usage.json')))
                for name in ('model-state.json', 'model-analysis-complete.json', 'model-code-complete.json'):
                    if (proposal_dir / name).is_file():
                        executor.send(proposal_dir / name, executor.base + '/campaign/' + str(Path(pending['response']).with_name(name)))
                # Submit last: the controller may seal evidence as soon as the response appears.
                executor.send(proposal_dir / 'proposal.json', executor.base + '/campaign/' + pending['response'])
                phase += 1
                event(job, 'submitted', phase=phase, progress=status)
                time.sleep(config['poll_seconds'])
        except BaseException as error:
            event(job, 'error', error=str(error))
            raise


def main(argv=None, root=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '_worker':
        worker(Path(argv[1]).resolve())
        return 0
    if argv and argv[0] == '_remote':
        remote_run(Path(argv[1]), argv[2])
        return 0
    if argv and argv[0] in ('status', 'resume', 'restart'):
        action = argv[0]
        reader = argparse.ArgumentParser(prog='afd-rsi optimize ' + action)
        reader.add_argument('job', type=Path)
        reader.add_argument('--follow', action='store_true')
        reader.add_argument('--json', action='store_true', help='Print raw state plus monitoring assessment')
        reader.add_argument('--reason', default='', help='restart only: why the controller was replaced')
        options = reader.parse_args(argv[1:])
        job = options.job.resolve()
        root = (root or Path(__file__).resolve().parents[2]).resolve()
        if action == 'restart':
            restart(job, root, options.reason)
            return watch(job, follow=options.follow, json_output=options.json)
        if action == 'resume' and read_json(job / 'status.json')['status'] != 'finished':
            if (job / 'worker.json').exists() and owned_alive(read_json(job / 'worker.json')['tag']):
                return watch(job, follow=options.follow, json_output=options.json)
            config = read_json(job / 'job.json')
            status = read_json(job / 'status.json')
            if status['status'] == 'waiting_model':
                write_json(job / 'status.json', {**status, 'phase': int(status.get('phase', 0)) + 1})
            config['retry_remote'] = True
            write_json(job / 'job.json', config)
            spawn(job)
        return watch(job, follow=options.follow, json_output=options.json)
    parser = argparse.ArgumentParser(prog='afd-rsi optimize')
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--executor', choices=('local', 'itask'), default='local')
    parser.add_argument('--task', help='Existing owned iTASK ordinary task; started idempotently')
    parser.add_argument('--workdir', help='Existing execution-host parent for a unique job directory')
    parser.add_argument('--python', default='python3', help='Execution-host Python')
    parser.add_argument('--setup-script', help='Execution-host environment script, e.g. CANN set_env.sh')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--follow', action='store_true', help='Watch background progress; Ctrl-C only detaches the viewer')
    args = parser.parse_args(argv)
    root = root or Path(__file__).resolve().parents[2]
    job = start(args, root)
    print(f'已提交后台任务：{job}\n尚不代表 iTASK 就绪或已开始优化。', flush=True)
    print('持续查看：' + shlex.join(['env', f'PYTHONPATH={root / "src"}', sys.executable,
                                   '-m', 'afd_rsi', 'optimize', 'status', str(job), '--follow']), flush=True)
    return watch(job, follow=args.follow) if args.follow else 0


if __name__ == '__main__':
    raise SystemExit(main())
