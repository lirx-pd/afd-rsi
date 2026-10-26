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
import uuid
from pathlib import Path

from . import campaign, skills
from .contracts import load_recipe
from .evidence import digest, read_json, write_json
from .runner import execute, host_identity, owned_alive
from .workspace import copy_source, inventory

MARKER = 'AFD_RSI_STATUS:'
TERMINAL = {'finished', 'error'}


def event(job, status, **fields):
    value = {**(read_json(job / 'status.json') if (job / 'status.json').exists() else {}),
             **fields, 'status': status, 'updated_at': time.time()}
    write_json(job / 'status.json', value)
    line = json.dumps(value, ensure_ascii=False)
    with (job / 'events.jsonl').open('a') as handle:
        handle.write(line + '\n')
    progress = value.get('progress', {})
    state = progress.get('state', {})
    step = progress.get('step') or {}
    service = step.get('service') or {}
    detail = '/'.join(str(x) for x in (step.get('side'), step.get('phase'), service.get('stage')) if x)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {status} "
          f"round={state.get('active') or '-'} calls={state.get('calls', 0)} "
          f"{detail} {state.get('status', '')} {value.get('error', '') or state.get('reason', '')}", flush=True)


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
        self.env = {**os.environ, 'ITASK_DISABLE_AUTO_UPDATE': '1'}

    def command(self, argv, **kwargs):
        result = subprocess.run(argv, env=self.env, capture_output=True, text=True, timeout=180, **kwargs)
        if result.returncode:
            raise RuntimeError(f'command failed: {shlex.join(argv[:4])}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
        return result.stdout

    def python(self, code, *args):
        setup = ('source ' + shlex.quote(self.config['setup_script']) + ' && ') if self.config['setup_script'] else ''
        command = setup + 'export PYTHONPATH=' + shlex.quote(self.base + '/controller/src') + '${PYTHONPATH:+:$PYTHONPATH}; exec ' + shlex.join([self.config['python'], '-c', code, *map(str, args)])
        argv = ['bash', '-c', command]
        if self.remote:
            argv = ['itask-guian', 'exec', self.config['task'], '--tty=false', '--stdin=false', '--', *argv]
        return self.command(argv)

    def send(self, source, destination):
        if self.remote:
            self.command(['itask-guian', 'cp', str(source), self.config['task'] + ':' + str(destination)])
        else:
            shutil.copyfile(source, destination)

    def fetch(self, source, destination):
        if self.remote:
            self.command(['itask-guian', 'cp', self.config['task'] + ':' + str(source), str(destination)])
        else:
            shutil.copyfile(source, destination)

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
    event(job, 'preparing', phase=0)
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
        executor.command(['itask-guian', 'auth', 'status'])
        executor.command(['itask-guian', 'start', config['task']])
    # mkdir is exclusive: interrupted/ambiguous bootstrap is not silently overwritten.
    executor.python('from pathlib import Path; import sys,json; b=Path(sys.argv[1]); b.mkdir(parents=True,exist_ok=True); marker=b/"bootstrap.json"; expected={"id":sys.argv[2],"sha256":sys.argv[3]}; assert (json.loads(marker.read_text())==expected if marker.exists() else not any(b.iterdir())), "foreign or changed bootstrap"; marker.write_text(json.dumps(expected))', executor.base, config['id'], digest(archive))
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
    if attempt and (run / 'attempts' / attempt / 'progress.json').exists():
        progress = read_json(run / 'attempts' / attempt / 'progress.json')
        if progress['steps']:
            step = progress['steps'][-1]
            result['step'] = {key: step.get(key) for key in ('side', 'phase', 'trial')}
            summary = run / step['trial'] / 'service/summary.json'
            if summary.exists():
                service = read_json(summary)
                result['step']['service'] = {key: service.get(key) for key in ('stage', 'test_results', 'elapsed_seconds')}
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
                raise RuntimeError('executor failed; inspect command receipt and recover via resume')
            state = read_json(run / 'search/state.json')
            archive_hash = None
            if state['status'] != 'running':
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
            write_json(directory / 'exit.json', {'code': 0, 'results_sha256': archive_hash})
        except BaseException as error:
            write_json(directory / 'exit.json', {'code': 1, 'error': str(error)})
            raise


def worker(job):
    with (job / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read_json(job / 'job.json')
        executor = Executor(config, job)
        try:
            stage(job, config, executor)
            phase = read_json(job / 'status.json').get('phase', 0)
            previous = None
            while True:
                status = executor.invoke('launch', str(phase))
                signature = json.dumps(status, sort_keys=True)
                if signature != previous:
                    event(job, 'running', phase=phase, execution=executor.base, progress=status)
                    previous = signature
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
                    if config['retry_remote']:
                        config['retry_remote'] = False
                        write_json(job / 'job.json', config)
                        phase += 1
                        event(job, 'recovering', phase=phase)
                        continue
                    raise RuntimeError(status['exit'].get('error', 'execution failed'))
                if status['state'].get('status') != 'running':
                    executor.fetch(executor.base + '/results.tar.gz', job / 'results.tar.gz')
                    if digest(job / 'results.tar.gz') != status['exit']['results_sha256']:
                        raise ValueError('downloaded result archive changed')
                    with tarfile.open(job / 'results.tar.gz') as tar:
                        manifest = json.load(tar.extractfile('results.manifest.json'))
                        for name, value in manifest.items():
                            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != value:
                                raise ValueError('result archive evidence changed')
                        tar.extractall(job / 'results', filter='data')
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
                if proposal_dir.exists():
                    if not receipt.exists() or read_json(receipt)['status'] != 'completed' or not read_json(receipt)['cleanup_complete']:
                        raise ValueError('interrupted proposer is not retried or refunded; inspect its owned process receipts')
                else:
                    # This finite process exits before any device experiment starts.
                    from .search import propose
                    propose(request, proposal_dir, job / 'controller', session_dir=job / 'codex-session')
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
    if argv and argv[0] in ('status', 'resume'):
        action = argv[0]
        reader = argparse.ArgumentParser(prog='afd-rsi optimize ' + action)
        reader.add_argument('job', type=Path)
        reader.add_argument('--follow', action='store_true')
        options = reader.parse_args(argv[1:])
        job = options.job.resolve()
        if action == 'resume' and read_json(job / 'status.json')['status'] != 'finished':
            if (job / 'worker.json').exists() and owned_alive(read_json(job / 'worker.json')['tag']):
                print('Worker is already running')
                return 0
            config = read_json(job / 'job.json')
            config['retry_remote'] = True
            write_json(job / 'job.json', config)
            spawn(job)
        previous = None
        while True:
            status = read_json(job / 'status.json')
            if status != previous:
                print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
                previous = status
            if not options.follow or status['status'] in TERMINAL:
                return 0
            time.sleep(read_json(job / 'job.json')['poll_seconds'])
    parser = argparse.ArgumentParser(prog='afd-rsi optimize')
    parser.add_argument('recipe', type=Path)
    parser.add_argument('--executor', choices=('local', 'itask'), default='local')
    parser.add_argument('--task', help='Existing owned iTASK ordinary task; started idempotently')
    parser.add_argument('--workdir', help='Existing execution-host parent for a unique job directory')
    parser.add_argument('--python', default='python3', help='Execution-host Python')
    parser.add_argument('--setup-script', help='Execution-host environment script, e.g. CANN set_env.sh')
    parser.add_argument('--poll-seconds', type=float, default=60)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    root = root or Path(__file__).resolve().parents[2]
    print(start(args, root))
    print('Started in background; use afd-rsi optimize status <output>. Progress: <output>/worker.log')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
