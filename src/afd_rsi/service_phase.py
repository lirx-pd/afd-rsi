"""A campaign gate around the frozen, real AFD E2E service lifecycle."""

import os
import math
import itertools
import sys
import threading
from pathlib import Path

from .evidence import digest, read_json, write_json
from .native import artifact_identity, runtime_env, verify_source
from .runner import execute, tagged_processes
from .contracts import positive


def evidence_files(directory):
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name not in ('observation.json', 'loaded-maps.json')}


def validate_summary(summary, expected_tests, *, require_clean_shutdown=True):
    if not isinstance(summary, dict) or not isinstance(summary.get('test_results'), dict):
        return False
    required = ('inference_status', 'cleanup_status')
    if not all(summary.get(key) == 'passed' for key in required):
        return False
    if require_clean_shutdown:
        if summary.get('status') != 'passed' or summary.get('shutdown_status') != 'passed':
            return False
    else:
        if summary.get('shutdown_status') not in ('passed', 'failed'):
            return False
        expected_status = 'failed' if summary['shutdown_status'] == 'failed' else 'passed'
        if summary.get('status') != expected_status or not all(
                isinstance(error, str) and error.startswith('shutdown:') for error in summary.get('errors', [])):
            return False
    return set(summary.get('test_results', {})) == set(expected_tests) and all(
        value == 'passed' for value in summary['test_results'].values())


def service_metrics(summary, configuration):
    """Extract real service measurements; never turn HTTP results into rank timings."""
    metrics = {}
    if 'gsm8k' in configuration['tests']:
        score = summary['gsm8k']
        if (type(score['samples']) is not int or score['samples'] != configuration['limit']
                or score['threshold'] != configuration['threshold']):
            raise ValueError('GSM8K sample count/threshold differs from the frozen service plan')
        value = score['strict_match']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('invalid GSM8K strict-match score')
        metrics['gsm8k'] = {key: score[key] for key in ('samples', 'strict_match', 'threshold')}
        if value < score['threshold']:
            raise ValueError('GSM8K strict-match below frozen threshold')
    if 'perf' in configuration['tests']:
        performance = summary['performance']
        if performance['status'] != 'passed' or performance['configuration'] != configuration['performance']:
            raise ValueError('service performance configuration/status mismatch')
        points = performance['points']
        lengths = configuration['performance']
        workloads = []
        if lengths['mode'] in ('matrix', 'both'):
            workloads.extend(('matrix', [i], [o]) for i, o in itertools.product(lengths['input_lens'], lengths['output_lens']))
        if lengths['mode'] in ('mixed', 'both'):
            workloads.append(('mixed', lengths['input_lens'], lengths['output_lens']))
        expected = list(itertools.product(workloads, lengths['concurrencies'], lengths['rps']))
        if len(points) != len(expected):
            raise ValueError('service performance points missing')
        metrics['performance'] = []
        for index, (point, (workload, concurrency, rate)) in enumerate(zip(points, expected, strict=True)):
            count = lengths['requests']
            if (point['status'] != 'passed' or point['requests'] != count or point['completed'] != count
                    or point['failed'] != 0 or point['cancelled'] != 0 or point['unsubmitted'] != 0
                    or point['metrics']['ttft_ms']['count'] != count
                    or point['phase'] != lengths['phase']
                    or (point['mode'], point['input_lens'], point['output_lens']) != workload
                    or point['concurrency'] != concurrency or point['rps'] != rate
                    or point['id'] != f'point-{index:03d}'
                    or point['warmup']['completed'] != point['warmup_requests']
                    or point['warmup_requests'] < lengths['warmup']
                    or point['warmup']['failed'] or point['warmup']['cancelled']):
                raise ValueError('service performance request evidence incomplete')
            ttft = {key: positive(point['metrics']['ttft_ms'][key], 'service TTFT') for key in ('mean', 'p50', 'p90')}
            metrics['performance'].append({key: point[key] for key in (
                'id', 'phase', 'input_lens', 'output_lens', 'concurrency', 'requests', 'warmup_requests')}
                | {'ttft_ms': ttft, 'scope': 'client HTTP time to first token; includes service overhead'})
    return metrics


def run_service(plan, run, trial, build, source_identity, timeout):
    config = plan['service']
    repo = (build / config['repo']).resolve(strict=True)
    if not repo.is_relative_to(build.resolve()):
        raise ValueError('service checkout escapes its candidate build')
    # Identity comes from the actual isolated build, not --repo metadata alone.
    source_lock = read_json(repo.parent / 'source.lock.json')
    if source_lock != source_identity:
        raise ValueError('service source does not match the evaluated side')
    verify_source(repo, source_lock)
    artifacts = artifact_identity(repo)
    frozen = run / 'foundation'
    driver = frozen / 'benchmarks/service/run.py'
    options = [*config['args'], '--repo', str(repo), '--python', sys.executable,
               '--cleanup-timeout', str(config['cleanup_seconds']), '--output', str(trial / 'service')]
    env = runtime_env(repo, os.environ.get('ASCEND_RT_VISIBLE_DEVICES', ''))
    # Explicit --device in the frozen arguments is resolved by the E2E runner.
    if not env['ASCEND_RT_VISIBLE_DEVICES']:
        del env['ASCEND_RT_VISIBLE_DEVICES']
    planning = execute([sys.executable, str(driver), 'plan', *options], repo,
                       trial / 'planning', min(timeout, 30), env)
    if planning['status'] != 'completed' or not planning['cleanup_complete']:
        raise ValueError('service plan failed')
    service_plan = read_json(trial / 'service/plan.json')
    expected_roles = {s['role']: len(s['devices']) for s in service_plan['services']}
    expected_tests = service_plan['configuration']['tests']
    stopped = threading.Event()
    mappings, monitor_errors, hashes = {}, [], {}

    def monitor():
        while not stopped.is_set():
            receipt = trial / 'execution/process.json'
            if receipt.exists():
                handles = []
                try:
                    handles = tagged_processes(read_json(receipt)['tag'])
                    for pid, _ in handles:
                        proc = Path('/proc') / str(pid)
                        try:
                            fields = dict(item.split(b'=', 1) for item in (proc / 'environ').read_bytes().split(b'\0') if b'=' in item)
                            role = fields.get(b'AFD_E2E_PROCESS_ROLE', b'').decode()
                            if role not in expected_roles:
                                continue
                            paths = set()
                            for line in (proc / 'maps').read_text().splitlines():
                                parts = line.split(maxsplit=5)
                                if len(parts) == 6 and parts[5].startswith('/'):
                                    path = Path(parts[5]).resolve()
                                    if path.name.startswith('_C_ascend') or (path.name == 'libcust_opapi.so' and 'afd-plugin' in path.parts):
                                        if not path.is_relative_to(repo):
                                            raise ValueError(f'foreign AFD library in service: {path}')
                                        paths.add(path)
                            for path in paths:
                                if path not in hashes:
                                    hashes[path] = digest(path)
                            actual = {str(p.relative_to(repo)): hashes[p] for p in paths}
                            if actual and any(artifacts['files'].get(name) != value for name, value in actual.items()):
                                raise ValueError('service loaded unexpected native artifacts')
                            if set(actual) >= set(artifacts['files']) - {'afd_plugin/__init__.py'}:
                                mappings[str(pid)] = {'role': role, 'files': actual}
                        except (FileNotFoundError, ProcessLookupError):
                            continue
                except Exception as error:
                    monitor_errors.append(str(error))
                finally:
                    for _, fd in handles:
                        os.close(fd)
            stopped.wait(1)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        process = execute([sys.executable, str(driver), 'run', *options], repo, trial / 'execution',
                          timeout, env, grace_seconds=config['cleanup_seconds'] + 15)
    finally:
        stopped.set()
        thread.join(timeout=5)
        write_json(trial / 'loaded-maps.json', {'processes': mappings, 'errors': monitor_errors})
    summary_path = trial / 'service/summary.json'
    if not summary_path.is_file():
        raise ValueError('service did not produce a final summary')
    summary = read_json(summary_path)
    if (not isinstance(summary, dict) or not summary.get('finished_at')
            or not {'status', 'inference_status', 'shutdown_status', 'cleanup_status', 'test_results'} <= summary.keys()
            or not isinstance(summary['test_results'], dict) or set(summary['test_results']) != set(expected_tests)):
        raise ValueError('incomplete service lifecycle evidence')
    verify_source(repo, source_lock)
    if artifact_identity(repo) != artifacts:
        raise ValueError('service artifacts changed during execution')
    counts = {role: sum(p['role'] == role for p in mappings.values()) for role in expected_roles}
    identity_ok = not monitor_errors and all(counts[role] >= count for role, count in expected_roles.items())
    require_shutdown = config.get('require_clean_shutdown', True)
    metrics = service_metrics(summary, service_plan['configuration']) if all(
        value == 'passed' for value in summary['test_results'].values()) else {}
    return {'process': process, 'checks_passed': validate_summary(summary, expected_tests, require_clean_shutdown=require_shutdown) and identity_ok,
            'service_metrics': metrics,
            'gate_policy': {'require_clean_shutdown': require_shutdown},
            'shutdown_errors_ignored': not require_shutdown and summary['shutdown_status'] == 'failed',
            'summary': summary, 'actual_identity_passed': identity_ok,
            'summary_hash': digest(summary_path), 'loaded_maps_hash': digest(trial / 'loaded-maps.json'),
            'evidence': evidence_files(trial / 'service')}
