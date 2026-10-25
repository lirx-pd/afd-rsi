import json
from pathlib import Path

import pytest

from afd_rsi import campaign, search
from afd_rsi.contracts import load_recipe
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.evaluate import evaluate
from afd_rsi.native import analyze_profile
from afd_rsi.service_phase import service_metrics

ROOT = Path(__file__).resolve().parents[1]


def traces(directory):
    for rank in (1, 2):
        p = directory / f'rank-{rank}/ASCEND_PROFILER_OUTPUT/kernel_details.csv'
        p.parent.mkdir(parents=True)
        p.write_text('Name,Duration(us)\nAfdMegaMoeFFN,10\nAfdMegaMoeFFN,30\nReduceSum,5\n')


def test_profile_ranks_duration_analysis_and_raw_identity(tmp_path):
    traces(tmp_path)
    result = analyze_profile(tmp_path, 1, 3)
    assert [r['rank'] for r in result['ffn_ranks']] == [1, 2]
    top = result['ffn_ranks'][0]['top_kernels'][0]
    assert top['name'] == 'AfdMegaMoeFFN' and top['mean_us'] == 20
    assert top['task_duration_share_pct'] == pytest.approx(100 * 40 / 45)
    assert result['evidence'][0]['sha256'] == digest(tmp_path / result['evidence'][0]['file'])
    # Two files for one rank cannot stand in for a missing rank.
    (tmp_path / 'rank-2').rename(tmp_path / 'rank-0')
    with pytest.raises(ValueError, match='FFN rank 2'):
        analyze_profile(tmp_path, 1, 3)


@pytest.mark.parametrize('value', ['nan', 'inf', '0', '-1'])
def test_profile_rejects_unusable_device_duration(tmp_path, value):
    traces(tmp_path)
    p = next((tmp_path / 'rank-1').rglob('*.csv'))
    p.write_text(p.read_text().replace('10\n', value + '\n'))
    with pytest.raises(ValueError, match='duration'):
        analyze_profile(tmp_path, 1, 3)


def service_result():
    settings = dict(phase='prefill', input_lens=[128], output_lens=[1], concurrencies=[1],
                    rps=[None], mode='matrix', requests=8, warmup=2)
    config = dict(tests=['gsm8k', 'perf'], limit=30, threshold=0.27, performance=settings)
    point = dict(id='point-000', phase='prefill', mode='matrix', input_lens=[128], output_lens=[1],
                 concurrency=1, rps=None, requests=8, completed=8, failed=0, cancelled=0,
                 unsubmitted=0, warmup_requests=2, warmup=dict(completed=2, failed=0, cancelled=0),
                 status='passed', metrics={'ttft_ms': dict(count=8, mean=10, p50=9, p90=12)})
    summary = dict(gsm8k=dict(samples=30, strict_match=0.8, threshold=0.27),
                   performance=dict(status='passed', configuration=settings, points=[point]))
    return config, summary


def test_service_extracts_exact_gsm8k_and_prefill_metrics():
    config, summary = service_result()
    result = service_metrics(summary, config)
    assert result['gsm8k']['samples'] == 30
    assert result['performance'][0]['ttft_ms']['p50'] == 9
    summary['gsm8k']['samples'] = 29
    with pytest.raises(ValueError, match='sample count'):
        service_metrics(summary, config)


@pytest.mark.parametrize('field,value', [('completed', 7), ('input_lens', [256]), ('warmup_requests', 0)])
def test_service_rejects_incomplete_or_changed_prefill(field, value):
    config, summary = service_result()
    summary['performance']['points'][0][field] = value
    with pytest.raises(ValueError, match='request evidence'):
        service_metrics(summary, config)


def test_profile_triggers_only_after_gates_and_before_unprofiled_benchmarks():
    plan = {'service': {}, 'native': {'profile': True}, 'repetitions': 2}
    jobs = campaign.phase_jobs(plan, ['baseline', 'candidate'])
    assert jobs[6] == ('candidate', 'profile', None)
    assert all(p != 'profile' for _, p, _ in jobs[:6])
    assert jobs[7] == ('baseline', 'benchmark', 0)
    assert campaign.phase_jobs(plan, ['baseline'])[3] == ('baseline', 'profile', None)
    assert not any(p == 'profile' for _, p, _ in campaign.phase_jobs({'repetitions': 2}, ['baseline']))


def test_recipes_freeze_30_gsm8k_and_one_prefill_point(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e.cli import parser
    from afd_e2e.config import from_args
    for execution in ('eager', 'graph'):
        p = tmp_path / f'{execution}.toml'
        p.write_text((ROOT / f'recipes/ascend_{execution}.toml').read_text().replace('../../afd-plugin', str(tmp_path)))
        plan = load_recipe(p)
        assert plan['native']['profile'] is True
        case = from_args(parser().parse_args(['plan', *plan['service']['args'], '--repo', str(tmp_path)]))
        config = json.loads(json.dumps(case.to_dict()))
        assert config['limit'] == 30 and set(config['tests']) == {'smoke', 'gsm8k', 'perf'}
        assert config['threshold'] == 0.9
        assert config['performance']['input_lens'] == [128,512,2048]
        assert config['performance']['concurrencies'] == [1,4,8]
        assert config['performance']['point_interval_seconds'] == 3
        assert config['test_interval_seconds'] == 5
        assert config['performance']['output_lens'] == [1]
        assert config['performance']['mode'] == 'matrix'


def test_profile_evidence_rechecked_during_resume(tmp_path):
    trial = tmp_path / 'trial'
    traces(trial / 'traces')
    observation = dict(status='valid', phase='profile', trial='trial', receipts={},
                       profiling=analyze_profile(trial / 'traces', 1, 3),
                       profile_files={str(p.relative_to(trial)): digest(p) for p in (trial / 'traces').rglob('*.csv')})
    plan = dict(native=dict(attn_ranks=1), ranks=3)
    campaign.verify_observation(tmp_path, observation, None, None, None, plan, check_runtime=False)
    p = next((trial / 'traces').rglob('*.csv'))
    p.write_text(p.read_text().replace(',10', ',11'))
    with pytest.raises(ValueError, match='profile evidence changed'):
        campaign.verify_observation(tmp_path, observation, None, None, None, plan, check_runtime=False)


def test_agent_request_contains_profile_service_metrics_and_local_strategy(tmp_path):
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text((ROOT / 'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source')))
    run = campaign.prepare(recipe, tmp_path / 'run', ROOT)
    plan = campaign.load(run, ROOT)
    attempt = run / 'attempts/qualification'
    config, summary = service_result()
    observation = dict(side='baseline', phase='service', status='valid', checks_passed=True,
                       service_metrics=service_metrics(summary, config))
    write_json(attempt / 'baseline.json', {'observations': [observation]})
    traces(tmp_path / 'traces')
    profile = analyze_profile(tmp_path / 'traces', 1, 3)
    write_json(attempt / 'progress.json', {'steps': [{'result': {'profiling': profile}}]})
    best = dict(source='workspaces/baseline', source_hash=plan['source_snapshot']['hash'],
                confirmation=None, baseline=attempt.name)
    context = search.request_context(run, plan, best, [], 1)
    fact = next(item for item in context['evidence'] if item['id'] == 'profile-incumbent')
    assert fact['facts']['ffn_ranks'][0]['top_kernels'][0]['name'] == 'AfdMegaMoeFFN'
    assert context['evidence'][0]['facts']['observations'][0]['service_metrics']['gsm8k']['samples'] == 30
    assert 'ascendc-perf-optimize' in context['knowledge']
    assert 'measurement/profiling.md' in context['knowledge']
    assert 'source-backed bottleneck' in context['instructions']
    trial = run / 'attempts/rejected/candidate/profile/t1/observation.json'
    write_json(trial, {'profiling': profile})
    record = dict(category='tiling', decision='rejected', reason='insufficient gain',
                  parent_hash=best['source_hash'], candidate_hash='candidate', hypothesis='minimal tile change',
                  evidence={str(trial.relative_to(run)): digest(trial)})
    following = search.request_context(run, plan, best, [record], 2)
    history = next(e for e in following['evidence'] if e['id'].startswith('profile-history-'))
    assert history['facts']['source_hash'] == 'candidate' and history['facts']['decision'] == 'rejected'
    assert history['facts']['ffn_ranks'] == profile['ffn_ranks']
    # Verbose logs must not displace measured profiles and previous failures.
    for index in range(90):
        log = run / f'logs/{index}/stdout.log'
        log.parent.mkdir(parents=True)
        log.write_text('x' * 4000)
        record['evidence'][str(log.relative_to(run))] = digest(log)
    noisy = search.request_context(run, plan, best, [record, record], 3)
    ids = {e['id'] for e in noisy['evidence']}
    assert {'profile-incumbent', 'history-0', 'history-1'} <= ids
    assert any(name.startswith('profile-history-') for name in ids)
    write_json(tmp_path / 'request.json', noisy)
    assert (tmp_path / 'request.json').stat().st_size <= plan['search']['max_context_bytes']
    # Local quoted dependencies are readable without expanding the edit scope.
    source = run / best['source']
    (source / 'kernel.cpp').write_text('#include "helper.h"\n#include "../outside.h"\n')
    (source / 'helper.h').write_text('#include "nested.h"\n')
    (source / 'nested.h').write_text('// local dependency\n')
    plan['allowed_files'] = ['kernel.cpp']
    closure = search.request_context(run, plan, best, [], 4)
    assert set(closure['source_files']) == {'kernel.cpp', 'helper.h', 'nested.h'}
    assert set(closure['read_only_source_files']) == {'helper.h', 'nested.h'}
    assert closure['allowed_files'] == ['kernel.cpp']
    # Noisy successful compiler logs must not crowd out a read-only include chain.
    (source / 'helper.h').write_text('#include "nested.h"\n' + '// helper\n' * 1400)
    (source / 'nested.h').write_text('// nested\n' * 1400)
    closure = search.request_context(run, plan, best, [record, record], 5)
    assert set(closure['source_files']) == {'kernel.cpp', 'helper.h', 'nested.h'}
    write_json(tmp_path / 'request.json', closure)
    assert (tmp_path / 'request.json').stat().st_size <= plan['search']['max_context_bytes']
    # Historical detail can be recovered from raw archives; keep current timelines.
    profile['pipeline'] = {'ffn_ranks': [{'summed_core_cycles': {'2': 100}, 'selected_cores': [{'core': 48}]}]}
    write_json(trial, {'profiling': profile})
    write_json(attempt / 'progress.json', {'steps': [{'result': {'profiling': profile}}]})
    bounded = search.request_context(run, plan, best, [record], 6)
    current = next(e for e in bounded['evidence'] if e['id'] == 'profile-incumbent')
    historical = next(e for e in bounded['evidence'] if e['id'].startswith('profile-history-'))
    assert current['facts']['pipeline']['ffn_ranks'][0]['selected_cores'] == [{'core': 48}]
    assert historical['facts']['pipeline']['ffn_ranks'] == [{'summed_core_cycles': {'2': 100}}]
    assert 'selected_cores' in read_json(trial)['profiling']['pipeline']['ffn_ranks'][0]
    plan['validation'] = 'operator-first'
    staged = search.request_context(run, plan, best, [record], 6)
    assert staged['proposal_requirements']['required_evidence_ids'] == ['baseline', 'profile-incumbent']
    assert any(e['id'] == 'profile-incumbent' for e in staged['evidence'])


def test_prefill_comparison_is_reported_without_changing_primary_acceptance():
    config, summary = service_result()
    metric = service_metrics(summary, config)
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=1, min_gain_pct=2, service={})
    observations = []
    for side, value in (('baseline', 10), ('candidate', 11)):
        observations.extend([
            dict(side=side, phase='correctness', status='valid', checks_passed=True),
            dict(side=side, phase='service', status='valid', checks_passed=True, service_metrics=metric),
            dict(side=side, phase='benchmark', status='valid', checks_passed=True, value=value)])
    verdict = evaluate(plan, observations)
    assert verdict['verdict'] == 'FAIL'
    assert verdict['service_performance_comparison'][0]['gain_pct'] == 0
    plan['native'] = {'profile': True}
    verdict = evaluate(plan, observations)
    assert 'qualified candidate profile required' in verdict['reasons']


def test_invalid_profile_analysis_cannot_leave_a_valid_observation(tmp_path, monkeypatch):
    from afd_rsi import native
    from afd_rsi.workspace import inventory
    source, build = tmp_path / 'source', tmp_path / 'build'
    source.mkdir()
    (source / 'input.cpp').write_text('input')
    write_json(build / 'artifacts.lock.json', {})
    plan = dict(commands={}, backend='ascend', timeout_seconds=10, expires_at=10**12,
                artifacts=[], native={'attn_ranks': 1}, ranks=3, workload_id='test')
    monkeypatch.setattr(campaign, 'load', lambda *args: plan)
    monkeypatch.setattr(campaign, 'artifacts', lambda *args, **kwargs: {})
    monkeypatch.setattr(campaign, 'validate_result', lambda *args, **kwargs: None)

    def execute(argv, cwd, output, timeout, env):
        write_json(Path(env['AFD_RSI_RESULT']), {'ranks': [{'checks': {'numeric': True}}], 'profiling': {}})
        return {'status': 'completed', 'cleanup_complete': True}

    def invalid(*args):
        raise ValueError('missing FFN device trace')

    monkeypatch.setattr(campaign, 'execute', execute)
    monkeypatch.setattr(native, 'analyze_profile', invalid)
    observation = campaign.command(plan, tmp_path, tmp_path / 'attempt', 'baseline', 'profile',
                                   inventory(source), build, ROOT, source=source, output=tmp_path / 'trial')
    assert observation['status'] == 'invalid'
    assert observation['reason'] == 'missing FFN device trace'
    assert read_json(tmp_path / 'recovery.json')['health'] == 'required'


@pytest.mark.parametrize('score', [float('nan'), 0.2])
def test_service_accuracy_cannot_pass_with_invalid_or_low_score(score):
    config, summary = service_result()
    summary['gsm8k']['strict_match'] = score
    with pytest.raises(ValueError, match='GSM8K'):
        service_metrics(summary, config)


def test_service_startup_failure_retains_real_error_without_missing_metric_exception(tmp_path, monkeypatch):
    from afd_rsi import service_phase
    build, trial = tmp_path / 'build', tmp_path / 'trial'
    (build / 'native/source').mkdir(parents=True)
    identity = {'hash': 'source', 'files': {}}
    write_json(build / 'native/source.lock.json', identity)
    monkeypatch.setattr(service_phase, 'verify_source', lambda *args: None)
    monkeypatch.setattr(service_phase, 'artifact_identity', lambda *args: {'files': {}})
    config = dict(repo='native/source', args=[], cleanup_seconds=1, require_clean_shutdown=False)
    summary = dict(finished_at='now', status='failed', inference_status='failed', shutdown_status='passed',
                   cleanup_status='passed', test_results={'gsm8k': 'not_started', 'perf': 'not_started'},
                   errors=['RuntimeError: EADDRINUSE'])

    def execute(argv, cwd, output, timeout, env, **kwargs):
        if argv[2] == 'plan':
            write_json(trial / 'service/plan.json', {'services': [{'role': 'ffn', 'devices': ['0']}],
                       'configuration': {'tests': ['gsm8k', 'perf']}})
            return dict(status='completed', cleanup_complete=True)
        write_json(trial / 'service/summary.json', summary)
        return dict(status='failed', cleanup_complete=True)

    monkeypatch.setattr(service_phase, 'execute', execute)
    result = service_phase.run_service({'service': config}, tmp_path, trial, build, identity, 10)
    assert result['checks_passed'] is False and result['service_metrics'] == {}
    assert result['process']['status'] == 'failed'
    assert result['summary']['errors'] == ['RuntimeError: EADDRINUSE']
    observation = dict(result, status='valid', phase='service', trial='trial', receipts={})
    campaign.verify_observation(tmp_path, observation, None, None, None, {}, check_runtime=False)


def test_internal_pipeline_trace_rejects_overflow_and_missing_cores(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/native'))
    from pipeline_trace import decode, TRACE_MAGIC
    from afd_rsi.native import analyze_pipeline
    data = [[0] * 2080 for _ in range(72)]
    for row in data:
        row[:4] = [TRACE_MAGIC, 1, 10, 30]
        row[32:36] = [12, 24, 8, 1]
    transaction = decode(data, {'source': 0, 'layer': 2, 'rows': 7})
    (tmp_path / 'identity.json').write_text('{}')
    (tmp_path / 'rank-1.json').write_text(json.dumps([transaction]))
    summary = analyze_pipeline(tmp_path, 1, 2)
    assert summary['ffn_ranks'][0]['summed_core_cycles']['8'] == 72 * 12
    selected = summary['ffn_ranks'][0]['selected_cores']
    assert [core['core'] for core in selected] == [0, 4, 8, 28, 47, 48, 60, 71]
    assert selected[0]['events'] == [[2, 14, 8, 1]]
    transaction['cores'][0]['events'].append([24, 25, 9, 3])
    transaction['cores'][0]['events'].append([25, 26, 9, 4])
    (tmp_path / 'rank-1.json').write_text(json.dumps([transaction]))
    selected = analyze_pipeline(tmp_path, 1, 2)['ffn_ranks'][0]['selected_cores']
    assert selected[0]['events'][-2:] == [[14, 15, 9, 3], [15, 16, 9, 4]]
    assert summary['ffn_ranks'][0]['first_transaction']['rows'] == 7
    data[0][1] = 513
    with pytest.raises(ValueError, match='overflowed'):
        decode(data, {})
    transaction['cores'].pop()
    (tmp_path / 'rank-1.json').write_text(json.dumps([transaction]))
    with pytest.raises(ValueError, match='incomplete pipeline cores'):
        analyze_pipeline(tmp_path, 1, 2)
