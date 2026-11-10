"""Exercise real controller stages; service calls are explicit CPU-only fixtures."""
import copy
from pathlib import Path

import pytest

from afd_rsi import campaign, search, service_phase
from afd_rsi.evidence import digest, read_json, write_json
from afd_rsi.evaluate import evaluate

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = dict(bottleneck='Synthetic latency field is 10.', mechanism='Reduce fixture latency to 8.',
                  prediction='Both paired runs report 8.', falsification='Either run remains at 10.',
                  risk='Numerical multiplier and bias must stay unchanged.')


@pytest.fixture
def staged(tmp_path, monkeypatch, request):
    candidate_ms = getattr(request, 'param', None)
    recipe = tmp_path / 'recipe.toml'
    text = (ROOT / 'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('backend = "cpu-synthetic"', 'backend = "cpu-synthetic"\nvalidation = "operator-first"')
    text = text.replace('repetitions = 4', 'repetitions = 2').replace('max_rounds = 9', 'max_rounds = 1')
    text += '\n[service]\nrepo = "."\nargs = ["--test", "smoke"]\ntimeout_seconds = 10\ncleanup_seconds = 1\n'
    if candidate_ms is not None:
        text += 'primary = { id = "point-000", metric = "ttft_p50_ms" }\n'
    recipe.write_text(text)
    calls = []

    if candidate_ms is not None:
        def fixture_metrics(summary, configuration, **_):
            point = dict(id='point-000', phase='prefill', input_lens=[512], output_lens=[1],
                         concurrency=1, requests=8)
            return {'performance': [{**point, 'ttft_ms': {'p50': summary['latency_ms']}}]}
        monkeypatch.setattr(service_phase, 'service_metrics', fixture_metrics)

    def fake_service(plan, run, trial, build, identity, timeout):
        calls.append(plan['service']['args'])
        summary = {'test_results': {'smoke': 'passed'}}
        if candidate_ms is not None:
            summary['latency_ms'] = 100 if identity['hash'] == plan['source_snapshot']['hash'] else candidate_ms
        write_json(trial / 'service/summary.json', summary)
        write_json(trial / 'service/plan.json', {'configuration': {'tests': ['smoke']}})
        write_json(trial / 'loaded-maps.json', {'fixture': 'no real service'})
        return dict(process={'status': 'completed', 'cleanup_complete': True}, checks_passed=True,
                    summary_hash=digest(trial / 'service/summary.json'),
                    loaded_maps_hash=digest(trial / 'loaded-maps.json'),
                    evidence=service_phase.evidence_files(trial / 'service'),
                    service_metrics=service_phase.service_metrics(summary, {}) if candidate_ms is not None else {})

    monkeypatch.setattr(service_phase, 'run_service', fake_service)
    return campaign.prepare(recipe, tmp_path / 'run', ROOT), calls


def respond(run, patch=None):
    proposal = dict(hypothesis='Synthetic latency decreases.', category='fixture', evidence=['baseline'], priority=3,
                    patch=patch or (ROOT / 'examples/faster.patch').read_text(), experiment=EXPERIMENT)
    pending = read_json(run / 'search/pending.json')
    write_json(run / pending['response'], dict(stop=False, reason='CPU fixture only.', proposals=[proposal]))


def test_screen_then_fresh_confirmation_then_service(staged):
    run, calls = staged
    search.search(run, ROOT, external=True)
    assert not calls  # Initial operator qualification and proposal need no model service.
    respond(run)
    search.search(run, ROOT)
    best = search.best_candidate(run, ROOT)
    assert best['confirmation'] is not None and len(calls) == 2
    screen = run / 'attempts' / best['screen']['attempt']
    final = run / 'attempts' / best['confirmation']['attempt']
    assert read_json(screen / 'verdict.json')['validation_level'] == 'operator'
    assert read_json(final / 'verdict.json')['validation_level'] == 'full'
    assert read_json(final / 'operator-gate.json')['verdict'] == 'PASS'
    assert all(step['phase'] not in ('service', 'profile') for step in read_json(screen / 'progress.json')['steps'])
    phases = [s['phase'] for s in read_json(final / 'progress.json')['steps']]
    assert phases.index('service') > max(i for i, p in enumerate(phases) if p == 'benchmark')
    # No repeated services on resume; all completed evidence is audited.
    search.search(run, ROOT)
    assert len(calls) == 2
    assert read_json(run / 'search/experience/0001.json')['experiment'] == EXPERIMENT


def test_service_runtime_debug_repeats_screen_and_full_confirmation(staged, monkeypatch):
    run, calls = staged
    search.search(run, ROOT, external=True)
    respond(run)
    original = service_phase.run_service
    failed = False
    def service(plan, run, trial, build, identity, timeout):
        nonlocal failed
        result = original(plan, run, trial, build, identity, timeout)
        if not failed and identity['hash'] != plan['source_snapshot']['hash']:
            failed = True
            summary = read_json(trial / 'service/summary.json') | {'inference_status': 'failed'}
            write_json(trial / 'service/summary.json', summary)
            result.update(process=dict(status='failed', cleanup_complete=True), checks_passed=False,
                          summary=summary, summary_hash=digest(trial / 'service/summary.json'),
                          evidence=service_phase.evidence_files(trial / 'service'))
        return result
    monkeypatch.setattr(service_phase, 'run_service', service)
    search.search(run, ROOT)
    record = read_json(run / 'search/rounds/0001/state.json')
    assert record['phase'] == 'awaiting_agent' and 'confirmation' in record['debug']
    original_attempts = {record['debug']['screen'], record['debug']['confirmation']}
    assert len(calls) == 2 and search.best_candidate(run, ROOT)['confirmation'] is None
    respond(run)
    search.search(run, ROOT)
    best = search.best_candidate(run, ROOT)
    assert best['confirmation'] is not None and len(calls) == 4
    assert not original_attempts & {best['screen']['attempt'], best['confirmation']['attempt']}
    assert read_json(run / 'search/state.json')['calls'] == 2


@pytest.mark.parametrize('staged,native_us,expected', [(80, 12, 'promoted'), (110, 8, 'rejected')], indirect=['staged'])
def test_search_uses_model_service_gain_over_synthetic_gain(staged, native_us, expected):
    run, calls = staged
    search.search(run, ROOT, external=True)
    patch = (ROOT / 'examples/faster.patch').read_text().replace('"synthetic_us": 8.0',
                                                                 f'"synthetic_us": {native_us}.0')
    respond(run, patch)
    search.search(run, ROOT)
    record = read_json(run / 'search/rounds/0001/state.json')
    assert record['decision'] == expected
    screen = run / 'attempts' / record['screen']
    verdict = read_json(screen / 'verdict.json')
    assert verdict['primary_source'] == 'service'
    assert verdict['gain_pct'] == pytest.approx(20 if expected == 'promoted' else -10)
    assert verdict['native_gain_pct'] == pytest.approx(-20 if native_us == 12 else 20)
    phases = [(step['side'], step['phase']) for step in read_json(screen / 'progress.json')['steps']]
    assert phases[-2:] == [('baseline', 'service'), ('candidate', 'service')]
    if expected == 'promoted':
        final = run / 'attempts' / record['confirmation']
        final_phases = [(step['side'], step['phase']) for step in read_json(final / 'progress.json')['steps']]
        assert final_phases[-2:] == [('candidate', 'service'), ('baseline', 'service')]
        assert len(calls) == 4
        assert all(args[-2:] == ['--test', 'smoke,perf'] for args in calls[:2])
        assert all(args[-2:] == ['--test', 'smoke'] for args in calls[2:])
    else:
        assert 'confirmation' not in record and len(calls) == 2


def test_confirmation_loses_gain_defers_all_services(staged, monkeypatch):
    run, calls = staged
    search.search(run, ROOT, external=True)
    respond(run)
    original = campaign.execute

    def regression(argv, cwd, output, timeout, env):
        result = original(argv, cwd, output, timeout, env)
        record = read_json(run / 'search/rounds/0001/state.json')
        if record['phase'] == 'confirming' and argv[-1] == 'benchmark' and env['AFD_RSI_SIDE'] == 'candidate':
            path = Path(env['AFD_RSI_RESULT'])
            data = read_json(path)
            for rank in data['ranks']:
                rank['value'] = 12
            write_json(path, data)
        return result

    monkeypatch.setattr(campaign, 'execute', regression)
    search.search(run, ROOT)
    record = read_json(run / 'search/rounds/0001/state.json')
    final = run / 'attempts' / record['confirmation']
    assert record['decision'] == 'rejected' and not calls
    assert read_json(final / 'operator-gate.json')['verdict'] == 'FAIL'
    assert search.best_candidate(run, ROOT)['confirmation'] is None
    campaign.audit_attempt(run, final, ROOT)
    write_json(final / 'operator-gate.json', {'verdict': 'PASS'})
    with pytest.raises(ValueError, match='operator gate'):
        campaign.audit_attempt(run, final, ROOT)


@pytest.mark.parametrize('numeric_failure', [False, True])
def test_rejected_screen_never_starts_service(staged, numeric_failure):
    run, calls = staged
    search.search(run, ROOT, external=True)
    patch = (ROOT / 'examples/faster.patch').read_text()
    patch = patch.replace('"synthetic_us": 8.0', '"synthetic_us": 12.0')
    if numeric_failure:
        patch = patch.replace('+{"synthetic_us": 12.0, "multiplier": 2, "bias": 1}',
                              '+{"synthetic_us": 12.0, "multiplier": 2, "bias": 2}')
    respond(run, patch)
    search.search(run, ROOT)
    assert not calls and search.best_candidate(run, ROOT)['confirmation'] is None
    record = read_json(run / 'search/rounds/0001/state.json')
    if numeric_failure:
        assert record['phase'] == 'awaiting_agent' and record['debug']['decision'] == 'rejected'
        assert read_json(run / 'search/state.json')['calls'] == 2
        record = record['debug']
    assert record['decision'] == 'rejected'
    assert 'confirmation' not in record


def test_operator_pass_does_not_satisfy_service_or_profile():
    plan = dict(backend='ascend', scope='operator', unit='us', repetitions=1, min_gain_pct=2,
                service={}, native={'profile': True}, validation='operator-first')
    observations = [dict(side=side, phase=phase, status='valid', checks_passed=True, value=value)
                    for side, value in [('baseline', 10), ('candidate', 8)]
                    for phase in ('correctness', 'benchmark')]
    assert evaluate(plan, observations, level='operator')['verdict'] == 'PASS'
    assert evaluate(plan, observations)['verdict'] == 'INCONCLUSIVE'
    jobs = campaign.phase_jobs(plan, ['baseline', 'candidate'], level='operator')
    assert not any(p in ('service', 'profile') for _, p, _ in jobs)
    jobs = campaign.phase_jobs(plan, ['baseline'])
    assert not any(p == 'service' for _, p, _ in jobs)
    assert jobs[-1] == ('baseline', 'profile', None)


def test_operator_case_regression_blocks_a_favorable_suite_mean():
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=2,
                min_gain_pct=2, native={'max_case_regression_pct': 5})
    observations = [dict(side=side, phase='correctness', status='valid', checks_passed=True)
                    for side in ('baseline', 'candidate')]
    for side in ('baseline', 'candidate'):
        cases = [(10, 10), (5, 11)][side == 'candidate']
        metrics = [dict(case=index, layer=2, sparse=False, tokens=128, median_us=value)
                   for index, value in enumerate(cases)]
        observations.extend(dict(side=side, phase='benchmark', status='valid', checks_passed=True,
                                 value=sum(cases) / 2, case_metrics=metrics) for _ in range(2))
    result = evaluate(plan, observations, level='operator')
    assert result['gain_pct'] == pytest.approx(20)
    assert result['verdict'] == 'FAIL'
    assert result['operator_case_comparison'][1]['median_gain_pct'] == pytest.approx(-10)
    observations[-1].pop('case_metrics')
    assert evaluate(plan, observations, level='operator')['verdict'] == 'INCONCLUSIVE'


def test_service_regression_needs_model_repeat_before_promotion():
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=1, min_gain_pct=2,
                service={'max_regression_pct': 5})
    point = dict(id='point-000', phase='prefill', input_lens=[512], output_lens=[1],
                 concurrency=1, requests=8)
    observations = []
    for side, native_us, ttft_ms in [('baseline', 10, 100), ('candidate', 8, 110)]:
        observations.extend([
            dict(side=side, phase='correctness', status='valid', checks_passed=True),
            dict(side=side, phase='benchmark', status='valid', checks_passed=True, value=native_us),
            dict(side=side, phase='service', status='valid', checks_passed=True,
                 service_metrics={'performance': [{**point, 'ttft_ms': {'p50': ttft_ms}}]}),
        ])
    result = evaluate(plan, observations)
    assert result['gain_pct'] == pytest.approx(20)
    assert result['verdict'] == 'INCONCLUSIVE'
    assert result['service_performance_comparison'][0]['gain_pct'] == pytest.approx(-10)


def test_service_primary_uses_model_gain_and_keeps_native_regressions_diagnostic():
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=1, min_gain_pct=2,
                native={'max_case_regression_pct': 5},
                service={'primary': {'id': 'point-000', 'metric': 'ttft_p50_ms'}})
    point = dict(id='point-000', phase='prefill', input_lens=[512], output_lens=[1],
                 concurrency=1, requests=8)
    observations = []
    for side, native_us, ttft_ms in [('baseline', 10, 100), ('candidate', 12, 80)]:
        observations.extend([
            dict(side=side, phase='correctness', status='valid', checks_passed=True),
            dict(side=side, phase='benchmark', status='valid', checks_passed=True,
                 value=native_us, case_metrics=[dict(case=0, layer=2, sparse=False,
                                                     tokens=129, median_us=native_us)]),
            dict(side=side, phase='service', status='valid', checks_passed=True,
                 service_metrics={'performance': [{**point, 'ttft_ms': {'p50': ttft_ms}}]}),
        ])
    result = evaluate(plan, observations, level='operator')
    assert result['verdict'] == 'PASS' and result['gain_pct'] == pytest.approx(20)
    assert result['native_gain_pct'] == pytest.approx(-20)
    assert result['operator_case_comparison'][0]['median_gain_pct'] == pytest.approx(-20)
    observations[-1]['service_metrics'] = {}
    assert evaluate(plan, observations, level='operator')['verdict'] == 'INCONCLUSIVE'


def test_service_primary_mixes_decode_and_prefill_without_hiding_regressions():
    plan = dict(backend='ascend', scope='native', unit='us', repetitions=1, min_gain_pct=2,
                service={'primary': [{'id': 'native-long-decode', 'metric': 'common_window_tpot_ms_p50'},
                                    {'id': 'mixed-prefill', 'metric': 'ttft_p50_ms'}],
                         'max_regression_pct': 5})
    observations = []
    for side, decode_ms, prefill_ms in [('baseline', 10, 100), ('candidate', 9, 80)]:
        scenarios = [dict(id='native-long-decode', configuration={'routing': 'native'},
                          per_dp=[], per_workload_group=[], ttft_ms={'mean': 50, 'p50': 50, 'p90': 50},
                          common_window_tpot_ms={'p50': decode_ms}, request_tpot_ms={'p50': decode_ms}),
                     dict(id='mixed-prefill', configuration={'routing': 'native'},
                          per_dp=[], per_workload_group=[],
                          ttft_ms={'mean': prefill_ms, 'p50': prefill_ms, 'p90': prefill_ms})]
        observations.extend([
            dict(side=side, phase='correctness', status='valid', checks_passed=True),
            dict(side=side, phase='benchmark', status='valid', checks_passed=True, value=10),
            dict(side=side, phase='service', status='valid', checks_passed=True,
                 service_metrics={'scenarios': scenarios}),
        ])
    result = evaluate(plan, observations, level='operator')
    assert result['verdict'] == 'PASS' and result['gain_pct'] == pytest.approx(15)
    assert len(result['primary_metric']) == 2
    observations[2]['service_metrics']['scenarios'][1]['input_tokens_per_s'] = 10
    observations[-1]['service_metrics']['scenarios'][1]['input_tokens_per_s'] = 9
    assert evaluate(plan, observations, level='operator')['verdict'] == 'PASS'
    assert evaluate(plan, observations)['verdict'] == 'INCONCLUSIVE'
    observations[-1]['service_metrics']['scenarios'][1]['ttft_ms']['p50'] = 110
    observations[-1]['service_metrics']['scenarios'][1]['ttft_ms']['mean'] = 110
    observations[-1]['service_metrics']['scenarios'][1]['ttft_ms']['p90'] = 110
    assert evaluate(plan, observations, level='operator')['verdict'] != 'PASS'


def test_evidence_and_falsification_are_required():
    requirements = dict(experiment_fields=list(EXPERIMENT), required_evidence_ids=['baseline', 'profile-incumbent'])
    value = dict(stop=False, reason='fixture', proposals=[dict(hypothesis='test', category='test',
                 evidence=['baseline', 'profile-incumbent'], priority=3, patch='diff', experiment=EXPERIMENT)])
    search.validate_proposal(value, set(requirements['required_evidence_ids']), requirements)
    missing = copy.deepcopy(value)
    missing['proposals'][0]['experiment']['falsification'] = ''
    with pytest.raises(ValueError, match='falsifiable'):
        search.validate_proposal(missing, set(requirements['required_evidence_ids']), requirements)

    missing = copy.deepcopy(value)
    missing['proposals'][0]['evidence'].remove('profile-incumbent')
    with pytest.raises(ValueError, match='incumbent'):
        search.validate_proposal(missing, set(requirements['required_evidence_ids']), requirements)


def test_final_service_failure_cannot_promote(staged, monkeypatch):
    run, calls = staged
    search.search(run, ROOT, external=True)
    respond(run)
    original = service_phase.run_service

    def failed_service(*args):
        result = original(*args)
        result['checks_passed'] = False
        return result

    monkeypatch.setattr(service_phase, 'run_service', failed_service)
    search.search(run, ROOT)
    assert len(calls) == 1
    assert search.best_candidate(run, ROOT)['confirmation'] is None
    record = read_json(run / 'search/rounds/0001/state.json')
    assert record['decision'] == 'rejected'
    final = run / 'attempts' / record['confirmation']
    assert read_json(final / 'operator-gate.json')['verdict'] == 'PASS'
    assert read_json(final / 'verdict.json')['verdict'] == 'FAIL'


def test_operator_gate_resume_does_not_repeat_native_trials(staged, monkeypatch):
    run, calls = staged
    search.search(run, ROOT, external=True)
    respond(run)
    original = campaign.command
    interrupted = False

    def interrupt(*args, **kwargs):
        nonlocal interrupted
        if args[4] == 'service' and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, 'command', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    record = read_json(run / 'search/rounds/0001/state.json')
    final = run / 'attempts' / record['confirmation']
    saved = {p: digest(p) for p in final.glob('*/benchmark/*/observation.json')}
    assert saved and not calls
    search.search(run, ROOT)
    assert len(calls) == 2 and search.best_candidate(run, ROOT)['confirmation'] is not None
    assert all(digest(p) == value for p, value in saved.items())
    # A successful operator gate cannot excuse omitted final service phases.
    progress = read_json(final / 'progress.json')
    progress['steps'] = [step for step in progress['steps'] if step['phase'] != 'service']
    write_json(final / 'progress.json', progress)
    with pytest.raises(ValueError, match='phases missing'):
        campaign.audit_attempt(run, final, ROOT)
