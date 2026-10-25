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
def staged(tmp_path, monkeypatch):
    recipe = tmp_path / 'recipe.toml'
    text = (ROOT / 'recipes/cpu_search.toml').read_text().replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('backend = "cpu-synthetic"', 'backend = "cpu-synthetic"\nvalidation = "operator-first"')
    text = text.replace('repetitions = 4', 'repetitions = 2').replace('max_rounds = 3', 'max_rounds = 1')
    text += '\n[service]\nrepo = "."\nargs = ["--test", "smoke"]\ntimeout_seconds = 10\ncleanup_seconds = 1\n'
    recipe.write_text(text)
    calls = []

    def fake_service(plan, run, trial, build, identity, timeout):
        calls.append(identity['hash'])
        summary = {'test_results': {'smoke': 'passed'}}
        write_json(trial / 'service/summary.json', summary)
        write_json(trial / 'service/plan.json', {'configuration': {'tests': ['smoke']}})
        write_json(trial / 'loaded-maps.json', {'fixture': 'no real service'})
        return dict(process={'status': 'completed', 'cleanup_complete': True}, checks_passed=True,
                    summary_hash=digest(trial / 'service/summary.json'),
                    loaded_maps_hash=digest(trial / 'loaded-maps.json'),
                    evidence=service_phase.evidence_files(trial / 'service'), service_metrics={})

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
