"""Real CPU controller runs exercise durable pool and recovery semantics."""
import copy
import difflib
from pathlib import Path

import pytest

from afd_rsi import campaign, search, search_policy as policy
from afd_rsi.evidence import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
REF = 'skills/ascendc-perf-optimize/SKILL.md'


def prepare(tmp_path, rounds=4, calls=1, model_calls=36):
    recipe = tmp_path / 'recipe.toml'
    text = (ROOT / 'recipes/cpu_adaptive_search.toml').read_text()
    text = text.replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('max_rounds = 9', f'max_rounds = {rounds}').replace('max_calls = 9', f'max_calls = {calls}')
    text = text.replace('repetitions = 4', 'repetitions = 2').replace('max_model_calls = 36', f'max_model_calls = {model_calls}')
    recipe.write_text(text)
    return campaign.prepare(recipe, tmp_path / 'run', ROOT)


def proposal(run, latency, mechanism='tile', priority=3, next_step=''):
    pending = read_json(run / 'search/pending.json')
    context = read_json(run / pending['request'])
    before = context['source_files']['kernel.json']
    import json
    data = json.loads(before)
    data['synthetic_us'] = latency
    after = json.dumps(data) + '\n'
    return dict(hypothesis='controlled synthetic latency', category=mechanism, priority=priority, evidence=['baseline'],
        patch=''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/kernel.json', tofile='b/kernel.json')),
        experiment={k: 'fixture ' + k for k in ('bottleneck','mechanism','prediction','falsification','risk')},
        direction=dict(id='tiling', mechanism=mechanism, step=2, refs=[REF], use='extend: CPU controller fixture only', next=next_step))


def respond(run, proposals):
    pending = read_json(run / 'search/pending.json')
    write_json(run / pending['response'], dict(stop=False, reason='controlled fixture', proposals=proposals))


def records(run):
    state = read_json(run / 'search/state.json')
    return [read_json(run / 'search/rounds' / n / 'state.json') for n in state['rounds']]


def test_pool_survives_exhausted_model_budget_and_resume(tmp_path, monkeypatch):
    run = prepare(tmp_path, rounds=3)
    search.search(run, ROOT, external=True)
    respond(run, [proposal(run, 11), proposal(run, 10.5, 'buffer', 2), proposal(run, 8, 'packing', 1)])
    original = search._start_attempt
    def interrupt(run, root, state, record, stage, round_dir=None):
        if round_dir and round_dir.name == '0002':
            raise KeyboardInterrupt()
        return original(run, root, state, record, stage, round_dir)
    monkeypatch.setattr(search, '_start_attempt', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    before = records(run)
    assert before[1]['reuse']['batch'] == '0001'
    first = digest(run / 'search/rounds/0001/state.json')
    monkeypatch.setattr(search, '_start_attempt', original)
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['calls'] == 1 and state['model_calls_reserved'] == 6
    assert len(state['rounds']) == 3 and records(run)[2]['decision'] == 'promoted'
    assert digest(run / 'search/rounds/0001/state.json') == first
    assert search.best_candidate(run, ROOT)['confirmation'] is not None
    search.search(run, ROOT)
    assert read_json(run / 'search/state.json') == state


def test_pool_origin_tamper_is_rejected(tmp_path, monkeypatch):
    run = prepare(tmp_path, rounds=2)
    search.search(run, ROOT, external=True)
    respond(run, [proposal(run, 11), proposal(run, 8, 'packing', 2)])
    original = search._start_attempt
    def interrupt(*args, **kwargs):
        directory = args[5] if len(args) > 5 else kwargs.get('round_dir')
        if directory and directory.name == '0002':
            raise KeyboardInterrupt()
        return original(*args, **kwargs)
    monkeypatch.setattr(search, '_start_attempt', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    path = run / 'search/rounds/0001/proposal.json'
    value = read_json(path)
    value['proposals'][1]['direction']['mechanism'] = 'tampered'
    write_json(path, value)
    with pytest.raises(ValueError, match='changed'):
        search.search(run, ROOT)


def test_negative_parent_compares_and_promotes_against_best(tmp_path):
    run = prepare(tmp_path, rounds=8, calls=2)
    search.search(run, ROOT, external=True)
    baseline = read_json(run / 'search/best.json')['source_hash']
    respond(run, [proposal(run, 10.2, next_step='then improve scheduling')])
    search.search(run, ROOT)
    request = read_json(run / read_json(run / 'search/pending.json')['request'])
    assert request['source_hash'] != baseline and request['exploration_parent']['baseline_hash'] == baseline
    respond(run, [proposal(run, 9.5, 'schedule')])
    search.search(run, ROOT)
    best = search.best_candidate(run, ROOT)
    assert best['parent_hash'] == baseline
    record = records(run)[1]
    assert record['parent']['source_hash'] != record['comparison']['source_hash']
    verdict = read_json(run / 'attempts' / record['screen'] / 'verdict.json')
    assert verdict['gain_pct'] == pytest.approx(5)
    assert record['branch_depth'] == 2


def test_review_uses_new_evidence_and_skill_catalog_allows_novel_direction(tmp_path):
    run = prepare(tmp_path, rounds=5, calls=2)
    search.search(run, ROOT, external=True)
    request = read_json(run / 'search/rounds/0001/request.json')
    assert request['direction_catalog']['skills'] and REF in request['knowledge']
    proposals = [proposal(run, value, mechanism) for value, mechanism in ((11,'tile'),(10.5,'buffer'),(10.3,'pack'))]
    proposals[2]['direction']['id'] = 'novel-algorithm'
    respond(run, proposals)
    search.search(run, ROOT)
    pending = read_json(run / 'search/pending.json')
    context = read_json(run / pending['request'])
    assert pending['call'] == 2 and context['strategy']['review_due']
    assert context['strategy']['stagnation'] == 3 and len(context['search_feedback']) == 3
    assert context['strategy']['mode'] == 'global-redesign'


def test_model_reservation_crash_cannot_charge_twice(tmp_path, monkeypatch):
    run = prepare(tmp_path)
    original = search._save_state
    def interrupt(run, state):
        original(run, state)
        if state.get('calls') == 1:
            raise KeyboardInterrupt()
    monkeypatch.setattr(search, '_save_state', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT, external=True)
    monkeypatch.setattr(search, '_save_state', original)
    search.search(run, ROOT, external=True)
    state = read_json(run / 'search/state.json')
    assert state['calls'] == 1 and state['model_calls_reserved'] == 6


def test_information_gate_uses_valid_evaluations_and_empirical_uncertainty():
    plan = dict(min_gain_pct=2, search=dict(info_floor=.1, margin=0, max_rounds=12, explore_max_fraction=.25))
    candidates = [dict(proposal=dict(priority=3, direction=dict(id='a'))), dict(proposal=dict(priority=2, direction=dict(id='b')))]
    rows = [dict(gain=g, baseline='best', direction=dict(id=d)) for d,g in [('a',1),('a',1),('a',1),('b',-8),('b',8),('b',-1)]]
    assert not policy.rank(candidates, rows, 'best', plan, 0)['explored']
    rows.append(dict(gain=1, baseline='best', direction=dict(id='a')))
    assert policy.rank(candidates, rows, 'best', plan, 0)['explored']
    assert not policy.rank(candidates, rows, 'best', plan, 3)['explored']
    assert not policy.rank(candidates, rows[:1] * 7, 'best', plan, 0)['explored']


@pytest.mark.parametrize('point', ['request', 'selection', 'attempt', 'promotion', 'ledger'])
def test_atomic_boundaries_resume_without_repeating_trials(tmp_path, monkeypatch, point):
    run = prepare(tmp_path, rounds=2)
    search.search(run, ROOT, external=True)
    respond(run, [proposal(run, 11), proposal(run, 8, 'packing', 2)])
    original = search.write_json
    interrupted = False
    def write(path, value):
        nonlocal interrupted
        original(path, value)
        second = '0002' in path.parts
        hit = ((point == 'request' and second and path.name == 'request.json') or
               (point == 'selection' and second and path.name == 'state.json' and value.get('phase') == 'proposed') or
               (point == 'attempt' and second and path.name == 'state.json' and 'screen' in value) or
               (point == 'promotion' and path.name == 'best.json' and value.get('confirmation')) or
               (point == 'ledger' and path == run / 'search/state.json' and len(value['rounds']) == 2 and value['active'] is None))
        if hit and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
    monkeypatch.setattr(search, 'write_json', write)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    assert interrupted
    receipts = {p: digest(p) for p in (run / 'attempts').rglob('process.json')}
    monkeypatch.setattr(search, 'write_json', original)
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['calls'] == 1 and records(run)[1]['decision'] == 'promoted'
    assert all(digest(path) == value for path, value in receipts.items())
    ledger = read_json(run / 'search/rounds/0002/routing.json')
    assert ledger['kind'] == policy.VERSION
    from afd_rsi.router import Ledger
    assert len(Ledger(run / 'search/ledger.jsonl').load().rows) == 2


def test_debug_reservation_and_original_pool_survive_resume(tmp_path, monkeypatch):
    import sys
    run = prepare(tmp_path, rounds=2, calls=2)
    search.search(run, ROOT, external=True)
    choices = [proposal(run, 11), proposal(run, 8, 'packing', 2)]
    respond(run, choices)
    execute, save = campaign.execute, search._save_state
    def failed_build(argv, cwd, output, timeout, env):
        if env.get('AFD_RSI_SIDE') == 'candidate' and argv[-1] == 'build' and '0001' in Path(cwd).parts and 'debug' not in Path(cwd).parts:
            argv = [sys.executable, '-c', 'raise RuntimeError("candidate compile failure")']
        return execute(argv, cwd, output, timeout, env)
    def interrupt(run, state):
        save(run, state)
        if state['calls'] == 2:
            raise KeyboardInterrupt()
    monkeypatch.setattr(campaign, 'execute', failed_build)
    monkeypatch.setattr(search, '_save_state', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    monkeypatch.setattr(search, '_save_state', save)
    search.search(run, ROOT)
    assert '/debug/' in read_json(run / 'search/pending.json')['request']
    respond(run, choices[:1])
    search.search(run, ROOT)
    state = read_json(run / 'search/state.json')
    assert state['calls'] == 2 and state['model_calls_reserved'] == 9
    assert records(run)[0]['debug']['decision'] == 'rejected'
    assert records(run)[1]['reuse']['batch'] == '0001'
    assert search.best_candidate(run, ROOT)['confirmation'] is not None


def test_confirmation_failure_remains_best_and_is_not_a_new_exploration_seed(tmp_path, monkeypatch):
    run = prepare(tmp_path, rounds=1)
    search.search(run, ROOT, external=True)
    respond(run, [proposal(run, 8, next_step='unsafe continuation must not be used')])
    original = campaign.execute
    def fail(argv, cwd, output, timeout, env):
        if env.get('AFD_RSI_SIDE') == 'candidate' and argv[-1] == 'correctness' and len(list((run/'attempts').glob('*/verdict.json'))) >= 1:
            import sys
            argv = [sys.executable, '-c', 'raise RuntimeError("numerical failure")']
        return original(argv, cwd, output, timeout, env)
    monkeypatch.setattr(campaign, 'execute', fail)
    search.search(run, ROOT)
    assert search.best_candidate(run, ROOT)['confirmation'] is None
    items = policy.rounds(run, read_json(run / 'search/state.json'))
    assert policy.measured(run, items[0]) is None


def test_invalid_debug_reply_can_resume_and_reuse_original_batch(tmp_path, monkeypatch):
    import sys
    run = prepare(tmp_path, rounds=2, calls=2)
    search.search(run, ROOT, external=True)
    choices = [proposal(run, 11), proposal(run, 8, 'packing', 2)]
    respond(run, choices)
    execute = campaign.execute
    def failed_build(argv, cwd, output, timeout, env):
        if env.get('AFD_RSI_SIDE') == 'candidate' and argv[-1] == 'build' and '0001' in Path(cwd).parts:
            argv = [sys.executable, '-c', 'raise RuntimeError("compile failure")']
        return execute(argv, cwd, output, timeout, env)
    monkeypatch.setattr(campaign, 'execute', failed_build)
    search.search(run, ROOT)
    assert '/debug/' in read_json(run / 'search/pending.json')['request']
    respond(run, [dict(choices[0], priority=0)])
    write = search.write_json
    def interrupt(path, value):
        write(path, value)
        if path.name == 'state.json' and value.get('decision') == 'invalid_agent':
            raise KeyboardInterrupt()
    monkeypatch.setattr(search, 'write_json', interrupt)
    with pytest.raises(KeyboardInterrupt):
        search.search(run, ROOT)
    monkeypatch.setattr(search, 'write_json', write)
    search.search(run, ROOT)
    assert records(run)[0]['decision'] == 'invalid_agent'
    assert records(run)[1]['decision'] == 'promoted'
    assert read_json(run / 'search/state.json')['calls'] == 2


def test_insufficient_completion_budget_stops_before_dispatch(tmp_path):
    run = prepare(tmp_path, model_calls=2)
    search.search(run, ROOT, external=True)
    state = read_json(run / 'search/state.json')
    assert state['status'] == 'budget_exhausted' and state['calls'] == 0
    assert state['rounds'] == [] and state['active'] is None
    assert not (run / 'search/pending.json').exists()


def test_completed_global_review_cannot_retrigger_without_new_measurements():
    plan = dict(min_gain_pct=2)
    best = dict(source_hash='best')
    rows = [dict(gain=-1, noise=0, baseline='best', promoted=False,
                 direction=dict(id='tiling', mechanism=str(i), step=2)) for i in range(3)]
    item = dict(record={}, request=dict(strategy=dict(mode='global-redesign', anchor='best', review_count=3)))
    assert not policy.strategy(plan, [item], rows, best)['review_due']
