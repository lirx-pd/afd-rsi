"""Direction-router tests.

The router itself is portable, so these run anywhere. The few cases that touch
``afd_rsi.search`` need the POSIX controller (``fcntl``) and are skipped
elsewhere rather than pretending to cover it.
"""

import copy

import pytest

from afd_rsi import contracts, router

WEIGHTS = router.DEFAULT_WEIGHTS


def ledger_at(tmp_path, name='ledger.jsonl'):
    return router.Ledger(tmp_path / name)


def fill(ledger, category, scores, features, *, foundation='f0', decision='rejected'):
    for score in scores:
        ledger.add(category=category, features=features, score=score,
                   decision=decision, foundation_hash=foundation)


def test_difficulty_is_a_frozen_weighted_sum():
    features = {'T': 1, 'S': 2, 'B': 0, 'R': 0, 'D': 0, 'M': 0, 'F': 0, 'C': 1}
    assert router.difficulty(features) == pytest.approx(
        WEIGHTS['T'] + 2 * WEIGHTS['S'] + WEIGHTS['C'])
    assert router.difficulty({}) == 0.0


def test_patch_features_only_count_added_lines_and_saturate():
    patch = ('diff --git a/kernel.cpp b/kernel.cpp\n'
             '--- a/kernel.cpp\n'
             '+++ b/kernel.cpp\n'
             '@@ -1 +1 @@\n'
             '-DataCopy(stride, offset, address);\n'
             '+DataCopy(src, dst);\n'
             '+stride = offset + address;\n'
             '+DataCopy(src, dst);\n')
    features = router.patch_features(patch)
    assert features['M'] == router.FEATURE_CAP, 'added data-movement evidence saturates'
    assert features['T'] == 0, 'the removed line must not contribute'
    assert set(features) == set(router.DIMENSIONS)


def test_normalized_score_is_monotone_neutral_and_bounded():
    assert router.normalized_score({'gain_pct': None}, 2.0) == 0.5
    assert router.normalized_score({'gain_pct': 0.0}, 2.0) == pytest.approx(0.5)
    low = router.normalized_score({'gain_pct': -4.0}, 2.0)
    mid = router.normalized_score({'gain_pct': 2.0}, 2.0)
    high = router.normalized_score({'gain_pct': 40.0}, 2.0)
    assert 0.0 < low < 0.5 < mid < high < 1.0
    # A threshold gain is a real but unremarkable win, not a saturated signal.
    assert 0.7 < mid < 0.8


def test_ledger_round_trips_and_detects_tampering(tmp_path):
    ledger = ledger_at(tmp_path)
    ledger.add(category='tiling', features={'B': 2}, score=0.7, decision='promoted',
               foundation_hash='f0', round_name='0001', gain_pct=3.5)
    reloaded = ledger_at(tmp_path).load()
    assert len(reloaded.rows) == 1
    assert reloaded.rows[0]['round'] == '0001'
    assert reloaded.rows[0]['gain_pct'] == 3.5
    reloaded.rows[0]['score'] = 0.99
    with pytest.raises(ValueError, match='changed'):
        reloaded.verify()


def test_ledger_rejects_a_damaged_line(tmp_path):
    path = tmp_path / 'ledger.jsonl'
    path.write_text('{"schema_version": 1, "category": "x", "record_hash": "nope"}\n')
    with pytest.raises(ValueError, match='changed'):
        router.Ledger(path).load()


def test_ability_falls_back_to_scalar_before_eight_dimensions_are_identified(tmp_path):
    ledger = ledger_at(tmp_path)
    fill(ledger, 'tiling', [0.7, 0.62], {'B': 2})
    ability = router.Ability(ledger, foundation_hash='f0')
    model = ability.fit('tiling')
    assert model['kind'] == 'scalar', 'two rows cannot identify eight dimensions'
    value, error = ability.predict('tiling', {'B': 2})
    assert 0.0 < value < 1.0 and error == 1.0
    assert ability.fit('unknown') is None
    assert ability.predict('unknown', {}) == (0.5, 1.0)


def test_ability_uses_the_vector_model_once_identified(tmp_path):
    ledger = ledger_at(tmp_path)
    for index in range(12):
        ledger.add(category='fusion', features={'F': index % 3 + 1, 'M': index % 2},
                   score=0.8 if index % 3 else 0.3, decision='promoted',
                   foundation_hash='f0')
    ability = router.Ability(ledger, foundation_hash='f0')
    assert ability.fit('fusion')['kind'] == 'vector'


def test_stale_foundation_rows_keep_half_weight(tmp_path):
    ledger = ledger_at(tmp_path)
    for index in range(12):
        ledger.add(category='fusion', features={'F': index % 3 + 1}, score=0.75,
                   decision='promoted', foundation_hash='old')
    # Twelve stale rows at half weight are worth six, below the nine needed, so
    # the model refuses to fit eight dimensions on aged evidence alone.
    stale = router.Ability(ledger, foundation_hash='new')
    assert stale.fit('fusion')['kind'] == 'scalar'
    fresh = router.Ability(ledger, foundation_hash='old')
    assert fresh.fit('fusion')['kind'] == 'vector'


def test_router_prefers_the_family_with_better_evidence(tmp_path):
    ledger = ledger_at(tmp_path)
    settings = {'router': 'auto'}
    for index in range(12):
        ledger.add(category='fusion', features={'F': index % 3 + 1},
                   score=0.85, decision='promoted', foundation_hash='f0')
        ledger.add(category='tiling', features={'F': index % 3 + 1},
                   score=0.2, decision='rejected', foundation_hash='f0')
    instance = router.Router(settings, ledger, foundation_hash='f0')
    decision = instance.rank([
        {'category': 'tiling', 'features': {'F': 2}, 'priority': 3, 'index': 0},
        {'category': 'fusion', 'features': {'F': 2}, 'priority': 1, 'index': 1},
    ])
    # Priority is a tie-break inside a family, not a global override.
    assert decision['selected'] == 'fusion'
    assert decision['index'] == 1
    assert decision['kind'] == 'vector'
    assert decision['explored'] is False


def test_router_breaks_priority_ties_inside_a_family_only(tmp_path):
    ledger = ledger_at(tmp_path)
    instance = router.Router({'router': 'auto'}, ledger, foundation_hash='f0')
    decision = instance.rank([
        {'category': 'tiling', 'features': {}, 'priority': 1, 'index': 0},
        {'category': 'tiling', 'features': {}, 'priority': 3, 'index': 1},
    ])
    assert decision['index'] == 1


def test_exploration_is_off_unless_a_recipe_opts_in(tmp_path):
    ledger = ledger_at(tmp_path)
    fill(ledger, 'fusion', [0.9] * 12, {'F': 1}, decision='promoted')
    items = [{'category': 'fusion', 'features': {'F': 1}, 'priority': 3, 'index': 0},
             {'category': 'tiling', 'features': {'F': 1}, 'priority': 1, 'index': 1}]
    quiet = router.Router({'router': 'auto'}, copy.deepcopy(ledger), foundation_hash='f0')
    assert quiet.rank(items)['explored'] is False
    assert quiet.rank(items)['selected'] == 'fusion'


def test_exploration_respects_its_budget_when_enabled(tmp_path):
    ledger = ledger_at(tmp_path)
    fill(ledger, 'fusion', [0.9] * 12, {'F': 1}, decision='promoted')
    items = [{'category': 'fusion', 'features': {'F': 1}, 'priority': 3, 'index': 0},
             {'category': 'tiling', 'features': {'F': 1}, 'priority': 1, 'index': 1}]
    instance = router.Router({'router': 'auto', 'info_floor': 0.05, 'margin': 0.0,
                              'explore_max_fraction': 0.25}, ledger, foundation_hash='f0')
    explored = 0
    for _ in range(8):
        explored += instance.rank(items)['explored']
    assert explored >= 1, 'an untested family with a wide bound is worth one look'
    assert explored <= 2, 'exploration may never take the majority of rounds'


def test_router_off_and_undeclared_campaigns_keep_the_old_rule(tmp_path):
    ledger = ledger_at(tmp_path)
    off = router.Router({'router': 'off'}, ledger, foundation_hash='f0')
    assert off.enabled is False
    legacy = router.Router({}, ledger, foundation_hash='f0')
    assert legacy.enabled is False, 'a campaign frozen before routing existed must not switch estimator'
    assert off.kind() == 'off'


def test_directive_names_an_untested_family_only(tmp_path):
    ledger = ledger_at(tmp_path)
    empty = router.Router({'router': 'auto'}, ledger, foundation_hash='f0')
    assert empty.directive() is None
    fill(ledger, 'fusion', [0.8] * 3, {'F': 1}, decision='promoted')
    stale = router.Router({'router': 'auto'}, ledger, foundation_hash='new')
    directive = stale.directive()
    assert directive['preferred_category'] == 'fusion'
    fresh = router.Router({'router': 'auto'}, ledger, foundation_hash='f0')
    assert fresh.directive() is None


def test_contracts_fill_router_defaults_into_the_frozen_plan(tmp_path):
    recipe = tmp_path / 'recipe.toml'
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'kernel.json').write_text('{}')
    recipe.write_text(
        'name = "r"\nbackend = "cpu-synthetic"\nsource = "source"\n'
        'workload_id = "w"\nscope = "s"\nunit = "us"\nmetric = "m"\n'
        'ranks = 1\nprimary_rank = 0\nrepetitions = 1\n'
        'timeout_seconds = 10\nbudget_seconds = 10\nmin_gain_pct = 1\n'
        'allowed_files = ["kernel.json"]\nartifacts = ["kernel.json"]\n'
        '[commands]\nbuild = ["{python}", "-c", "pass"]\n'
        'correctness = ["{python}", "{foundation}/benchmarks/x.py"]\n'
        'benchmark = ["{python}", "{foundation}/benchmarks/x.py"]\n'
        '[search]\nprovider = "codex"\n'
        'max_rounds = 6\nmax_calls = 6\nagent_timeout_seconds = 10\n'
        'max_context_bytes = 8192\nexperience_dirs = []\n')
    config = contracts.load_recipe(recipe)
    search = config['search']
    assert search['router'] == 'auto'
    assert search['info_floor'] == 0.0, 'exploration must be an explicit opt-in'
    assert search['margin'] == 0.02
    assert search['explore_max_fraction'] == 0.25
    assert search['features']['weights'] == {name: float(WEIGHTS[name])
                                             for name in sorted(WEIGHTS)}
    assert contracts.agent_driver(search['agent_command']) == 'benchmarks/agents/codex_agent.py'


@pytest.mark.parametrize('block', [
    'router = "sometimes"',
    'router = "auto"\ninfo_floor = 2.0',
    'router = "auto"\n[search.features]\nweights = { T = 1.0 }',
    'router = "auto"\n[search.features]\nweights = { T = 1.0, S = 1.0, B = 1.0, R = 1.0, D = 1.0, M = 1.0, F = 1.0, C = 0.0 }',
])
def test_contracts_reject_invalid_router_settings(tmp_path, block):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'kernel.json').write_text('{}')
    head = ('name = "r"\nbackend = "cpu-synthetic"\nsource = "source"\n'
            'workload_id = "w"\nscope = "s"\nunit = "us"\nmetric = "m"\n'
            'ranks = 1\nprimary_rank = 0\nrepetitions = 1\n'
            'timeout_seconds = 10\nbudget_seconds = 10\nmin_gain_pct = 1\n'
            'allowed_files = ["kernel.json"]\nartifacts = ["kernel.json"]\n'
            '[commands]\nbuild = ["{python}", "-c", "pass"]\n'
            'correctness = ["{python}", "{foundation}/benchmarks/x.py"]\n'
            'benchmark = ["{python}", "{foundation}/benchmarks/x.py"]\n'
            '[search]\nprovider = "codex"\n'
            'max_rounds = 6\nmax_calls = 6\nagent_timeout_seconds = 10\n'
            'max_context_bytes = 8192\nexperience_dirs = []\n')
    recipe = tmp_path / 'recipe.toml'
    # The weights table belongs after the [search] keys, so rewrite the tail.
    recipe.write_text(head + block + '\n')
    with pytest.raises(ValueError):
        contracts.load_recipe(recipe)


def _search_module():
    try:
        from afd_rsi import search
    except ImportError as error:                    # pragma: no cover - non-POSIX hosts
        pytest.skip(f'the campaign controller needs a POSIX host: {error}')
    return search


def test_select_proposal_keeps_the_legacy_rule_without_a_router(tmp_path):
    search = _search_module()
    ledger = ledger_at(tmp_path)
    first = {'category': 'latency', 'priority': 3, 'patch': 'p1'}
    other = {'category': 'other', 'priority': 2, 'patch': 'p2'}
    history = [{'parent_hash': 'old', 'patch_hash': 'unused',
                'category': 'latency', 'decision': 'rejected'}] * 2
    assert search.select_proposal([first, other], history, 'new') == other
    seen = [{'parent_hash': 'same', 'patch_hash': search.identity('p1'),
             'category': 'latency', 'decision': 'rejected'}]
    assert search.select_proposal([first], seen, 'same') is None
    assert search.select_proposal([first], seen, 'different') == first
    legacy = router.Router({}, ledger, foundation_hash='f0')
    assert search.select_proposal([first, other], history, 'new', legacy) == other


def test_route_proposals_reports_the_decision_and_uses_evidence(tmp_path):
    search = _search_module()
    ledger = ledger_at(tmp_path)
    for index in range(12):
        ledger.add(category='fusion', features={'F': index % 3 + 1},
                   score=0.85, decision='promoted', foundation_hash='f0')
        ledger.add(category='tiling', features={'F': index % 3 + 1},
                   score=0.2, decision='rejected', foundation_hash='f0')
    instance = router.Router({'router': 'auto'}, ledger, foundation_hash='f0')
    ranked = {'category': 'tiling', 'priority': 3, 'patch': '+DataCopy(src, dst);\n'}
    better = {'category': 'fusion', 'priority': 1, 'patch': '+DataCopy(src, dst);\n'}
    selected, decision = search.route_proposals([ranked, better], [], 'new', instance)
    assert selected is better
    assert decision['scores']['fusion']['expected'] > decision['scores']['tiling']['expected']
    duplicate = [{'parent_hash': 'new', 'patch_hash': search.identity(better['patch'].strip()),
                  'category': 'fusion', 'decision': 'rejected'}]
    assert search.route_proposals([better], duplicate, 'new', instance) == (None, None)


def test_fingerprint_only_tracks_routing_when_a_recipe_declares_it():
    search = _search_module()
    plan = {'source_snapshot': {'hash': 'a'}, 'foundation_hash': 'b', 'controller_hash': 'c',
            'backend': 'cpu-synthetic', 'workload_id': 'w', 'scope': 's', 'unit': 'u',
            'metric': 'm', 'ranks': 1, 'primary_rank': 0, 'repetitions': 1,
            'min_gain_pct': 1.0, 'commands': {}, 'artifacts': [], 'allowed_files': [],
            'search': {}}
    legacy = search.fingerprint(plan)
    # A campaign frozen before routing existed keeps its historical digest, so
    # experience recorded then stays loadable.
    assert legacy == search.fingerprint({**plan, 'search': {}})
    declared = search.fingerprint({**plan, 'search': {'router': 'auto',
                                                      'features': {'weights': dict(WEIGHTS)},
                                                      'info_floor': 0.0, 'margin': 0.02,
                                                      'explore_max_fraction': 0.25}})
    assert declared != legacy
    changed = search.fingerprint({**plan, 'search': {'router': 'auto',
                                                     'features': {'weights': {**WEIGHTS, 'C': 0.5}},
                                                     'info_floor': 0.0, 'margin': 0.02,
                                                     'explore_max_fraction': 0.25}})
    assert changed != declared, 'different weights must not reuse the same experience'


def test_ledger_interrupted_replacement_preserves_complete_prefix(tmp_path, monkeypatch):
    from afd_rsi import evidence
    ledger = ledger_at(tmp_path)
    ledger.add(category='first', features={}, score=.5, decision='rejected', foundation_hash='f0')
    before = ledger.path.read_bytes()
    original = evidence.os.replace
    def interrupt(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(evidence.os, 'replace', interrupt)
    with pytest.raises(KeyboardInterrupt):
        ledger.add(category='next', features={}, score=.6, decision='rejected', foundation_hash='f0')
    assert ledger.path.read_bytes() == before
    assert len(ledger_at(tmp_path).load().rows) == 1
    monkeypatch.setattr(evidence.os, 'replace', original)
    ledger.add(category='next', features={}, score=.6, decision='rejected', foundation_hash='f0')
    assert len(ledger_at(tmp_path).load().rows) == 2
