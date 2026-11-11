"""Recipe contracts that must hold on every host, without the Linux controller.

The benefit campaigns decide whether MegaMoe actually earns its keep, so the
measurement configuration is a contract rather than a per-run choice.
"""

from pathlib import Path

import pytest

from afd_rsi import contracts

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['ascend_eager', 'ascend_graph', 'ascend_pipeline_scenarios',
                                  'cpu_adaptive_search'])
def test_adaptive_recipes_double_model_call_budget(name):
    plan = contracts.load_recipe(ROOT / 'recipes' / f'{name}.toml')
    assert plan['search']['max_model_calls'] == 72


@pytest.mark.parametrize('limit', [None, 11])
def test_model_call_budget_default_preserves_explicit_override(tmp_path, limit):
    text = (ROOT / 'recipes/cpu_adaptive_search.toml').read_text()
    text = text.replace('../examples/source', str(ROOT / 'examples/source'))
    text = text.replace('max_model_calls = 72', '' if limit is None else f'max_model_calls = {limit}')
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text(text)
    plan = contracts.load_recipe(recipe)
    assert plan['search']['max_model_calls'] == (72 if limit is None else limit)


def test_pipeline_recipe_covers_model_shape_cohort_boundaries(tmp_path):
    recipe = tmp_path / 'pipeline.toml'
    recipe.write_text((ROOT / 'recipes/ascend_pipeline_scenarios.toml').read_text().replace('../../afd-plugin', str(tmp_path)))
    plan = contracts.load_recipe(recipe)
    assert (plan['native']['hidden'], plan['native']['intermediate'], plan['native']['experts']) == (4096, 2048, 64)
    assert plan['native']['lengths'] == [128, 513, 1025, 2050, 4100, 8125]
    assert len(plan['service']['primary']) == 2
    assert plan['search']['agent_timeout_seconds'] == 3600


def test_dbo_detection_covers_both_spellings():
    assert contracts.dbo_enabled(['--dbo', 'on'])
    assert contracts.dbo_enabled(['--dbo=on'])
    assert contracts.dbo_enabled(['--dbo', 'ON']), 'a recipe may shout'
    assert not contracts.dbo_enabled(['--dbo', 'off'])
    assert not contracts.dbo_enabled(['--dbo=off'])
    assert not contracts.dbo_enabled(['--execution', 'eager'])
    assert not contracts.dbo_enabled(['--dbo']), 'a trailing flag names no value'


def test_service_args_cannot_enable_dbo(tmp_path):
    """DBO overlaps Attention with FFN, so it would flatten the measured gain."""
    source = (ROOT / 'recipes/ascend_eager.toml').read_text()
    source = source.replace('source = "../../afd-plugin"', f'source = "{tmp_path.as_posix()}"')
    recipe = tmp_path / 'recipe.toml'
    recipe.write_text(source.replace('"--dbo", "off"', '"--dbo", "on"'))
    with pytest.raises(ValueError, match='--dbo off'):
        contracts.load_recipe(recipe)


def test_shipped_benefit_recipes_keep_dbo_off():
    """Every campaign that decides whether MegaMoe earns its keep runs without DBO."""
    recipes = sorted((ROOT / 'recipes').glob('ascend_*.toml'))
    assert recipes, 'no benefit recipe found'
    for path in recipes:
        plan = contracts.load_recipe(path)
        if 'service' not in plan:
            continue
        args = plan['service']['args']
        assert not contracts.dbo_enabled(args), path.name
        # Explicitly off, not merely unset: a recipe has to say so.
        assert any(argument in ('--dbo=off', 'off') for argument in args), \
            f'{path.name} does not state --dbo off'
        assert not any('dbo' in argument.lower() and argument not in ('--dbo', 'off')
                       for argument in args), \
            f'{path.name} still passes DBO-only tuning arguments'


def test_service_rule_does_not_disturb_other_recipes():
    # The synthetic recipe has no service stage; it must keep loading unchanged.
    plan = contracts.load_recipe(ROOT / 'recipes/cpu_search.toml')
    assert 'service' not in plan


def test_ascend_search_recipes_freeze_real_model_primary_and_small_screen(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'benchmarks/service'))
    from afd_e2e.cli import parser
    for path in sorted((ROOT / 'recipes').glob('ascend_*.toml')):
        plan = contracts.load_recipe(path)
        primary = plan['service']['primary']
        assert plan['validation'] == 'operator-first'
        components = primary if isinstance(primary, list) else [primary]
        assert all(item['metric'] in ('ttft_p50_ms', 'common_window_tpot_ms_p50') for item in components)
        if path.stem == 'ascend_pipeline_scenarios':
            assert plan['service']['screen_scenarios'] == ['native-long-decode', 'mixed-prefill']
        else:
            assert components[0]['id'] == 'point-008'
        screen_args = [*plan['service']['args'], '--test', 'smoke,perf']
        if plan['service'].get('screen_scenarios'):
            screen_args += ['--perf-scenarios', ','.join(plan['service']['screen_scenarios'])]
        selected = parser().parse_args(['plan', *screen_args])
        assert selected.test == 'smoke,perf'
        if path.stem == 'ascend_pipeline_scenarios':
            assert selected.perf_scenarios == 'native-long-decode,mixed-prefill'
