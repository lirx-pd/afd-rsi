"""Recipe contracts that must hold on every host, without the Linux controller.

The benefit campaigns decide whether MegaMoe actually earns its keep, so the
measurement configuration is a contract rather than a per-run choice.
"""

from pathlib import Path

import pytest

from afd_rsi import contracts

ROOT = Path(__file__).resolve().parents[1]


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
