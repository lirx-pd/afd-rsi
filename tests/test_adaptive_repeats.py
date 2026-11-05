"""The measurement decides whether it needs more repetitions."""
import ast
import json
import os
import statistics
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / 'benchmarks/native/campaign_driver.py'


def driver_namespace():
    """Load only the measurement helpers, without running a device stage."""
    source = DRIVER.read_text()
    tree = ast.parse(source)
    wanted = {'case_spread', 'densify'}
    blocks = [textwrap.dedent(ast.get_source_segment(source, node))
              for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert len(blocks) == len(wanted), 'helpers moved; update this probe'
    namespace = {'os': os, 'statistics': statistics, 'Path': Path}
    exec(compile('\n\n'.join(blocks), str(DRIVER), 'exec'), namespace)
    return namespace


def sample(case, values):
    return {'timing': [{'case': case, 'transaction_us': value} for value in values]}


def test_spread_grows_with_the_noisiest_case():
    namespace = driver_namespace()
    quiet = {'reports': [sample(0, [100, 101, 99]), sample(1, [200, 202, 198])]}
    noisy = {'reports': [sample(0, [100, 140, 90]), sample(1, [200, 201, 199])]}
    # (101-99)/100 and (202-198)/200 are both 2%; the worst case is reported.
    assert namespace['case_spread'](quiet) == pytest.approx(2.0, abs=0.1)
    assert namespace['case_spread'](noisy) > 20
    assert namespace['case_spread']({'reports': [sample(0, [100])]}) == 0


def test_densify_remeasures_only_when_spread_exceeds_the_limit(tmp_path, monkeypatch):
    namespace = driver_namespace()
    calls = []

    def launch(name, argv, *, mega, repeats):
        calls.append((name, repeats, list(argv)))
        spread = 30 if name == 'transaction' else 2
        return {'reports': [sample(0, [100, 100 + spread, 100])]}

    namespace['launch'] = launch
    namespace['output'] = tmp_path
    namespace['write_json'] = lambda path, value: Path(path).write_text(json.dumps(value))
    argv = ['python', 'fixture.py', '--measure', '--repeats', '4']
    settings = {'repeats': 4}
    monkeypatch.delenv('AFD_RSI_NO_ADAPTIVE_REPEATS', raising=False)
    monkeypatch.setenv('AFD_RSI_SPREAD_LIMIT', '5')
    monkeypatch.setenv('AFD_RSI_REPEAT_FACTOR', '3')

    namespace['densify']({'reports': [sample(0, [100, 130, 100])]}, argv, settings)
    assert calls == [('transaction-dense', 12, argv[:2] + ['12'])] or calls[0][:2] == ('transaction-dense', 12)
    assert calls[0][2][calls[0][2].index('--repeats') + 1] == '12'
    record = json.loads((tmp_path / 'adaptive-repeats.json').read_text())
    assert record['used'] == 'dense' and record['dense_repeats'] == 12
    assert record['worst_case_spread_pct'] > 5

    calls.clear()
    namespace['densify']({'reports': [sample(0, [100, 101, 100])]}, argv, settings)
    assert calls == [], 'a quiet first pass is used as measured'
    record = json.loads((tmp_path / 'adaptive-repeats.json').read_text())
    assert record['used'] == 'configured'


def test_spread_names_a_wrong_container_instead_of_a_key_error():
    """The fixture result shape is asserted here, so a key drift fails loudly."""
    namespace = driver_namespace()
    with pytest.raises(ValueError, match='carries no reports list'):
        namespace['case_spread']({'ranks': []})


def test_densify_rejects_unresolved_noise_without_picking_a_quiet_pass(tmp_path, monkeypatch):
    namespace = driver_namespace()
    namespace['output'] = tmp_path
    namespace['write_json'] = lambda path, value: Path(path).write_text(json.dumps(value))
    monkeypatch.delenv('AFD_RSI_NO_ADAPTIVE_REPEATS', raising=False)
    monkeypatch.setenv('AFD_RSI_SPREAD_LIMIT', '5')

    def launch(name, argv, *, mega, repeats):
        spread = 8 if name == 'transaction' else 90
        return {'reports': [sample(0, [100, 100 + spread, 100])]}

    namespace['launch'] = launch
    with pytest.raises(RuntimeError, match='timing spread'):
        namespace['densify']({'reports': [sample(0, [100, 108, 100])]},
                             ['f', '--repeats', '2'], {'repeats': 2})
    record = json.loads((tmp_path / 'adaptive-repeats.json').read_text())
    assert record['used'] == 'dense' and record['resolved'] is False
    assert record['dense_spread_pct'] > record['worst_case_spread_pct']


def test_densify_can_be_disabled_by_environment(tmp_path, monkeypatch):
    namespace = driver_namespace()
    namespace['launch'] = lambda *args, **kwargs: pytest.fail('adaptive repeats are disabled')
    namespace['output'] = tmp_path
    namespace['write_json'] = lambda path, value: Path(path).write_text('{}')
    monkeypatch.setenv('AFD_RSI_NO_ADAPTIVE_REPEATS', '1')
    monkeypatch.setenv('AFD_RSI_SPREAD_LIMIT', '0.1')
    with pytest.raises(RuntimeError, match='timing spread'):
        namespace['densify']({'reports': [sample(0, [100, 200, 100])]},
                             ['x', '--repeats', '4'], {'repeats': 4})


def test_service_primary_keeps_noisy_native_timing_as_diagnostic(tmp_path, monkeypatch):
    namespace = driver_namespace()
    namespace['launch'] = lambda *args, **kwargs: pytest.fail('diagnostic timing must not be rerun')
    namespace['output'] = tmp_path
    namespace['write_json'] = lambda path, value: Path(path).write_text(json.dumps(value))
    monkeypatch.setenv('AFD_RSI_SPREAD_LIMIT', '5')
    native = {'reports': [sample(0, [100, 150, 100])]}
    assert namespace['densify'](native, ['x', '--repeats', '2'], {'repeats': 2}, diagnostic_only=True) is native
    record = json.loads((tmp_path / 'adaptive-repeats.json').read_text())
    assert record['diagnostic_only'] and record['worst_case_spread_pct'] > 5
