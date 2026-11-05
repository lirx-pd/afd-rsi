import copy
import json
from pathlib import Path

import pytest

from afd_rsi.native import fixture_result, profile_evidence, timing_summary, write_report


IDENTITY = {"files": {"binding.so": "fixed"}, "kernel_artifacts": {"kernel.o": "fixed"}}
CASES = [(2, False, 7), (5, True, 7), (2, False, 7), (5, False, 16),
         (2, True, 16), (5, True, 129), (2, False, 129)]


def result(mega=False, value=100):
    reports = []
    for rank in range(3):
        reports.append({"rank": rank, "mega_moe": mega, "identity": copy.deepcopy(IDENTITY),
                        "cleanup_complete": True, "layered_reference_close" if mega else "eager_exact_match": True,
                        "timing": [{"case": i, "layer": layer, "sparse": sparse, "tokens": tokens,
                                    "transaction_us": value}
                                   for i, (layer, sparse, tokens) in enumerate(CASES)] if rank == 0 else []})
    return {"result": "PASS", "reports": reports}


def test_native_requires_complete_loaded_identity_and_measurements(tmp_path):
    log = tmp_path / "stdout.log"
    data = result()
    log.write_text('not-json\n' + json.dumps(data) + '\n')
    assert fixture_result(log, 3, IDENTITY, False, 1, 1) == data
    for mutation in ("rank", "identity", "nonfinite", "missing", "cleanup", "case", "numerical"):
        broken = copy.deepcopy(data)
        if mutation == "rank":
            broken["reports"].pop()
        elif mutation == "identity":
            broken["reports"][1]["identity"]["files"]["binding.so"] = "foreign"
        elif mutation == "nonfinite":
            broken["reports"][0]["timing"][0]["transaction_us"] = float("nan")
        elif mutation == "missing":
            broken["reports"][0]["timing"].pop()
        elif mutation == "cleanup":
            broken["reports"][2]["cleanup_complete"] = False
        elif mutation == "case":
            broken["reports"][0]["timing"][0]["tokens"] = 8
        else:
            broken["reports"][0]["eager_exact_match"] = False
        log.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            fixture_result(log, 3, IDENTITY, False, 1, 1)


def test_native_result_checks_frozen_custom_lengths(tmp_path):
    data = result(mega=True)
    data['reports'][0]['timing'] = [
        {'case': index, 'layer': layer, 'sparse': sparse, 'tokens': tokens,
         'transaction_us': 100}
        for index, (layer, sparse, tokens) in enumerate(
            [(2, False, 128), (5, True, 128), (2, False, 512), (5, True, 512)])]
    log = tmp_path / 'stdout.log'
    log.write_text(json.dumps(data))
    assert fixture_result(log, 3, IDENTITY, True, 1, 1, lengths='128,512') == data
    data['reports'][0]['timing'][-1]['tokens'] = 511
    log.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='workload mismatch'):
        fixture_result(log, 3, IDENTITY, True, 1, 1, lengths='128,512')


def test_paired_summary_keeps_slow_samples_and_regressions():
    observations = [
        {"phase": "pair-0-layered", "result": result(value=100)},
        {"phase": "pair-0-mega", "result": result(True, value=80)},
        {"phase": "pair-1-mega", "result": result(True, value=120)},
        {"phase": "pair-1-layered", "result": result(value=100)},
    ]
    summary = timing_summary(observations, 2, 1)
    assert len(summary) == 7
    assert summary[0]["gain_pct_by_trial"] == [20, -20]
    assert summary[0]["candidate_us_by_trial"] == [80, 120]
    assert summary[0]["median_gain_pct"] == 0


def test_megamoe_baseline_and_candidate_have_distinct_trial_labels():
    trials = [{"phase": "pair-0-baseline", "result": result(True, value=100)},
              {"phase": "pair-0-candidate", "result": result(True, value=120)}]
    summary = timing_summary(trials, 1, 1, ("baseline", "candidate"))
    assert summary[0]["baseline_us_by_trial"] == [100]
    assert summary[0]["candidate_us_by_trial"] == [120]
    assert summary[0]["median_gain_pct"] == -20


def test_native_report_does_not_turn_consistency_into_performance_pass(tmp_path):
    trials = [{"phase": "pair-0-layered", "result": result(value=100)},
              {"phase": "pair-0-mega", "result": result(True, value=120)}]
    write_report(tmp_path, {"claim": "synthetic eager transaction", "correctness": "PASS",
                           "timing": timing_summary(trials, 1, 1), "profiler": []})
    report = (tmp_path / "report.md").read_text()
    assert "-20.00%" in report
    assert "No automatic promotion" in report
    assert "[observations](observations.json)" in report


def test_stage_shares_exclude_the_dispatch_wait(tmp_path):
    """The FFN idles until Attention dispatches; that wait must not hide the work."""
    from afd_rsi.native import named_stage_cycles
    totals = {'47': 8700, '40': 1200, '41': 50, '43': 60, '42': 30, '45': 40, '44': 20}
    rows = {row['stage']: row for row in named_stage_cycles(totals)}
    assert rows['sync_in']['share_pct'] == pytest.approx(100 * 8700 / 10100, abs=0.1)
    assert rows['sync_in']['active_share_pct'] is None
    # Active shares answer where the FFN's own cycles go.
    assert rows['round_prepare']['active_share_pct'] == pytest.approx(100 * 1200 / 1400, abs=0.1)
    assert rows['activation']['active_share_pct'] == pytest.approx(100 * 60 / 1400, abs=0.1)
    assert sum(row['share_pct'] for row in rows.values()) == pytest.approx(100.0, abs=0.2)
    assert named_stage_cycles({}) == []


def test_pull_substages_keep_nested_waits_out_of_parent_totals():
    from afd_rsi.native import named_stage_cycles, named_substage_cycles
    totals = {'40': 100, '45': 50, '47': 300, '60': 80, '61': 30,
              '62': 10, '65': 5, '66': 5, '70': 20, '71': 10, '72': 5, '73': 5}
    top = {row['stage']: row for row in named_stage_cycles(totals)}
    assert top['round_prepare']['core_cycles'] == 100
    assert top['round_prepare']['active_share_pct'] == pytest.approx(66.67, abs=0.01)
    detail = named_substage_cycles(totals)
    stages = {row['stage']: row for row in detail['stages']}
    assert stages['rx_select']['parent_share_pct'] == 80
    assert stages['rx_header_validation']['parent'] == 'rx_select'
    assert detail['residual_cycles']['rx_select'] == 50
    assert detail['residual_cycles']['round_prepare'] == 0
    assert detail['residual_cycles']['combine'] == 10
    with pytest.raises(ValueError, match='exceeds rx_select'):
        named_substage_cycles({**totals, '61': 81})


def test_profiler_requires_device_kernel_evidence_on_both_ffn_ranks(tmp_path):
    for rank in (1, 2):
        path = tmp_path / f"rank-{rank}" / "kernel_details.csv"
        path.parent.mkdir()
        path.write_text('Name,Duration(us)\nAfdMegaMoePullNative_kernel,12\n')
    assert len(profile_evidence(tmp_path, range(1, 3))) == 2
    (tmp_path / "rank-2/kernel_details.csv").write_text('Name\nCPU_launch_AfdOther\n')
    with pytest.raises(ValueError, match="FFN rank 2"):
        profile_evidence(tmp_path, range(1, 3))


def test_profiler_does_not_count_attention_comm_as_ffn_evidence(tmp_path):
    """The comm kernel also runs on Attention; that must not satisfy a FFN rank."""
    attention = tmp_path / "rank-0/kernel_details.csv"
    attention.parent.mkdir()
    attention.write_text('Name,Duration(us)\nAfdMegaMoeComm,9\n')
    ffn = tmp_path / "rank-1/kernel_details.csv"
    ffn.parent.mkdir()
    ffn.write_text('Name,Duration(us)\nAfdMegaMoePullNative,12\n')
    with pytest.raises(ValueError, match="FFN rank 2"):
        profile_evidence(tmp_path, range(1, 3))
    assert len(profile_evidence(tmp_path, range(1, 2))) == 1


def test_build_snapshot_sets_version_and_preserves_input(monkeypatch, tmp_path):
    from afd_rsi import native
    source = tmp_path / "input"
    source.mkdir()
    (source / "setup.py").write_text("# input\n")
    output = tmp_path / "build"
    observed = {}

    def execute(argv, cwd, logs, timeout, env):
        observed.update(env=env, cwd=cwd, argv=argv)
        return {"status": "completed", "cleanup_complete": True}

    monkeypatch.setattr(native, "execute", execute)
    monkeypatch.setattr(native, "artifact_identity", lambda repo: IDENTITY)
    target = native.build(source, output, 10)
    assert observed["cwd"] == target and target != source
    assert observed["argv"][2:4] == ["egg_info", "build_ext"]
    assert observed["env"]["AFD_SKIP_ACLNN_BUILD"] == "0"
    assert observed["env"]["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_AFD_PLUGIN"].startswith("0.0.0+rsi.")
    assert (source / "setup.py").read_text() == "# input\n"
    assert Path(output / "artifacts.json").is_file()


def test_native_mapping_distinguishes_afd_from_other_vendors(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks/native"))
    from identity import mapped_api_paths
    root = tmp_path / "measured"
    api = root / "vendors/afd-plugin/op_api/lib/libcust_opapi.so"
    fla = tmp_path / "vendors/fla/op_api/lib/libcust_opapi.so"
    foreign = tmp_path / "other/vendors/afd-plugin/op_api/lib/libcust_opapi.so"

    def maps(*paths):
        return '\n'.join(f'0000-1111 r--p 000000 00:00 0 {p}' for p in paths)

    assert mapped_api_paths(root, maps(api, fla)) == ({api}, {fla})
    with pytest.raises(RuntimeError, match="foreign"):
        mapped_api_paths(root, maps(api, foreign, fla))
    with pytest.raises(RuntimeError, match="did not map"):
        mapped_api_paths(root, maps(fla))


@pytest.mark.parametrize('execution', ['eager', 'graph'])
def test_native_compare_routes_distinct_baseline_and_candidate(monkeypatch, tmp_path, execution):
    from types import SimpleNamespace
    from afd_rsi import native
    root = tmp_path / 'framework'
    for directory in ('skills','knowledge','benchmarks','src'):
        (root/directory).mkdir(parents=True)
    (root/'skills/manifest.json').write_text('{}')
    (root/'src/controller.py').write_text('# frozen\n')
    baseline, candidate = tmp_path/'baseline', tmp_path/'candidate'
    for repo in (baseline, candidate):
        repo.mkdir()
        (repo/'source.txt').write_text(repo.name)
    calls = []

    def identity(repo):
        return {'files':{'binding.so':repo.name},'kernel_artifacts':{'kernel.o':repo.name}}

    def execute(argv, cwd, output, timeout, env):
        calls.append((cwd, '--mega-moe' in argv, env['AFD_RSI_TARGET'], '--full-graph' in argv))
        output.mkdir(parents=True)
        data = result('--mega-moe' in argv, 100 if cwd == baseline else 120)
        for report in data['reports']:
            report['identity'] = identity(cwd)
            report['execution'] = 'graph' if '--full-graph' in argv else 'eager'
            if report['rank']:
                report['graph'] = {'capture_count': 2, 'replay_count': 8,
                                   'python_transaction_disabled': True, 'stable_exposed_addresses': 4}
            if '--measure' not in argv:
                report['timing'] = []
        (output/'stdout.log').write_text(json.dumps(data))
        return {'status':'completed','cleanup_complete':True,'exit_code':0}

    monkeypatch.setattr(native.skills, 'validate', lambda _: None)
    monkeypatch.setattr(native, 'artifact_identity', identity)
    monkeypatch.setattr(native, 'execute', execute)
    output = tmp_path/'comparison'
    args = SimpleNamespace(baseline_repo=baseline, attn_ranks=1, pairs=1, profile=False,
                           per_group=False, hidden=256, experts=4, intermediate=512,
                           timeout=30, repeats=1, warmups=0, execution=execution)
    native.compare(candidate, output, '0,1,2', root, args)
    assert [(repo, mega) for repo, mega, _, _ in calls] == [
        (baseline,False),(baseline,True),(candidate,True),(baseline,True),(candidate,True)]
    assert all(str(repo) == target for repo, _, target, _ in calls)
    assert [graph for _, _, _, graph in calls] == [False] + [execution == 'graph'] * 4
    summary = json.loads((output/'summary.json').read_text())
    assert summary['baseline_mode'] == 'MegaMoe'
    assert summary['timing'][0]['median_gain_pct'] == -20
    plan = json.loads((output/'plan.lock.json').read_text())
    assert plan['sides']['baseline']['source']['hash'] != plan['sides']['candidate']['source']['hash']
    assert 'Baseline: MegaMoe' in (output/'report.md').read_text()


def test_full_graph_requires_capture_replay_and_no_eager_fallback(tmp_path):
    log = tmp_path / 'stdout.log'
    data = result(True)
    for report in data['reports']:
        report['execution'] = 'graph'
        if report['rank']:
            report['graph'] = {'capture_count': 2, 'replay_count': 8,
                               'python_transaction_disabled': True, 'stable_exposed_addresses': 4}
    log.write_text(json.dumps(data))
    fixture_result(log, 3, IDENTITY, True, 1, 1, full_graph=True)
    for field, value in [('capture_count', 0), ('replay_count', 0), ('python_transaction_disabled', False), ('stable_exposed_addresses', 0)]:
        broken = copy.deepcopy(data)
        broken['reports'][1]['graph'][field] = value
        log.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match='FULL'):
            fixture_result(log, 3, IDENTITY, True, 1, 1, full_graph=True)
    data['reports'][0]['execution'] = 'eager'
    log.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='FULL'):
        fixture_result(log, 3, IDENTITY, True, 1, 1, full_graph=True)


def test_graph_recipe_freezes_matching_native_and_service_modes(tmp_path):
    from afd_rsi.contracts import load_recipe
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / 'target'
    source.mkdir()
    recipe = tmp_path / 'graph.toml'
    text = (root / 'recipes/ascend_graph.toml').read_text().replace('../../afd-plugin', str(source))
    recipe.write_text(text)
    plan = load_recipe(recipe)
    assert plan['native']['execution'] == 'graph'
    recipe.write_text(text.replace('"--execution", "graph"', '"--execution", "eager"'))
    with pytest.raises(ValueError, match='must match'):
        load_recipe(recipe)
