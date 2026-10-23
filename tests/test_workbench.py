import copy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from afd_rsi import campaign, skills
from afd_rsi.contracts import load_recipe, validate_result
from afd_rsi.evidence import read_json, write_json
from afd_rsi.runner import execute, group_alive
from afd_rsi.workspace import inventory


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run(tmp_path):
    return campaign.prepare(ROOT / "recipes/cpu_demo.toml", tmp_path / "run", ROOT)


def candidate(run, **changes):
    path = run / "workspaces/candidate/kernel.json"
    spec = read_json(path)
    write_json(path, {**spec, **changes})


def verdict(run):
    attempt = campaign.run_comparison(run, ROOT)
    return read_json(attempt / "verdict.json"), attempt


@pytest.mark.parametrize("nested", [False, True])
def test_comparison_exports_reproducible_patch_without_editing_original(run, tmp_path, nested):
    if nested:
        subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    original = inventory(ROOT / "examples/source")
    campaign.apply_patch(run, ROOT / "examples/faster.patch", ROOT)
    result, attempt = verdict(run)
    assert result["verdict"] == "PASS"
    assert result["gain_pct"] == pytest.approx(20)
    assert result["claim"] == "synthetic controller fixture only"
    observations = read_json(attempt / "observations.json")
    assert [x["side"] for x in observations if x["phase"] == "benchmark"] == [
        "baseline", "candidate", "candidate", "baseline",
        "baseline", "candidate", "candidate", "baseline"]
    destination = campaign.export(run, tmp_path / "export", ROOT)
    replay = tmp_path / "replay"
    replay.mkdir()
    shutil.copy2(run / "workspaces/baseline/kernel.json", replay / "kernel.json")
    subprocess.run(["git", f"--git-dir={replay / '.git'}", f"--work-tree={replay}",
                    "apply", "-p2", str(destination / "change.patch")], cwd=replay, check=True)
    assert inventory(replay) == inventory(run / "workspaces/candidate")
    assert inventory(ROOT / "examples/source") == original


@pytest.mark.parametrize("changes", [{"synthetic_us": 12}, {"synthetic_us": 8, "bias": 2}])
def test_slow_or_wrong_candidate_is_rejected(run, changes):
    candidate(run, **changes)
    result, _ = verdict(run)
    assert result["verdict"] == "FAIL"


def test_raw_result_tampering_blocks_report(run):
    candidate(run, synthetic_us=8)
    _, attempt = verdict(run)
    observation = read_json(attempt / "observations.json")[0]
    (run / observation["trial"] / "result.json").write_text("{}")
    with pytest.raises(ValueError, match="raw result changed"):
        campaign.report(run, ROOT)


def test_unallowed_patch_has_no_side_effect(run, tmp_path):
    patch = tmp_path / "bad.patch"
    patch.write_text("diff --git a/other b/other\nnew file mode 100644\n--- /dev/null\n+++ b/other\n@@ -0,0 +1 @@\n+bad\n")
    before = inventory(run / "workspaces/candidate")
    with pytest.raises(ValueError, match="outside allowed_files"):
        campaign.apply_patch(run, patch, ROOT)
    assert inventory(run / "workspaces/candidate") == before


def test_frozen_contract_and_knowledge_cannot_change(run):
    plan = run / "plan.lock.json"
    data = read_json(plan)
    data["min_gain_pct"] = 0
    write_json(plan, data)
    with pytest.raises(ValueError, match="frozen plan changed"):
        campaign.load(run, ROOT)


def test_expired_budget_does_not_launch_commands(run):
    # Exercise the command boundary directly with an expired in-memory deadline.
    plan = campaign.load(run, ROOT)
    plan["expires_at"] = 0
    attempt = run / "attempts/expired"
    result = campaign.command(plan, run, attempt, "baseline", "build", plan["source_snapshot"],
                              attempt / "build", ROOT)
    assert result["status"] == "invalid"
    assert result["reason"] == "campaign budget exhausted"
    assert not list(attempt.rglob("process.json"))


def test_missing_rank_data_is_inconclusive(run):
    plan = campaign.load(run, ROOT)
    # Frozen driver must fail closed if it returns incomplete rank evidence.
    from afd_rsi.evaluate import evaluate
    result = evaluate(plan, [{"status": "invalid", "reason": "missing rank results"}])
    assert result["verdict"] == "INCONCLUSIVE"


@pytest.fixture
def protocol():
    plan = load_recipe(ROOT / "recipes/cpu_demo.toml")
    artifacts = {"kernel.json": "f" * 64}
    result = {key: plan[key] for key in ("backend", "workload_id", "scope", "unit", "metric")}
    result["ranks"] = [{"rank": i, "source_hash": "source", "artifacts": artifacts,
                        "checks": {"numeric": True}, "value": 2} for i in range(2)]
    return plan, artifacts, result


@pytest.mark.parametrize("bad", ["missing", "duplicate", "bool_rank", "artifact", "source",
                                  "nan", "inf", "zero", "bool_metric", "string_check", "workload"])
def test_invalid_result_never_becomes_valid(protocol, bad):
    plan, artifacts, data = copy.deepcopy(protocol)
    if bad == "missing":
        data["ranks"].pop()
    elif bad == "duplicate":
        data["ranks"][1]["rank"] = 0
    elif bad == "bool_rank":
        data["ranks"][1]["rank"] = True
    elif bad == "artifact":
        data["ranks"][1]["artifacts"] = {}
    elif bad == "source":
        data["ranks"][1]["source_hash"] = "wrong"
    elif bad == "string_check":
        data["ranks"][0]["checks"] = {"numeric": "true"}
    elif bad == "workload":
        data["workload_id"] = "other"
    else:
        data["ranks"][0]["value"] = {"nan": float("nan"), "inf": float("inf"),
                                     "zero": 0, "bool_metric": True}[bad]
    with pytest.raises(ValueError):
        validate_result(data, plan, "source", artifacts, measured=True)


def test_context_is_self_contained_and_detects_resource_tamper(run):
    foundation = run / "foundation"
    text = skills.context(foundation, "afd-megamoe")
    assert "cam_async_routed_only_compact_v2" in text
    (foundation / "knowledge/afd/compact-v2.md").write_text("modified")
    with pytest.raises(ValueError, match="differs from manifest"):
        skills.context(foundation, "afd-megamoe")


def test_source_symlink_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "external").symlink_to(ROOT / "README.md")
    with pytest.raises(ValueError, match="symlink"):
        inventory(source)


def test_timeout_terminates_parent_and_child(tmp_path):
    code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
    result = execute([sys.executable, "-c", code], tmp_path, tmp_path / "trial", 0.4, {})
    assert result["status"] == "timeout"
    assert result["cleanup_complete"]
    assert not group_alive(result["pid"])


def test_successful_parent_leaving_child_is_not_success(tmp_path):
    code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])"
    result = execute([sys.executable, "-c", code], tmp_path, tmp_path / "trial", 5, {})
    assert result["status"] == "orphaned_children"
    assert result["cleanup_complete"]


def test_candidate_changes_after_evaluation_cannot_be_exported(run, tmp_path):
    candidate(run, synthetic_us=8)
    verdict(run)
    candidate(run, synthetic_us=7)
    with pytest.raises(ValueError, match="changed after evaluation"):
        campaign.export(run, tmp_path / "export", ROOT)


@pytest.mark.parametrize("mutation", ["source", "artifact", "rank"])
def test_changed_source_artifact_or_missing_rank_blocks_comparison(run, monkeypatch, mutation):
    original_execute = campaign.execute

    def execute_with_changed_evidence(argv, cwd, output, timeout, env):
        process = original_execute(argv, cwd, output, timeout, env)
        if mutation == "source":
            (cwd / "kernel.json").write_text("changed during command")
        elif argv[-1] == "correctness":
            if mutation == "artifact":
                (Path(env["AFD_RSI_BUILD_DIR"]) / "kernel.json").write_text("changed artifact")
            else:
                path = Path(env["AFD_RSI_RESULT"])
                data = read_json(path)
                data["ranks"].pop()
                write_json(path, data)
        return process

    monkeypatch.setattr(campaign, "execute", execute_with_changed_evidence)
    result, _ = verdict(run)
    assert result["verdict"] == "INCONCLUSIVE"


def test_incomplete_cleanup_blocks_next_launch(run):
    write_json(run / "attempts/old/process.json", {"status": "cleanup_timeout", "cleanup_complete": False})
    with pytest.raises(ValueError, match="unfinished process receipt"):
        campaign.run_comparison(run, ROOT)


def test_campaign_output_cannot_be_inside_foundation(tmp_path):
    with pytest.raises(ValueError, match="outside foundation"):
        campaign.prepare(ROOT / "recipes/cpu_demo.toml", ROOT / "skills/unsafe-output", ROOT)
    assert not (ROOT / "skills/unsafe-output").exists()


def test_unfrozen_validator_is_rejected(tmp_path):
    source = (ROOT / "recipes/cpu_demo.toml").read_text()
    source = source.replace('source = "../examples/source"', f'source = "{ROOT / "examples/source"}"')
    source = source.replace('correctness = ["{python}", "{foundation}/benchmarks/fixtures/cpu_driver.py",',
                            'correctness = ["{python}", "{repo}/validator.py",')
    recipe = tmp_path / "recipe.toml"
    recipe.write_text(source)
    with pytest.raises(ValueError, match="frozen repository benchmark"):
        load_recipe(recipe)
