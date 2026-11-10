from pathlib import Path

import pytest

from afd_rsi import campaign
from afd_rsi.evidence import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    run = campaign.prepare(ROOT / "recipes/cpu_demo.toml", tmp_path / "run", ROOT)
    calls = []
    execute = campaign.execute

    def track(argv, cwd, output, timeout, env):
        calls.append((env["AFD_RSI_SIDE"], argv[-1]))
        return execute(argv, cwd, output, timeout, env)

    monkeypatch.setattr(campaign, "execute", track)
    return run, calls


def correctness(attempt, side="baseline"):
    return next(step["result"] for step in read_json(attempt / "progress.json")["steps"]
                if (step["side"], step["phase"]) == (side, "correctness"))


def test_baseline_reused_across_screen_confirmation_and_resume(setup):
    run, calls = setup
    original = campaign.run_baseline(run, ROOT)
    path = run / "workspaces/candidate/kernel.json"
    write_json(path, {**read_json(path), "synthetic_us": 8})
    screen = campaign.run_comparison(run, ROOT, level="operator")
    confirmation = campaign.run_comparison(run, ROOT)
    for attempt in (screen, confirmation):
        result = correctness(attempt)
        assert result["reused_from"]["trial"] == correctness(original)["trial"]
        assert not (run / result["trial"] / "process.json").exists()
        campaign.audit_attempt(run, attempt, ROOT)
    assert calls.count(("baseline", "correctness")) == 1
    assert calls.count(("candidate", "correctness")) == 2
    assert calls.count(("baseline", "benchmark")) == 12
    # Crash after recording the reused observation but before acknowledging it.
    state = read_json(confirmation / "progress.json")
    step = next(s for s in state["steps"] if (s["side"], s["phase"]) == ("baseline", "correctness"))
    del step["result"]
    state["status"] = "running"
    write_json(confirmation / "progress.json", state)
    before = list(calls)
    campaign._continue(run, confirmation, ROOT)
    assert calls == before
    campaign.audit_attempt(run, confirmation, ROOT)


def test_promoted_candidate_reuses_its_own_correctness(setup):
    run, calls = setup
    source = run / "workspaces/candidate"
    write_json(source / "kernel.json", {**read_json(source / "kernel.json"), "synthetic_us": 8})
    previous = campaign.run_comparison(run, ROOT)
    attempt = campaign.run_comparison(run, ROOT, baseline_source=source)
    assert correctness(attempt)["reused_from"]["trial"] == correctness(previous, "candidate")["trial"]
    assert calls.count(("baseline", "correctness")) == 1
    assert calls.count(("candidate", "correctness")) == 2


def test_native_reference_outputs_follow_reused_correctness(setup, monkeypatch):
    run, calls = setup
    load, artifacts, execute = campaign.load, campaign.artifacts, campaign.execute
    monkeypatch.setattr(campaign, "load", lambda *a: {**load(*a), "native": {"attn_ranks": 1}})
    monkeypatch.setattr(campaign, "artifacts", lambda *a, **kw: artifacts(*a))

    def native_fixture(argv, cwd, output, timeout, env):
        attempt = Path(env["AFD_RSI_ATTEMPT"])
        reference = attempt / "reference.rank0.pt"
        if (env["AFD_RSI_SIDE"], argv[-1]) == ("baseline", "correctness"):
            reference.write_bytes(b"sealed native reference fixture")
            write_json(attempt / "reference.lock.json", {str(reference): campaign.digest(reference)})
        elif argv[-1] in ("correctness", "benchmark"):
            assert reference.read_bytes() == b"sealed native reference fixture"
            assert str(reference) in read_json(attempt / "reference.lock.json")
        return execute(argv, cwd, output, timeout, env)

    monkeypatch.setattr(campaign, "execute", native_fixture)
    original = campaign.run_baseline(run, ROOT)
    source = run / "workspaces/candidate"
    write_json(source / "kernel.json", {**read_json(source / "kernel.json"), "synthetic_us": 8})
    comparison = campaign.run_comparison(run, ROOT)
    promoted = campaign.run_comparison(run, ROOT, baseline_source=source)
    assert correctness(comparison)["reference_files"] == correctness(original)["reference_files"]
    assert correctness(promoted)["reused_from"]["trial"] == correctness(comparison, "candidate")["trial"]
    assert calls.count(("baseline", "correctness")) == 1
    campaign.audit_attempt(run, promoted, ROOT)
    (promoted / "reference.rank0.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="reference outputs changed"):
        campaign.audit_attempt(run, promoted, ROOT)


@pytest.mark.parametrize("mismatch", ["environment", "host", "source", "artifact", "failed", "incomplete"])
def test_mismatch_does_not_reuse(setup, monkeypatch, mismatch):
    run, calls = setup
    previous = campaign.run_baseline(run, ROOT)
    baseline_source = None
    if mismatch == "environment":
        monkeypatch.setenv("ASCEND_LAUNCH_BLOCKING", "1")
    elif mismatch == "host":
        monkeypatch.setattr(campaign, "host_identity", lambda: {"host": "different", "boot": "new"})
    elif mismatch == "source":
        baseline_source = run / "workspaces/candidate"
        path = baseline_source / "kernel.json"
        write_json(path, {**read_json(path), "synthetic_us": 9})
    elif mismatch == "artifact":
        original = campaign.artifacts
        monkeypatch.setattr(campaign, "artifacts", lambda *a, **kw: {**original(*a, **kw), "other": "new"})
    else:
        state = read_json(previous / "progress.json")
        if mismatch == "incomplete":
            state["status"] = "running"
        else:
            next(s for s in state["steps"] if s["phase"] == "correctness")["result"]["checks_passed"] = False
        write_json(previous / "progress.json", state)
    # Direct construction isolates cache selection from resume's unfinished-attempt guard.
    attempt = campaign._new_attempt(run, ROOT, "baseline", baseline_source=baseline_source)
    campaign._continue(run, attempt, ROOT)
    assert "reused_from" not in correctness(attempt)
    assert calls.count(("baseline", "correctness")) == 2


@pytest.mark.parametrize("file", ["result.json", "process.json", "observation.json"])
def test_reused_evidence_is_rechecked_during_audit(setup, file):
    run, _ = setup
    original = campaign.run_baseline(run, ROOT)
    comparison = campaign.run_comparison(run, ROOT)
    (run / correctness(original)["trial"] / file).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        campaign.audit_attempt(run, comparison, ROOT)


def test_corrupt_cache_cannot_become_pass(setup):
    run, calls = setup
    original = campaign.run_baseline(run, ROOT)
    (run / correctness(original)["trial"] / "result.json").write_text("{}")
    comparison = campaign.run_comparison(run, ROOT)
    result = correctness(comparison)
    assert result["status"] == "invalid"
    assert "raw result changed" in result["reason"]
    assert calls.count(("baseline", "correctness")) == 1
