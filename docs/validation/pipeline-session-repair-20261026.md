# Pipeline diagnostic and persistent Codex session repair — 2026-10-26

## Original failure

Job: `runs/pipeline-20261025-191954`, task `afd-mrv2-async`.
The job ended at 2026-10-25 19:33 Asia/Shanghai with search status
`baseline_failed`, zero proposal calls, zero candidate rounds and no best.
The outer `finished` means the controller finished handling the campaign; it
does not mean baseline qualification or optimization succeeded.

Production build and native numerical/FULL checks passed. Both native timing
trials were valid. The isolated diagnostic build failed before any profile was
collected:

```text
afd_mega_moe_ffn.cpp:235:32: error: use of undeclared identifier 'validation'
        flags.trace.End(31, 0, validation);
```

The overlay expected the old inline `layerId` validation and conditional prepare
stage. The expert-packed target moved validation into `ValidateTransaction` and
changed preparation to `PrepareTile`. Unchecked string replacements missed the
begin markers while still inserting end markers. This is a framework diagnostic
adapter failure, not evidence of production MegaMoe numerical failure.

## Repair

- Support both known pipeline layouts with exact-count checked insertion anchors.
  Validate all anchors before writing either diagnostic source file.
- Archive compiler stdout/stderr and receipts without copying build sources or
  binaries. Build failures now identify their stderr path; progress includes the
  search outcome/reason so a baseline failure remains visible after completion.
- Persist one Codex conversation per campaign, retaining its exact thread ID.
  Resume it across Astra analysis, Sol coding and subsequent rounds. Keep a stable
  workspace, update the authoritative request, record each turn independently and
  exit the CLI before device experiments. Never use `--last`, ephemeral sessions
  or an implicit new conversation when resume fails.
- Propagate the session directory through unified optimize, direct search and the
  manual propose entry. Retain dispatch/deadline limits and interrupted-call rules.

Original campaign/controller/validators and the plugin worktree remain unchanged.
Use a new campaign to adopt the repair; resuming the old frozen job cannot install
this maintenance change.

## Validation evidence

Artifacts: `results/afd-rsi-repair-20261026/` in the parent workspace.
The original compiler stderr is preserved there as `original-compiler-stderr.log`.

Actual Codex CLI 0.164.0 validation completed three finite calls: Sol medium,
Astra xhigh, Sol xhigh. All emitted thread ID
`01a1759f-b885-7ac3-bea9-01cfdefdf7a5`; both resumed calls recalled the first turn's
random marker without being supplied that marker again. This tests provider/login,
CLI option placement, schema output and real cross-model conversation restoration.
It is not an optimization result.

The isolated NPU validation reuses the failed campaign's immutable production
artifacts and reference outputs, builds the repaired diagnostic source in a new
maintenance directory and runs the existing profile entry. An initial maintenance
script overwrote CANN's PYTHONPATH and failed importing msopgen; that failed attempt
is preserved. The second script prepends the controller path to CANN's environment.
Final results follow.

Local regression: **160 passed, 2 skipped**, 527.67 seconds. The skips are the
existing WSL localhost binding restriction and missing local httpx dependency.
Final overlay/session targeted tests: **8 passed**; native/session checks:
**16 passed**. Foundation closure: **8 skills, 64 files**. `git diff --check` passed.

Final NPU maintenance validation **passed (exit 0)** using the final uploaded
controller/foundation, verified equal to the current checkout. Three-device health,
independent diagnostic build, seven numerical workload cases with internal tracing,
production FULL profiler execution and strict profile reduction all completed.
FFN ranks 1 and 2 recorded 8 and 7 transactions respectively (a workload case can
split into multiple transactions), all with 72 per-core traces. Every packed
metadata-validation span (code 31) belongs to Vector core 0 and surrounds the
actual validation call. Pipeline and production processes completed cleanup.

The evidence archive contains 88 files, 368610 compressed bytes, with SHA256
`9c43258bd2105f63f0a21ed4e7687222c825ba932144896ddab2f788fba99ff3`.
The downloaded archive and every manifest entry were verified. Original and
maintenance failed attempts are preserved; no task was stopped or deleted.
This validates the diagnostic adapter on the failed run's frozen synthetic
workload, not an optimization gain, model accuracy or the full service matrix.
A new optimization campaign was not started.
