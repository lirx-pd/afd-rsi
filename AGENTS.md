# AFD-RSI development contract

Use the repository's `skills/` and `knowledge/` content. Do not resolve knowledge
through global skills, other checkouts, URLs, or downloads. Read the skill for
the current phase and its declared repository-local resources.

Keep execution local to Linux. The controller imports no accelerator libraries.
Never edit the target's original worktree, the CANN installation, frozen
validators, campaign gates, or the knowledge snapshot during candidate search.
Put build output in `AFD_RSI_BUILD_DIR`, outside the candidate source directory.

Preserve failed attempts and raw observations. A CPU synthetic fixture proves
controller contracts only. NPU correctness and performance need real device
evidence with the same workload, source, artifacts and effective configuration.

Run `python -m pytest` and `PYTHONPATH=src python -m afd_rsi check` after changes.
Foundation maintenance requires `PYTHONPATH=src python tools/freeze_foundation.py`
and inspection of the manifest diff. Do not regenerate a manifest to conceal an
unexpected change. There is no automatic knowledge promotion or framework edit.
