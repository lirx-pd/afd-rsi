---
name: ascendc-code-review
description: Review A3 MegaMoe changes for correctness and integration risks.
---

# ascendc-code-review

Scope: local A3 MegaMoe development. Read these repository-owned resources for
this task; no global skill or online lookup is needed.

- [review-debug](../../knowledge/troubleshooting/review-debug.md)
- [memory-synchronization](../../knowledge/api/memory-synchronization.md)
- [compact-v2](../../knowledge/afd/compact-v2.md)

Perform a local sequential review: read scope and connected code, enumerate applicable checks, inspect each, verify line locations, and report severity/trigger/evidence/fix. Explicitly list device checks not executed. No plugin, external tool or subagent is required. Do not modify frozen acceptance gates.

Use the campaign snapshot and actual target facts. Missing evidence remains
unknown. Foundation and evaluator updates are maintenance work, outside the
candidate edit scope.
