---
name: npu-arch
description: Confirm A3 architecture and resources before designing or building kernels.
---

# npu-arch

Scope: local A3 MegaMoe development. Read these repository-owned resources for
this task; no global skill or online lookup is needed.

- [a3](../../knowledge/architecture/a3.md)

Collect exact SoC, SDK and visible core resources. Match architecture and execution route. If unknown or unsupported, return the missing facts rather than invent a compatible implementation. Output platform facts, constraints and validation gaps.

Use the campaign snapshot and actual target facts. Missing evidence remains
unknown. Foundation and evaluator updates are maintenance work, outside the
candidate edit scope.
