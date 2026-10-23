---
name: ascendc-performance-best-practices
description: Implement measured A3 GMM, quantization, memory and pipeline improvements.
---

# ascendc-performance-best-practices

Scope: local A3 MegaMoe development. Read these repository-owned resources for
this task; no global skill or online lookup is needed.

- [tiling-pipeline](../../knowledge/performance/tiling-pipeline.md)
- [w4a8](../../knowledge/gmm-quantization/w4a8.md)
- [memory-synchronization](../../knowledge/api/memory-synchronization.md)

Choose an optimization from the diagnosed phase and target architecture. Account for additional bytes, memory, synchronization and tails. Change one primary mechanism with its required callers. Compare whole-work performance, not only the optimized stage. Preserve rounding and all protocol invariants.

Use the campaign snapshot and actual target facts. Missing evidence remains
unknown. Foundation and evaluator updates are maintenance work, outside the
candidate edit scope.

Detailed local technical material (read for the corresponding implementation):

- [arch22-pipeline](../../knowledge/performance/arch22-pipeline.md)
- [arch22-reuse-split](../../knowledge/performance/arch22-reuse-split.md)
