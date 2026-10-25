---
name: ascendc-perf-optimize
description: Analyze MegaMoe performance with tiling and layered pipeline evidence.
---

# ascendc-perf-optimize

Scope: local A3 MegaMoe development. Read these repository-owned resources for
this task; no global skill or online lookup is needed.

- [expert-packed tiles and service limits](../../knowledge/performance/megamoe-expert-packing-20261025.md)
- [tiling-pipeline](../../knowledge/performance/tiling-pipeline.md)
- [reviewed A3 experiments](../../knowledge/performance/megamoe-a3-experiments.md)
- [campaign profiling](../../knowledge/measurement/profiling.md)
- [contract](../../knowledge/measurement/contract.md)

Apply the four stages in order: tiling, inter-device, inter-core, intra-core. Mark inapplicable stages and missing inputs. Do not prescribe chunk balancing without actual tiling and valid isolated controls. Output measured bottleneck, alternatives, falsifiable hypothesis and experiment.

Use the campaign snapshot and actual target facts. Missing evidence remains
unknown. Foundation and evaluator updates are maintenance work, outside the
candidate edit scope.
