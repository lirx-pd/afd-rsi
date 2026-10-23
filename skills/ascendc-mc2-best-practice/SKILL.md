---
name: ascendc-mc2-best-practice
description: Design or change A3 CAM communication/computation fusion.
---

# ascendc-mc2-best-practice

Scope: local A3 MegaMoe development. Read these repository-owned resources for
this task; no global skill or online lookup is needed.

- [window-protocol](../../knowledge/mc2/window-protocol.md)
- [compact-v2](../../knowledge/afd/compact-v2.md)
- [memory-synchronization](../../knowledge/api/memory-synchronization.md)

Read both A/F sides. Map state writers/readers and buffer reuse. Separate communication protocol from compute implementation. Derive resource and visibility constraints, then propose a same-protocol candidate. Protocol changes require a new campaign. Output dataflow, synchronization obligations and paired validation cases.

Use the campaign snapshot and actual target facts. Missing evidence remains
unknown. Foundation and evaluator updates are maintenance work, outside the
candidate edit scope.

Detailed local technical material (read for the corresponding implementation):

- [mte-address-access](../../knowledge/api/mte-address-access.md)
- [sync-and-visibility](../../knowledge/api/sync-and-visibility.md)
