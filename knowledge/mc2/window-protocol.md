# CAM window and state contract

Scope: mapped MTE peer windows used by the compact_v2 AFD path. This is not a
symmetric EP collective or an arbitrary network transport contract.

Use the current context/address helper supplied by the target implementation;
do not hardcode another platform's window structure offsets. Do not replace
token-level window operations with an opaque collective and assume equivalent
ready/count/order semantics. Communication resource creation belongs to the
connector; tensors, workspace and graph anchors have their own owners.

Trace payload, scale, route indices, counts and ready fields together. For each
field identify its only writer or explicit reduction, readers, publication
order, reset condition and integer range. Zero-token work must still complete
its protocol. Clearing ready while a consumer retains input is a reuse bug.

Model a transaction across both roles. An early FFN output write does not mean
Attention can finish before all required FFN contributions. A completion tensor
can be a dependency/lifetime anchor rather than a remotely observed completion.
After a work item has been consumed, runtime fallback cannot safely receive it
again through a different backend.
