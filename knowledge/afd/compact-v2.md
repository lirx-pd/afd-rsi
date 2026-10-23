# AFD compact_v2 MegaMoe contract

Scope: cam_async_routed_only_compact_v2. Initial campaign keeps protocol and
quantization unchanged. All facts must be checked against the experiment's
source snapshot when the target revision changes.

Attention ranks occupy [0,A), FFN ranks [A,A+F); distinguish role-local rank,
communication world rank and local device ordinal. Attention TP width divides
A. First supported FFN scope is TP=1; this does not mean Attention TP must be 1.

Only routed experts are sent. Expert 0 is valid; no shared-expert reserved slot
is assumed. The sender already deduplicates activation for token -> destination
FFN rank; expert occurrences still need indices and expert input organization.
Shared experts and final top-k weighting remain on Attention.

The selected FFN metadata is int64[5+T+E_local*T]: total routed count, Attention
TP-group base, layer ID, expert interval, T prefixes and T*E_local counts. Chunk
rows are the sum of counts for the selected chunk, not the metadata total or
allocation capacity. Organization and return order preserve expert, TP-source
and route occurrence semantics. Layer ID is device-selected through a checked
layer-to-weight-slot map; invalid layer cannot dereference weights.

Publish payload/count/index before ready, including zero work. Input ready is
cleared only after relevant consumers finish; final expert chunk advances the
source group. Combine publishes completion after all required output writes.
Attention retains routing state until matching combine receive is complete.

The phase-1 MegaMoe scope is DR -> W13/SwiGLU/quant -> W2 -> CS inside a bounded
mixed kernel. Keep compact_v2 and disable experimental FFN two-stream. One
launch does not prove overlap: inspect actual stage barriers and measure.
