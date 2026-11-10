# Runtime and source/build identity

Scope: local A3 Async CAM integration. At campaign start trace the actual
connector -> runner -> executor -> native schema -> tiling -> kernel route.
Names alone do not prove dispatch. Record effective environment/config flags,
loaded library paths, hashes, kernel names, topology and per-rank version facts.

MegaMoe selection has moved across AFD revisions, so read it from the target
rather than assuming the old switch. Newer sources enable the FFN pull-native
path through connector extra config (`ffn_mega_moe`, requiring dynamicQuant=1)
and run it as `afd_mega_moe_pull_native` paired with
`afd_mega_moe_comm`; older sources used the AFD_ASYNC_CAM_MEGA_MOE environment
variable and a fused `afd_mega_moe_ffn` kernel. Verify that contract against the
current target. Compare eager with eager and FFN FULL with the same FULL
configuration. Attention's graph/DBO constraints are independent and must be
checked rather than inferred from old branches.

Current Async CAM always uses the Layered W4A8/W8A8 executor. The old
AFD_ASYNC_CAM_LAYERED_GMM environment switch and FFN two-stream options are
removed. Attention pending-state checks use connector.has_pending_attention,
which covers both Push queues and Pull windows; do not inspect their private
session storage. The native fixture retains the old dict fallback for older sources.

Capture requires stable weight/workspace addresses, correct mutability and side
effect registration, matched startup traffic and all-rank configuration. A Meta
schema or capture pass does not prove communication or replay correctness.

Build into experiment-owned output. Never install over SDK/shared operators.
Separate working directories do not isolate Python import paths or dynamic
libraries. Start fresh processes for each side. Independently hash declared
artifacts before/after each command and require every rank to report the same
expected source/artifact identity. A benchmark's declaration is a trust boundary:
its implementation must actually inspect loaded native libraries for NPU use.
