# Runtime and source/build identity

Scope: local A3 Async CAM integration. At campaign start trace the actual
connector -> runner -> executor -> native schema -> tiling -> kernel route.
Names alone do not prove dispatch. Record effective environment/config flags,
loaded library paths, hashes, kernel names, topology and per-rank version facts.

MegaMoe selection in the source used to author this bundle uses
AFD_ASYNC_CAM_MEGA_MOE with the layered W4A8 executor and ffn_two_stream=false.
Verify that contract against the current target. Compare eager with eager and
FFN FULL with the same FULL configuration. Attention's graph/DBO constraints
are independent and must be checked rather than inferred from old branches.

Capture requires stable weight/workspace addresses, correct mutability and side
effect registration, matched startup traffic and all-rank configuration. A Meta
schema or capture pass does not prove communication or replay correctness.

Build into experiment-owned output. Never install over SDK/shared operators.
Separate working directories do not isolate Python import paths or dynamic
libraries. Start fresh processes for each side. Independently hash declared
artifacts before/after each command and require every rank to report the same
expected source/artifact identity. A benchmark's declaration is a trust boundary:
its implementation must actually inspect loaded native libraries for NPU use.
