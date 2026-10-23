# Eager native validation contract

The native fixture and its provenance/license live in `benchmarks/native/`.
Use native build to copy source, build with the target's own entry, and keep all
outputs in experiment-owned storage. A source snapshot lacks Git metadata;
use its content hash as the setuptools-scm local build version.

Run layered references, MegaMoe correctness, paired AB/BA timing, and profiling
in separate fresh processes. Validate loaded binding/API mappings and device
object hashes on every rank against expected build artifacts before interpreting
results. Device kernel names in exported NPU traces establish actual dispatch.

For manual candidate evaluation provide separate built baseline and candidate
repositories. Freeze both identities. Use the baseline's layered implementation
as the common numerical reference, then validate both MegaMoe implementations
and measure baseline/candidate in alternating order. Never regenerate the common
reference from candidate code. Without a baseline repository the entry diagnoses
layered versus MegaMoe in one build. Reports distinguish these two modes.

Default cases: H256/I512, E_local4, two FFN TP1, top-k2, two layer IDs, 7/16/129
tokens and sparse/empty expert work. Optional wide shapes retain synthetic
weights and do not establish real model/topology correctness. Two Attention
sources remain serialized by harness barriers, not simultaneous incoming fan-in.

Attention NPU events measure complete DS->FFN->CR transactions in one clock
domain. Host output validation follows the ending event. Disable FFN's per-work
host observations during timing; use separate diagnostic/profiler runs. Keep
all samples and per-case regressions. No kernel-time or service-throughput claim
can be derived from these transaction timings.

Missing ranks, actual-library mismatches, changed source/artifacts, malformed
or nonfinite samples and incomplete cleanup invalidate the attempt. Failed
attempts remain inspectable. Current native compare is a manual diagnostic
entry, separate from automatic campaign promotion and crash recovery.

FULL graph uses the same fixture with --full-graph, and eager layered references.
Require explicit graph execution on all ranks, capture/replay counts on every
FFN rank, stable exposed addresses, and disabled Python transactions. Timing
replays the graph without diagnostic D2H copies or buffer poisoning. A graph
request must never fall back to eager. Freeze native.execution and matching
service --execution in separate recipes; eager and graph evidence are distinct.

Async CAM service graph also means FFN explicit FULL with Attention eager.
FULL_DECODE_ONLY on FFN and graph on Attention violate the target validator.
MegaMoe metadata starts at offset zero for the tiled executor with completion;
the original executor layout places metadata after aligned X/scales. Freeze
the target source/artifact identities; do not mix layouts or comparison modes.
