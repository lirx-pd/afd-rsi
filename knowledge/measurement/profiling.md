# Campaign profiling and small prefill

The Ascend eager/graph recipes freeze smoke, 300 official GSM8K strict-match samples and a
prefill matrix: 128/512/2048 input token IDs, one output token, concurrency 1/4/8,
eight measured requests and two unmeasured warmups per point. Routing can raise
warmups to cover all Attention endpoints. GSM8K strict-match defaults to 0.27 for DeepSeek
V2 Lite and 284/300 for DeepSeek V4 Flash; the V4 recipes use that default.
This small sample is a functional gate, not production accuracy qualification.
Keep seed 42, eight few-shot examples and 512 maximum generated tokens fixed.
Use the existing offline cache or explicitly configure a prepared local dataset.

Service observations and reports retain GSM8K counts/scores and client HTTP TTFT
mean/p50/p90. Prefill TTFT includes scheduling, Attention, FFN, networking and
client overhead. The eager/graph recipes use the 2048-token, concurrency-8 TTFT
p50 as the frozen model-service primary metric, with independent confirmation. Eight
requests demonstrate the measurement path, not a statistically reliable gain.

With native.profile=true, the initial operator-first baseline is profiled once
after native qualification and before the first proposal. This is diagnostic,
not service qualification. Ordinary candidate screens run no profiler. Only a
fresh confirmation that repeats the model-service gain and passes the configured
service gates runs the final candidate profile. Legacy full recipes keep their
previous profiling order. Completed profiles are audited and reused on resume.
A failed or interrupted profile blocks qualification and requires the existing
NPU health barrier. Do not spend a diagnostic rebuild on each unpromising patch.

The child uses torch_npu.profiler on the same native seven-case workload and
execution mode. Every FFN rank needs its own kernel_details.csv containing actual
Mega device tasks, with positive finite Duration(us). Archive raw traces,
process receipts, source/library identities and hashes. Keep profiler execution
separate from ordinary benchmark samples; never compare profiled latencies as
unprofiled gains.

Which task proves the Mega path depends on the revision: newer sources run the
pull-native FFN as `AfdMegaMoePullNative` together with `AfdMegaMoeComm`, older
ones had one fused `AfdMegaMoeFFN` task. The controller accepts any of those
names and rejects a profile that shows none.

The controller summarizes count, total/mean device duration and top task shares
per FFN rank. Summed task duration is not elapsed critical-path time or PMU busy
time. A fused Mega task does not expose its internal Cube/Vector/MTE/communication
split. Sparse cases intentionally distribute unequal work; different rank totals
alone do not establish imbalance. Kernel timing ranks investigation targets; the
Agent must combine it with actual source/tiling to state a falsifiable hypothesis.

Requests include incumbent profile analysis and recent candidate profile evidence,
as well as service metrics and the repository-owned performance strategy skill.
After rejection, history and diagnostic evidence guide the next proposal. Validate
every proposal through the unchanged gates. Native synthetic profiling does not
explain model prefill TTFT by itself. Stage/PMU attribution needs additional
instrumentation and controlled experiments before stronger bottleneck claims.

Campaign profile additionally creates a separate A3 diagnostic build with bounded
per-core GetSystemCycle records. An owning scratch allocation reserves 72 tails,
2048 records per core and a checked guard. Overflow, missing cores, malformed
intervals or incorrect diagnostic outputs block the profile phase. This eager
seven-case diagnostic is explicitly distinct from FULL replay timing.
Stage IDs 0..6 are RECEIVED, PRE1, G1, PRE2, G2, OUTPUT, SENT;
7 is activation; 8+stage is dependency wait; 15+stage is arrival fence/drain;
22+stage is arrival publication; 29 entry, 30 DR setup, 31 metadata validation,
32 transport finish, 33 final synchronization. Sequence identifies the tile.
Preserve raw per-core intervals: summed spans double-count overlap and are not
elapsed time. GetSystemCycle clocks must not be assumed cross-core synchronized.
Compare waits and work on each core before attributing a pipeline bottleneck.
Request summaries retain selected cores through sequences 1..4, covering the
first reuse of the original two-slot diagnostic target. Historical profiles retain
phase totals and raw archive identities; detailed timelines stay in the archived
observations.
Current profile timelines, source dependencies and knowledge precede log tails
in the context budget. The limit includes the actual indented request file.
The diagnostic source and binary identities are archived separately; production
service and timing continue to use the immutable uninstrumented artifacts.

For the newer three-slot, expert-packed target, see the
[reviewed external study](../performance/megamoe-expert-packing-20261025.md).
Its first-16-tile probe is separate from this driver. Check actual target layout,
clock assumptions and diagnostic support before reusing a probe. A positive finite
CANN duration can still underflow beyond the capture envelope; preserve invalid
rows and apply the frozen failure policy rather than deleting evidence to pass.

Diagnostic overlays support the legacy prepare pipeline and the expert-packed
PrepareTile/validated-entry pipeline. Check every insertion anchor and its exact
count before writing either diagnostic file. Missing anchors are unsupported
source, not permission to emit partial instrumentation. Entry and validation
intervals must pair in the kernel entry scope; never match the helper's layerId.
Retain diagnostic compiler stdout/stderr and process receipts in result archives.

The pull-native overlay records the top-level receive round: `round_prepare`,
`pull_rows`, `gmm1`, `activation`, `gmm2`, `combine`, `round_complete`,
`sync_in`, `sync_out` and `rx_decode` (codes 40..49). Eight selected cores record
events; every core writes the header the decoder requires. The nested breakdown
adds `rx_select` (60), its `rx_header_validation` (61), `rx_metadata` (62),
`rx_commit_check` (63), `rx_group_plan` (64), `rx_publish` (65), `rx_layer_map`
(66), `output_local` (70), `tx_remote` (71), `output_sync` (72),
`output_wait_gmm2` (73), GMM input waits (74..75), and `rx_source_wait` (76).
The latter ends when a complete source is ready and contains header validation.
The report keeps parent
and child cycle sums separate and gives each child's share of its parent;
`residual_cycles` contains unmarked work and trace overhead. `rx_select` spans
source polling plus header checks. Its residual after `rx_header_validation`
is polling, state reads and instrumentation overhead, not a pure wait timer.
`tx_remote` includes the final DMA drain; earlier `Pump` intervals can enqueue
asynchronous copies. Compare events on the same core. Summed intervals are not
critical-path latency. `active_share_pct` and `ffn_core_cycles` exclude `sync_in`
and the source-readiness interval except measured header validation; FFN internal
dependency waits remain included. Older traces without code 76 conservatively
exclude all of `rx_select` except header validation, including some FFN selection
work. Without code 60 the FFN shares are unavailable. These are per-core sums,
not request or kernel latency. The instrumented tree is a separate eager diagnostic build;
production timing and graph replay stay uninstrumented.

`overlap` compares sampled Cube/Vector, Vector/Vector and Cube/Cube core pairs
within the same receive round. The entry (`sync_in`) and exit (`sync_out`)
barrier spans bound the offset between the two core clocks; their intersection
gives a range, not an exact alignment. Overlap is reported as a minimum and
maximum over that range, conditional on equal clock rates and a stable offset
within the round. Inconsistent or missing barriers make the pair unavailable.
These assumptions are not independent clock calibration. GMM comparisons also
report overlap after removing marked input or activation waits, when those wait
markers exist. `pull_rows` covers Pull/RX and packing; compare it with GMM1 to
see potential prefetch/compute overlap. Activation may still wait internally. `tx_remote` represents
enqueue and drain calls, not the lifetime of asynchronous DMA on the link.
Intervals describe sampled call spans, not Cube/Vector hardware occupancy,
complete critical-path latency, or proof of performance gain. Rounds are
aggregated across captured fixture cases; consult raw records for context.
