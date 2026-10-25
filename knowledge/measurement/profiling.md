# Campaign profiling and small prefill

The Ascend recipes freeze smoke, 30 official GSM8K strict-match samples and a
prefill matrix: 128/512/2048 input token IDs, one output token, concurrency 1/4/8,
eight measured requests and two unmeasured warmups per point. Routing can raise
warmups to cover all Attention endpoints. GSM8K strict-match defaults to 0.27 for DeepSeek
V2 Lite and 0.9 for DeepSeek V4 Flash; the V4 recipes explicitly freeze 0.9.
This small sample is a functional gate, not production accuracy qualification.
Keep seed 42, eight few-shot examples and 512 maximum generated tokens fixed.
Use the existing offline cache or explicitly configure a prepared local dataset.

Service observations and reports retain GSM8K counts/scores and client HTTP TTFT
mean/p50/p90. Prefill TTFT includes scheduling, Attention, FFN, networking and
client overhead. It is an auxiliary performance indicator; promotion still uses
the frozen paired native transaction metric and independent confirmation. Eight
requests demonstrate the measurement path, not a statistically reliable gain.

With native.profile=true, the initial operator-first baseline is profiled once
after native qualification and before the first proposal. This is diagnostic,
not service qualification. Ordinary candidate screens run no profiler. Only a
fresh confirmation that also repeats the native gain and passes the configured
service gates runs the final candidate profile. Legacy full recipes keep their
previous profiling order. Completed profiles are audited and reused on resume.
A failed or interrupted profile blocks qualification and requires the existing
NPU health barrier. Do not spend a diagnostic rebuild on each unpromising patch.

The child uses torch_npu.profiler on the same native seven-case workload and
execution mode. Every FFN rank needs its own kernel_details.csv containing actual
AfdMegaMoeFFN device tasks, with positive finite Duration(us). Archive raw traces,
process receipts, source/library identities and hashes. Keep profiler execution
separate from ordinary benchmark samples; never compare profiled latencies as
unprofiled gains.

The controller summarizes count, total/mean device duration and top task shares
per FFN rank. Summed task duration is not elapsed critical-path time or PMU busy
time. A fused MegaMoe task does not expose its internal Cube/Vector/MTE/communication
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
512 records per core and a checked guard. Overflow, missing cores, malformed
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
first reuse of the current two-slot ring. Historical profiles retain phase totals
and raw archive identities; detailed timelines stay in the archived observations.
Current profile timelines, source dependencies and knowledge precede log tails
in the context budget. The limit includes the actual indented request file.
The diagnostic source and binary identities are archived separately; production
service and timing continue to use the immutable uninstrumented artifacts.
