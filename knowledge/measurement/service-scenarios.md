# Current native-routing service targets

The pipeline-scenarios recipe selects exactly native-long-decode, mixed-prefill
and heavy-prefill. Small prefill matrices and the superseded 16K burst point are
not selected. The [embedded workload record](native-service-targets.json) captures
the completed external experiment specifications and provenance. Shapes, request
mix and offered rate are reproduced; historical prompt text and measured gains
are not imported as a new RSI baseline. Token prompts use this harness's fixed seed.

- native-long-decode: six simultaneous 32768-input/512-output requests; six warmups
  with 32 output tokens. Require one-token SSE events and a common decode window.
- mixed-prefill: 24 requests, eight 16384 inputs and sixteen 512 inputs, output 1,
  QPS3, three unmeasured shape-matched warmups.
- heavy-prefill: 128 requests, each 65536 input and 1 output token, QPS3, three
  warmups. No small client concurrency cap: allow up to 128 outstanding requests.

All three use native routing without DP affinity headers. Client workload groups
are not actual DP ranks. Require complete per-engine prompt/generation counters,
nonnegative increments, exact aggregate usage, feasible per-engine token mixes
and zero preemptions; report actual routing distribution. Archive raw requests,
warmup and measured counter snapshots. Acceptance recomputes summaries with the
frozen validator. Retain failures instead of silently shortening input or workload.
Common-window TPOT and output tokens/s, aggregate/group prefill TTFT and input
throughput enter reports and Agent context. Higher throughput means improvement;
lower latency means improvement. Native screening/promotion gates remain unchanged.

The historical fixed-DP protocols remain available explicitly and are described
below; their results are not directly comparable to native-routing experiments.

# Fixed-DP prefill and long-context decode

These are optional final service performance scenarios. Keep native operator
screening and fresh confirmation ahead of expensive model work. One A/F startup
runs smoke, GSM8K, the selected matrix and both scenarios. Do not restart services
between scenario points or reuse loaded libraries across source versions.

Fixed-DP prefill: pin the supported X-data-parallel-rank header. Default DP0 input
is 16384 tokens; every other Attention DP uses 512. Warm one exact-shape request
per DP, then send eight measured requests per DP at aggregate QPS3 in round-robin
DP order. Generate one output token. Report TTFT separately for every DP; a single
pooled median hides the heavy/light difference. Require exact per-engine prompt
and generation counter deltas and zero preemptions. Headers alone are not routing
proof. Missing engine labels, counter resets or unrelated traffic invalidate the
scenario. Metrics can publish late; allow a bounded drain, never fabricate counts.

Long decode: default 32768 input and 512 output tokens, two concurrent requests
per DP. Warm two same-context requests per DP with 32 outputs. Disable prefix
caching. Require exact prompt and output usage and zero preemptions. Save one
client monotonic timestamp per emitted token ID. Require stream interval one;
merged multi-token SSE events cannot supply a token timing distribution.

Keep request TPOT separate from common-window TPOT. The common decode-only window
begins one second after the latest first token and ends one second before the
earliest final token. Every request must contribute at least 128 token events.
Within that shared window compute each request's time between its first/last
retained token divided by its retained inter-token count, then report the mean
and percentiles across requests. Empty or short windows fail, rather than falling
back to whole-request averages. HTTP/SSE delivery includes network/client effects;
it is not hardware kernel latency, exact device step time or proof of overlap.
No cross-device clock synchronization is inferred from client timestamps.

Archive warmup/measured JSONL and before-warmup/before/after Prometheus text.
Recompute summaries from these raw files during acceptance and resume auditing.
Both scenarios must finish if selected; a failure cannot be replaced by a favorable
matrix point. Agent context and reports retain per-DP TTFT and both TPOT scopes.
The scenarios measure integration behavior; they do not prove actual DBO, graph
replay or physical compute/communication overlap. Use separate device evidence
for those claims. The native primary metric and promotion thresholds stay frozen;
service scenario comparisons are auxiliary unless a future frozen policy explicitly
makes them the objective. Never claim model improvement from a native-only PASS.
