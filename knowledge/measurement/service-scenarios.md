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
