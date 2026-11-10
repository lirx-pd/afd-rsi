# Frozen search and experience rules

New Ascend recipes declare `policy="adaptive-v1"`. The original sealed request
and response of each proposal batch form a persistent candidate pool. Each
selected candidate gets its own round state, isolated source and attempts;
unselected proposals are reused without another model dispatch. Resume verifies
request/response hashes, selected proposal membership, parent identity and the
comparison baseline. Completed phases and charged calls are never repeated.
Model budget exhaustion does not discard pending candidates that fit the test budget.

Use the [direction catalog](../performance/search-directions.json) and the local
performance skills as starting points, not a whitelist. Every adaptive proposal
names direction.id, mechanism, step (1 local, 2 coordinated, 3 architectural),
refs, use and next. Cite exact frozen knowledge paths and explain applicability,
extension or a source-backed new mechanism. Individual referenced skill and
knowledge bodies remain available as read-only context file IDs even if the
optional combined knowledge text is trimmed. New ideas do not automatically
change frozen knowledge or acceptance gates. Historical native improvements
are not current model-service evidence.

Adaptive scope follows measured gain trends, not round number. The first
analysis and a changed incumbent get a global diagnosis. Three consecutive
valid sub-threshold/noise-limited results across at least two mechanisms trigger
a global review in place of the next analysis. The review cursor is bound to
its baseline and valid evaluation count, so unchanged evidence cannot repeatedly
trigger review. Candidate compile, numerical or environment failures are not
performance observations. Recent gain momentum is a small-sample ranking
heuristic, not a causal gradient or a measured hardware bound.

A valid slower intermediate may supply patch source for one continuation when
`direction.next` specifies a concrete mechanism, cumulative primary regression
is at most 5 percent, and exploration budget remains. The continuation has depth
two from the qualified best. Patch parent and comparison baseline are separate:
all screening/confirmation gains compare directly with qualified best, and only
two complete PASS results replace it. Information selections and continuations
share a candidate-count quota, floor(max_rounds * explore_max_fraction).
Incomplete or incorrect candidates cannot become exploration parents.

`max_rounds` bounds candidate rounds; `max_calls` bounds proposal/debug dispatches.
`max_model_calls` reserves at most three returned completions for ordinary Codex,
four for DeepSeek, and one for code-only debug before dispatch. Conservative
reservations are never refunded after interruption; actual usage is separately
reported, with absent usage explicitly unknown. Before opening another candidate,
reserve screening and independent confirmation using frozen phase ceilings.
The original wall deadline remains authoritative. There is no hard token budget.
`info_floor=0` disables information exploration; a positive value must be at most
0.25 and also needs more than six valid candidate evaluations and empirical
score-scale variability (three recent observations per considered direction).
Unknown directions have no invented confidence bound. Repeats, confirmation and
failed implementation attempts do not add extra valid evaluation counts.

Recipes without policy retain the legacy rules below. Their frozen snapshots
and historical fingerprints are not migrated. The CPU adaptive recipe is a
controller fixture; it makes no accelerator performance claim.

The Agent proposes hypotheses and complete unified diffs. The controller alone
applies allowed-file patches to isolated source copies, builds, validates and
accepts them. Never edit validators, gates, the original checkout or this
knowledge snapshot to make a candidate pass. Codex runs in read-only mode with a
structured JSON response; the existing CLI login stays on the proposer host.

Qualify the baseline before a model call. With validation=operator-first, qualify
native numerics and timing and collect the incumbent profile once; initial
baseline qualification does not claim model/service correctness. When the recipe
sets service.primary, screen a patch with native correctness and paired timing,
then compare the frozen real-model service metric with smoke and the selected
performance scenario only. A fresh confirmation repeats the native checks and
model metric with reversed service order, and runs GSM8K plus all configured
service scenarios. Both service gains
must pass the frozen threshold; native timing is diagnostic and cannot veto a
model gain. Run the final candidate profile after the confirmation service gates.
Two matching model-service PASS results plus full final qualification can replace best.
A recipe without service.primary retains its native timing objective and gate.
A numerical, timing or final service failure cannot promote a candidate. Ordinary
full recipes retain the previous full-validation order. Old snapshots are immutable.
HTTP inference success cannot replace cleanup evidence. A frozen service policy may
set require_clean_shutdown=false to defer errors classified by the E2E runner as
shutdown-only. Preserve the failed raw lifecycle summary and explicitly record
that waiver; inference, every configured test, loaded identity and cleanup must
still pass. The Agent cannot change this policy during search.

Agent-dispatch reservations and round state are persisted before dispatch. Wall
time is the original campaign deadline; resume cannot reset any budget. A file
exchange may separate the proposer from the executor, but it cannot change the
frozen workload, timeout or acceptance rules. Preserve raw model events and the
submitted proposal alongside experiment evidence. Token usage is an observation,
not an enforced token budget; Agent dispatches and wall time are bounded.
Ordinary dispatches analyze once and then code. Codex allows one extra coding
call if local patch application fails; DeepSeek allows one recovery per stage,
shared by context reads, output truncation, schema and patch failures (at most four completions
per ordinary dispatch). Retry prompts name response properties, not schema keys.
The proposer host needs Git for disposable-copy patch preflight. No preflight
modifies the original checkout or replaces the controller's allowed-file checks.
Follow the
[Codex model policy](codex-models.md) for role assignment and audit records.

Candidate compile/runtime or native numerical failures receive at most one
code-only debug dispatch in the same round, charged before dispatch against
max_calls and the original deadline. Baseline/environment validation failures,
valid slow results and final accuracy rejections do not trigger debug. Preserve
the failed patch, logs and attempts under the round; the debug request includes
the original hypothesis and failure receipts, not a new architecture analysis.
Return a complete replacement patch against the original parent; keep the
hypothesis, category and experiment unchanged. A second implementation failure
ends the round. A repair must start fresh screening and independent confirmation,
including all final service gates; previous partial passes cannot qualify it.
Device health recovery still precedes any new device trial. Resume never grants
another debug or refunds a reserved call. If budget is exhausted, record why the
debug was skipped. Preserve the latest two history summaries ahead of optional
knowledge and older profiles, so the next direction can learn from failure.

Native builds capture the CMake-resolved MC2/Catlass include closure and generated
activation into a read-only dependency bundle bound to source and artifact hashes.
Build observations seal that bundle; cache reuse checks installed header bytes.
Repository protocol/layout sources in allowed_files must not be silently trimmed.
The request transport includes the bundle; Codex materializes individual sources,
dependencies, evidence and knowledge with a small manifest for selective reading.
Generic compiler/AscendC includes outside this closure remain explicitly unresolved.
Before coding, verify completion counts, Finalize, idle-core drains and activation
slicing required by the chosen design. Return stop=true, proposals=[] and
needs_context naming missing contracts. The controller reports context_required,
not exhausted optimization directions, and spends no further dispatch. Repair the
evidence in a new campaign; frozen historical requests are never rewritten.
Protocol v2 gives HTTP models a bounded read from the frozen context catalog;
only the selected source/contracts are repeated for coding. Critical MC2/generated
contracts and the complete request source closure remain inline during analysis.
Original request facts (including sampled events) and SDK bodies remain retrievable,
with unchanged stage/wait/overlap summaries; full device traces remain in the archive.
If the required seed or selected coding context exceeds the budget, return
context_required before that model stage. See [agent providers](agent-providers.md).

Recover only processes with the trial's ownership tag, matching host and boot.
Use pidfds rather than signalling a reused numeric PID. Preserve the original
receipt and write a separate reconciliation record. Verify completed phases and
reuse them; retry unfinished phases in new trial directories. After an Ascend
execution failure/interruption, run the frozen device compute health check
before any new device trial. A successful basic check does not prove that every
communication failure is recoverable; a failed check stops scheduling. Never
reset devices or kill unrelated workloads automatically.

Each experience separates the proposed hypothesis from its measured decision
and includes source/patch identities and evidence hashes. Reuse only matching
source, foundation, controller, workload, topology, service and gate contracts.
Verify hashes before loading lessons. Suppress an already tested patch against
the same parent. Penalize failed categories and reward confirmed promotions
when ranking the next proposals. A category score guides exploration; it is not
proof of causality. Never automatically promote an experience into frozen
skills or knowledge. Maintainers review such changes outside candidate search.

Priority 3 is preferred and 1 is fallback. The controller selects the maximum
of priority plus category promotions minus category failures; list order does
not override that score. Do not describe an unselected proposal as an experiment.

Unified background execution and deep-to-incremental strategy are specified in
[orchestration](orchestration.md). The controller still owns all acceptance.
When a recipe reserves `global_redesign_round`, that dispatch compares the FFN
RX/pull, packing, GMM, activation and output/TX path after source readiness and may
replace or add allowed kernel files as one coherent design. Do not mistake a
producer wait for an FFN hotspot, or a large diff for a measured speedup; the same numerical,
native and model-service gates apply.
Compare FFN stage overlap and internal waits before using a stage's cycle share
as a priority: a large concurrent stage may already be hidden. Sampled-core
overlap bounds and TX call spans are diagnostic, not a full critical path or
asynchronous communication utilization; validate any masking hypothesis using
unprofiled native and model-service measurements.

Under operator-first, proposals require experiment.bottleneck, mechanism,
prediction, falsification and risk. Cite baseline measurements and the incumbent
profile when supplied. Explain the source dependency and an observable change;
stop when evidence is insufficient. A coherent overlap change must identify the
communication/computation stages, ownership, waits and lifetimes being changed.
No profile duration alone proves physical overlap. Preserve this experiment in
history so a failed prediction affects the next direction, not just the patch.
The controller validates structure and evidence IDs, not the truth of a hypothesis.
