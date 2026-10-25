# Frozen search and experience rules

The Agent proposes hypotheses and complete unified diffs. The controller alone
applies allowed-file patches to isolated source copies, builds, validates and
accepts them. Never edit validators, gates, the original checkout or this
knowledge snapshot to make a candidate pass. Codex runs in read-only mode with a
structured JSON response; the existing CLI login stays on the proposer host.

Qualify the baseline before a model call. A candidate must pass correctness,
every configured real service gate, and all paired native measurements. A PASS
is only a screen: repeat the whole comparison in a fresh attempt against the
same incumbent. Only two matching PASS results can replace best. A failed,
missing, timed-out or incompatible observation cannot promote a candidate.
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
The built-in adapter uses at most two model calls per dispatch; follow the
[Codex model policy](codex-models.md) for role assignment and audit records.

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
