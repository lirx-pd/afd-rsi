# One-command optimization orchestration

The optimize front end freezes a controller/foundation snapshot and starts a
background worker. It supports local Linux and the optional iTASK CLI executor.
The execution-host controller owns prepare/search/resume, device experiments,
health barriers, decisions and export. The proposer host retains its Codex CLI
login; credentials are never shipped. Core acceptance has no platform imports.

Codex analysis and patch generation are finite proposal processes. After they
exit, the worker submits the proposal and launches the execution-host controller
in a detached process. Poll with ordinary timed process/file operations, default
60 seconds, never an idle model conversation. Report phase changes concisely;
preserve detailed events, command receipts, request hashes and model usage.
Starting a job returns immediately after the controller snapshot is ready.

Early rounds (approximately the first two thirds) use deep optimization: coherent
changes to tile scheduling, buffer lifetimes or coordinated tiling may span
allowed files. Later rounds use incremental optimization of the current best.
For one round, use deep mode. Strategy changes only proposal guidance: allowed
paths, published interfaces, numerical/service gates, gain threshold and fresh
confirmation requirements remain unchanged. Depth is not a line-count target.

Within each tested source/artifact version start the A/F service once, then run
smoke, GSM8K30 and the performance matrix. Settle between suites and points,
outside measured requests. Monitor service liveness during suite settling;
cancellation interrupts settling and still cleans up owned processes. Do not
reuse an old loaded service across different candidate libraries. Failed matrix
points remain failed; attempt remaining independent points where possible.
Overall performance still fails if any required point fails.

The front end preserves a unique execution directory, controller identity,
source snapshot, per-launch intent, process ownership and atomic results.
Explicit resume reattaches to the same job; it must not start a second worker,
reset campaign budgets or silently retry an interrupted proposer. A transport
failure is not proof that the remote experiment stopped. Keep remote data and
inspect receipts before recovery. Downloaded archives and their file hashes are
checked before reporting completion. Oversized transfers fail visibly and leave
raw evidence on the execution host. Task shutdown/deletion is never automatic.
