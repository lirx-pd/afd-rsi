# Local review and failure diagnosis

Scope: A3 Host/Tiling/Kernel and local A/F runtime changes. Review the complete
connected path, not only the changed lines. Check input bounds, integer width,
workspace intervals, per-core ownership, zero work, tails, signal pairing,
visibility, graph mutability, loaded artifact identity and error propagation.
Report concrete failure conditions and source locations. Distinguish observed
failure, static risk and unknown hardware behavior.

For a hang preserve the same attempt's input ID, rank/device, kernel, tiling,
time and logs. Check blocked peers, absent ready, barrier participants and
buffer reuse. For an AIC/AIV fault preserve full available device error evidence
and the matching binary; do not mix PC/register logs across builds. For host
crashes retain the stack and same input. For suspected OOB use the installed
platform's memory diagnostics in a separate diagnostic attempt when available.
No diagnostic tool is assumed installed by the controller.

Timeout triggers process-group cleanup for this attempt only. Killing host
processes does not certify device health. No global pkill or automatic device
reset. If cleanup or health is unknown, stop before launching more work.
Repair is validated with the original failing input and repeated launch/replay,
then correctness and ordinary performance without diagnostic instrumentation.
