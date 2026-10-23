# Measurement contract

Scope: local controlled experiments. Freeze workload, route distribution,
capacity, quantization, graph mode, effective flags, seeds, warmups, repeats,
units, clock domain, primary rank and thresholds before candidate search.

Kernel Task Duration, complete FFN work duration, Attention A->F->A transaction
and service request latency are separate metrics. Measure a transaction in one
Attention clock domain with all relevant stream dependencies. Do not subtract
unsynchronized device clocks or assume max heterogeneous rank kernel duration
is end-to-end latency. Host elapsed includes its declared control/launch scope.

Alternate AB/BA using the same payload and independent trial groups. Keep all
valid slow samples. Confirm screened winners with fresh groups. Ranks within
one launch are not independent repeats. Keep per-case regressions visible.
Timing instrumentation/profiler runs are separate from uninstrumented totals.
PMU active time is not critical-path elapsed. Logical DMA bytes are not physical
link bytes. Calibrate hardware cycle conversion against the exact target.

For communication attribution hold batch_info/payload and bytes per sender
fixed; separate single-segment completion, target fan-in and concurrent GMM.
Ready backlog is not simultaneous arrival. Check feasible AIC/AIV budgets and
core-start spread before attributing slowdown or overlap to contention.

Missing ranks, malformed/nonfinite data, mismatched identities and interrupted
trials cannot become PASS. A completed numerical failure is FAIL even if later
performance evidence is absent. CPU fixture values are synthetic, never NPU gain.
