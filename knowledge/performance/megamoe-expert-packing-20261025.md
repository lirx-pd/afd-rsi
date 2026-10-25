# A3 MegaMoE: expert-packed tiles, 2026-10-25

This is manually reviewed external evidence and reusable experiment guidance.
It is **not** a compatible dynamic `search/experience` record, an RSI promotion
verdict or proof that a future candidate will improve performance. Existing
campaign snapshots, recipes, gates and the pinned reference header remain unchanged.
The [local evidence record](megamoe-expert-packing-20261025.json) preserves exact
values, source identities, artifact hashes and measurement scope.

## Diagnose the pipeline unit before tuning overlap

The previous kernel already had three slots, two-tile PRE1 lookahead, early input
refill, 16 communication AIVs and 32 compute AIVs. It nevertheless ran one expert
per tile, with at most 128 rows. Every short expert repeated Matmul initialization
and seven stage publications. With N4096/baseN256, an expert of at most 64 original
rows has only 16 M/N blocks for 24 AICs. Separate experts repeatedly leave eight
AICs without a matrix block.

A V4 light-prefill snapshot had 343 routed rows in 43 nonempty experts; other
snapshots had 498 rows in 25 experts and five decode rows in three experts.
These are local transactions, not the heavy DP's complete request. The useful
hypothesis was to pack adjacent experts in one bounded tile and expose the
expert range to the existing grouped Matmul block rotation (`preCount`).

The three 512-row slots retain expert/TP row order. A tile records first expert,
first-expert offset, last expert, row count and sequence, with slot-local counts.
Transport partitions **packed** rows modulo 16; short experts no longer reset
communication ownership to lane zero. MSD quantization and Matmul are reused.
Do not assume numerical equivalence from identical diagonal experts: exercise
row-varying inputs, distinct expert scales/bias, TP intersections, zero experts,
empty FFN, partial experts, old chunk continuation and eager/FULL replay.

## Preserve ownership when combining experts

- G2 completion releases tile inputs for RX of the replacement tile. Output `y`
  has a separate lifetime: wait for `SENT(sequence - MEGA_RING_SLOTS)` before reuse.
- GMM2 output selects each row's expert bias from **immutable transaction counts**.
  Slot `tileCounts` may already describe the replacement tile after G2 release.
- Preserve all participant counts, monotonic sequences, MTE3/FIX drains, scalar
  cache visibility and final compact-v2 completion. Metadata remains live until
  transport drains; one source still owns an invocation.
- Rebuild workspace binding, host tiling and kernel together, then restart and
  recapture graphs. The workspace row constant changes allocation and offsets.
  `copy2` can restore old header timestamps: explicitly invalidate affected host
  and binding objects and verify installed kernel/library hashes after rebuilding.
- For H4096/I2048/E64/TP4, scratch grows from 35572736 to 141750272 bytes per FFN
  rank, adding 101.26 MiB. Route-index UB grows from 256 to 1024 bytes and needs
  the host bound check. The shared 16 MiB system workspace does not grow.

512 rows is a measured A3/V4 candidate, not an architecture-wide optimum. The
GMM base M/N/K stays 128/256/256 in this per-channel case. Actual host output is
24 AICs, 48 AIVs and 196352 bytes of UB. Matmul API `usedCoreNum=1` describes the
per-core instance, not a one-AIC kernel. Read the actual target source and SDK;
the repository's older pinned reference header is not the current target layout.

## Keep native, service and diagnostic claims separate

Native: H4096/I2048, E64 per FFN rank, top-k6, four Attention TP ranks plus four
FFN ranks, ready-input real DS/MegaMoE/CR with synthetic weights. Each Attention TP rank has
2048, hot129 or one local input row respectively. Check the numerical oracle
and allocation guard on every call. One extra warmup precedes 12 calls; use the
last five CANN task durations, averaged per FFN rank and then take the maximum.
These runs use `msprof` task-based PipeUtilization with no internal phase probes:
they are not unprofiled A-to-F-to-A transaction measurements. Confirmation repeats
the candidate only; there is no paired AB/BA result or fresh control.

| Native case | Control us | Packed first / confirmation us | Mean reduction |
| --- | ---: | ---: | ---: |
| balanced2048 | 6844.996 | 3443.720 / 3443.628 | 49.69% |
| hot129 | 1710.024 | 1039.428 / 1039.524 | 39.21% |
| decode1 | 315.712 | 215.052 / 215.716 | 31.78% |

Service: V4 Flash W4A8, 43 layers, Attention DP3/TP4 eager + DBO, FFN EP4 FULL,
32K chunked prefill, no prefix cache or FFN two-stream. Decode/prefill DBO
thresholds are 2/512. Profiling is disabled during service timing.

- Imbalanced prefill: 24 requests at QPS3, eight requests per DP; DP0 inputs are
  16384 tokens, DP1/2 inputs 512, output one token. Pin the DP with the supported
  HTTP header and verify the actual 32:1:1 input ratio using engine counters.
  Exclude one exact-shape warmup per DP.
- Long decode: six simultaneous requests, two per DP, each 32768 input + 512
  output tokens; warm with six same-context requests and 32 output tokens.
  Use the common window from one second after the last first token to one second
  before the first final token. Require all six active, at least 128 window tokens
  per request, zero preemptions, actual tiny-token two-stage DBO on all 12
  Attention ranks and FULL replay on all four FFN ranks. This excludes prefill
  interference and records single-token SSE arrivals rather than merged deltas.

| Service path | Heavy prefill TTFT ms | Light prefill TTFT ms | Common decode TPOT ms |
| --- | ---: | ---: | ---: |
| Unfused | 1624.621 | 248.302 | 254.310 |
| Previous MegaMoE | 2111.745 | 341.994 | 257.821 |
| 512 rows, single expert | 1628.900 | 265.109 | 245.977 |
| 512 rows, packed experts | 1621.757 | 244.760 | 245.320 |

Packed experts remove the old MegaMoE prefill regression. Relative to unfused,
prefill is effectively at parity and decode TPOT is 3.53% lower **in this run**.
Controls are frozen earlier service runs; neither reverse tests nor statistical
significance were established. The small decode difference between the two
512-row candidates is not established as significant. Native gains do not imply
model gains of the same size.

The same 300 GSM8K documents and prompts score 96.00% against a saved 95.6667%.
The reference used DP2TP4/EP8; the new run uses DP3TP4/EP4 plus DBO. Three answers
improve, two worsen and 169 generated texts change. Preserve the frozen score
gate; this is not bitwise equivalence or a causal accuracy improvement.

## Read overlap and utilization evidence carefully

The limiting-rank native MAC time ratio changes from 14.32% to 28.52%, while AIV
scalar ratio changes from 80.96% to 64.22%. These PMU ratios are separate from
software stage spans. In the full small-token probe, packing six tiles into one
reduces Cube mean stage wait from 127.59 to 61.21 us. A single tile has zero
inter-tile overlap and still finishes faster: eliminate fragmentation and
synchronization overhead before optimizing an overlap percentage.

The 72-core probe reports 12752 software dependency/ownership checks. Long
transactions record only sequences 1..16; decode1 covers its full transaction.
Do not compare different-sized prefix overlap totals as whole-transaction gains.
Spans include completion fences and are not simultaneous DMA/MAC activity.
Do not assume cross-core or cross-die clock synchronization; retain clock-domain
and calibration evidence before interpreting an interval intersection physically.

Twenty-eight model profiles retain one CANN duration underflow. A positive,
finite duration can still be invalid: check the capture envelope and preserve
the raw row. Exclusion here affects descriptive diagnostic aggregates only;
it must not turn a failed formal profile gate into PASS. PMU `N/A` is missing
information, not zero activity.

Two 43-transaction graph slices have 344 unfused device tasks versus 88 MegaMoE
tasks, and 86 versus 26 host graph launches. Four-transaction graph batching
already existed before packing. The observed ACL graph-call time saving is
about 0.31 ms over 86 transactions; packing's incremental benefit comes from
internal scheduling, not another reduction of those graph-call counts.
DR waiting, Attention and DBO scheduling remain on the service path.

## Use this experience in a new campaign

Freeze the current parent, workspace ABI, route distribution and effective graph
configuration. Predict fewer tile epochs and less initialization/wait time, while
leaving model-level gain unknown. Reject the hypothesis on numerical/control
failure, forbidden memory reuse, missing rebuild identity, native regression or
failure of unchanged service/confirmation gates. Keep the single-expert 512-row
ablation and any negative or neutral cases. Re-run the configured paired controls
and independent confirmation before claiming an RSI promotion.

Packing needs coordinated changes to the pipeline, GMM2 output epilogue and shared
workspace header. A one-header-only campaign cannot adopt it by relaxing old
allowed files; declare the scope in a new frozen recipe. This imported knowledge
does not upgrade the repository diagnostic driver, infer support for packed tiles,
replace its old reference header or alter existing workload/acceptance defaults.

Evidence archive: `mega-moe-deep-opt-results.tar.gz`, 49906443 bytes, 1138 verified
manifest entries, SHA256
`35fcd0f9b3223030fd22992a6137bf5929809017205207cbedd38cd14c0f3afa`.
Control kernel:
`fbc03d657ca009c65b1f6073ec4f4d7ffc31b970413c07775e1ebe143130cb2e`.
Packed kernel:
`16008a55303c0027e8cfb9ab68f793e724c30b6841e3dc0b19bd6e2e87ad16ea`.
The local evidence record contains per-file identities; consuming this guidance
does not require another checkout or an external archive download.
