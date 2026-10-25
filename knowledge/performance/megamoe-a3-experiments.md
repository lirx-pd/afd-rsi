# Reviewed A3 MegaMoe experiments, 2026-10-24

These are manually reviewed prior measurements, not compatible dynamic experience
records or proof that a future candidate will improve performance. Do not repeat
the isolated patches below. Combine them only with a new falsifiable reason.

Baseline source inventory SHA256:
`4449945484c15ae54e10ef6388cb17029a7034dc7ed30a89eff57a33b0cea582`.
A3, H4096/I2048, seven-case synthetic native A/F transaction, FFN FULL, three
paired unprofiled repetitions. Gates: native numerics, real V4 Flash W4A8
service with 30 GSM8K strict-match samples >=0.9, independent profiling, then
>=2% native median gain and fresh confirmation. HTTP prefill TTFT is auxiliary.

1. Inactive OutputTile lanes: return before `pipe.Reset()` when
   `GetBlockIdx() - Cam::MEGA_COMM_AIVS >= rows`, preserving the caller's
   unconditional OUTPUT arrival. Candidate inventory
   `c1343eee1dcb375cd8b1f180ea97e72de6cf699f7e6eeb13d0faf01b36a96e42`.
   All functional gates passed; baseline/candidate GSM8K 27/30 and 28/30.
   Native baseline means [396.9921406,396.9385720,401.0785688] us;
   candidate [395.0950000,382.1114283,392.6653575] us.
   Framework median gain 1.08989138%, FAIL. Diagnostic OUTPUT summed core cycles
   fell 48.14%/71.78% on FFN ranks 1/2; this did not establish whole-transaction
   gain. No confirmation or promotion.

2. ComputePipeline reorder: G2(current) before G1(next), all waits/arrivals and
   ring slot reuse retained. Functional gates passed, both GSM8K 28/30.
   Native baseline means [395.0957156,391.9503582,390.6478571] us;
   candidate [395.0389272,387.1232109,389.8953575] us.
   Framework median gain 0.5243012888%, FAIL. Device core-48 sequence records
   confirm the actual order changed. Prefill mean 113.651 ->115.186 ms regressed.
   No confirmation or promotion.

Diagnostic per-core clocks have no cross-core alignment. Summed phase cycles
are neither the critical path nor PMU activity. Inspect producer/consumer helper
source before changing cache invalidation or quantization. Do not infer
coherence from numerical success on a previous unrelated patch.


3. Remove PrepareTile's whole-scalar-cache invalidation, keeping ComputeStage
   and activation invalidations and all synchronization. Candidate inventory
   `3b957ea08f3357b598590964c6567ae88cc06a86707314cc580f22109e82815b`.
   Functional/FULL/service/profile gates passed; GSM8K baseline 28/30, candidate
   27/30. Native means baseline [413.6650014,397.6328559,400.9464293] us,
   candidate [391.5953572,408.0967842,400.5649997] us. Gain 0.0951323%, FAIL.
   PRE2 summed cycles fell on both FFN ranks; whole-transaction gain did not
   qualify. Prefill mean 112.899 ->114.296 ms. No confirmation or promotion.

4. Cube baseN=192, coordinated in SetShape, SetFixSplit and MegaGmmParams,
   for both GMMs. Candidate inventory
   `e0b86254d202ece09f4396b4163e2fd43aec43c137ff399c2e7b0156962b5018`.
   Screen gain 5.7775120%, PASS. All seven case summaries improved 3.20-6.67%.
   Fresh full confirmation gain 1.5335875%, FAIL; no promotion. Functional,
   service and independent profiles passed in both attempts. Screen GSM8K both
   27/30; confirmation both 28/30. Prefill mean worsened 113.372 ->116.344 ms
   in screening but improved 114.743 ->113.225 ms in confirmation, so there
   is no established model prefill improvement. Diagnostic AIC 64..69 performed
   real GMM work after the tiling change; no uniform per-core speedup or hardware
   bound is established. A favorable screen never overrides failed confirmation.

5. Skip PipelineFlags::Arrive's root-id0 private arrival store, whose lane is
   not polled. Preserve member/leader waits, fences and final global publication.
   Candidate inventory
   `cd601b92d825cbbbc3b00c5d8fdc207ce67b7f6195e6dcc2182d09c92120c393`.
   Functional/FULL/service/profile gates passed, GSM8K both 27/30. Native means
   baseline [392.3482134,413.8282173,396.2382131] us; candidate
   [389.6478569,394.9953568,402.1328565] us. Gain 0.3136639%, FAIL. Prefill
   mean 114.227 ->113.257 ms is auxiliary only. No confirmation or promotion.

All candidates above were evaluated independently against the same original
source. Rejected patches are not retained in the production checkout. An
unselected active-vector-participation proposal has no device result and must
not be described as a measured optimization. New combinations need a distinct
source-backed hypothesis and unchanged independent acceptance gates.


6. Refill replacement input after the old G2 global completion, before waiting
   for old OUTPUT/Send. Protect reused output y with SENT(sequence-RING_SLOTS),
   including terminal tiles. Candidate inventory
   `64cd64ce7612a7c49b4dda3e6666f5c45c2c17c4e10509f3004f4f44a6c4350b`.
   One pipeline header, no arithmetic, quantization, tiling or ABI change.
   Screen 3.6713127% PASS; fresh complete confirmation 2.5219115% PASS, promoted.
   All native/FULL, service and independent profiling gates passed twice.
   Candidate GSM8K 27/30 in screening and 28/30 in confirmation. Four non-sparse
   cases improved 5.28-9.43% in screening and 3.16-6.25% in confirmation. Sparse
   case medians changed -0.57%, -0.86%, +1.47% in confirmation; this is not a
   universal improvement. Prefill results did not establish model end-to-end gain.
   Matched later-tile input-ready waits fell sharply on both FFN ranks in both
   separate diagnostic runs; AIC48 PRE1 wait totals approximately halved. These
   own-core spans include setup/waits, not PMU utilization or cross-core overlap.
   Refill input and output completion have distinct lifetimes: preserve G2 input
   release, y/SENT output protection and all arrival visibility/participant rules.
   This only begins to reduce pipeline gaps; it does not prove continuous Cube
   activity. Future combinations require new hypotheses against this new parent.
