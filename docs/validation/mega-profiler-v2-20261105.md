# Pull MegaMoE profiler v2: A3 diagnostic

The `20261105-0030` baseline was replayed with the same seven-case W4A8
serialized transaction fixture on iTASK `afd-v032-mega-20261104b` (910A3).
Production source hash: `49e47532347a4653d4858046a52624799303e2b99e94ef3f938721569da3e4e8`.
The instrumented source was built separately; both diagnostic FFN ranks completed
with the reference outputs, and the build and run process receipts have exit code 0.

| Selected-core cycle sums | FFN rank 1 | FFN rank 2 |
| --- | ---: | ---: |
| `round_prepare` | 12,653,861 | 12,375,183 |
| `rx_select` | 12,650,405 (99.97% of prepare) | 12,371,308 (99.97%) |
| `rx_header_validation` (inside select) | 220 | 687 |
| `rx_metadata` | 517 | 958 |
| `rx_commit_check` | 220 | 418 |
| `rx_group_plan` | 211 | 202 |
| `rx_publish` | 93 | 88 |
| `rx_layer_map` | 298 | 272 |

`Transport::Select()` loops over candidate Attention sources until the expected
COMMIT is visible. Its residual after header validation is 12,650,185 and
12,370,621 cycles respectively. Thus the previous `round_prepare` hotspot is
overwhelmingly source selection/polling, not metadata processing or group planning.
Rank 1's continuing round spent only 25 cycles in `rx_select`; the first round
dominated the sum. These spans include state reads and trace overhead, so the
residual is not a calibrated pure-wait duration.

The same diagnostic separates `pull_rows`, GMM1, activation, GMM2, local output,
remote TX, output synchronization, and input waits. GMM1 input waits account for
82.90% / 88.82% of GMM1 selected-core cycle sums, and GMM2 activation waits
for 89.04% / 84.86% of GMM2. Local output is 4.04% / 2.43% of combine; remote
TX is 25.17% / 28.30%; output synchronization is 21.34% / 20.31%. These
intervals are nested and overlap other cores. They are diagnostic call spans,
not additive kernel latency or a model-level speedup. TX `Pump` may enqueue
asynchronous DMA; `Finalize` drains it.

Raw result, complete device traces, rank events and process receipts are under
`runs/megamoe-codex-20261105-0030/diagnostics/profiler-v2/` (local run artifact,
ignored by Git). The transient iTASK copy was used only for this validation.
