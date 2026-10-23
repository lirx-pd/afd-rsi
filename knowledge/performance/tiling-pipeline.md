# Tiling and pipeline optimization procedure

Scope: A3 MegaMoe. Start from measured headroom and the real call path.

1. Tiling model: record each GMM M/N/K, dtype/layout, block/tile sizes, core
   assignment, buffer lifetimes/bytes and tails. Obtain actual host tiling data.
2. Inter-device pipeline: if communication is present, collect semantically
   valid isolated communication/compute controls before choosing chunking.
   Disabling a producer must not leave its consumers waiting forever. AFD
   disaggregated roles have no automatic symmetric-EP local-expert fast path.
3. Inter-core pipeline: map producers, consumers, flags and barrier participants;
   estimate extra traffic and scratch before removing synchronization.
4. Intra-core pipeline: identify Scalar/Vector/Cube/MTE limiting phases, paired
   events and buffer reuse. Correct the tiling model with the measured evidence.

For small M, test tile/packing/launch overhead hypotheses. For large M, inspect
GMM efficiency, redundant GM traffic, pipeline bubbles and imbalance. Evaluate
retaining Cube and changing its tiling before proposing an AIV engine rewrite.
Do not transfer arch35-only instruction tricks to A3.

Each candidate names its hypothesis, supporting evidence, expected metric,
falsifier, scope and caller/build/test impact. First attempt the smallest
complete change. A local stage improvement is accepted only when the declared
whole-work metric and regression gates also pass. Preserve negative results.
