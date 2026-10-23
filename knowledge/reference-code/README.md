# A3 workspace reference

The [workspace header](mega_moe_workspace.h) is an actual content snapshot of
the AFD MegaMoe workspace definition, including tile/alignment constants and
stage scratch formulas. Its copyright and license are retained in [LICENSE](LICENSE).
Exact source revision, worktree status boundary and content hash are recorded
in the repository's knowledge provenance record.

Use it to review caller propagation when experimenting with tile sizes. It is
not a production header dependency of the controller and is not a claim that
the target's current ABI matches this snapshot. Before a target experiment,
compare its config indices, layout fields, capacity and scratch calculations.
Do not change only MEGA_TILE_ROWS without matching tiling, strides and tails.
