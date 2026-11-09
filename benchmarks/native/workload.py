"""Deterministic routes shared by dispatch and expected FFN metadata."""


def routing_ids(tokens, experts, sparse, *, balanced=False):
    if sparse:
        return [[0, 1] for _ in range(tokens)]
    if balanced:
        # Exactly tokens rows per FFN, including odd cohort boundaries.
        return [[row % experts, experts + row % experts] for row in range(tokens)]
    return [[(2 * row) % (2 * experts), (2 * row + 1) % (2 * experts)] for row in range(tokens)]
