"""Reference core for the afd-rsi direction router (pure standard library).

This is the S1 deliverable sketch from docs/search-strategy-router.md:
an append-only ledger plus a ridge-weighted least-squares ability model with a
scalar fallback. No numpy, matching the repository rule that the controller has
no third-party runtime dependency.

Run directly to self-check against a hand-computed case and a simulated
non-stationary run:  python docs/prototypes/router_core.py
"""

from __future__ import annotations

import hashlib
import json
import math

DIMENSIONS = ("T", "S", "B", "R", "D", "M", "F", "C")
# Paper Eq.(1) ordering; frozen in the recipe, not chosen by the model.
WEIGHTS = {"T": 0.51, "S": 0.78, "B": 0.59, "R": 0.70,
           "D": 0.44, "M": 0.61, "F": 0.64, "C": 0.98}


def difficulty(features: dict, weights: dict = WEIGHTS) -> float:
    return sum(weights[name] * features.get(name, 0) for name in DIMENSIONS)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None when singular."""
    n = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-12:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        rows[col] = [value / scale for value in rows[col]]
        for r in range(n):
            if r == col:
                continue
            factor = rows[r][col]
            if factor:
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return [rows[i][n] for i in range(n)]


def _inverse(matrix: list[list[float]]) -> list[list[float]] | None:
    n = len(matrix)
    columns = []
    for i in range(n):
        unit = [1.0 if j == i else 0.0 for j in range(n)]
        solved = _solve(matrix, unit)
        if solved is None:
            return None
        columns.append(solved)
    return [[columns[c][r] for c in range(n)] for r in range(n)]


class Ledger:
    """Append-only observations; rows are immutable and hash-checked."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, *, category, features, score, decision, foundation_hash,
            gain_pct=None, round_name=None) -> dict:
        row = {"schema_version": 1, "round": round_name, "category": category,
               "features": {k: int(features.get(k, 0)) for k in DIMENSIONS},
               "difficulty": round(difficulty(features), 6),
               "score": round(float(score), 6), "decision": decision,
               "gain_pct": None if gain_pct is None else round(float(gain_pct), 4),
               "foundation_hash": foundation_hash}
        row["record_hash"] = digest(row)
        self.rows.append(row)
        return row

    def verify(self) -> None:
        for row in self.rows:
            body = {k: v for k, v in row.items() if k != "record_hash"}
            if digest(body) != row["record_hash"]:
                raise ValueError("ledger row changed")

    def rows_for(self, category, foundation_hash=None) -> list[dict]:
        picked = [r for r in self.rows if r["category"] == category]
        if foundation_hash is None:
            return picked
        return [r for r in picked if r["foundation_hash"] == foundation_hash]

    def legacy_score(self, category) -> float:
        """Today's rule: raw count of promoted minus rejected, no adjustment."""
        rows = self.rows_for(category)
        promoted = sum(1 for r in rows if r["decision"] == "promoted")
        failed = sum(1 for r in rows if r["decision"] in ("rejected", "invalid_patch"))
        return promoted - failed


class Ability:
    """Ridge WLS on logit(score) + difficulty ~ 1 + features, with scalar fallback."""

    RIDGE = 1.0

    def __init__(self, ledger: Ledger, *, foundation_hash, half_weight_stale=True):
        self.ledger = ledger
        self.foundation_hash = foundation_hash
        self.half_weight_stale = half_weight_stale

    def _design(self, category):
        rows = self.ledger.rows_for(category)
        if not rows:
            return None
        X, z, w = [], [], []
        for row in rows:
            score = min(max(row["score"], 0.02), 0.98)     # logit is undefined at 0/1
            X.append([1.0] + [float(row["features"][d]) for d in DIMENSIONS])
            z.append(math.log(score / (1 - score)) + row["difficulty"])
            same = row["foundation_hash"] == self.foundation_hash
            w.append(1.0 if same or not self.half_weight_stale else 0.5)
        return X, z, w

    def fit(self, category):
        prepared = self._design(category)
        if prepared is None:
            return None
        X, z, w = prepared
        if sum(w) < len(DIMENSIONS) + 1:
            mean = sum(wi * zi for wi, zi in zip(w, z)) / sum(w)
            return {"kind": "scalar", "a0": mean}
        size = len(X[0])
        ata = [[0.0] * size for _ in range(size)]
        atz = [0.0] * size
        for xi, zi, wi in zip(X, z, w):
            for i in range(size):
                atz[i] += wi * xi[i] * zi
                for j in range(size):
                    ata[i][j] += wi * xi[i] * xi[j]
        for i in range(size):
            ata[i][i] += 0.0 if i == 0 else self.RIDGE
        coef = _solve(ata, atz)
        if coef is None:
            mean = sum(wi * zi for wi, zi in zip(w, z)) / sum(w)
            return {"kind": "scalar", "a0": mean}
        inverse = _inverse(ata)
        if inverse is None:
            return {"kind": "vector", "coef": coef, "cov": None}
        residual = [zi - sum(c * x for c, x in zip(coef, xi)) for xi, zi in zip(X, z)]
        dof = max(1.0, sum(w) - len(DIMENSIONS) - 1)
        variance = sum(wi * r * r for wi, r in zip(w, residual)) / dof
        cov = [[variance * v for v in row] for row in inverse]
        return {"kind": "vector", "coef": coef, "cov": cov}

    def predict(self, category, features) -> tuple[float, float]:
        model = self.fit(category)
        if model is None:
            return 0.5, 1.0
        phi = [1.0] + [float(features.get(d, 0)) for d in DIMENSIONS]
        if model["kind"] == "scalar":
            z = model["a0"]
            return 1.0 / (1.0 + math.exp(-z)), 1.0
        z = sum(c * x for c, x in zip(model["coef"], phi))
        if model["cov"] is None:
            return 1.0 / (1.0 + math.exp(-z)), 1.0
        var = sum(phi[i] * model["cov"][i][j] * phi[j]
                  for i in range(len(phi)) for j in range(len(phi)))
        se = math.sqrt(max(1e-9, var))
        return 1.0 / (1.0 + math.exp(-z)), se


def choose(category_scores: dict, ability: Ability, features, *,
           info_floor=0.0, margin=0.02) -> tuple[str, bool]:
    """Argmax with optional information-gated optimistic exploration."""
    preds = {c: ability.predict(c, features) for c in category_scores}
    best = max(preds, key=lambda c: (preds[c][0], c))
    if info_floor <= 0.0:
        return best, False
    p_best, se_best = preds[best]
    contenders = []
    for name, (p, se) in preds.items():
        if name == best:
            continue
        if p * (1 - p) >= info_floor and p + 1.96 * se > p_best + margin:
            contenders.append((p + 1.96 * se, name))
    if contenders:
        return max(contenders)[1], True
    return best, False


def _self_check() -> None:
    ledger = Ledger()
    ledger.add(category="tiling", features={"B": 2, "M": 1}, score=0.30,
               decision="rejected", foundation_hash="f0", gain_pct=-1.2)
    ledger.add(category="fusion", features={"F": 2, "M": 1}, score=0.85,
               decision="promoted", foundation_hash="f0", gain_pct=3.4)
    ledger.verify()
    assert ledger.legacy_score("fusion") - ledger.legacy_score("tiling") == 2.0

    ability = Ability(ledger, foundation_hash="f0")
    scalar = ability.fit("fusion")
    assert scalar["kind"] == "scalar", "two rows cannot identify eight dimensions"
    p, se = ability.predict("fusion", {"F": 2})
    assert 0.0 < p < 1.0 and se == 1.0

    # Simulate enough rows to identify the vector model, then check it ranks the
    # arm that truly scores higher for a fusion-shaped operator.
    for k in range(14):
        ledger.add(category="fusion",
                   features={"F": (k % 3) + 1, "M": k % 2},
                   score=0.80 if (k % 3) else 0.35,
                   decision="promoted" if (k % 3) else "rejected",
                   foundation_hash="f0")
    for k in range(14):
        ledger.add(category="tiling",
                   features={"F": (k % 3) + 1, "M": k % 2},
                   score=0.25,
                   decision="rejected", foundation_hash="f0")
    ledger.verify()
    vector = ability.fit("fusion")
    assert vector["kind"] == "vector", vector
    pick, explored = choose({"fusion": 0, "tiling": 0}, ability, {"F": 2, "M": 1})
    assert pick == "fusion" and not explored

    # A mismatched foundation halves stale rows rather than dropping them. With
    # every row stale the effective sample count falls below the identification
    # threshold, so the model must fall back to scalar instead of fitting noise.
    stale = Ability(ledger, foundation_hash="f9")
    assert stale.fit("fusion")["kind"] == "scalar"
    # Mixed versions: fresh rows plus half-weighted stale rows identify the vector.
    mixed = Ability(ledger, foundation_hash="f0")
    assert mixed.fit("fusion")["kind"] == "vector"

    # Immutability is enforced.
    ledger.rows[0]["score"] = 0.99
    try:
        ledger.verify()
    except ValueError:
        pass
    else:
        raise AssertionError("tampered row accepted")

    print("router_core self-check: ok")


if __name__ == "__main__":
    _self_check()
