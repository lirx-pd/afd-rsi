"""Difficulty-adjusted routing over frozen campaign evidence.

The controller has one real decision per round: which of the proposed change
families to spend the next expensive attempt on. Before this module that
decision was `priority + category_successes - category_failures`
(``search.select_proposal``), a raw count that punishes hard categories for
ever failing and rewards categories that merely happened to be tried early.

This module estimates per-category ability with a continuous item-response
model over an eight-dimensional patch profile, so a family is not judged by how
often it was tried but by how well it performs on changes of this shape, at
this difficulty. A scalar fallback keeps the estimate honest while there are
too few rows to identify eight dimensions: with a three-round recipe the model
never leaves the fallback, which is the intended behaviour.

Design notes:

* ``score`` is continuous in the measured gain, not a PASS/FAIL flag. A hard but
  rewarding direction can therefore outrank an easy one; a Pass-only signal
  would bias the router toward trivial edits, because difficulty is subtracted
  from ability exactly as in the paper's Eq.(3).
* The ledger is append-only and every row is hash-checked. A row produced under
  another foundation keeps half weight, so a knowledge or skill update ages old
  evidence instead of silently trusting it.
* Nothing here decides acceptance. It only orders proposals; promotion still
  requires the two independent PASS attempts enforced by ``search``.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

DIMENSIONS = ("T", "S", "B", "R", "D", "M", "F", "C")

# Frozen default difficulty weights, same ordering and magnitude as CAKE Eq.(1):
# tiling, scheduling depth, blocking, reduction, dtype spread, data movement,
# fusion breadth, cube intensity. A recipe may override them, but they are part
# of the plan fingerprint, so changing them invalidates cross-campaign reuse.
DEFAULT_WEIGHTS = {"T": 0.51, "S": 0.78, "B": 0.59, "R": 0.70,
                   "D": 0.44, "M": 0.61, "F": 0.64, "C": 0.98}

# Patch-text evidence per dimension. Counted on added lines only: removed lines
# describe the old implementation, not the direction being proposed.
FEATURE_PATTERNS = {
    "T": (r"tile", r"tiling", r"basem\b", r"basen\b"),
    "S": (r"pipeline", r"\bqueue", r"setflag", r"waitflag", r"barrier"),
    "B": (r"\bblock", r"basek\b", r"blockidx"),
    "R": (r"reduce", r"atomic", r"accum", r"\bsum\b"),
    "D": (r"w4a8", r"int8", r"fp16", r"bf16", r"quant", r"\bcast\b"),
    "M": (r"datacopy", r"stride", r"\boffset", r"address", r"gm_to_ub", r"ub_to_gm"),
    "C": (r"matmul", r"mmad", r"\bcube\b"),
}
FUSION_MARKERS = (r"dispatch", r"combine", r"\bffn\b", r"\battn\b", r"\bmoe\b", r"\bgate\b")

# A single dimension saturates quickly; the paper's features are small integers.
FEATURE_CAP = 3
RIDGE = 1.0
INFO_FLOOR_DEFAULT = 0.0          # exploration off unless a recipe opts in
MARGIN_DEFAULT = 0.02
EXPLORE_FRACTION_DEFAULT = 0.25


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def difficulty(features: dict, weights: dict = DEFAULT_WEIGHTS) -> float:
    """Scalar operator difficulty b = w . phi."""
    return round(sum(weights[name] * features.get(name, 0) for name in DIMENSIONS), 6)


def patch_features(patch: str) -> dict:
    """Eight-dimensional profile of what a proposed patch touches."""
    added = [line[1:] for line in patch.splitlines()
             if line.startswith("+") and not line.startswith("+++")]
    text = "\n".join(added)
    features = {}
    for name in DIMENSIONS:
        if name == "F":
            features[name] = min(FEATURE_CAP, sum(
                1 for marker in FUSION_MARKERS if re.search(marker, text, re.IGNORECASE)))
            continue
        hits = 0
        for pattern in FEATURE_PATTERNS[name]:
            hits += len(re.findall(pattern, text, re.IGNORECASE))
        features[name] = min(FEATURE_CAP, hits)
    return features


def normalized_score(verdict: dict, min_gain_pct: float) -> float:
    """Map a measured gain onto (0, 1) without letting a single round saturate.

    ``gain`` equal to the frozen threshold sits near 0.73, zero gain at 0.5, and
    a loss below it. INCONCLUSIVE rounds carry no gain and stay neutral.
    """
    gain = verdict.get("gain_pct")
    if gain is None or not math.isfinite(gain):
        return 0.5
    scale = max(float(min_gain_pct), 0.5)
    value = 0.5 + 0.5 * math.tanh(gain / (2 * scale))
    return round(min(max(value, 0.05), 0.95), 6)


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None when singular."""
    size = len(vector)
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(rows[r][column]))
        if abs(rows[pivot][column]) < 1e-12:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for r in range(size):
            if r == column:
                continue
            factor = rows[r][column]
            if factor:
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[column])]
    return [rows[i][size] for i in range(size)]


def _inverse(matrix: list[list[float]]) -> list[list[float]] | None:
    size = len(matrix)
    columns = []
    for i in range(size):
        solved = _solve(matrix, [1.0 if j == i else 0.0 for j in range(size)])
        if solved is None:
            return None
        columns.append(solved)
    return [[columns[c][r] for c in range(size)] for r in range(size)]


class Ledger:
    """Append-only observations, one row per evaluated candidate."""

    SCHEMA = 1

    def __init__(self, path: Path):
        self.path = path
        self.rows: list[dict] = []

    def load(self) -> "Ledger":
        if not self.path.is_file():
            return self
        self.rows = []
        for number, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"ledger line {number} is not JSON: {error}") from error
            self._verify_row(row, number)
            self.rows.append(row)
        return self

    def _verify_row(self, row: dict, number: int) -> None:
        if row.get("schema_version") != self.SCHEMA:
            raise ValueError(f"unsupported ledger schema on line {number}")
        body = {key: value for key, value in row.items() if key != "record_hash"}
        if digest(body) != row.get("record_hash"):
            raise ValueError(f"ledger row {number} changed")

    def verify(self) -> None:
        for number, row in enumerate(self.rows, 1):
            self._verify_row(row, number)

    def add(self, *, category: str, features: dict, score: float, decision: str,
            foundation_hash: str, round_name: str | None = None,
            gain_pct: float | None = None, stage: str | None = None) -> dict:
        row = {"schema_version": self.SCHEMA, "round": round_name, "category": category,
               "features": {name: int(features.get(name, 0)) for name in DIMENSIONS},
               "difficulty": difficulty(features),
               "score": round(float(score), 6), "decision": decision,
               "stage": stage,
               "gain_pct": None if gain_pct is None else round(float(gain_pct), 4),
               "foundation_hash": foundation_hash}
        row["record_hash"] = digest(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from .evidence import write_text
        # Atomic replacement keeps a killed append from leaving half a JSON record.
        previous = self.path.read_text() if self.path.exists() else ''
        if previous and not previous.endswith('\n'):
            previous += '\n'
        write_text(self.path, previous + json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        self.rows.append(row)
        return row

    def categories(self) -> list[str]:
        return sorted({row["category"] for row in self.rows})

    def legacy_score(self, category: str) -> float:
        """The pre-router rule, kept for fallback and for tests."""
        rows = [row for row in self.rows if row["category"] == category]
        promoted = sum(1 for row in rows if row["decision"] == "promoted")
        failed = sum(1 for row in rows if row["decision"] in ("rejected", "invalid_patch"))
        return promoted - failed


class Ability:
    """Ridge weighted least squares on logit(score) + difficulty ~ 1 + phi."""

    RIDGE = RIDGE

    def __init__(self, ledger: Ledger, *, foundation_hash: str, half_weight_stale: bool = True):
        self.ledger = ledger
        self.foundation_hash = foundation_hash
        self.half_weight_stale = half_weight_stale

    def _prepare(self, category: str):
        rows = [row for row in self.ledger.rows if row["category"] == category]
        if not rows:
            return None
        design, target, weights = [], [], []
        for row in rows:
            # logit is undefined at the clipping bounds, so clamp first.
            score = min(max(float(row["score"]), 0.02), 0.98)
            design.append([1.0] + [float(row["features"][name]) for name in DIMENSIONS])
            target.append(math.log(score / (1 - score)) + float(row["difficulty"]))
            same = row["foundation_hash"] == self.foundation_hash
            weights.append(1.0 if same or not self.half_weight_stale else 0.5)
        return design, target, weights

    def fit(self, category: str) -> dict | None:
        prepared = self._prepare(category)
        if prepared is None:
            return None
        design, target, weights = prepared
        total = sum(weights)
        if total < len(DIMENSIONS) + 1:
            return {"kind": "scalar",
                    "a0": sum(w * t for w, t in zip(weights, target)) / total}
        width = len(DIMENSIONS) + 1
        gram = [[0.0] * width for _ in range(width)]
        moment = [0.0] * width
        for row, value, weight in zip(design, target, weights):
            for i in range(width):
                moment[i] += weight * row[i] * value
                for j in range(width):
                    gram[i][j] += weight * row[i] * row[j]
        for i in range(1, width):        # the intercept stays unpenalized
            gram[i][i] += self.RIDGE
        coef = _solve(gram, moment)
        if coef is None:
            return {"kind": "scalar",
                    "a0": sum(w * t for w, t in zip(weights, target)) / total}
        inverse = _inverse(gram)
        if inverse is None:
            return {"kind": "vector", "coef": coef, "cov": None}
        residual = [value - sum(c * x for c, x in zip(coef, row))
                    for row, value in zip(design, target)]
        dof = max(1.0, total - len(DIMENSIONS) - 1)
        variance = sum(w * r * r for w, r in zip(weights, residual)) / dof
        return {"kind": "vector", "coef": coef,
                "cov": [[variance * cell for cell in row] for row in inverse]}

    def predict(self, category: str, features: dict) -> tuple[float, float]:
        """Return (expected score, standard error); uninformative when empty."""
        model = self.fit(category)
        if model is None:
            return 0.5, 1.0
        phi = [1.0] + [float(features.get(name, 0)) for name in DIMENSIONS]
        if model["kind"] == "scalar":
            return _sigmoid(model["a0"] - difficulty(features)), 1.0
        linear = sum(c * x for c, x in zip(model["coef"], phi))
        value = _sigmoid(linear - difficulty(features))
        if model["cov"] is None:
            return value, 1.0
        variance = sum(phi[i] * model["cov"][i][j] * phi[j]
                       for i in range(len(phi)) for j in range(len(phi)))
        return value, math.sqrt(max(1e-9, variance))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


class Router:
    """Order proposals by estimated ability; optionally name a category to explore."""

    def __init__(self, settings: dict, ledger: Ledger, *, foundation_hash: str):
        self.settings = settings or {}
        self.ledger = ledger
        self.foundation_hash = foundation_hash
        self.weights = (self.settings.get("features") or {}).get("weights") or DEFAULT_WEIGHTS
        self.mode = self.settings.get("router", "auto")
        # A campaign frozen before routing existed has no declared settings and
        # no ledger. It keeps the historical selection rule rather than ranking
        # every category from an empty, uninformative model.
        self.declared = "router" in self.settings
        self.info_floor = float(self.settings.get("info_floor", INFO_FLOOR_DEFAULT))
        self.margin = float(self.settings.get("margin", MARGIN_DEFAULT))
        self.explore_fraction = float(
            self.settings.get("explore_max_fraction", EXPLORE_FRACTION_DEFAULT))
        self.ability = Ability(ledger, foundation_hash=foundation_hash)
        # Resuming a campaign must not hand exploration a fresh budget, so the
        # round counter continues from what the ledger already recorded.
        self.rounds = len({row.get("round") for row in ledger.rows if row.get("round")})
        self.explored_rounds = 0

    @property
    def enabled(self) -> bool:
        return self.declared and self.mode != "off"

    def features(self, patch: str) -> dict:
        return patch_features(patch)

    def predict(self, category: str, features: dict) -> tuple[float, float]:
        return self.ability.predict(category, features)

    def _explore_allowed(self) -> bool:
        if self.info_floor <= 0.0 or self.explore_fraction <= 0.0:
            return False
        # Every round is expensive, so exploration may never take the majority.
        return self.explored_rounds < max(1, int(self.rounds * self.explore_fraction))

    def rank(self, items: list[dict]) -> dict:
        """Order candidate items and record why.

        ``items`` hold ``{"category", "features", "priority", "index"}``. Within a
        category the highest priority wins; categories are then ordered by
        estimated ability, with at most one information-gated exploration slot.
        """
        best_per_category: dict[str, dict] = {}
        for item in items:
            current = best_per_category.get(item["category"])
            key = (item["priority"], -item["index"])
            if current is None or key > (current["priority"], -current["index"]):
                best_per_category[item["category"]] = item
        if not best_per_category:
            raise ValueError("no candidate to route")
        scores = {}
        for category, item in best_per_category.items():
            value, error = self.predict(category, item["features"])
            scores[category] = {"expected": round(value, 6), "se": round(error, 6),
                                "difficulty": difficulty(item["features"], self.weights),
                                "rows": sum(1 for row in self.ledger.rows
                                            if row["category"] == category)}
        order = sorted(scores, key=lambda name: (scores[name]["expected"], name), reverse=True)
        selected = order[0]
        explored = False
        if self._explore_allowed() and len(order) > 1:
            best = scores[selected]["expected"]
            contenders = [(scores[name]["expected"] + 1.96 * scores[name]["se"], name)
                          for name in order[1:]
                          if scores[name]["expected"] * (1 - scores[name]["expected"]) >= self.info_floor
                          and scores[name]["expected"] + 1.96 * scores[name]["se"] > best + self.margin]
            if contenders:
                selected = max(contenders)[1]
                explored = True
        self.rounds += 1
        if explored:
            self.explored_rounds += 1
        return {"selected": selected, "explored": explored, "scores": scores,
                "order": order, "index": best_per_category[selected]["index"],
                "kind": self._kind(), "explore_budget": self.explore_fraction}

    def _kind(self) -> str:
        """Which estimator is in force, by effective sample weight per category."""
        if not self.ledger.rows:
            return "empty"
        for name in self.ledger.categories():
            total = sum(1.0 if row["foundation_hash"] == self.foundation_hash else 0.5
                        for row in self.ledger.rows if row["category"] == name)
            if total >= len(DIMENSIONS) + 1:
                return "vector"
        return "scalar"

    def kind(self) -> str:
        return "off" if not self.enabled else self._kind()

    def directive(self) -> dict | None:
        """Category to request next round when its ability is untested or stale.

        The controller cannot author patches, so exploration is expressed as a
        preference in the next request rather than a forced choice.
        """
        categories = self.ledger.categories()
        if not categories:
            return None
        fresh = [name for name in categories
                 if not any(row["category"] == name and row["foundation_hash"] == self.foundation_hash
                            for row in self.ledger.rows)]
        if not fresh:
            return None
        return {"preferred_category": fresh[0],
                "reason": "no evidence for this change family under the frozen foundation",
                "categories": fresh}
