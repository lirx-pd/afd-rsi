"""Prototype: CAKE-style strategy-family router for afd-rsi search.

Maps CAKE's L0-L3 loop onto afd-rsi's real decision point. In afd-rsi the
controller never chooses *what to try*: the model proposes 1-3 patches and
`select_proposal` ranks them by `priority + category_successes - category_failures`
(search.py:294-305). One arm sampled per round, no difficulty adjustment, no
re-measurement after a knowledge/skill change.

Here the controller chooses the strategy family (arm) to assign each round, and
the model must propose inside it. Policies:
  fixed    today: constant ordering by raw per-category score
  exploit  difficulty-adjusted ability (item-response), argmax
  router   ability + bounded re-measurement/optimistic exploration

Non-stationarity is the interesting case: a knowledge/skill update lands
mid-run and changes which family is actually best, exactly as CAKE's
(track, skill_version) half-weight rule anticipates.
"""

import numpy as np

D = 8
W = np.array([0.51, 0.78, 0.59, 0.70, 0.44, 0.61, 0.64, 0.98])
N_ARMS = 5
FAMILIES = ["tiling", "fusion", "vectorize", "host-tiling", "sync-protocol"]

BASE = np.array([
    [0.55, 0.10, 0.10, 0.15, 0.05, -0.85, 0.05, 0.25],   # tiling
    [0.40, 0.05, 1.55, 0.05, 0.05, -0.15, 0.05, 0.15],   # fusion
    [0.50, 0.05, 0.30, 0.05, 0.05, 0.85, 0.05, 0.85],    # vectorize
    [0.45, 0.60, 0.20, 1.30, 0.05, 0.10, 0.05, 0.20],    # host-tiling
    [0.35, 0.90, 0.15, 0.15, 0.05, 0.20, 0.05, 0.40],    # sync-protocol
])
A0 = np.array([0.55, 0.40, 0.50, 0.42, 0.38])

SHIFT_AT = 70
SHIFT = np.zeros((N_ARMS, D))
SHIFT[1, 2] = 1.10      # a new fusion/scheduling knowledge note makes arm 1 strong
SHIFT[3, 1] = -1.10     # a host-tiling contract change makes arm 3 stale

BLOCK = 8               # re-measurement block: one explore per N rounds after a change


def rate(arm, phi, b, shifted):
    theta = BASE[arm] + (SHIFT[arm] if shifted else 0.0)
    return 1.0 / (1.0 + np.exp(-(A0[arm] + theta @ phi - b)))


def draw_phi(rng):
    phi = rng.integers(0, 3, size=D).astype(float)
    phi[rng.integers(0, D)] = rng.integers(2, 4)
    return phi


def noisy(p, rng):
    return float(np.clip(p + rng.normal(0.0, 0.40 * p * (1 - p) + 0.012), 0.02, 0.98))


class Ledger:
    def __init__(self):
        self.rows = [[] for _ in range(N_ARMS)]

    def add(self, arm, phi, b, y, version):
        self.rows[arm].append((phi, b, y, version))

    def count(self, arm, version=None):
        return len(self.rows[arm]) if version is None else sum(
            1 for r in self.rows[arm] if r[3] == version)

    def raw_mean(self, arm):
        rows = self.rows[arm]
        return float(np.mean([r[2] for r in rows])) if rows else 0.0

    def last_version(self, arm):
        return self.rows[arm][-1][3] if self.rows[arm] else None

    def fit(self, arm, version):
        rows = self.rows[arm]
        if not rows:
            return None
        X = np.array([[1.0, *r[0]] for r in rows])
        z = np.array([np.log(r[2] / (1 - r[2])) + r[1] for r in rows])
        w = np.array([1.0 if r[3] == version else 0.5 for r in rows])   # half weight
        if w.sum() < D + 1:
            return ("scalar", float(np.average(z, weights=w)))
        Xw = X * w[:, None]
        pen = np.eye(D + 1)
        pen[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ Xw + pen, Xw.T @ z)
        resid = z - X @ coef
        var = float((w * resid ** 2).sum() / max(1.0, w.sum() - D - 1))
        cov = var * np.linalg.inv(X.T @ Xw + pen)
        return ("full", coef, cov)

    def predict(self, arm, phi, b, version):
        model = self.fit(arm, version)
        if model is None:
            return 0.5, 1.0
        if model[0] == "scalar":
            return float(1 / (1 + np.exp(-(model[1] - b)))), 1.0
        _, coef, cov = model
        x = np.array([1.0, *phi])
        se = float(np.sqrt(max(1e-9, x @ cov @ x)))
        return float(1 / (1 + np.exp(-(float(x @ coef) - b)))), se


def pick_fixed(ledger, phi, b, version, step):
    return max(range(N_ARMS), key=lambda a: (ledger.raw_mean(a), -a)), False


def pick_exploit(ledger, phi, b, version, step):
    return max(range(N_ARMS), key=lambda a: (ledger.predict(a, phi, b, version)[0], -a)), False


def pick_router(ledger, phi, b, version, step):
    preds = {a: ledger.predict(a, phi, b, version) for a in range(N_ARMS)}
    best = max(range(N_ARMS), key=lambda a: (preds[a][0], -a))
    if step % BLOCK:
        return best, False
    # exploration slot: only stale-after-change arms, or a real statistical contender
    p_best, se_best = preds[best]
    cands = []
    for a in range(N_ARMS):
        if a == best:
            continue
        p, se = preds[a]
        stale = ledger.last_version(a) is not None and ledger.last_version(a) != version
        under = ledger.count(a, version) < 2
        contender = p + 1.96 * se > p_best + 1.96 * se_best
        if (stale or under) and p * (1 - p) >= 0.05:
            cands.append((p + 1.96 * se, -a, a))
        elif contender:
            cands.append((p + 1.96 * se, -a, a))
    if cands:
        return max(cands)[2], True
    return best, False


PICK = {"fixed": pick_fixed, "exploit": pick_exploit, "router": pick_router}


def run(policy, *, rounds=140, seed=5):
    rng = np.random.default_rng(seed)
    ledger = Ledger()
    hits = pre_regret = post_regret = 0.0
    explores = 0
    for step in range(rounds):
        shifted = step >= SHIFT_AT
        version = 1 if shifted else 0
        phi = draw_phi(rng)
        b = float(W @ phi)
        truth = {a: rate(a, phi, b, shifted) for a in range(N_ARMS)}
        oracle = max(range(N_ARMS), key=lambda a: truth[a])
        pick, explored = PICK[policy](ledger, phi, b, version, step)
        explores += explored
        hits += pick == oracle
        gap = truth[oracle] - truth[pick]
        if shifted:
            post_regret += gap
        else:
            pre_regret += gap
        ledger.add(pick, phi, b, noisy(truth[pick], rng), version)
    return {"hit": hits / rounds, "pre": pre_regret / SHIFT_AT,
            "post": post_regret / (rounds - SHIFT_AT), "explore": explores / rounds}


if __name__ == "__main__":
    seeds = (5, 11, 23, 41, 97)
    print(f"{'policy':9s} {'hit':>6s} {'regret<shift':>13s} {'regret>=shift':>14s} {'explore':>8s}")
    for policy in ("fixed", "exploit", "router"):
        vals = [run(policy, seed=s) for s in seeds]
        print(f"{policy:9s} "
              f"{np.mean([v['hit'] for v in vals]):6.3f} "
              f"{np.mean([v['pre'] for v in vals]):13.4f} "
              f"{np.mean([v['post'] for v in vals]):14.4f} "
              f"{np.mean([v['explore'] for v in vals]):8.3f}")
    print()
    print("per seed (hit / post-shift regret):")
    for s in seeds:
        row = {p: run(p, seed=s) for p in ("fixed", "exploit", "router")}
        print(f"  seed {s:3d}  " + "  ".join(
            f"{p}={row[p]['hit']:.3f}/{row[p]['post']:.4f}" for p in row))
