"""Sensitivity: when does CAKE-style exploration actually earn its budget?

Three regimes, same ledger/model as afd-rsi-search-router-sim.py:
  separable  arms clearly different per feature dimension (the paper's Fig.3 case)
  near_tied  arms nearly indistinguishable -> raw-mean ordering is noise
  drift      separable + a knowledge update mid-run changes which arm is best

Reports mean and worst-case (max over seeds) regret, so tail risk is visible,
which is what matters when one round costs a build + AB/BA + service gate.
"""

import numpy as np

D = 8
W = np.array([0.51, 0.78, 0.59, 0.70, 0.44, 0.61, 0.64, 0.98])
N = 5
A0 = np.array([0.55, 0.40, 0.50, 0.42, 0.38])

SEPARABLE = np.array([
    [0.55, 0.10, 0.10, 0.15, 0.05, -0.85, 0.05, 0.25],
    [0.40, 0.05, 1.55, 0.05, 0.05, -0.15, 0.05, 0.15],
    [0.50, 0.05, 0.30, 0.05, 0.05, 0.85, 0.05, 0.85],
    [0.45, 0.60, 0.20, 1.30, 0.05, 0.10, 0.05, 0.20],
    [0.35, 0.90, 0.15, 0.15, 0.05, 0.20, 0.05, 0.40],
])
CLOSE = SEPARABLE * 0.18 + 0.18

SHIFT = np.zeros((N, D))
SHIFT[1, 2] = 1.10
SHIFT[3, 1] = -1.10
SHIFT_AT = 70
BLOCK = 8


def make_rate(theta):
    def rate(arm, phi, b, shifted):
        t = theta[arm] + (SHIFT[arm] if shifted else 0.0)
        return 1.0 / (1.0 + np.exp(-(A0[arm] + t @ phi - b)))
    return rate


def draw_phi(rng):
    phi = rng.integers(0, 3, size=D).astype(float)
    phi[rng.integers(0, D)] = rng.integers(2, 4)
    return phi


def noisy(p, rng):
    return float(np.clip(p + rng.normal(0.0, 0.40 * p * (1 - p) + 0.012), 0.02, 0.98))


class Ledger:
    def __init__(self):
        self.rows = [[] for _ in range(N)]

    def add(self, a, phi, b, y, v):
        self.rows[a].append((phi, b, y, v))

    def count(self, a, v=None):
        return len(self.rows[a]) if v is None else sum(1 for r in self.rows[a] if r[3] == v)

    def mean(self, a):
        return float(np.mean([r[2] for r in self.rows[a]])) if self.rows[a] else 0.0

    def last(self, a):
        return self.rows[a][-1][3] if self.rows[a] else None

    def fit(self, a, v):
        rows = self.rows[a]
        if not rows:
            return None
        X = np.array([[1.0, *r[0]] for r in rows])
        z = np.array([np.log(r[2] / (1 - r[2])) + r[1] for r in rows])
        w = np.array([1.0 if r[3] == v else 0.5 for r in rows])
        if w.sum() < D + 1:
            return ("s", float(np.average(z, weights=w)))
        Xw = X * w[:, None]
        pen = np.eye(D + 1)
        pen[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ Xw + pen, Xw.T @ z)
        r = z - X @ coef
        var = float((w * r ** 2).sum() / max(1.0, w.sum() - D - 1))
        return ("f", coef, var * np.linalg.inv(X.T @ Xw + pen))

    def predict(self, a, phi, b, v):
        m = self.fit(a, v)
        if m is None:
            return 0.5, 1.0
        if m[0] == "s":
            return float(1 / (1 + np.exp(-(m[1] - b)))), 1.0
        _, coef, cov = m
        x = np.array([1.0, *phi])
        se = float(np.sqrt(max(1e-9, x @ cov @ x)))
        return float(1 / (1 + np.exp(-(float(x @ coef) - b)))), se


def pick(policy, led, phi, b, v, step):
    if policy == "fixed":
        return max(range(N), key=lambda a: (led.mean(a), -a)), False
    preds = {a: led.predict(a, phi, b, v) for a in range(N)}
    best = max(range(N), key=lambda a: (preds[a][0], -a))
    if policy == "exploit" or step % BLOCK:
        return best, False
    pb, sb = preds[best]
    cands = []
    for a in range(N):
        if a == best:
            continue
        p, se = preds[a]
        stale = led.last(a) is not None and led.last(a) != v
        if (stale or led.count(a, v) < 2) and p * (1 - p) >= 0.05:
            cands.append((p + 1.96 * se, -a, a))
        elif p + 1.96 * se > pb + 1.96 * sb:
            cands.append((p + 1.96 * se, -a, a))
    return (max(cands)[2], True) if cands else (best, False)


def run(policy, theta, *, rounds, drift, seed):
    rng = np.random.default_rng(seed)
    rate = make_rate(theta)
    led = Ledger()
    regret = 0.0
    for step in range(rounds):
        shifted = drift and step >= SHIFT_AT
        v = 1 if shifted else 0
        phi = draw_phi(rng)
        b = float(W @ phi)
        truth = {a: rate(a, phi, b, shifted) for a in range(N)}
        oracle = max(range(N), key=lambda a: truth[a])
        a, _ = pick(policy, led, phi, b, v, step)
        regret += truth[oracle] - truth[a]
        led.add(a, phi, b, noisy(truth[a], rng), v)
    return regret / rounds


REGIMES = {
    "separable": (SEPARABLE, False, 100),
    "near_tied": (CLOSE, False, 100),
    "drift": (SEPARABLE, True, 140),
}

if __name__ == "__main__":
    seeds = range(101, 121)
    print(f"{'regime':10s} {'policy':8s} {'mean regret':>12s} {'worst seed':>11s} {'vs fixed':>9s}")
    for name, (theta, drift, rounds) in REGIMES.items():
        base = None
        for policy in ("fixed", "exploit", "router"):
            vals = np.array([run(policy, theta, rounds=rounds, drift=drift, seed=s) for s in seeds])
            if base is None:
                base = vals.mean()
            print(f"{name:10s} {policy:8s} {vals.mean():12.4f} {vals.max():11.4f} "
                  f"{100 * (vals.mean() - base) / base:+8.1f}%")
        print()
