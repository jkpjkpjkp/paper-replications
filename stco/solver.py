"""Equations (8), (11), and Lemmas 3.7--3.8 of Yao & Zhang (2026).

The paper's Algorithm 2 prints w^j in the ADMM w update. Lemma 3.7 and
the proximal subproblem (12) require the FIXED outer anchor w^k instead.
The implementation follows the subproblem and lemma. An exact scalar-dual
prox is also supplied so the very small paper step size is practical.
"""

from dataclasses import dataclass
import numpy as np
from numba import njit


@njit(cache=True)
def project_simplex(z):
    u = np.sort(z)[::-1]
    total = 0.0
    theta = 0.0
    for j in range(len(z)):
        total += u[j]
        candidate = (total - 1.0) / (j + 1)
        if u[j] > candidate:
            theta = candidate
    return np.maximum(z - theta, 0.0)


@njit(cache=True)
def _prox_exact(z, h, tau):
    """min_{w in simplex} .5||w-z||² + tau||w-h||_1, h >= 0.

    w_i(nu) = max(0, h_i + soft(z_i - nu - h_i, tau)). Its sum is
    continuous piecewise affine, with three knots per coordinate. Sweep
    the knots to solve sum_i w_i(nu)=1 without iterative optimization.
    """
    m = len(z)
    knots = np.empty(3 * m)
    slopes = np.empty(3 * m)
    for i in range(m):
        knots[3*i] = z[i] - h[i] - tau
        slopes[3*i] = -1.0
        knots[3*i+1] = z[i] - h[i] + tau
        slopes[3*i+1] = 1.0
        knots[3*i+2] = z[i] + tau
        slopes[3*i+2] = -1.0
    order = np.argsort(knots)
    intercept = np.sum(z) - m * tau
    slope = float(m)
    nu = knots[order[-1]]
    for j in order:
        knot = knots[j]
        if slope > 0:
            root = (intercept - 1.0) / slope
            if root <= knot + 1e-15:
                nu = root
                break
        intercept += slopes[j] * knot
        slope += slopes[j]
    v = z - nu - h
    return np.maximum(h + np.sign(v) * np.maximum(np.abs(v) - tau, 0.0), 0.0)


def prox_exact(z, h, tau):
    z, h = np.asarray(z, dtype=float), np.asarray(h, dtype=float)
    if z.ndim != 1 or z.shape != h.shape or np.min(h) < 0 or tau < 0:
        raise ValueError("z and nonnegative h must be equal-length vectors; tau >= 0")
    return _prox_exact(z, h, tau)


def prox_admm(anchor, q, h, v, *, alpha=.0025, penalty=8e-6,
              lam=.025, rho=.618, tol=1e-10, max_iter=100_000,
              stopping="residual", literal_algorithm2=False):
    """ADMM reference; fixed anchor, no warm start across outer problems.

    stopping='relative' reproduces Algorithm 2's inner iterate test;
    stopping='residual' additionally checks primal/dual feasibility.
    literal_algorithm2=True is an audit option, not equation (11).
    """
    anchor, q, h, v = [np.asarray(x, dtype=float) for x in (anchor, q, h, v)]
    w, d, mu = anchor.copy(), anchor.copy(), np.zeros_like(anchor)
    for j in range(1, max_iter + 1):
        old_w, old_d = w.copy(), d.copy()
        center = old_w if literal_algorithm2 else anchor
        w = project_simplex((center / alpha + rho*d + q + penalty*v - mu)
                            / (1/alpha + rho))
        phi = w + mu / rho - h
        d = h + np.sign(phi) * np.maximum(np.abs(phi) - lam / rho, 0)
        mu += rho * (w - d)
        rel = np.linalg.norm(w-old_w) / np.linalg.norm(w)
        primal = np.linalg.norm(w-d)
        dual = rho*np.linalg.norm(d-old_d)
        if rel <= tol and (stopping == "relative" or max(primal, dual) <= tol):
            return w, {"iterations": j, "primal": primal, "dual": dual,
                       "relative_step": rel, "converged": True}
    return w, {"iterations": j, "primal": primal, "dual": dual,
               "relative_step": rel, "converged": False}


@njit(cache=True)
def _topk(w, k):
    # Stable tie-breaking: lower asset index wins, including initial uniform w.
    order = np.argsort(-w, kind="mergesort")
    v = np.zeros(len(w))
    for i in order[:k]:
        v[i] = 1.0
    return v


@njit(cache=True)
def _regime(w, h):
    r = np.empty(len(w), dtype=np.int64)
    for i in range(len(w)):
        if w[i] <= 2e-13:
            r[i] = 0
        elif abs(w[i] - h[i]) <= 2e-13:
            r[i] = 1
        elif w[i] < h[i]:
            r[i] = 2
        else:
            r[i] = 3
    return r


@njit(cache=True)
def _jump_bound(w, delta, h, v, budget):
    """Steps before crossing a prox-region or top-K membership boundary.

    Inside a region, repeated proximal steps are a translation because the
    simplex projection Jacobian on the free coordinates is idempotent.
    Stop conservatively before the first event, then resume explicit steps.
    """
    bound = float(budget)
    for i in range(len(w)):
        if abs(delta[i]) < 1e-17:
            continue
        if delta[i] > 0 and w[i] < h[i] - 2e-13:
            bound = min(bound, (h[i]-w[i])/delta[i])
        if delta[i] < 0:
            target = h[i] if w[i] > h[i] + 2e-13 else 0.0
            if w[i] > target + 2e-13:
                bound = min(bound, (target-w[i])/delta[i])
        if v[i] == 1:
            for j in range(len(w)):
                if v[j] == 0 and delta[j] > delta[i] + 1e-17:
                    bound = min(bound, max(0.0, w[i]-w[j])/(delta[j]-delta[i]))
    return max(0, int(np.floor(bound)) - 2)


@njit(cache=True)
def _solve_exact(q, h, k, penalty, lam, alpha, tol, max_iter, accelerate,
                 initial):
    w = initial.copy()
    iterations, evaluations, skipped = 0, 0, 0
    prior_delta = np.zeros(len(w))
    prior_v = np.zeros(len(w))
    prior_regime = np.full(len(w), -1, dtype=np.int64)
    converged = False
    while iterations < max_iter:
        v = _topk(w, k)
        a = q + penalty*v
        # Subtracting a common linear coefficient changes only a constant.
        a = a - np.mean(a)
        new = _prox_exact(w + alpha*a, h, alpha*lam)
        delta = new - w
        iterations += 1
        evaluations += 1
        rel = np.linalg.norm(delta) / np.linalg.norm(new)
        if rel <= tol:
            w = new
            converged = True
            break
        regime = _regime(new, h)
        if (accelerate and np.all(v == prior_v) and
                np.all(regime == prior_regime) and
                np.max(np.abs(delta-prior_delta)) < 2e-14 and
                np.all(_topk(new, k) == v)):
            jump = _jump_bound(new, delta, h, v, max_iter-iterations)
            if jump > 0:
                new = new + jump*delta
                iterations += jump
                skipped += jump
        prior_delta = delta
        prior_v = v
        prior_regime = regime
        w = new
    return w, iterations, evaluations, skipped, converged


@dataclass(frozen=True)
class PSGAConfig:
    k: int = 10
    penalty: float = 8e-6
    alpha: float = .0025
    rho: float = .618
    tol: float = 1e-8
    max_iter: int = 100_000_000
    accelerate: bool = True


def solve_psga(q, h, lam, config=None, *, initial=None):
    """Algorithm 1 with exact solutions to the ADMM subproblem (11).

    Event skipping preserves the fixed-step outer iterates algebraically;
    set accelerate=False for direct PSGA, or use prox_admm as the reference
    inner solver. No hard thresholding or cardinality repair is applied.
    """
    cfg = config or PSGAConfig()
    q, h = np.asarray(q, dtype=float), np.asarray(h, dtype=float)
    if q.ndim != 1 or q.shape != h.shape or not np.all(np.isfinite(q)):
        raise ValueError("q and h must be finite equal-length vectors")
    if np.min(h) < 0 or not np.isclose(np.sum(h), 1, atol=1e-10):
        raise ValueError("h must belong to the simplex")
    if not (1 <= cfg.k <= len(q)) or lam < 0 or cfg.penalty < 0 or cfg.alpha <= 0:
        raise ValueError("invalid k, lambda, penalty or alpha")
    initial = np.ones(len(q))/len(q) if initial is None else np.asarray(initial, float)
    if np.min(initial) < 0 or not np.isclose(initial.sum(), 1):
        raise ValueError("initial must belong to the simplex")
    w, n, ev, skipped, ok = _solve_exact(q, h, cfg.k, cfg.penalty, lam,
                                       cfg.alpha, cfg.tol, cfg.max_iter,
                                       cfg.accelerate, initial)
    return w, {"iterations": n, "prox_evaluations": ev, "skipped": skipped,
               "converged": ok}


def objective(w, q, h, lam, penalty, k):
    return float(-np.dot(q, w) + lam*np.abs(w-h).sum()
                 - penalty*np.sort(np.abs(w))[-k:].sum())
