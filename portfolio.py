"""Portfolio experiments from arXiv:2411.16829v2, Sections 4.2 and E.2.

The optimization variable lies on the long-only, fully invested simplex.
All objectives are convex; the simplex first-order gap supplies an independent
upper bound on suboptimality. Returns are decimal weekly returns.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import digamma, logsumexp
from scipy.stats import invwishart, multivariate_t

RADII = np.array([1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, .001, .002,
                  .005, .01, .02, .05, .1, .2, .5, 1.])
METHODS = ("BASPE", "BASPP", "BDRO")


@dataclass
class Posterior:
    mean: np.ndarray
    kappa: float
    nu: float
    psi: np.ndarray

    @property
    def covariance_pe(self):
        return self.psi / self.nu

    @property
    def predictive_df(self):
        return self.nu - len(self.mean) + 1

    @property
    def predictive_shape(self):
        return self.psi * (self.kappa + 1) / (self.kappa * self.predictive_df)

    def offset(self, upstream_indexing=False):
        """Expected KL offset; optional historical off-by-one in upstream G.

        Upstream sums psi(a+(1-i)/2) with i=0,...,D-1 instead of i=1,...,D.
        Its portfolio runner uses epsilon_minus_G directly, so this discrepancy
        does not change any reproduced portfolio weights.
        """
        d = len(self.mean)
        indices = np.arange(d) if upstream_indexing else np.arange(1, d + 1)
        return float(.5 * (d * np.log(self.nu / 2) + d / self.kappa
                           - digamma((self.nu + 1 - indices) / 2).sum()))


def fit_posterior(data):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or not np.isfinite(data).all() or len(data) == 0:
        raise ValueError("Expected a nonempty finite observations-by-assets matrix")
    n, d = data.shape
    # Authors' prior: mean=0, nu=D+1, kappa=nu+D+2, Psi=I.
    kappa = float(2 * d + 3 + n)
    mean = data.sum(axis=0) / kappa
    psi = np.eye(d) + data.T @ data - kappa * np.outer(mean, mean)
    return Posterior(mean, kappa, float(d + 1 + n), (psi + psi.T) / 2)


def sample_predictive(post, count, seed):
    # Match upstream's RNG reset per (window, method, radius).
    return multivariate_t.rvs(loc=post.mean, shape=post.predictive_shape,
                             df=post.predictive_df, size=(1, count),
                             random_state=np.random.default_rng(seed)).reshape(count, -1)


def sample_covariances(post, count, seed):
    return invwishart(df=post.nu, scale=post.psi,
                      seed=np.random.default_rng(seed)).rvs(count).reshape(
                          count, len(post.mean), len(post.mean))


def gaussian_objective(x, mean, covariances, radius):
    """PE: one covariance. BDRO: sample average of posterior standard deviations."""
    cx = np.einsum("mij,j->mi", covariances, x, optimize=False)
    sd = np.sqrt(np.einsum("mi,i->m", cx, x))
    factor = np.sqrt(2 * radius)
    return (-mean @ x + factor * sd.mean(),
            -mean + factor * np.mean(cx / sd[:, None], axis=0))


def entropic_risk(losses, radius):
    """Exact finite-support KL worst case, via a scalar tilted-distribution solve.

    Returns dual value, worst-case probabilities, and temperature gamma.
    Center and scale losses to avoid cancellation at small radii.
    """
    losses = np.asarray(losses, dtype=float)
    if radius < 0:
        raise ValueError("KL radius must be nonnegative")
    m = len(losses)
    if radius == 0 or np.ptp(losses) < 1e-15:
        return float(losses.mean()), np.full(m, 1 / m), np.inf
    maximum = losses.max()
    top = losses == maximum
    if radius >= np.log(m / top.sum()):
        return float(maximum), top / top.sum(), 0.
    scale = losses.std()
    z = (losses - maximum) / scale

    def tilted(beta):
        logits = beta * z
        lse = logsumexp(logits)
        q = np.exp(logits - lse)
        kl = float(q @ logits - lse + np.log(m))
        return kl, q

    upper = max(1., np.sqrt(2 * radius))
    while tilted(upper)[0] < radius:
        upper *= 2
    beta = brentq(lambda b: tilted(b)[0] - radius, 0., upper,
                  xtol=1e-13, rtol=1e-13)
    kl, q = tilted(beta)
    gamma = scale / beta
    value = float(q @ losses + gamma * (radius - kl))
    return value, q, gamma


def predictive_objective(x, samples, radius):
    value, q, _ = entropic_risk(-samples @ x, radius)
    return value, -samples.T @ q


def solve_simplex(objective, dim, initial=None, gap_tolerance=1e-7):
    """SLSQP with analytic gradient, checked by a convex first-order certificate."""
    start = perf_counter()
    x0 = np.full(dim, 1 / dim) if initial is None else np.array(initial, copy=True)
    constraint = {"type": "eq", "fun": lambda x: x.sum() - 1,
                  "jac": lambda x: np.ones(dim)}

    def scaled(x):
        value, gradient = objective(x)
        return value * 100., gradient * 100.

    for ftol in (1e-11, 1e-14):
        result = minimize(scaled, x0, jac=True, method="SLSQP",
                          bounds=[(0., 1.)] * dim, constraints=constraint,
                          options={"ftol": ftol, "maxiter": 1000})
        x = np.maximum(result.x, 0.)
        x /= x.sum()
        value, gradient = objective(x)
        gap = max(0., float(gradient @ x - gradient.min()))
        if gap <= gap_tolerance:
            return {"weights": x, "objective": float(value), "gap": gap,
                    "iterations": int(result.nit), "solver_success": bool(result.success),
                    "solve_seconds": perf_counter() - start}
        x0 = x
    raise RuntimeError(f"Uncertified optimization: gap={gap:.3g}; {result.message}")


def solve_conic(mean, covariance_samples, samples, radius):
    """Independent CVXPY conic reference, with scaling suited to the radius."""
    import cvxpy as cp
    d = len(mean)
    x = cp.Variable(d)
    constraints = [x >= 0, cp.sum(x) == 1]
    if samples is None:
        risk = cp.sum(cp.hstack([
            cp.norm(np.linalg.cholesky(cov).T @ x)
            for cov in covariance_samples])) / len(covariance_samples)
        expr = -mean @ x + np.sqrt(2 * radius) * risk
        problem = cp.Problem(cp.Minimize(100 * expr), constraints)
        solver = "CLARABEL"
    elif radius >= 1e-4:
        gamma = cp.Variable(nonneg=True)
        losses = -100 * samples @ x
        expr = (gamma * (radius - np.log(len(samples)))
                + cp.perspective(cp.log_sum_exp(losses), gamma,
                                 f_recession=cp.max(losses)))
        problem = cp.Problem(cp.Minimize(expr), constraints)
        solver = "CLARABEL"
    else:
        m = len(samples)
        gamma = cp.Variable(nonneg=True)
        t = cp.Variable()
        u = cp.Variable(m)
        constraints += [cp.ExpCone(-100 * samples @ x - t, np.ones(m) * gamma, u),
                        cp.sum(u) / m <= gamma]
        expr = t + gamma * radius
        problem = cp.Problem(cp.Minimize(expr), constraints)
        solver = "CLARABEL" if m < 500 else "ECOS"
    if solver == "CLARABEL":
        problem.solve(solver=solver, tol_gap_abs=1e-9, tol_gap_rel=1e-9,
                      tol_feas=1e-9, max_iter=300)
    else:
        problem.solve(solver=solver, abstol=1e-9, reltol=1e-9,
                      feastol=1e-9, max_iters=300)
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(problem.status)
    return {"weights": x.value, "objective": problem.value / 100,
            "status": problem.status, "solver": solver,
            "solve_seconds": problem.solver_stats.solve_time}
