"""Causal backtests with the self-financing transaction equation (5)."""

from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from scipy.io import loadmat
from .solver import PSGAConfig, _solve_exact

DATASETS = {"NYSE(O)": "nyse-o.mat", "NYSE(N)": "nyse-n.mat",
            "TSE": "tse.mat", "MSCI": "msci.mat"}
SHAPES = {"NYSE(O)": (5651, 36), "NYSE(N)": (6431, 23),
          "TSE": (1259, 88), "MSCI": (1043, 24)}


def load_dataset(name, root=None):
    root = Path(root or Path(__file__).resolve().parents[1]/"upstream/STCO-code")
    p = root/DATASETS[name]
    try:
        x = loadmat(p)["data"]
    except NotImplementedError:
        import h5py
        with h5py.File(p) as f:
            x = f["data"][:].T
    x = np.ascontiguousarray(x, dtype=float)
    if x.shape != SHAPES[name] or not np.isfinite(x).all() or x.min() <= 0:
        raise ValueError(f"Invalid benchmark matrix {name}: {x.shape}")
    return x


@njit(cache=True)
def transaction_fraction(h, w, gamma):
    """Unique root in [0,1]: theta+gamma*||h-theta*w||_1=1.

    h=0 for the initial purchase, so theta=1/(1+gamma).
    gamma is per dollar bought OR sold; no factor 1/2 is introduced.
    """
    if gamma == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo+hi)/2
        if mid + gamma*np.sum(np.abs(h-mid*w)) > 1:
            hi = mid
        else:
            lo = mid
    return (lo+hi)/2


@njit(cache=True)
def predict(x, last_observed, strategy, window):
    if strategy == 1:
        return 1/x[last_observed]
    # Expanding startup convention inferred from author STCO2_fun.m.
    n = min(window-1, last_observed+1)
    total, product = np.ones(x.shape[1]), np.ones(x.shape[1])
    for lag in range(n):
        product = product / x[last_observed-lag]
        total += product
    return total/(n+1)


@njit(cache=True)
def _backtest(x, gamma, strategy, window, k, c, alpha, tol, max_iter, accelerate):
    tmax, m = x.shape
    weights = np.empty((tmax, m))
    theta, gross, wealth = np.ones(tmax), np.ones(tmax), np.ones(tmax)
    iterations, evals = np.zeros(tmax, np.int64), np.zeros(tmax, np.int64)
    converged = np.ones(tmax, np.bool_)
    h, initial = np.zeros(m), np.ones(m)/m
    capital = 1.0
    for t in range(tmax):
        w = initial.copy()
        if t > 0:
            q = predict(x, t-1, strategy, window)
            w, n, ev, skipped, ok = _solve_exact(q, h, k, c, 10*gamma, alpha,
                                                tol, max_iter, accelerate, initial)
            iterations[t], evals[t], converged[t] = n, ev, ok
        theta[t] = transaction_fraction(h, w, gamma)
        gross[t] = np.dot(w, x[t])
        capital *= theta[t]*gross[t]
        wealth[t] = capital
        weights[t] = w
        h = w*x[t]/gross[t]
    return weights, theta, gross, wealth, iterations, evals, converged


def backtest(x, gamma, strategy=1, window=4, config=None):
    cfg = config or PSGAConfig()
    if not (0 <= gamma < 1) or strategy not in [1, 2] or window < 2:
        raise ValueError("invalid transaction rate, strategy or window")
    if not 1 <= cfg.k <= x.shape[1]:
        raise ValueError("k must be between 1 and the number of assets")
    arrays = _backtest(np.ascontiguousarray(x), gamma, strategy, window, cfg.k,
                       cfg.penalty, cfg.alpha, cfg.tol, cfg.max_iter, cfg.accelerate)
    return dict(zip(["weights", "theta", "gross", "wealth", "iterations",
                     "prox_evaluations", "converged"], arrays))


def fixed_weight_backtest(x, w, gamma):
    tmax = len(x)
    theta = np.ones(tmax)
    gross = x@w
    h = np.zeros(len(w))
    for t in range(tmax):
        theta[t] = transaction_fraction(h, w, gamma)
        h = w*x[t]/gross[t]
    return {"wealth": np.cumprod(theta*gross), "gross": gross, "theta": theta,
            "weights": np.tile(w, (tmax, 1))}


def ubah_backtest(x, gamma):
    values = np.cumprod(x, axis=0)
    wealth = values.mean(axis=1)/(1+gamma)
    gross = np.r_[values[0].mean(), wealth[1:]/wealth[:-1]]
    theta = np.ones(len(x)); theta[0] = 1/(1+gamma)
    weights = np.vstack([np.ones(x.shape[1])/x.shape[1],
                         values[:-1]/values[:-1].sum(axis=1, keepdims=True)])
    return {"wealth": wealth, "gross": gross, "theta": theta, "weights": weights}


def best_stock(x):
    return np.eye(x.shape[1])[np.log(x).sum(axis=0).argmax()]


def bcrp_weights(x):
    """Hindsight log-optimal constant weights before transaction costs."""
    from scipy.optimize import minimize
    m = x.shape[1]
    result = minimize(lambda w: -np.log(x@w).mean(), np.ones(m)/m,
                      jac=lambda w: -(x/(x@w)[:, None]).mean(axis=0),
                      constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                   "jac": lambda w: np.ones(m)},
                      bounds=[(0, 1)]*m, method="SLSQP",
                      options={"maxiter": 10_000, "ftol": 1e-14})
    if not result.success:
        raise RuntimeError(result.message)
    w = np.maximum(result.x, 0); w /= w.sum()
    # First-order optimality: every asset's directional derivative <= 1.
    residual = float(np.max((x/(x@w)[:, None]).mean(axis=0))-1)
    if residual > 2e-6:
        raise RuntimeError(f"BCRP KKT residual {residual}")
    return w, residual


def metrics(result, market):
    # Paper B.4 defines r_t=(gross_t-1)*theta_t, which omits the direct
    # transaction loss theta_t-1. Both this and self-financing returns saved.
    r = (result["gross"]-1)*result["theta"]
    rm = (market["gross"]-1)*market["theta"]
    beta = float(np.cov(r, rm, ddof=1)[0, 1]/np.var(rm, ddof=1))
    net = result["gross"]*result["theta"]-1
    wealth = result["wealth"]
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    support = np.sum(result["weights"] > 1e-8, axis=1)
    return {"wealth": float(wealth[-1]), "paper_MER": float(np.mean(r-rm)),
            "paper_alpha": float(r.mean()-beta*rm.mean()), "paper_beta": beta,
            "net_daily_mean": float(net.mean()),
            "net_sharpe_252": float(np.sqrt(252)*net.mean()/net.std(ddof=1)),
            "max_drawdown": float(np.max(1-wealth/peak)),
            "support_mean": float(support.mean()),
            "support_std": float(support.std(ddof=1)),
            "support_max": int(support.max()),
            "support_mean_after_initial": float(support[1:].mean()),
            "simplex_error": float(np.abs(result["weights"].sum(axis=1)-1).max()),
            "minimum_weight": float(result["weights"].min())}
