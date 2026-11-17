from __future__ import annotations

import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def hac_mean(values, lag=6):
    y = np.asarray(values, dtype=float)
    if not np.isfinite(y).all() or len(y) < max(12, lag + 2):
        raise ValueError("Need a finite, sufficiently long return series")
    model = sm.OLS(y, np.ones((len(y), 1)), missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
    lo, hi = model.conf_int(alpha=0.05)[0]
    return {"n_months": len(y), "mean_annual": float(model.params[0] * 12),
            "mean_se_annual": float(model.bse[0] * 12), "mean_t": float(model.tvalues[0]),
            "mean_p": float(model.pvalues[0]), "mean_ci_low": float(lo * 12),
            "mean_ci_high": float(hi * 12), "hac_lag": lag}


def factor_alpha(values, factors, lag=6):
    y = np.asarray(values, dtype=float)
    X = np.asarray(factors, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError("Missing returns or factors")
    model = sm.OLS(y, sm.add_constant(X, has_constant="add"), missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
    low, high = model.conf_int()[0]
    b_low, b_high = model.conf_int()[1]
    return {"alpha_annual": float(model.params[0] * 12), "alpha_t": float(model.tvalues[0]),
            "alpha_p": float(model.pvalues[0]), "alpha_ci_low": float(low * 12),
            "alpha_ci_high": float(high * 12), "market_beta": float(model.params[1]),
            "market_beta_ci_low": float(b_low), "market_beta_ci_high": float(b_high),
            "r_squared": float(model.rsquared), "factor_loadings": model.params[1:].tolist()}


def performance(values, lag=6):
    y = np.asarray(values, dtype=float)
    result = hac_mean(y, lag)
    nav = np.cumprod(1 + y)
    peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    result.update({"cagr": float(nav[-1]**(12 / len(y)) - 1),
                   "volatility_annual": float(y.std(ddof=1) * np.sqrt(12)),
                   "sharpe": float(y.mean() / y.std(ddof=1) * np.sqrt(12)) if y.std(ddof=1) > 0 else 0,
                   "max_drawdown": float((nav / peaks - 1).min()),
                   "positive_month_fraction": float((y > 0).mean())})
    return result


def stationary_indices(n, repetitions=5000, average_block=12, seed=20261106):
    rng = np.random.default_rng(seed)
    indices = np.empty((repetitions, n), dtype=np.int32)
    indices[:, 0] = rng.integers(n, size=repetitions)
    for t in range(1, n):
        restart = rng.random(repetitions) < 1.0 / average_block
        fresh = rng.integers(n, size=repetitions)
        indices[:, t] = np.where(restart, fresh, (indices[:, t - 1] + 1) % n)
    return indices


def bootstrap_mean(values, indices):
    y = np.asarray(values, dtype=float)
    sampled = y[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975]) * 12
    centered = sampled - y.mean()
    p = (1 + np.count_nonzero(abs(centered) >= abs(y.mean()))) / (len(sampled) + 1)
    return {"mean_annual": float(y.mean() * 12), "ci_low": float(low), "ci_high": float(high),
            "centered_two_sided_p": float(p), "repetitions": len(sampled)}


def holm(pvalues):
    return multipletests(np.asarray(pvalues), alpha=0.05, method="holm")[1].tolist()
