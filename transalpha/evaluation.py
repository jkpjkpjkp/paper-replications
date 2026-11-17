"""Cross-sectional metrics and an explicit 100%-gross, dollar-neutral portfolio."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 1e-14 else 0.0


def liquidation_notional(old, new):
    """Dollar amount of long sales and short covers, including sign crossings."""
    return float(np.maximum(np.maximum(old, 0) - np.maximum(new, 0), 0).sum()
                 + np.maximum(np.maximum(-old, 0) - np.maximum(-new, 0), 0).sum())


def rebalance_fee(old, target, rate):
    # Target weights are relative to NAV after the rebalance fee.
    fee = 0.0
    for _ in range(30):
        updated = rate * liquidation_notional(old, target * (1 - fee))
        if abs(updated - fee) < 1e-14:
            return updated
        fee = updated
    raise RuntimeError("Fee fixed point did not converge")


def summarize_returns(returns, annualization=252):
    r = np.asarray(returns, float)
    nav = np.r_[1.0, np.cumprod(1 + r)]
    return {"annualized_return": float(nav[-1] ** (annualization / len(r)) - 1),
            "sharpe": float(np.sqrt(annualization) * r.mean() / r.std(ddof=1)) if r.std(ddof=1) else 0.,
            "maximum_drawdown": float(np.max(1 - nav / np.maximum.accumulate(nav))),
            "total_return": float(nav[-1] - 1), "mean_daily_return": float(r.mean())}


def evaluate(predictions, targets, dates, symbols, topk=50, cost_rate=0.001, annualization=252,
             portfolio_returns=None, execution_tradable=None):
    old = np.zeros(len(symbols), np.float64)
    rows = []
    for t, (p, y, date) in enumerate(zip(predictions, targets, dates)):
        eligible = np.flatnonzero(np.isfinite(p))
        frozen = np.zeros_like(old, dtype=bool)
        if execution_tradable is not None:
            # Trading status is observed at execution t+1, after the signal at t.
            tradable = execution_tradable[t]
            frozen = ~tradable & (old != 0)
            eligible = eligible[tradable[eligible]]
        if len(eligible) < 2 * topk:
            raise ValueError("Insufficient stocks for disjoint long/short portfolios")
        order = eligible[np.argsort(p[eligible], kind="stable")]
        weights = np.zeros_like(old)
        weights[frozen] = old[frozen]
        long_budget = max(0., .5 - weights[weights > 0].sum())
        short_budget = max(0., .5 + weights[weights < 0].sum())
        weights[order[-topk:]], weights[order[:topk]] = long_budget / topk, -short_budget / topk
        selected = weights != 0
        accounting_y = y if portfolio_returns is None else portfolio_returns[t]
        if not np.isfinite(accounting_y[selected]).all():
            raise ValueError("Selected security has unknown future return on " + str(date))
        r = np.where(np.isfinite(accounting_y), accounting_y, 0)
        if frozen.any():
            # Frozen dollar holdings cannot be rescaled by the rebalance fee.
            fee = 0.
            for _ in range(30):
                new = weights * (1 - fee)
                new[frozen] = old[frozen]
                fee = cost_rate * liquidation_notional(old, new)
            executed_weights = new / (1 - fee)
        else:
            fee = rebalance_fee(old, weights, cost_rate)
            executed_weights = weights
        gross = float(executed_weights @ r)
        turnover = float(np.abs(executed_weights * (1 - fee) - old).sum())
        net = (1 - fee) * (1 + gross) - 1
        old = executed_weights * (1 + r) / (1 + gross)
        terminal_fee = 0.0
        if t == len(dates) - 1:
            terminal_fee = cost_rate * np.abs(old).sum()
            net = (1 + net) * (1 - terminal_fee) - 1
        valid = np.isfinite(p) & np.isfinite(y)
        ic = correlation(p[valid], y[valid])
        ric = correlation(rankdata(p[valid]), rankdata(y[valid]))
        rows.append({"signal_date": str(date), "ic": ic, "rank_ic": ric,
                     "gross_return": gross, "net_return": net, "rebalance_fee": fee,
                     "terminal_fee": terminal_fee, "traded_notional_over_nav": turnover,
                     "liquidated_notional_over_nav": fee / cost_rate if cost_rate else 0.,
                     "forecastable_stocks": len(eligible), "labelled_stocks": int(valid.sum()),
                     "frozen_positions": int(frozen.sum()), "blocked_forecasts": int(np.isfinite(p).sum() - len(eligible)),
                     "gross_exposure": float(np.abs(executed_weights).sum()),
                     "net_exposure": float(executed_weights.sum()),
                     "long_return": float(r[order[-topk:]].mean()),
                     "short_underlying_return": float(r[order[:topk]].mean())})
    daily = pd.DataFrame(rows)
    metrics = {"days": len(daily), "ic": float(daily.ic.mean()), "ic_std": float(daily.ic.std(ddof=0)),
               "rank_ic": float(daily.rank_ic.mean()), "rank_ic_std": float(daily.rank_ic.std(ddof=0)),
               "icir": float(daily.ic.mean() / daily.ic.std(ddof=0)) if daily.ic.std(ddof=0) else None,
               "rank_icir": float(daily.rank_ic.mean() / daily.rank_ic.std(ddof=0)) if daily.rank_ic.std(ddof=0) else None,
               "mean_daily_traded_notional_over_nav": float(daily.traded_notional_over_nav.mean()),
               "mean_daily_liquidated_notional_over_nav": float(daily.liquidated_notional_over_nav.mean()),
               "gross": summarize_returns(daily.gross_return, annualization),
               "net": summarize_returns(daily.net_return, annualization),
               "cost_rate": cost_rate, "gross_exposure": 1.0, "net_exposure_at_rebalance": 0.0,
               "maximum_absolute_net_exposure": float(daily.net_exposure.abs().max()),
               "maximum_gross_exposure": float(daily.gross_exposure.max()),
               "frozen_position_days": int(daily.frozen_positions.sum()),
               "blocked_forecast_days": int(daily.blocked_forecasts.sum()),
               "information_ratio": None,
               "information_ratio_reason": "Paper does not identify benchmark or active-return convention"}
    return metrics, daily


def block_bootstrap_mean(values, block=10, replicates=5000, seed=2026):
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    starts = rng.integers(len(a), size=(replicates, int(np.ceil(len(a) / block))))
    ids = (starts[..., None] + np.arange(block)) % len(a)
    means = a[ids.reshape(replicates, -1)[:, :len(a)]].mean(1)
    return {"mean": float(a.mean()), "ci95": np.quantile(means, [.025, .975]).tolist(),
            "block_sessions": block, "replicates": replicates}
