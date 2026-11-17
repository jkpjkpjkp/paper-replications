from __future__ import annotations

import numpy as np
import pandas as pd


def equal_long_short(permnos, scores, fraction):
    permnos, scores = np.asarray(permnos), np.asarray(scores)
    if len(permnos) < 20 or not np.isfinite(scores).all():
        raise ValueError("Need at least 20 finite scores")
    if len(np.unique(permnos)) != len(permnos):
        raise ValueError("Duplicate security IDs")
    count = int(np.floor(len(scores) * fraction))
    if count < 1 or 2 * count > len(scores):
        raise ValueError("Invalid portfolio fraction")
    order = np.lexsort((permnos, scores))
    weights = np.zeros(len(scores), dtype=np.float64)
    weights[order[:count]] = -1.0 / count
    weights[order[-count:]] = 1.0 / count
    assert abs(weights.sum()) < 1e-10
    assert abs(np.abs(weights).sum() - 2) < 1e-10
    return weights


def portfolio_path(holdings, one_way_bps, annual_borrow, adverse_missing=False):
    """Monthly P&L and turnover, preserving prior NAV-normalized drifted weights.

    Holdings columns: month, permno, weight, realized_return, terminal.
    Zero-weight assets need not be present. No label availability filter is used.
    """
    drift = {}
    output = []
    for month, current in holdings.groupby("month", sort=True):
        ids = current.permno.to_numpy()
        weights = current.weight.to_numpy(dtype=float)
        assert abs(weights.sum()) < 1e-10 and abs(abs(weights).sum() - 2) < 1e-10
        realized = current.realized_return.to_numpy(dtype=float)
        missing = ~np.isfinite(realized)
        unresolved = (current.unresolved_payoff.to_numpy(dtype=bool)
                      if "unresolved_payoff" in current else np.zeros(len(current), dtype=bool))
        returns = np.where(missing, 0, realized)
        if adverse_missing:
            uncertain = missing | unresolved
            returns = np.where(uncertain & (weights > 0), np.minimum(returns, -1), returns)
            returns = np.where(uncertain & (weights < 0), np.maximum(returns, 1), returns)
        gross = float(weights @ returns)
        target = dict(zip(ids, weights))
        union = set(target) | set(drift)
        turnover = sum(abs(target.get(s, 0) - drift.get(s, 0)) for s in union)
        short_notional = float(-weights[weights < 0].sum())
        cost = turnover * one_way_bps / 10000
        borrow = short_notional * annual_borrow / 12
        net = gross - cost - borrow
        if 1 + net <= 0:
            raise FloatingPointError("Portfolio NAV exhausted")
        terminal = current.terminal.to_numpy(dtype=bool) | (missing & adverse_missing)
        ending = weights * (1 + returns) / (1 + net)
        ending[terminal] = 0
        drift = {s: float(w) for s, w in zip(ids, ending) if w != 0}
        long = weights > 0
        output.append({"month": int(month), "gross_return": gross, "return": net,
                       "turnover": turnover, "transaction_cost": cost, "borrow_cost": borrow,
                       "long_return": float(weights[long] @ returns[long]),
                       "short_underlying_return": float(-weights[~long] @ returns[~long]),
                       "n_long": int(long.sum()), "n_short": int((~long).sum()),
                       "net_exposure": float(weights.sum()), "gross_exposure": float(abs(weights).sum()),
                       "missing_count": int(missing.sum()), "missing_gross_weight": float(abs(weights[missing]).sum()),
                       "unresolved_payoff_count": int(unresolved.sum()),
                       "unresolved_payoff_gross_weight": float(abs(weights[unresolved]).sum()),
                       "terminal_count": int(terminal.sum())})
    return pd.DataFrame(output)
