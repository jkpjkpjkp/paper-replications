"""Conventional weighted D10-D1 with auditable delayed CRSP accounting."""
import numpy as np
import pandas as pd
from limt.neutral import neutral_projection


def decile_targets(ids, score, size, n):
    ids, score, size = np.asarray(ids), np.asarray(score), np.asarray(size, dtype=float)
    assert len(ids) >= 10 and len(np.unique(ids)) == len(ids)
    assert np.isfinite(score).all() and np.isfinite(size).all() and (size > 0).all()
    order = np.lexsort((ids, score))
    buckets = np.minimum(9, np.arange(len(ids)) * 10 // len(ids))
    weights = np.zeros((10, n))
    for decile in range(10):
        idx = order[buckets == decile]
        weights[decile, ids[idx]] = size[idx] / size[idx].sum()
    np.testing.assert_allclose(weights.sum(1), 1)
    return weights


def backtest(panel, predictions, weighting="value", cost_bp=0., adverse_missing=False, save_weights=False):
    n = len(panel.symbols)
    nav, previous = 1e9, np.zeros(n)
    date_map = {str(d): i for i, d in enumerate(panel.dates)}
    symbol_map = {str(s): i for i, s in enumerate(panel.symbols)}
    groups = list(predictions.groupby("date", sort=True))
    rows, holdings, deciles = [], [], []
    last_t = None
    for number, (date, frame) in enumerate(groups):
        t = date_map[str(date)]
        if last_t is not None:
            assert t == last_t + 1, "Missing signal session would invalidate position drift accounting"
        last_t = t
        ids = np.array([symbol_map[str(s)] for s in frame.symbol])
        assert np.asarray(panel.member[ids, t]).all()
        assert (frame.realized_date.to_numpy() == panel.dates[t + 2]).all()
        size = panel.capitalization[ids, t] if weighting == "value" else panel.market[ids, t, 1]
        if weighting not in ["value", "volume"]:
            raise ValueError(weighting)
        target = decile_targets(ids, frame.pred_return.to_numpy(), size, n)
        desired = nav * (target[9] - target[0])
        low, high = np.minimum(previous, 0), np.maximum(previous, 0)
        low = np.minimum(low, desired)
        high = np.maximum(high, desired)
        tradable = np.asarray(panel.execution_tradable[:, t + 1])
        lower = np.where(tradable, low, previous)
        upper = np.where(tradable, high, previous)
        current = neutral_projection(desired, lower, upper, tolerance=max(1e-7, nav * 1e-11))
        assert np.allclose(current[~tradable], previous[~tradable], atol=1e-7)
        trades = current - previous
        traded = float(abs(trades).sum())
        total = np.asarray(panel.total_return[:, t + 2], dtype=float)
        price = np.asarray(panel.price_return[:, t + 2], dtype=float)
        missing = ~np.isfinite(total)
        payoff = np.where(missing, 0, total)
        if adverse_missing:
            payoff = np.where(missing & (current > 0), -1., np.where(missing & (current < 0), 1., payoff))
        pnl = float(current @ payoff)
        start_nav = nav
        ending = current * (1 + np.where(np.isfinite(price), price, 0))
        ending[np.asarray(panel.terminal[:, t + 2])] = 0
        liquidation = float(abs(ending).sum()) if number == len(groups) - 1 else 0.
        fee = (traded + liquidation) * cost_bp / 1e4
        nav += pnl - fee
        if nav <= 0:
            raise FloatingPointError(f"Insolvent {weighting}/{cost_bp} on {date}")
        row = dict(signal_date=str(date), execution_date=str(panel.dates[t + 1]), date=str(panel.dates[t + 2]),
            return_gross=pnl / start_nav, return_net=(pnl - fee) / start_nav, start_nav=start_nav, end_nav=nav,
            pnl=pnl, fees=fee, traded_dollars=traded, final_liquidation_dollars=liquidation,
            turnover=traded / start_nav, gross_exposure=float(abs(current).sum() / start_nav),
            net_exposure=float(current.sum() / start_nav), n_long=int((current > 1e-6).sum()),
            n_short=int((current < -1e-6).sum()), missing_held_count=int((missing & (abs(current) > 1e-6)).sum()),
            missing_gross_weight=float(abs(current[missing]).sum() / start_nav),
            target_deviation_l1=float(abs(current - desired).sum() / start_nav))
        assert abs(row["net_exposure"]) < 1e-8
        rows.append(row)
        if save_weights:
            holdings.append(current / start_nav)
        for d in range(10):
            deciles.append(dict(signal_date=str(date), date=str(panel.dates[t + 2]), decile=d + 1,
                target_return=float(target[d] @ np.where(missing, 0, total)),
                missing_weight=float(target[d, missing].sum()), n_stocks=int((target[d] > 0).sum()),
                note="Unconstrained signal-time target decile, before costs; actual spread uses executable ledger"))
        previous = ending if number < len(groups) - 1 else np.zeros(n)
    ledger = pd.DataFrame(rows)
    np.testing.assert_allclose(ledger.end_nav, ledger.start_nav + ledger.pnl - ledger.fees, rtol=1e-12)
    np.testing.assert_allclose(1e9 * np.cumprod(1 + ledger.return_net), ledger.end_nav, rtol=1e-11)
    return ledger, np.array(holdings) if save_weights else None, pd.DataFrame(deciles)
