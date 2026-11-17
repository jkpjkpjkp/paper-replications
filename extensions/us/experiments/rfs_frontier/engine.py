"""Pure portfolio mechanics for the zero-cost RFS audit."""
import numpy as np
import pandas as pd
import scipy.stats as st
from limt.neutral import neutral_projection

def rank_target(scores):
    """Tie-neutral ranks with disjoint, matched half-NAV legs."""
    ranks = st.rankdata(np.asarray(scores, float), method="average")
    centered = ranks - (len(ranks) + 1) / 2
    positive = np.maximum(centered, 0)
    negative = np.maximum(-centered, 0)
    if positive.sum() == 0:
        return centered
    return .5 * positive / positive.sum() - .5 * negative / negative.sum()


def rank_backtest(panel, predictions, adverse_missing=False, save_weights=False):
    """Same timing, execution box and cash accounting as base backtest."""
    n = len(panel.symbols)
    nav = 2e9
    previous = np.zeros(n)
    date_map = {str(d): i for i, d in enumerate(panel.dates)}
    symbol_map = {str(s): i for i, s in enumerate(panel.symbols)}
    rows, held = [], []
    groups = list(predictions.groupby("date", sort=True))
    for number, (date, frame) in enumerate(groups):
        t = date_map[str(date)]
        ids = np.array([symbol_map[str(s)] for s in frame.symbol])
        assert (frame.realized_date.to_numpy() == panel.dates[t + 2]).all()
        assert np.isfinite(frame.pred_return).all()
        raw = np.zeros(n)
        raw[ids] = nav * rank_target(frame.pred_return)
        low = np.minimum(previous, 0.)
        high = np.maximum(previous, 0.)
        high[raw > 0] = np.maximum(high[raw > 0], raw[raw > 0])
        low[raw < 0] = np.minimum(low[raw < 0], raw[raw < 0])
        cap = np.where(panel.execution_tradable[:, t + 1],
                       10 * max(nav, abs(previous).sum()), 0.)
        lower = np.maximum(previous - cap, low)
        upper = np.minimum(previous + cap, high)
        current = neutral_projection(raw, lower, upper,
                                     tolerance=max(1e-7, nav * 1e-11))
        trade = current - previous
        assert (abs(trade) <= cap + 1.).all()
        buy = float(np.maximum(trade, 0).sum())
        sell = float(np.maximum(-trade, 0).sum())
        total = np.array(panel.total_return[:, t + 2], dtype=float)
        missing = ~np.isfinite(total)
        total[missing] = 0.
        if adverse_missing:
            total[missing & (current > 0)] = -1.
            total[missing & (current < 0)] = 1.
        price = np.nan_to_num(np.array(panel.price_return[:, t + 2], float), nan=0.)
        pnl = float(current @ total)
        ending = current * (1 + price)
        ending[panel.terminal[:, t + 2]] = 0.
        liquidation_buy = liquidation_sell = 0.
        if number == len(groups) - 1:
            liquidation_buy = float(np.maximum(-ending, 0).sum())
            liquidation_sell = float(np.maximum(ending, 0).sum())
            ending[:] = 0.
        start_nav = nav
        nav += pnl
        assert nav > 0
        row = dict(signal_date=str(date), execution_date=str(panel.dates[t + 1]),
                   date=str(panel.dates[t + 2]), return_gross=pnl / start_nav,
                   return_net=pnl / start_nav, start_nav=start_nav, end_nav=nav,
                   pnl=pnl, fees=0., buy_value=buy, sell_value=sell,
                   liquidation_buy=liquidation_buy, liquidation_sell=liquidation_sell,
                   long_value=float(np.maximum(current, 0).sum()),
                   short_value=float(np.maximum(-current, 0).sum()),
                   net_exposure=float(current.sum() / start_nav),
                   gross_exposure=float(abs(current).sum() / start_nav),
                   missing_return_gross_weight=float(abs(current[missing]).sum() / start_nav),
                   missing_held_count=int(((abs(current) > 1e-6) & missing).sum()),
                   end_net_drift=float(ending.sum() / nav),
                   n_long=int((current > 1e-6).sum()), n_short=int((current < -1e-6).sum()))
        assert abs(row["net_exposure"]) < 1e-8
        rows.append(row)
        if save_weights:
            held.append(current / start_nav)
        previous = ending
    result = pd.DataFrame(rows)
    np.testing.assert_allclose(result.end_nav, result.start_nav + result.pnl, rtol=1e-12)
    np.testing.assert_allclose(2e9 * np.cumprod(1 + result.return_net), result.end_nav, rtol=1e-11)
    return result, np.array(held) if save_weights else None


def utility_flow(x, gamma):
    x = np.asarray(x, float)
    return x - gamma / 2 * (x - x.mean()) ** 2
