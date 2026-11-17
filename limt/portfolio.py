"""Causal APO weights and a declared close-execution backtest convention.

This is an independent implementation: the author repository has no APO engine.
"""
import numpy as np
import pandas as pd


def linear_norm(x):
    x = np.asarray(x, dtype=float)
    if not len(x):
        return x
    if not np.isfinite(x).all():
        raise ValueError("Portfolio inputs must be finite")
    shifted = x - x.min()
    return shifted / shifted.sum() if shifted.sum() > 0 else np.full(len(x), 1 / len(x))


def observed_moments(market, member, t):
    """Eq22 moments measured from realized observations at signal date t."""
    volume = market[:, :, 1]
    high, low, vwap = [market[:, :, j] for j in (4, 5, 6)]
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.where(volume[:, t-5:t+1] > 0, volume[:, t-5:t+1], np.nan))
        shock = logv[:, -1] - logv[:, :5].mean(axis=1)
        sigma = (high[:, t] - low[:, t]) / vwap[:, t]
        logmean = logv[:, 1:].mean(axis=1)
    stats = []
    for values in (shock, sigma):
        values = values[member[:, t] & np.isfinite(values)]
        stats.append((float(values.mean()), float(values.std(ddof=1))))
    return stats, logmean


def apo_weights(pred, ids, market, member, t, r=0.1, participation=0.02,
                capital=1e9, volume_multiplier=100.0):
    stats, logmean = observed_moments(market, member, t)
    (mv, sv), (ms, ss) = stats
    with np.errstate(over="raise", invalid="raise"):
        amount = np.exp(pred[:, 1] * sv + mv + logmean[ids]) * market[ids, t, 6] * volume_multiplier
    sigma = pred[:, 2] * ss + ms
    raw = r * linear_norm(-sigma) + (1-r) * linear_norm(amount)
    observed_amount = market[ids, t, 1] * market[ids, t, 6] * volume_multiplier
    # Eq26 clips targets; unallocated capital stays in cash.
    return np.minimum(raw, participation * observed_amount / capital)


def clip_trades(previous, desired, amount, participation=0.02):
    capacity = participation * np.nan_to_num(amount, nan=0., posinf=0., neginf=0.).clip(min=0)
    requested = desired - previous
    filled = np.clip(requested, -capacity, capacity)
    return previous + filled, requested, filled


def portfolio_metrics(daily, turnover=None, requested=None, filled=None):
    daily = np.asarray(daily, dtype=float)
    if not len(daily) or not np.isfinite(daily).all():
        raise ValueError("Portfolio metrics require finite, nonempty daily returns")
    nav = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    maxdd = float(np.max(1-nav/peak))
    annual = float(daily.mean() * 252)
    vol = float(daily.std(ddof=1) * np.sqrt(252)) if len(daily)>1 else 0.
    downside = float(np.sqrt(np.mean(np.minimum(daily, 0)**2)) * np.sqrt(252))
    losses = -daily[daily < 0].sum()
    return dict(days=len(daily), cumulative_return=float(nav[-1]-1), annualized_return=annual,
        sharpe=annual/vol if vol else None, max_drawdown=maxdd, downside_vol=downside,
        sortino=annual/downside if downside else None, calmar=annual/maxdd if maxdd else None,
        profit_loss_ratio=float(daily[daily>0].sum()/losses) if losses else None,
        mean_turnover=float(np.mean(turnover)) if turnover is not None else None,
        filled_value_ratio=float(np.sum(filled)/np.sum(requested)) if requested is not None and np.sum(requested)>0 else None)


def backtest(panel, predictions, strategy="apo", r=0.1, capital=1e9,
             fee=0.001, participation=0.02, volume_multiplier=100., benchmark=None):
    n = len(panel.symbols)
    positions = np.zeros(n)
    equity = 2 * capital
    rows = []
    symbol_map = {s:i for i,s in enumerate(panel.symbols)}
    date_map = {s:i for i,s in enumerate(panel.dates)}
    # Carry last observed close for valuation only. Missing trading volume blocks
    # transactions. This cannot recover delisting returns absent from the source.
    close = pd.DataFrame(np.asarray(panel.market[:, :, 0]).T).ffill().to_numpy().T
    full_orders = orders = 0
    retained = set()
    for signal, frame in predictions.groupby("date", sort=True):
        t = date_map[signal]
        ids = np.array([symbol_map[s] for s in frame.symbol])
        pred = frame[["pred_return", "pred_volume", "pred_volatility"]].to_numpy()
        # Liquidity inputs needed for sizing must be available as of t.
        stats, logmean = observed_moments(panel.market, panel.member, t)
        good = np.isfinite(pred).all(axis=1) & np.isfinite(logmean[ids]) & (panel.market[ids,t,6]>0)
        ids, pred = ids[good], pred[good]
        rank = np.argsort(pred[:,0], kind="stable")
        desired = np.zeros(n)
        if strategy == "top50":
            available = set(ids.tolist())
            ranked_ids = ids[rank[::-1]].tolist()
            held_rank = [i for i in ranked_ids if i in retained and i in available]
            # Qlib-style TopkDropout: replace up to20 only where a higher-scored
            # unheld candidate enters the union's top50.
            new = [i for i in ranked_ids if i not in retained][:max(0, 50-len(held_rank)+20)]
            union = set(held_rank + new)
            chosen = [i for i in ranked_ids if i in union][:50]
            retained = set(chosen)
            if chosen:
                desired[chosen] = capital / len(chosen)
        else:
            count = max(1, len(ids)//10)
            for side, ix in [(1,rank[-count:]), (-1,rank[:count])]:
                group = ids[ix]
                if strategy == "apo":
                    weights = apo_weights(pred[ix], group, panel.market, panel.member, t,
                        r, participation, capital, volume_multiplier)
                elif strategy == "equal":
                    weights = np.full(len(group), 1/len(group))
                elif strategy == "rank":
                    weights = linear_norm(pred[ix,0] * side)
                elif strategy == "risk_parity":
                    c = close[group, t-20:t+1]
                    sigma = np.nanstd(c[:,1:]/c[:,:-1]-1, axis=1, ddof=1)
                    inv = 1/np.maximum(sigma, 1e-8)
                    weights = inv/inv.sum()
                else:
                    raise ValueError(strategy)
                # Fair target cap applies to every sizing rule.
                amount_t = panel.market[group,t,1]*panel.market[group,t,6]*volume_multiplier
                weights = np.minimum(weights, participation*amount_t/capital)
                desired[group] = side * capital * weights
        # Orders from t are executed at t+1 close; fill caps use observed t+1
        # turnover, then holdings earn close[t+2]/close[t+1]-1.
        amount = panel.market[:, t+1, 1]*panel.market[:,t+1,6]*volume_multiplier
        executable = np.isfinite(panel.market[:,t+1,0]) & (panel.market[:,t+1,1]>0)
        amount = np.where(executable, amount, 0)
        current, request, fill = clip_trades(positions, desired, amount, participation)
        requested = np.abs(request).sum()
        traded = np.abs(fill).sum()
        active_orders = np.abs(request)>1e-6
        orders += int(active_orders.sum())
        full_orders += int(((np.abs(fill-request)<1e-6) & active_orders).sum())
        ret = np.nan_to_num(close[:,t+2]/close[:,t+1]-1, nan=0.)
        fees = fee * traded
        pnl = float(np.dot(current, ret)-fees)
        denom = equity if strategy != "top50" else (equity-capital)
        daily = pnl/denom
        equity += pnl
        positions = current*(1+ret)
        row = dict(signal_date=signal, execution_date=panel.dates[t+1],
            date=panel.dates[t+2], return_net=daily, fees=fees,
            turnover=traded/(2*capital if strategy != "top50" else capital),
            requested_value=requested, filled_value=traded,
            gross_exposure=float(np.abs(current).sum()/(2*capital if strategy != "top50" else capital)))
        if benchmark is not None:
            row["benchmark_return"] = float(benchmark[t+2])
        rows.append(row)
    frame = pd.DataFrame(rows)
    metrics = portfolio_metrics(frame.return_net, frame.turnover,
                               frame.requested_value, frame.filled_value)
    metrics["order_success_rate"] = full_orders/orders if orders else None
    metrics["mean_gross_exposure"] = float(frame.gross_exposure.mean())
    if benchmark is not None:
        excess = frame.return_net-frame.benchmark_return
        metrics["excess_annualized_return"] = float(excess.mean()*252)
        metrics["excess_information_ratio"] = float(excess.mean()/excess.std(ddof=1)*np.sqrt(252))
    return frame, metrics
