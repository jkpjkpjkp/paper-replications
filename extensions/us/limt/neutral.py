"""Equal-dollar long/short execution with separate buy and sell fees."""
import numpy as np
import pandas as pd
from limt.portfolio import portfolio_metrics


def neutral_projection(desired, lower, upper, tolerance=1e-6):
    """Euclidean projection onto the executable box intersected with sum(x)=0.

    The single Lagrange multiplier solves x=clip(desired-lambda,lower,upper).
    Infeasible neutrality is an error, never a silently unhedged portfolio.
    """
    desired,lower,upper=[np.asarray(x,dtype=float) for x in [desired,lower,upper]]
    if not all(np.isfinite(x).all() for x in [desired,lower,upper]):
        raise ValueError('Neutrality inputs must be finite')
    if (lower>upper).any():raise ValueError('Invalid executable position bounds')
    if lower.sum()>tolerance or upper.sum() < -tolerance:
        raise ValueError('Dollar neutrality is infeasible under current trade/position bounds')
    left=float(np.min(desired-upper));right=float(np.max(desired-lower))
    for _ in range(90):
        mid=(left+right)/2
        positions=np.clip(desired-mid,lower,upper)
        net=float(positions.sum())
        if abs(net)<=tolerance:return positions
        if net>0:left=mid
        else:right=mid
    positions=np.clip(desired-(left+right)/2,lower,upper)
    if abs(positions.sum())>tolerance*10:raise ArithmeticError('Neutral projection did not converge')
    return positions


def neutral_backtest(panel,predictions,buy_fee=.0005,sell_fee=.0015,capital=1e9,
                     participation=.02,volume_multiplier=100.):
    """Same signal timing and cash budget as the original engine, with zero
    post-trade dollar exposure and equal-weighted decile targets. Liquidity can
    reduce deployed capital; it never changes the neutrality requirement.
    """
    n=len(panel.symbols);previous=np.zeros(n);equity=2*capital
    dates={s:i for i,s in enumerate(panel.dates)}
    symbols={s:i for i,s in enumerate(panel.symbols)}
    close=pd.DataFrame(np.asarray(panel.market[:,:,0]).T).ffill().to_numpy().T
    rows=[]
    for date,frame in predictions.groupby('date',sort=True):
        t=dates[date]
        ids=np.array([symbols[s] for s in frame.symbol])
        score=frame.pred_return.to_numpy()
        good=np.isfinite(score)&np.isfinite(panel.market[ids,t,6])&(panel.market[ids,t,1]>0)
        ids,score=ids[good],score[good]
        order=np.argsort(score,kind='stable');count=max(1,len(ids)//10)
        long=ids[order[-count:]];short=ids[order[:count]]
        target_cap=participation*panel.market[:,t,1]*panel.market[:,t,6]*volume_multiplier
        target_cap=np.nan_to_num(target_cap,nan=0.).clip(min=0)
        raw=np.zeros(n)
        raw[long]=np.minimum(capital/count,target_cap[long])
        raw[short]=-np.minimum(capital/count,target_cap[short])
        desired=raw.copy()
        long_total=float(raw[long].sum());short_total=float(-raw[short].sum())
        matched=min(long_total,short_total)
        if long_total>0:desired[long]*=matched/long_total
        if short_total>0:desired[short]*=matched/short_total
        # Non-candidates can only liquidate existing positions. Candidates can
        # fill toward their raw capped equal-weight target, preserving room to
        # balance a frozen position on the opposite side.
        position_low=np.minimum(previous,0.);position_high=np.maximum(previous,0.)
        position_high[long]=np.maximum(position_high[long],raw[long])
        position_low[short]=np.minimum(position_low[short],raw[short])
        amount=panel.market[:,t+1,1]*panel.market[:,t+1,6]*volume_multiplier
        executable=np.isfinite(panel.market[:,t+1,0])&(panel.market[:,t+1,1]>0)
        capacity=np.asarray(participation*np.where(executable,np.nan_to_num(amount,nan=0.),0.).clip(min=0),dtype=np.float64)
        lower=np.maximum(previous-capacity,position_low)
        upper=np.minimum(previous+capacity,position_high)
        try:
            current=neutral_projection(desired,lower,upper,tolerance=max(1e-7,capital*1e-10))
        except ValueError as e:
            raise ValueError(f'{date}: {e}') from e
        trade=current-previous
        excess=np.abs(trade)-capacity
        if np.max(excess)>1e-3:raise ArithmeticError(f'{date}: trade capacity exceeded by {np.max(excess)} CNY')
        buys=float(np.maximum(trade,0).sum());sells=float(np.maximum(-trade,0).sum())
        fees=buy_fee*buys+sell_fee*sells
        returns=np.nan_to_num(close[:,t+2]/close[:,t+1]-1,nan=0.)
        pnl=float(np.dot(current,returns)-fees)
        long_value=float(np.maximum(current,0).sum());short_value=float(np.maximum(-current,0).sum())
        net=float(current.sum()/equity)
        assert abs(net)<1e-8
        daily=pnl/equity;equity+=pnl
        rows.append(dict(signal_date=date,execution_date=panel.dates[t+1],date=panel.dates[t+2],
            return_net=daily,fees=fees,buy_value=buys,sell_value=sells,turnover=(buys+sells)/(2*capital),
            long_notional=long_value,short_notional=short_value,net_exposure=net,
            gross_notional_ratio=(long_value+short_value)/(2*capital)))
        previous=current*(1+returns)
    frame=pd.DataFrame(rows)
    metrics=portfolio_metrics(frame.return_net,frame.turnover)
    metrics.update(max_absolute_net_exposure=float(frame.net_exposure.abs().max()),
        mean_long_notional=float(frame.long_notional.mean()),mean_short_notional=float(frame.short_notional.mean()),
        mean_gross_notional_ratio=float(frame.gross_notional_ratio.mean()),total_fees=float(frame.fees.sum()),
        buy_fee=buy_fee,sell_fee=sell_fee)
    return frame,metrics
