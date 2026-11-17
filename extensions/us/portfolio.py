"""Calendar-time neutral dollars/cash accounting with CRSP total returns."""
import numpy as np
import pandas as pd
from limt.neutral import neutral_projection
from limt.portfolio import apo_weights


def backtest(panel,predictions,strategy='equal',buy_bp=0.,adverse_missing=False,save_weights=False):
    n=len(panel.symbols);nav=2e9;previous=np.zeros(n)
    date_map={str(d):i for i,d in enumerate(panel.dates)}
    symbol_map={str(s):i for i,s in enumerate(panel.symbols)}
    rows=[];held_weights=[]
    groups=list(predictions.groupby('date',sort=True))
    for number,(date,frame) in enumerate(groups):
        t=date_map[str(date)]
        ids=np.array([symbol_map[str(s)] for s in frame.symbol])
        pred=frame[['pred_return','pred_volume','pred_volatility']].to_numpy()
        assert np.isfinite(pred).all()
        assert (frame.realized_date.to_numpy()==panel.dates[t+2]).all()
        if strategy=='apo':
            known_volume=np.asarray(panel.market[ids,t-4:t+1,1])
            usable=(np.isfinite(known_volume)&(known_volume>0)).all(1)&(panel.market[ids,t,6]>0)
            ids,pred=ids[usable],pred[usable]
        order=np.lexsort((ids,pred[:,0]));count=max(1,len(ids)//10)
        long,short=order[-count:],order[:count]
        raw=np.zeros(n)
        for sign,ix in [(1,long),(-1,short)]:
            group=ids[ix]
            if strategy=='equal':
                raw[group]=sign*nav/2/count
            elif strategy=='apo':
                w=apo_weights(pred[ix],group,panel.market,panel.member,t,
                    r=.1,participation=.02,capital=1e9,volume_multiplier=1.)
                raw[group]=sign*1e9*w
            else:raise ValueError(strategy)
        long_total=np.maximum(raw,0).sum();short_total=np.maximum(-raw,0).sum()
        desired=raw.copy();matched=min(long_total,short_total)
        if long_total>0:desired[raw>0]*=matched/long_total
        if short_total>0:desired[raw<0]*=matched/short_total
        low=np.minimum(previous,0.);high=np.maximum(previous,0.)
        high[ids[long]]=np.maximum(high[ids[long]],raw[ids[long]])
        low[ids[short]]=np.minimum(low[ids[short]],raw[ids[short]])
        if strategy=='apo':
            cap=np.asarray(.02*panel.market[:,t+1,1]*panel.market[:,t+1,6],dtype=float)
            cap=np.nan_to_num(cap,nan=0.,posinf=0.,neginf=0.).clip(min=0)
        else:cap=np.full(n,10*max(nav,abs(previous).sum()))
        tradable=np.asarray(panel.execution_tradable[:,t+1])
        cap=np.where(tradable,cap,0.)
        lower=np.maximum(previous-cap,low);upper=np.minimum(previous+cap,high)
        current=neutral_projection(desired,lower,upper,tolerance=max(1e-7,nav*1e-11))
        trades=current-previous
        assert (abs(trades)<=cap+1.).all()
        buy=float(np.maximum(trades,0).sum());sell=float(np.maximum(-trades,0).sum())
        fee=buy*buy_bp/1e4
        total=np.asarray(panel.total_return[:,t+2],dtype=float)
        price=np.asarray(panel.price_return[:,t+2],dtype=float)
        missing=~np.isfinite(total)
        missing_weight=float(abs(current[missing]).sum()/nav)
        total=np.where(missing,0.,total)
        if adverse_missing:
            total=np.where(missing & (current>0),-1.,np.where(missing & (current<0),1.,total))
        price=np.where(np.isfinite(price),price,0.)
        pnl=float(current@total);start_nav=nav
        ending=current*(1+price)
        # Total P&L includes cash income/terminal value; the remaining stock
        # position is extinguished on terminal events, without double counting.
        ending[np.asarray(panel.terminal[:,t+2])]=0.
        liquidation_buy=0.
        if number==len(groups)-1:
            liquidation_buy=float(np.maximum(-ending,0).sum())
            fee+=liquidation_buy*buy_bp/1e4
            ending[:]=0.
        nav+=pnl-fee
        if nav<=0:raise FloatingPointError('NAV exhausted')
        row=dict(signal_date=str(date),execution_date=str(panel.dates[t+1]),date=str(panel.dates[t+2]),
            return_gross=pnl/start_nav,return_net=(pnl-fee)/start_nav,start_nav=start_nav,end_nav=nav,
            pnl=pnl,fees=fee,buy_value=buy,sell_value=sell,liquidation_buy=liquidation_buy,
            long_value=float(np.maximum(current,0).sum()),short_value=float(np.maximum(-current,0).sum()),
            net_exposure=float(current.sum()/start_nav),gross_exposure=float(abs(current).sum()/start_nav),
            missing_return_gross_weight=missing_weight,missing_held_count=int(((abs(current)>1e-6)&missing).sum()),
            end_net_drift=float(ending.sum()/nav),n_long=int((current>1e-6).sum()),n_short=int((current< -1e-6).sum()))
        assert abs(row['net_exposure'])<1e-8
        rows.append(row)
        if save_weights:held_weights.append(current/start_nav)
        previous=ending
    result=pd.DataFrame(rows)
    # Independently reconcile self-financing cash/stock P&L and NAV recurrence.
    np.testing.assert_allclose(result.end_nav,result.start_nav+result.pnl-result.fees,rtol=1e-12)
    np.testing.assert_allclose(2e9*np.cumprod(1+result.return_net),result.end_nav,rtol=1e-11)
    np.testing.assert_allclose(result.long_value,result.short_value,atol=2.)
    return result,np.array(held_weights) if save_weights else None
