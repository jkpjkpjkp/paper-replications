#!/usr/bin/env python3
"""Exact same-weights gross/net decomposition and covariance identity."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
OUT=HERE/'diagnostics';OUT.mkdir(exist_ok=True)

def run(folder,label):
    z=dict(np.load(folder/'daily.npz'));spec=json.loads((folder/'models.json').read_text())
    dates=pd.DatetimeIndex(z['dates']);lookup={s['id']:i for i,s in enumerate(spec)}
    rows=[];corr=[];matrix=[]
    for ci,cost in enumerate(z['costs']):
        for period,(lo,hi) in {'validation':(2005,2014),'holdout':(2015,2025),'all_oos':(2005,2025)}.items():
            mask=(dates.year>=lo)&(dates.year<=hi)
            for m,s in enumerate(spec):
                if not s['cycle']:continue
                b=lookup[s['id'].replace('__cycle','__base')]
                r=z['returns'][mask,ci,m];base=z['returns'][mask,ci,b];delta=r-base
                gross=(z['gross_return'][mask,ci,m]-z['gross_return'][mask,ci,b]).mean()*252
                fee=(z['fees'][mask,ci,m]-z['fees'][mask,ci,b]).mean()*252
                borrow=(z['borrow'][mask,ci,m]-z['borrow'][mask,ci,b]).mean()*252
                net=delta.mean()*252
                assert abs(net-(gross-fee-borrow))<1e-12
                vdelta=delta.var(ddof=1)*252;cov=2*np.cov(base,delta,ddof=1)[0,1]*252
                vchange=r.var(ddof=1)*252-base.var(ddof=1)*252
                assert abs(vchange-(vdelta+cov))<1e-12
                rows.append(dict(horizon=label,cost_bps=cost*10000,period=period,factor=s['factor'],method=s['method'],delta_net=net,delta_gross_same_weights=gross,delta_trading_fees=fee,delta_borrow=borrow,delta_turnover=fee/cost if cost else (z['turnover'][mask,ci,m]-z['turnover'][mask,ci,b]).mean()*252,cycle_gross=z['gross_return'][mask,ci,m].mean()*252,base_gross=z['gross_return'][mask,ci,b].mean()*252,cycle_net=r.mean()*252,base_net=base.mean()*252,variance_change=vchange,standalone_increment_variance=vdelta,covariance_term=cov,base_increment_correlation=np.corrcoef(base,delta)[0,1],cycle_base_correlation=np.corrcoef(r,base)[0,1],tracking_error=np.sqrt(vdelta),increment_information_ratio=net/np.sqrt(vdelta) if vdelta else np.nan))
            p=[i for i,s in enumerate(spec) if s['method']=='plain' and s['factor']!='Mom12m']
            for a in p:
                for b in p:
                    common=mask&(z['active'][:,ci,a]>0)&(z['active'][:,ci,b]>0)
                    corr.append(dict(horizon=label,cost_bps=cost*10000,period=period,factor_a=spec[a]['factor'],factor_b=spec[b]['factor'],correlation=np.corrcoef(z['returns'][mask,ci,a],z['returns'][mask,ci,b])[0,1],active_common_correlation=np.corrcoef(z['returns'][common,ci,a],z['returns'][common,ci,b])[0,1],common_days=int(common.sum())))
    return pd.DataFrame(rows),pd.DataFrame(corr)

if __name__=='__main__':
    a=[];b=[]
    for label,folder in [('daily',HERE/'results'),('five_day',HERE/'results_h5')]:
        if not (folder/'daily.npz').exists():continue
        aa,bb=run(folder,label);a.append(aa);b.append(bb)
    d=pd.concat(a,ignore_index=True);d.to_csv(OUT/'cost_risk_decomposition.csv',index=False)
    pd.concat(b,ignore_index=True).to_csv(OUT/'plain_factor_return_correlations.csv',index=False)
    q=d.query("horizon=='daily' and cost_bps==5 and period=='holdout'")
    summary=q.groupby('factor')[['delta_net','delta_gross_same_weights','delta_trading_fees','delta_borrow','delta_turnover','tracking_error','base_increment_correlation','cycle_base_correlation']].mean()
    summary.to_csv(OUT/'cost_risk_by_factor.csv')
    print(summary.to_string())
    print('Positive gross / positive net',sum(q.delta_gross_same_weights>0),sum(q.delta_net>0))
