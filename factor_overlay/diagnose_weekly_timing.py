#!/usr/bin/env python3
"""Fixed original-clock weekly portfolio vs actual weekly factor baselines."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import pandas as pd
import duckdb
from data import HERE, ROOT, FACTORS, load_year
from portfolio import Ledger
from diagnose_correlations import corr,hac_mean

OUT=HERE/'diagnostics'

def main():
    OUT.mkdir(exist_ok=True)
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    cal=cal[cal.date.dt.year.between(2005,2025)]
    ledger=Ledger(1,.0005)
    c=duckdb.connect();c.execute('SET threads=4')
    results=[];dates=[]
    for year in range(2005,2026):
        z=load_year(year)
        ix=z['idx'];starts=np.r_[0,np.flatnonzero(np.diff(ix))+1];ends=np.r_[starts[1:],len(ix)]
        emap={int(ix[a]):(a,b) for a,b in zip(starts,ends)}
        raw=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        for key in ['ret','retx']:raw[key]=np.ma.filled(raw[key],np.nan)
        ix=raw['idx'];starts=np.r_[0,np.flatnonzero(np.diff(ix))+1];ends=np.r_[starts[1:],len(ix)]
        mmap={int(ix[a]):(a,b) for a,b in zip(starts,ends)}
        for day in cal[cal.date.dt.year.eq(year)].itertuples():
            idx=int(day.idx);a,b=mmap[idx]
            ids=None;w=None
            if idx%5==0:
                ea,eb=emap.get(idx,(0,0));ids=z['permno'][ea:eb];clock=z['clock'][ea:eb]
                long=(clock>=-5)&(clock<=-1);short=(clock>=-40)&(clock<=-21)
                w=np.zeros((1,len(ids)))
                if min(long.sum(),short.sum())>=20:
                    w[0,long]=1/long.sum();w[0,short]=-1/short.sum()
            result=ledger.step(raw['permno'][a:b],raw['ret'][a:b],raw['retx'][a:b],ids,w,
                               terminal=idx==int(cal.idx.iloc[-1]))
            results.append(result);dates.append(day.date)
        print(f'Fixed weekly timing ledger {year}',flush=True)
    arrays={k:np.stack([r[k] for r in results])[:,0] for k in results[0]}
    arrays.update(dates=np.array(dates,dtype='datetime64[D]'),idx=cal.idx.to_numpy(),cost=.0005)
    np.savez_compressed(OUT/'timing_daily.npz',**arrays)
    df=pd.DataFrame({k:v for k,v in arrays.items() if np.ndim(v)==1});df.to_csv(OUT/'timing_daily.csv',index=False)
    base=dict(np.load(HERE/'results/daily.npz'))
    assert np.array_equal(base['dates'],arrays['dates'])
    specs=json.loads((HERE/'results/models.json').read_text());lookup={s['id']:i for i,s in enumerate(specs)}
    cost_idx=int(np.flatnonzero(np.isclose(base['costs'],.0005))[0])
    years=pd.DatetimeIndex(arrays['dates']).year
    periods={'all_2005_2025':(2005,2025),'2005_2014':(2005,2014),'2015_2025':(2015,2025)}
    rows=[];performance=[]
    for period,(first,last) in periods.items():
        mask=(years>=first)&(years<=last)
        for series,key in [('gross_same_cost_holdings','gross_return'),('net','returns')]:
            r=arrays[key][mask];variance=float(r.var(ddof=1));vol=np.sqrt(variance)
            performance.append(dict(period=period,series=series,n=len(r),mean_daily=float(r.mean()),
                mean_bps_day=float(r.mean()*1e4),variance_daily=variance,sd_daily=float(vol),
                annual_mean=float(r.mean()*252),annual_vol=float(vol*np.sqrt(252)),
                sharpe=float(r.mean()/vol*np.sqrt(252)),active_fraction=float(arrays['active'][mask].mean()),
                annual_turnover=float(arrays['turnover'][mask].mean()*252),
                annual_fees=float(arrays['fees'][mask].mean()*252),annual_borrow=float(arrays['borrow'][mask].mean()*252)))
            for f in FACTORS:
                j=lookup[f+'__plain']
                for support in ['all_dates','both_active']:
                    use=mask.copy()
                    if support=='both_active':use &= (arrays['active']>0)&(base['active'][:,cost_idx,j]>0)
                    x=base[key][use,cost_idx,j];y=arrays[key][use];idx=arrays['idx'][use]
                    rho=corr(x,y)
                    if len(x)>2 and np.isfinite(rho):
                        zx=(x-x.mean())/x.std();zy=(y-y.mean())/y.std()
                        psi=zx*zy-rho*(zx*zx+zy*zy)/2
                        se=hac_mean(psi,idx)['se']
                    else:se=np.nan
                    rows.append(dict(period=period,factor=f,series=series,support=support,n=len(x),
                        fraction_period=float(use.sum()/mask.sum()),correlation=rho,corr_hac_se=se,
                        corr_ci_low=max(-1,rho-1.95996398454*se),corr_ci_high=min(1,rho+1.95996398454*se),
                        baseline_mean_bps_day=float(x.mean()*1e4),timing_mean_bps_day=float(y.mean()*1e4),
                        baseline_variance_daily=float(x.var(ddof=1)),timing_variance_daily=float(y.var(ddof=1)),
                        covariance_daily=float(np.cov(x,y,ddof=1)[0,1])))
    pd.DataFrame(rows).to_csv(OUT/'weekly_timing_return_correlations.csv',index=False)
    pd.DataFrame(performance).to_csv(OUT/'weekly_timing_performance.csv',index=False)
    definition=dict(status='Fixed diagnostic requested after primary results; no parameter/model selection',
        timing_rule='Original clock: long[-5,-1],short[-40,-21],equalweight,minimum20each,otherwisecash',
        universe='All existing eligible regular65/>=5analyst names; no factor availability restriction. This differs from each factor-specific finite universe.',
        schedule='Target before return on market idx%5==0; same schedule and initial date as actual baseline backtests',
        accounting='Shared audited Ledger; same full-panel marks, fixed-share drift, forced missing exits, terminal liquidation',
        costs='5bp per dollar traded and1%annual short borrow; exact cost feedback',
        gross_definition='Gross return is daily price/distribution P&L from5bp-scenario actual holdings; fee/borrow deductions excluded from that day numerator but affect holdings via prior NAV. Not the separate zero-cost portfolio.',
        comparators='Actual factor_overlay/results/daily.npz plain model rows, cost5bp, exact identical calendar dates',
        support='all_dates includes cash zeros; both_active retains only dates both actual portfolios held positions',
        statistical_uncertainty='Descriptive95% delta-method Bartlett HAC60 correlation intervals in original trading-session distances; no multiplicity or causal claim',
        n_days=len(dates),max_accounting_error=ledger.max_accounting_error,
        bankrupt=bool(ledger.dead.any()),terminal_holdings=int(ledger.hold.size))
    (OUT/'weekly_timing_definitions.json').write_text(json.dumps(definition,indent=2))
    q=pd.DataFrame(rows)
    print(q.query("period=='2015_2025' and support=='all_dates'")[['factor','series','n','correlation','baseline_mean_bps_day','timing_mean_bps_day']].to_string(index=False))
    print(pd.DataFrame(performance).query("period=='2015_2025'").to_string(index=False))

if __name__=='__main__':main()
