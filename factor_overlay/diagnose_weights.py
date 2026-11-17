#!/usr/bin/env python3
"""Post-result mechanism diagnostic: cross the fixed ridge mean/risk forecasts.

No fit or tuning occurs here. All four cells use the identical annual ridge
models from the main experiment. This is a plug-in attribution diagnostic, not
a causal decomposition: the variance models use their own mean residuals, and
cost-aware weights and holdings respond jointly and nonlinearly.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import json,pickle,time
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from data import HERE,ROOT,FACTORS,load_year,features
from portfolio import Ledger,allocate_leg

OUT=HERE/'results/risk_mean_ablation'
VARIANTS=['base','cycle_mean_only','cycle_risk_only','cycle_both']
MEAN_SOURCE=[0,1,0,1]
RISK_SOURCE=[0,0,1,1]
COST=.0005
SPECS=[dict(id=f+'__'+v,factor=f,variant=v,cycle_mean=bool(MEAN_SOURCE[k]),cycle_risk=bool(RISK_SOURCE[k]))
       for f in FACTORS for k,v in enumerate(VARIANTS)]


def log(text):print(time.strftime('%H:%M:%S'),text,flush=True)


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'design.json').write_text(json.dumps(dict(
        created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='post-result mechanism diagnostic; no selection or tuning',
        method='fixed existing annual ridge fits; four crossed mean/risk input cells',
        variants=VARIANTS,cost_bps=5,borrow_annual=.01,period=[2005,2025],
        interpretation='Plug-in channels, not causal effects. Variance models are fitted to their own mean-model residuals. Nonlinear cost-aware optimizer creates interactions. Controls must reproduce main ridge returns.',
        diagnostics='Optimizer target weights before common post-fee NAV scaling. No-trade means proximal solution equals prior own-NAV weight; actual dollar transactions can remain due post-fee scaling.'),indent=2))
    calendar=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    calendar=calendar[calendar.date.dt.year.between(2005,2025)]
    ledger=Ledger(len(SPECS),COST);daily=[];dates=[];diags=[]
    c=duckdb.connect();c.execute('SET threads=4')
    for year in range(2005,2026):
        fitted=pickle.loads((HERE/f'fits/{year}.pkl').read_bytes())
        z=load_year(year);rr=np.flatnonzero(z['idx']%5==0)
        z={k:v[rr] for k,v in z.items()}
        mu=np.zeros((len(SPECS),len(rr)),np.float32);sigma=np.full_like(mu,.03)
        for j,factor in enumerate(FACTORS):
            rows=np.flatnonzero(z['leg'][:,j]!=0)
            xb=features(z,j,rows);xc=features(z,j,rows,cycle=True)
            m0,s0=fitted[factor+'__ridge__base'].predict(xb)
            m1,s1=fitted[factor+'__ridge__cycle'].predict(xc)
            for k in range(4):
                mu[j*4+k,rows]=[m0,m1][MEAN_SOURCE[k]]
                sigma[j*4+k,rows]=[s0,s1][RISK_SOURCE[k]]
        ri=z['idx'];starts=np.r_[0,np.flatnonzero(np.diff(ri))+1];ends=np.r_[starts[1:],len(ri)]
        rmap={int(ri[a]):(a,b) for a,b in zip(starts,ends)}
        market=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        for key in ['ret','retx']:market[key]=np.ma.filled(market[key],np.nan)
        mi=market['idx'];ss=np.r_[0,np.flatnonzero(np.diff(mi))+1];ee=np.r_[ss[1:],len(mi)]
        mmap={int(mi[a]):(a,b) for a,b in zip(ss,ee)}
        for day in calendar[calendar.date.dt.year.eq(year)].itertuples():
            idx=int(day.idx);ma,mb=mmap[idx];ids=None;weights=None
            if idx%5==0:
                a,b=rmap.get(idx,(0,0));ids=z['permno'][a:b]
                old=ledger.lookup(ids);weights=np.zeros((len(SPECS),len(ids)))
                for j,factor in enumerate(FACTORS):
                    leg=z['leg'][a:b,j];long=np.flatnonzero(leg==1);short=np.flatnonzero(leg==-1)
                    if min(len(long),len(short))<(5 if factor=='RDIPO' else 20):continue
                    mods=np.arange(j*4,j*4+4)
                    for sign,rows in [(1,long),(-1,short)]:
                        previous=sign*old[np.ix_(mods,rows)]
                        m=mu[np.ix_(mods,a+rows)].astype(float)
                        s=sigma[np.ix_(mods,a+rows)].astype(float)
                        alloc=allocate_leg(m,s,previous,COST)
                        weights[np.ix_(mods,rows)]=sign*alloc
                        n=len(rows);base=1/n;lower=.5*base;upper=min(1.5*base,.2)
                        for k,model in enumerate(mods):
                            interior=(alloc[k]>lower+1e-10)&(alloc[k]<upper-1e-10)
                            row=dict(date=str(day.date.date()),year=year,factor=factor,variant=VARIANTS[k],leg=sign,n=n,
                                     abs_multiplier_tilt=float(np.mean(abs(alloc[k]/base-1))),
                                     lower_fraction=float(np.mean(abs(alloc[k]-lower)<1e-9)),
                                     upper_fraction=float(np.mean(abs(alloc[k]-upper)<1e-9)),
                                     optimizer_no_trade_fraction=float(np.mean((abs(alloc[k]-previous[k])<1e-9)&interior)),
                                     allocation_l1_to_base=float(abs(alloc[k]-alloc[0]).sum()),
                                     forecast_mean=float(m[k].mean()),forecast_mean_sd=float(m[k].std()),
                                     forecast_rms_mean_change=float(np.sqrt(np.mean((m[k]-m[0])**2))),
                                     forecast_mean_variance=float(np.mean(s[k]**2)),
                                     forecast_rms_logvariance_change=float(np.sqrt(np.mean((np.log(s[k]**2)-np.log(s[0]**2))**2))))
                            diags.append(row)
            out=ledger.step(market['permno'][ma:mb],market['ret'][ma:mb],market['retx'][ma:mb],ids,weights,terminal=idx==int(calendar.idx.iloc[-1]))
            daily.append(out);dates.append(day.date)
        log(f'ablation {year} complete')
    arrays={k:np.stack([a[k] for a in daily]) for k in daily[0]}
    arrays.update(dates=np.array(dates,dtype='datetime64[D]'))
    np.savez_compressed(OUT/'daily.npz',**arrays)
    pd.DataFrame(diags).to_csv(OUT/'rebalance_diagnostics.csv',index=False)
    (OUT/'models.json').write_text(json.dumps(SPECS,indent=2))
    # Exact path reproduction is essential before interpreting the cross cells.
    reference=np.load(HERE/'results/daily.npz')
    refspec=json.loads((HERE/'results/models.json').read_text());reflookup={s['id']:i for i,s in enumerate(refspec)}
    ci=int(np.flatnonzero(np.isclose(reference['costs'],COST))[0])
    assert np.array_equal(reference['dates'],arrays['dates'])
    checks=[]
    for j,factor in enumerate(FACTORS):
        for k,suffix in [(0,'base'),(3,'cycle')]:
            refid=reflookup[factor+'__ridge__'+suffix]
            err=float(np.max(abs(arrays['returns'][:,j*4+k]-reference['returns'][:,ci,refid])))
            turnerr=float(np.max(abs(arrays['turnover'][:,j*4+k]-reference['turnover'][:,ci,refid])))
            assert err<1e-10 and turnerr<1e-9,(factor,suffix,err,turnerr)
            checks.append(dict(factor=factor,variant=VARIANTS[k],return_max_difference=err,turnover_max_difference=turnerr))
    (OUT/'validation.json').write_text(json.dumps(dict(reference_checks=checks,max_accounting_error=ledger.max_accounting_error,
        terminal_holdings=int(ledger.hold.size),bankrupt=ledger.dead.tolist()),indent=2))
    summarize(arrays,pd.DataFrame(diags))


def summarize(arrays,diagnostics):
    years=pd.DatetimeIndex(arrays['dates']).year
    stats=[];effects=[]
    for period,low,high in [('validation',2005,2014),('holdout',2015,2025),('all_oos',2005,2025)]:
        mask=(years>=low)&(years<=high)
        r=arrays['returns'][mask]
        for model,spec in enumerate(SPECS):
            rr=r[:,model];var=252*rr.var(ddof=1);mean=252*rr.mean()
            base_model=(model//4)*4;base=r[:,base_model]
            stats.append(dict(**spec,period=period,annual_mean=mean,annual_vol=np.sqrt(var),
                              sharpe=mean/np.sqrt(var) if var>0 else np.nan,
                              cagr=np.prod(1+rr)**(252/len(rr))-1,cer3=mean-1.5*var,
                              delta_mean_vs_base=252*(rr-base).mean(),
                              delta_cer3_vs_base=252*(rr-base).mean()-1.5*(var-252*base.var(ddof=1)),
                              annual_gross_return=252*arrays['gross_return'][mask,model].mean(),
                              annual_fees=252*arrays['fees'][mask,model].mean(),
                              annual_borrow=252*arrays['borrow'][mask,model].mean(),
                              annual_turnover=252*arrays['turnover'][mask,model].mean(),
                              correlation_with_base=float(np.corrcoef(rr,base)[0,1])))
        for j,factor in enumerate(FACTORS):
            baseline,meanonly,riskonly,both=r[:,j*4:j*4+4].T
            effects.append(dict(factor=factor,period=period,
                cycle_mean_only_increment=252*(meanonly-baseline).mean(),
                cycle_risk_only_increment=252*(riskonly-baseline).mean(),
                both_increment=252*(both-baseline).mean(),
                interaction_increment=252*(both-meanonly-riskonly+baseline).mean()))
    pd.DataFrame(stats).to_csv(OUT/'performance.csv',index=False)
    effect=pd.DataFrame(effects);effect.to_csv(OUT/'channel_decomposition.csv',index=False)
    avg=diagnostics[diagnostics.year>=2015].groupby(['factor','variant','leg']).mean(numeric_only=True)
    avg.to_csv(OUT/'holdout_weight_diagnostics.csv')
    note='Post-result plug-in channel diagnostic; annual arithmetic net returns. No new tuning.\n'+effect[effect.period.eq('holdout')].to_string(index=False)+'\n'
    (OUT/'summary.txt').write_text(note)
    print(note,flush=True)


if __name__=='__main__':run()
