#!/usr/bin/env python3
"""Independent scalar-dollar replay of the integrated walk-forward execution.

Reuses the fitted forecasts and observed data, but does not call production
Ledger/allocate_leg. Scipy Brent root finding solves both scalar multipliers
and post-fee equity. This checks production indexing/rounding and integration,
not the investment hypothesis or historical point-in-time data provenance.
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from scipy.optimize import brentq
from data import HERE, ROOT, FACTORS, load_year, features


def independent_leg(mu, sigma, previous, cost=.0005):
    n=len(mu);base=1/n
    sigma=np.clip(np.nan_to_num(sigma,nan=.03,posinf=.2,neginf=.005),.005,.2)
    quadratic=50*sigma**2/base
    lower=.5*base;upper=min(1.5*base,.2)
    if abs(n*upper-1)<1e-12:return np.full(n,base)
    def allocation(lam):
        # Explicit three-way proximal solution, separately from production
        # sign/max formula, with the leg-total Lagrange multiplier.
        center=base+(5*mu-lam)/quadratic
        threshold=cost/quadratic
        candidate=np.where(center>previous+threshold,center-threshold,
                           np.where(center<previous-threshold,center+threshold,previous))
        return np.minimum(np.maximum(candidate,lower),upper)
    lo=float(np.min(5*mu-quadratic*(upper-base)-cost))-1e-10
    hi=float(np.max(5*mu-quadratic*(lower-base)+cost))+1e-10
    lam=brentq(lambda v:allocation(v).sum()-1,lo,hi,xtol=1e-15)
    out=allocation(lam)
    assert abs(out.sum()-1)<1e-10
    return out


def replay_prefix(days=20):
    source=HERE/'results/daily.npz'
    if not source.exists():raise FileNotFoundError('Run evaluation before validation')
    z=load_year(2005)
    fitted=pickle.loads((HERE/'fits/2005.pkl').read_bytes())
    names=['Cash__plain','Cash__ridge__cycle']
    models=json.loads((HERE/'results/models.json').read_text())
    mids=[next(i for i,m in enumerate(models) if m['id']==name) for name in names]
    factor=FACTORS.index('Cash');cost=.0005
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    cal=cal[cal.date.dt.year.eq(2005)].head(days)
    market=duckdb.connect().execute(f'''SELECT idx,permno,ret,retx FROM read_parquet(
      '{ROOT}/coverage5/results_regular65/panel/2005.parquet')
      WHERE idx<={int(cal.idx.max())} ORDER BY idx,permno''').df()
    with np.load(source) as obj:
        arrays={k:obj[k][:days] for k in ['returns','turnover','fees','borrow','gross_exposure','net_exposure','n_long','n_short','forced_exit','missing']}
        assert np.array_equal(obj['dates'][:days],cal.date.to_numpy(dtype='datetime64[D]'))
        ci=int(np.flatnonzero(np.isclose(obj['costs'],cost))[0])
    nav=np.ones(len(names));held=[{},{}]
    errors={key:0. for key in arrays}
    observations=[]
    for t,day in enumerate(cal.itertuples()):
        frame=market[market.idx.eq(day.idx)]
        today={int(row.permno):(float(row.ret),float(row.retx)) for row in frame.itertuples()}
        rows=np.flatnonzero(z['idx']==int(day.idx))
        ids=z['permno'][rows];legs=z['leg'][rows,factor]
        for j,name in enumerate(names):
            oldnav=nav[j];old=held[j]
            forced=sum(v!=0 and p not in today for p,v in old.items())
            if int(day.idx)%5==0:
                target={}
                if min(np.count_nonzero(legs==1),np.count_nonzero(legs==-1))>=20:
                    for side in [1,-1]:
                        rr=np.flatnonzero(legs==side)
                        if j==0:weights=np.full(len(rr),1/len(rr))
                        else:
                            x=features(z,factor,rows[rr],cycle=True)
                            # Production stores forecast arrays in float32.
                            mu,sigma=fitted[name].predict(x)
                            mu=mu.astype(np.float32).astype(float)
                            sigma=sigma.astype(np.float32).astype(float)
                            previous=np.array([side*old.get(int(p),0)/oldnav for p in ids[rr]])
                            weights=independent_leg(mu,sigma,previous,cost)
                        target.update({int(p):side*float(w) for p,w in zip(ids[rr],weights)})
                union=set(old)|set(target)
                def fee_equation(afterfee):
                    return afterfee+cost*sum(abs(afterfee*target.get(p,0)-old.get(p,0)) for p in union)-oldnav
                equity=brentq(fee_equation,0,oldnav,xtol=1e-15)
                positions={p:equity*w for p,w in target.items()}
            else:positions={p:v for p,v in old.items() if p in today}
            turnover=sum(abs(positions.get(p,0)-old.get(p,0)) for p in set(positions)|set(old))
            fee=cost*turnover
            borrow=.01/252*sum(-v for v in positions.values() if v<0)
            gross=0.;end={};missing=0
            for p,v in positions.items():
                r,rx=today[p]
                if not np.isfinite(r):missing+=v!=0;r=0
                if not np.isfinite(rx):rx=r
                gross+=v*r;end[p]=v*(1+rx)
            nav[j]=oldnav+gross-fee-borrow
            held[j]=end
            expected=dict(returns=nav[j]/oldnav-1,turnover=turnover/oldnav,fees=fee/oldnav,
                          borrow=borrow/oldnav,gross_exposure=sum(abs(v) for v in positions.values())/oldnav,
                          net_exposure=sum(positions.values())/oldnav,
                          n_long=sum(v>1e-15 for v in positions.values()),n_short=sum(v< -1e-15 for v in positions.values()),
                          forced_exit=forced,missing=missing)
            for key,val in expected.items():
                diff=abs(val-arrays[key][t,ci,mids[j]])
                errors[key]=max(errors[key],float(diff))
            observations.append(dict(date=str(day.date.date()),model=name,reference_nav=nav[j],**expected))
    assert max(errors.values())<1e-9,errors
    pd.DataFrame(observations).to_csv(HERE/'results/independent_prefix.csv',index=False)
    return dict(models=names,days=days,cost_bps=5,comparison_max_errors=errors)


def chronology():
    records=json.loads((HERE/'results/fit_audit.json').read_text())
    for r in records:
        assert r['training_last_year']==r['prediction_year']-1
        assert r['training_first_year']==r['prediction_year']-10
    # Match reported training row counts to actual timestamped sample records.
    with np.load(HERE/'data/train.npz') as t:
        for year in [2005,2015,2025]:
            for j,factor in enumerate(FACTORS):
                expected=int(np.sum((t['year']>=year-10)&(t['year']<year)&(t['factor']==j)))
                subset=[r for r in records if r['prediction_year']==year and r['id'].startswith(factor+'__')]
                assert len(subset)==12
                assert all(r['n_train']==expected for r in subset)
    execution=json.loads((HERE/'results/execution_audit.json').read_text())
    assert execution['terminal_holdings']==[0,0,0]
    assert execution['max_accounting_error']<1e-10
    return dict(annual_model_fits=len(records),all_training_windows_strictly_past=True,
                sampled_training_count_checks=3*6*12,
                terminal_holdings=execution['terminal_holdings'],
                execution_max_accounting_error=execution['max_accounting_error'])


if __name__=='__main__':
    result=dict(prefix_replay=replay_prefix(),chronology=chronology())
    (HERE/'results/integrated_validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
