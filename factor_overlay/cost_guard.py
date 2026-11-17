#!/usr/bin/env python3
"""Post-result strict local execution-cost gate for five-day cycle forecasts.

For each rebalance, compute baseline and enhanced targets from the SAME guard
holdings. Accept the enhanced target only if its exact transaction fee is lower
than the baseline target's fee; otherwise execute the baseline target. This is
a conditional-action rule, not a guarantee about cumulative fees on separately
evolved portfolios. All 36 existing families are evaluated, without selection.
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
from scipy.optimize import brentq
from data import HERE,ROOT,FACTORS,load_year,features
from models import METHODS
from portfolio import Ledger,allocate_leg

OUT=HERE/'results_cost_guard';COST=.0005
PAIRS=[dict(factor=f,method=m,id=f+'__'+m) for f in FACTORS for m in METHODS]
SPECS=[dict(**pair,variant=kind,model=pair['id']+'__'+kind) for pair in PAIRS for kind in ['baseline','guard']]


def fee_fraction(weights,old,exit_gross,cost=COST):
    """Exact rebalance fee on prior NAV, including targets outside current rows."""
    weights=np.asarray(weights,float);old=np.asarray(old,float)
    x=np.ones(len(weights))
    for _ in range(15):
        fee=cost*(abs(x[:,None]*weights-old).sum(axis=1)+exit_gross)
        x=1-fee
    fee=cost*(abs(x[:,None]*weights-old).sum(axis=1)+exit_gross)
    assert np.max(abs(x+fee-1))<1e-11
    return fee


def validate_rule():
    rng=np.random.default_rng(92178);errors=[]
    for _ in range(25):
        n=20;old=rng.normal(0,.04,n);w=rng.normal(0,.04,n);exit_gross=float(rng.uniform(0,.4))
        observed=fee_fraction(w[None,:],old[None,:],np.array([exit_gross]))[0]
        x=brentq(lambda a:a+COST*(abs(a*w-old).sum()+exit_gross)-1,0,1,xtol=1e-15)
        expected=COST*(abs(x*w-old).sum()+exit_gross)
        errors.append(abs(observed-expected))
    # From identical holdings a retained target must be accepted against a
    # full reversal, while swapping those candidates must reject it.
    old=np.array([[.5,.5,-.5,-.5]])
    retained=old.copy();reversed=-old
    cheap=fee_fraction(retained,old,np.array([0.]))[0]
    expensive=fee_fraction(reversed,old,np.array([0.]))[0]
    assert cheap<expensive-1e-12
    assert not(expensive<cheap-1e-12)
    assert max(errors)<1e-12
    return dict(independent_fee_cases=25,fee_max_difference=float(max(errors)),
                synthetic_accept_reject_pass=True)


def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    design=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
       status='Post-result mechanism follow-up; all36 existing families, no tuning or winner selection',
       training='Existing direct-five-day annual fits_h5, strict preceding10years',
       policy='From the same guard pretrade holdings, compute no-cycle target B and cycle target E using their existing forecasts and identical optimizer. Exact fee f=c*(sum|x*w-h|+exiting gross), x=1-f. Accept E iff fE<fB-1e-12, otherwise B.',
       cost_bps=5,borrow_annual=.01,evaluation=[2005,2025],
       caveat='Local-fee reduction relative to the contemporaneous counterfactual does not imply cumulative fee reduction relative to a baseline book with different past holdings. No cumulative budget constraint in this variant.',
       fixed_controls='Same eligibility/minimum count, weekly rebalance,+100/-100 target, caps, annual model parameters, full-return marking, terminal liquidation',
       validation=validate_rule())
    (OUT/'design.json').write_text(json.dumps(design,indent=2))
    (OUT/'models.json').write_text(json.dumps(SPECS,indent=2))
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    cal=cal[cal.date.dt.year.between(2005,2025)]
    ledger=Ledger(len(SPECS),COST);daily=[];dates=[];decisions=[];maxfeeerror=0.
    c=duckdb.connect();c.execute('SET threads=4')
    for year in range(2005,2026):
        fitted=pickle.loads((HERE/f'fits_h5/{year}.pkl').read_bytes())
        z=load_year(year);rr=np.flatnonzero(z['idx']%5==0);z={k:v[rr] for k,v in z.items()}
        mb=np.zeros((len(PAIRS),len(rr)),np.float32);me=np.zeros_like(mb)
        sb=np.full_like(mb,.03);se=np.full_like(mb,.03)
        for j,factor in enumerate(FACTORS):
            rows=np.flatnonzero(z['leg'][:,j]!=0);xb=features(z,j,rows);xc=features(z,j,rows,cycle=True)
            for k,method in enumerate(METHODS):
                p=j*6+k
                mb[p,rows],sb[p,rows]=fitted[factor+'__'+method+'__base'].predict(xb)
                me[p,rows],se[p,rows]=fitted[factor+'__'+method+'__cycle'].predict(xc)
        ri=z['idx'];ss=np.r_[0,np.flatnonzero(np.diff(ri))+1];ee=np.r_[ss[1:],len(ri)]
        rmap={int(ri[a]):(a,b) for a,b in zip(ss,ee)}
        market=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        for key in ['ret','retx']:market[key]=np.ma.filled(market[key],np.nan)
        mi=market['idx'];ss=np.r_[0,np.flatnonzero(np.diff(mi))+1];ee=np.r_[ss[1:],len(mi)]
        mmap={int(mi[a]):(a,b) for a,b in zip(ss,ee)}
        for day in cal[cal.date.dt.year.eq(year)].itertuples():
            idx=int(day.idx);ma,mz=mmap[idx];ids=None;weights=None;selectedfee=None
            terminal=idx==int(cal.idx.iloc[-1])
            if idx%5==0:
                a,b=rmap.get(idx,(0,0));ids=z['permno'][a:b]
                old=ledger.lookup(ids);weights=np.zeros((len(SPECS),len(ids)))
                wb=np.zeros((len(PAIRS),len(ids)));we=np.zeros_like(wb);active=np.zeros(len(PAIRS),bool)
                for j,factor in enumerate(FACTORS):
                    leg=z['leg'][a:b,j];long=np.flatnonzero(leg==1);short=np.flatnonzero(leg==-1)
                    if min(len(long),len(short))<(5 if factor=='RDIPO' else 20):continue
                    pairs=np.arange(j*6,j*6+6);bases=2*pairs;guards=bases+1;active[pairs]=True
                    for sign,rows in [(1,long),(-1,short)]:
                        bmu=mb[np.ix_(pairs,a+rows)];bsigma=sb[np.ix_(pairs,a+rows)]
                        emu=me[np.ix_(pairs,a+rows)];esigma=se[np.ix_(pairs,a+rows)]
                        weights[np.ix_(bases,rows)]=sign*allocate_leg(bmu,bsigma,sign*old[np.ix_(bases,rows)],COST)
                        wb[np.ix_(pairs,rows)]=sign*allocate_leg(bmu,bsigma,sign*old[np.ix_(guards,rows)],COST)
                        we[np.ix_(pairs,rows)]=sign*allocate_leg(emu,esigma,sign*old[np.ix_(guards,rows)],COST)
                guards=np.arange(1,len(SPECS),2)
                exiting=~np.isin(ledger.ids,ids)
                exitgross=abs(ledger.hold[np.ix_(guards,np.flatnonzero(exiting))]).sum(axis=1)
                fb=fee_fraction(wb,old[guards],exitgross);fe=fee_fraction(we,old[guards],exitgross)
                accept=fe<fb-1e-12
                weights[guards]=np.where(accept[:,None],we,wb)
                selectedfee=np.where(accept,fe,fb)
                assert np.all(selectedfee<=fb+1e-12)
                for p,pair in enumerate(PAIRS):
                    decisions.append(dict(date=str(day.date.date()),year=year,**pair,active=bool(active[p]),accepted=bool(accept[p]),
                        fallback_fee=float(fb[p]),enhanced_fee=float(fe[p]),selected_fee=float(selectedfee[p]),
                        local_fee_saving=float(fb[p]-selectedfee[p]),
                        target_l1_difference=float(abs(we[p]-wb[p]).sum())))
            out=ledger.step(market['permno'][ma:mz],market['ret'][ma:mz],market['retx'][ma:mz],ids,weights,terminal=terminal)
            if selectedfee is not None and not terminal:
                maxfeeerror=max(maxfeeerror,float(abs(out['fees'][1::2]-selectedfee).max()))
            daily.append(out);dates.append(day.date)
        log(f'cost guard {year} complete')
    arrays={k:np.stack([d[k] for d in daily]) for k in daily[0]}
    arrays['dates']=np.array(dates,dtype='datetime64[D]')
    np.savez_compressed(OUT/'daily.npz',**arrays)
    pd.DataFrame(decisions).to_csv(OUT/'decisions.csv',index=False)
    ref=np.load(HERE/'results_h5/daily.npz');refs=json.loads((HERE/'results_h5/models.json').read_text());lookup={s['id']:i for i,s in enumerate(refs)}
    ci=int(np.flatnonzero(np.isclose(ref['costs'],COST))[0]);assert np.array_equal(arrays['dates'],ref['dates'])
    checks=[]
    for p,pair in enumerate(PAIRS):
        ri=lookup[pair['id']+'__base']
        re=float(abs(arrays['returns'][:,2*p]-ref['returns'][:,ci,ri]).max())
        te=float(abs(arrays['turnover'][:,2*p]-ref['turnover'][:,ci,ri]).max())
        assert re<1e-10 and te<1e-9,(pair,re,te)
        checks.append(dict(**pair,return_max_error=re,turnover_max_error=te))
    fits=[r for y in range(2005,2026) for r in json.loads((HERE/f'fits_h5/{y}.json').read_text())]
    assert all(r['training_last_year']<r['prediction_year'] for r in fits)
    assert maxfeeerror<1e-10
    (OUT/'validation.json').write_text(json.dumps(dict(reference_checks=checks,selected_fee_max_error=maxfeeerror,
        strict_local_veto_violations=0,all_fits_strictly_past=True,max_accounting_error=ledger.max_accounting_error,
        terminal_holdings=int(ledger.hold.size),bankrupt=ledger.dead.tolist(),synthetic_rule=design['validation']),indent=2))
    summarize(arrays,pd.DataFrame(decisions))


def summarize(z,decisions):
    from scipy.stats import norm
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests
    years=pd.DatetimeIndex(z['dates']).year;rows=[]
    for period,lo,hi in [('validation',2005,2014),('holdout',2015,2025),('all_oos',2005,2025)]:
        mask=(years>=lo)&(years<=hi)
        for p,pair in enumerate(PAIRS):
            base=2*p;guard=base+1;b=z['returns'][mask,base];g=z['returns'][mask,guard];d=g-b
            fit=sm.OLS(d,np.ones((len(d),1))).fit(cov_type='HAC',cov_kwds={'maxlags':60})
            own=sm.OLS(g,np.ones((len(g),1))).fit(cov_type='HAC',cov_kwds={'maxlags':60})
            conf=fit.conf_int()[0]*252
            dec=decisions[(decisions.year>=lo)&(decisions.year<=hi)&decisions.id.eq(pair['id'])]
            rows.append(dict(**pair,period=period,base_annual_mean=b.mean()*252,guard_annual_mean=g.mean()*252,
                delta_annual_mean=d.mean()*252,ci_low=conf[0],ci_high=conf[1],p=fit.pvalues[0],own_p=own.pvalues[0],
                base_annual_vol=b.std(ddof=1)*np.sqrt(252),guard_annual_vol=g.std(ddof=1)*np.sqrt(252),
                delta_cer3=d.mean()*252-1.5*252*(g.var(ddof=1)-b.var(ddof=1)),
                base_annual_fees=z['fees'][mask,base].mean()*252,guard_annual_fees=z['fees'][mask,guard].mean()*252,
                delta_annual_fees=(z['fees'][mask,guard]-z['fees'][mask,base]).mean()*252,
                delta_annual_gross=(z['gross_return'][mask,guard]-z['gross_return'][mask,base]).mean()*252,
                active_rebalances=int(dec.active.sum()),accepted_rebalances=int(dec.accepted.sum()),
                acceptance_fraction=float(dec.loc[dec.active,'accepted'].mean()),
                sum_local_fee_saving=float(dec.local_fee_saving.sum()),
                mean_annual_local_fee_saving=float(dec.local_fee_saving.sum()/(hi-lo+1))))
    stats=pd.DataFrame(rows)
    for _,g in stats.groupby('period'):
        stats.loc[g.index,'holm_p']=multipletests(g.p,method='holm')[1]
        stats.loc[g.index,'own_holm_p']=multipletests(g.own_p,method='holm')[1]
    stats['joint_success']=(stats.delta_annual_mean>0)&(stats.holm_p<.05)&(stats.guard_annual_mean>0)&(stats.own_holm_p<.05)
    stats.to_csv(OUT/'performance.csv',index=False)
    h=stats[stats.period.eq('holdout')]
    summary=dict(holdout_positive_increments=int((h.delta_annual_mean>0).sum()),
        holdout_significant_positive=int(((h.delta_annual_mean>0)&(h.holm_p<.05)).sum()),
        holdout_joint_success=int(h.joint_success.sum()),
        holdout_cumulative_fees_lower=int((h.delta_annual_fees<0).sum()),
        mean_acceptance_fraction=float(h.acceptance_fraction.mean()),
        max_fee_increase_bps=float(h.delta_annual_fees.max()*10000),
        caveat='All accepted cycle targets strictly reduce contemporaneous fees versus fallback from SAME holdings. This does not guarantee lower cumulative fees versus separately evolved baseline; see counts.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':run()
