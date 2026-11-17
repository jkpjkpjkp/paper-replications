#!/usr/bin/env python3
"""All-eligible cross-sectional correlations with realized return labels.

Future returns define outcomes only; no portfolio or predictive feature changes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import itertools,json
import numpy as np
import pandas as pd
import duckdb
from scipy.stats import rankdata
from data import HERE,ROOT,FACTORS,load_year
from diagnose_correlations import corr,hac_mean

OUT=HERE/'diagnostics'
NAMES=FACTORS+['clock_score','return_1d','return_5d']

def labels5(z,year,c):
    raw=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
    key=raw['idx'].astype(np.int64)*1000000+raw['permno'].astype(np.int64)
    assert np.all(np.diff(key)>0)
    ret=np.ma.filled(raw['ret'],np.nan);retx=np.ma.filled(raw['retx'],np.nan)
    pos=np.ones(len(z['idx']));payoff=np.zeros(len(pos));forced=np.zeros(len(pos),bool)
    missing=0;returns=[];exreturns=[];present=[]
    for h in range(5):
        target=(z['idx'].astype(np.int64)+h)*1000000+z['permno'].astype(np.int64)
        loc=np.searchsorted(key,target)
        found=(loc<len(key))&(key[np.minimum(loc,len(key)-1)]==target)
        forced|=(pos!=0)&~found;pos[~found]=0
        rr=np.zeros(len(pos));rx=np.zeros(len(pos));rr[found]=ret[loc[found]];rx[found]=retx[loc[found]]
        missing+=int(((pos!=0)&found&~np.isfinite(rr)).sum())
        rr=np.where(np.isfinite(rr),rr,0.);rx=np.where(np.isfinite(rx),rx,rr)
        payoff+=pos*rr;pos*=1+rx
        returns.append(rr);exreturns.append(rx);present.append(found)
    # Independently compute fixed-share cashflow products for deterministic rows.
    checks=[]
    for j in np.linspace(0,len(pos)-1,min(5,len(pos)),dtype=int):
        value=0.;mark=1.
        for h in range(5):
            if not present[h][j]:break
            value+=mark*returns[h][j];mark*=1+exreturns[h][j]
        checks.append(abs(value-payoff[j]));assert checks[-1]<1e-12
    return payoff,dict(year=year,eligible_weekly_rows=len(pos),forced_missing_listing_labels=int(forced.sum()),
                        missing_return_marks=missing,scalar_checks=len(checks),max_scalar_error=max(checks))

def main():
    OUT.mkdir(exist_ok=True)
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    end=cal.assign(year=cal.date.dt.year).groupby('year').idx.max().to_dict()
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='5GB'")
    pairs=list(itertools.combinations_with_replacement(range(len(NAMES)),2))
    rows=[];audits=[]
    for year in range(2005,2026):
        a=load_year(year);keep=(a['idx']%5==0)&(a['idx']+4<=end[year])
        z={k:v[keep] for k,v in a.items()};del a
        five,audit=labels5(z,year,c);audits.append(audit)
        clock=z['clock'];ts=np.where((clock>=-5)&(clock<=-1),1.,np.where((clock>=-40)&(clock<=-21),-1.,0.))
        values=np.column_stack([z['rank'],ts,z['ret'],five]).astype(float)
        values[:,5]=np.where(np.isfinite(z['rank'][:,5]),z['leg'][:,5],np.nan)
        idx=z['idx'];starts=np.r_[0,np.flatnonzero(np.diff(idx))+1];ends=np.r_[starts[1:],len(idx)]
        for start,stop in zip(starts,ends):
            v=values[start:stop]
            for i,j in pairs:
                valid=np.isfinite(v[:,i])&np.isfinite(v[:,j]);n=int(valid.sum())
                if n<20:continue
                x=v[valid,i];y=v[valid,j];p=corr(x,y)
                if not np.isfinite(p):continue
                s=corr(rankdata(x),rankdata(y))
                rows.append(dict(year=year,idx=int(idx[start]),variable1=NAMES[i],variable2=NAMES[j],
                                 n_names=n,pearson=p,spearman=s))
        print(f'Groundtruth all-eligible correlations {year}: {len(five):,} labels',flush=True)
    daily=pd.DataFrame(rows);daily.to_csv(OUT/'groundtruth_signal_correlations_daily.csv',index=False)
    summary=[]
    for period,first,last in [('2005_2017',2005,2017),('2018_2025',2018,2025)]:
        subset=daily[daily.year.between(first,last)]
        for (one,two),g in subset.groupby(['variable1','variable2'],sort=False):
            for metric in ['pearson','spearman']:
                h=hac_mean(g[metric],g.idx//5,lag=12)
                summary.append(dict(period=period,variable1=one,variable2=two,metric=metric,
                    n_dates=h['n'],mean_pair_names=float(g.n_names.mean()),mean_correlation=h['mean'],
                    ci_low=h['lo'],ci_high=h['hi'],hac_se=h['se'],p=h['p']))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'groundtruth_signal_correlations.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'groundtruth_label_audit.csv',index=False)
    audit=dict(status='Post-result descriptive diagnostics, no refit or new portfolio selection',
        sample='ALL eligible rows on idx%5==0, before any factor-tail selection; >=20 pairwise finite names per date, nonzero variance; calendar-year crossing five-day labels purged',
        score='Economic signed source-month percentile rank for first5factors; RDIPO+1raw0,-1raw1; fixed originalclock+1pre1..5,-1pre21..40,0otherwise',
        groundtruth='return_1d is raw observed stock total return; return_5d is exact5session fixed-share gross underlying payoff including distributions retained as cash, not annualized. Future prices used only for outcome labels.',
        missing='Missing observed nextday return excluded from its pairwise correlations. Five-day label missing observed marks0, missingretx effective totalret, first missinglisting closesatlastmark; no survivor filter.',
        correlation='Within-date Pearson or Spearman on same pairwise-complete names, then equalweight average across dates. Pearson subtracts cross-sectional means, removing a common stock-return market level; not pooled level/trend correlation.',
        uncertainty='HAC12 on original five-session grid including gaps for omitted dates; descriptive95% intervals/p, no multiplicity-adjusted discovery claims',
        limits='Pairwise supports differ by predictor; matrix need not be positive semidefinite and is not an optimizer covariance input. These are in-period realized information coefficients, not model OOSprediction correlations; do not establish nonlinear conditional alpha.2018cutoff chosen after outcomes.',
        variables=NAMES,periods=['2005_2017','2018_2025'],total_label_rows=sum(x['eligible_weekly_rows'] for x in audits),
        scalar_label_checks=sum(x['scalar_checks'] for x in audits),max_scalar_label_error=max(x['max_scalar_error'] for x in audits))
    (OUT/'groundtruth_correlation_definitions.json').write_text(json.dumps(audit,indent=2))
    print(summary[(summary.variable2=='return_5d')&(summary.variable1.isin(NAMES[:7]))][['period','variable1','metric','n_dates','mean_correlation','ci_low','ci_high']].to_string(index=False))

if __name__=='__main__':main()
