#!/usr/bin/env python3
"""Post-result date splits of existing portfolios only; no refits/selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from data import HERE,FACTORS
from diagnose_correlations import hac_mean

OUT=HERE/'diagnostics'
PERIODS={'2005_2017':(2005,2017),'2018_2025':(2018,2025),
         '2005_2014':(2005,2014),'2015_2017':(2015,2017)}

def load_book(folder):
    z=dict(np.load(HERE/folder/'daily.npz'))
    specs=json.loads((HERE/folder/'models.json').read_text())
    if 'costs' in z:
        ci=int(np.flatnonzero(np.isclose(z['costs'],.0005))[0])
        z={k:(v[:,ci,:] if np.ndim(v)==3 else v) for k,v in z.items()}
    return z,specs

def mean_test(x,idx):
    h=hac_mean(x,idx)
    return dict(mean_daily=h['mean'],annual_mean=h['mean']*252,
                ci_low=h['lo']*252,ci_high=h['hi']*252,p=h['p'])

def main():
    OUT.mkdir(exist_ok=True)
    books={}; pairs=[]
    for family,folder in [('h1','results'),('h5','results_h5'),('cost_guard','results_cost_guard')]:
        z,spec=load_book(folder);books[family]=(z,spec)
        if family=='cost_guard':
            for j,s in enumerate(spec):
                if s['variant']=='guard':pairs.append((family,s['factor'],s['method'],j-1,j))
        else:
            lookup={s['id']:i for i,s in enumerate(spec)}
            for j,s in enumerate(spec):
                if s['cycle']:pairs.append((family,s['factor'],s['method'],lookup[s['id'].replace('__cycle','__base')],j))
    dates=books['h1'][0]['dates']; years=pd.DatetimeIndex(dates).year
    for z,_ in books.values():assert np.array_equal(dates,z['dates'])
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date']).set_index('date')
    idx=cal.loc[pd.DatetimeIndex(dates),'idx'].to_numpy()
    rows=[];annual=[];breaks=[]
    for family,factor,method,b,e in pairs:
        z=books[family][0]
        for series,key in [('net','returns'),('gross_same_cost_holdings','gross_return')]:
            baseline=z[key][:,b];enhanced=z[key][:,e];delta=enhanced-baseline
            for period,(first,last) in PERIODS.items():
                use=(years>=first)&(years<=last);x=baseline[use];y=enhanced[use]
                h=mean_test(delta[use],idx[use]);own=mean_test(y,idx[use])
                rows.append(dict(period=period,family=family,factor=factor,method=method,series=series,
                    n_days=int(use.sum()),n_years=last-first+1,baseline_mean=float(x.mean()*252),
                    enhanced_mean=float(y.mean()*252),delta_mean=h['annual_mean'],
                    ci_low=h['ci_low'],ci_high=h['ci_high'],p=h['p'],own_p=own['p'],
                    own_ci_low=own['ci_low'],own_ci_high=own['ci_high'],joint_p=max(h['p'],own['p']),
                    baseline_sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(252)),
                    enhanced_sharpe=float(y.mean()/y.std(ddof=1)*np.sqrt(252)),
                    delta_fee=float((z['fees'][use,e]-z['fees'][use,b]).mean()*252),
                    delta_borrow=float((z['borrow'][use,e]-z['borrow'][use,b]).mean()*252),
                    delta_turnover=float((z['turnover'][use,e]-z['turnover'][use,b]).mean()*252)))
            # Same fixed strategy, post-minus-pre paired increment: not comparing
            # two winners. HAC regression accounts for the shared date boundary.
            fit=sm.OLS(delta,np.column_stack([np.ones(len(delta)),years>=2018])).fit(
                cov_type='HAC',cov_kwds={'maxlags':60})
            ci=fit.conf_int()[1]*252
            breaks.append(dict(family=family,factor=factor,method=method,series=series,
                post_minus_pre_delta=float(fit.params[1]*252),ci_low=float(ci[0]),
                ci_high=float(ci[1]),p=float(fit.pvalues[1])))
        for year in range(2005,2026):
            use=years==year
            annual.append(dict(year=year,n_days=int(use.sum()),family=family,factor=factor,method=method,
                baseline_net_mean=float(z['returns'][use,b].mean()*252),
                enhanced_net_mean=float(z['returns'][use,e].mean()*252),
                delta_net=float((z['returns'][use,e]-z['returns'][use,b]).mean()*252),
                delta_gross=float((z['gross_return'][use,e]-z['gross_return'][use,b]).mean()*252),
                delta_fees=float((z['fees'][use,e]-z['fees'][use,b]).mean()*252)))
    tests=pd.DataFrame(rows);breaks=pd.DataFrame(breaks)
    for _,g in tests.groupby(['period','series']):
        assert len(g)==108
        for col,dest in [('p','holm108_p'),('own_p','own_holm108_p'),('joint_p','joint_holm108_p')]:
            tests.loc[g.index,dest]=multipletests(g[col],method='holm')[1]
        h=g[g.family!='cost_guard'];assert len(h)==72
        for col,dest in [('p','holm72_p'),('own_p','own_holm72_p'),('joint_p','joint_holm72_p')]:
            tests.loc[h.index,dest]=multipletests(h[col],method='holm')[1]
    tests['joint_success108']=(tests.delta_mean>0)&(tests.enhanced_mean>0)&(tests.joint_holm108_p<.05)
    tests['joint_success72']=(tests.delta_mean>0)&(tests.enhanced_mean>0)&(tests.joint_holm72_p<.05)
    for _,g in breaks.groupby('series'):
        breaks.loc[g.index,'holm108_p']=multipletests(g.p,method='holm')[1]
        h=g[g.family!='cost_guard'];breaks.loc[h.index,'holm72_p']=multipletests(h.p,method='holm')[1]
    tests.to_csv(OUT/'pre2018_paired_inference.csv',index=False)
    breaks.to_csv(OUT/'pre2018_change_in_increment.csv',index=False)
    pd.DataFrame(annual).to_csv(OUT/'pre2018_annual_increments.csv',index=False)
    summary=[]
    for (period,family,series),g in tests.groupby(['period','family','series']):
        summary.append(dict(period=period,family=family,series=series,n_days=int(g.n_days.iloc[0]),n_models=len(g),
            positive_increments=int((g.delta_mean>0).sum()),
            positive_significant72=int(((g.delta_mean>0)&(g.holm72_p<.05)).sum()),
            positive_significant108=int(((g.delta_mean>0)&(g.holm108_p<.05)).sum()),
            joint_success108=int(g.joint_success108.sum()),median_increment=float(g.delta_mean.median()),
            mean_increment=float(g.delta_mean.mean()),min_increment=float(g.delta_mean.min()),
            max_increment=float(g.delta_mean.max()),median_own_return=float(g.enhanced_mean.median()),
            median_delta_fee=float(g.delta_fee.median())))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'pre2018_summary.csv',index=False)
    # Actual plain baselines are the same in both horizons; do not duplicate.
    z,spec=books['h1'];baseline=[]
    timing=dict(np.load(OUT/'timing_daily.npz'));assert np.array_equal(dates,timing['dates'])
    serieslist=[(s['factor'],{k:z[k][:,j] for k in ['returns','gross_return','fees','borrow','active']})
                for j,s in enumerate(spec) if s['method']=='plain' and s['factor'] in FACTORS]
    serieslist.append(('original_clock',timing))
    for label,a in serieslist:
        for period,(first,last) in PERIODS.items():
            use=(years>=first)&(years<=last)
            for series,key in [('net','returns'),('gross_same_cost_holdings','gross_return')]:
                r=a[key][use];h=mean_test(r,idx[use]);nav=np.cumprod(1+r)
                baseline.append(dict(factor=label,period=period,series=series,n_days=len(r),**h,
                    variance_daily=float(r.var(ddof=1)),sd_daily=float(r.std(ddof=1)),
                    sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(252)),
                    cagr=float(nav[-1]**(252/len(r))-1),active=float(a['active'][use].mean()),
                    annual_fees=float(a['fees'][use].mean()*252),annual_borrow=float(a['borrow'][use].mean()*252)))
    baseline=pd.DataFrame(baseline);baseline.to_csv(OUT/'pre2018_baseline_moments.csv',index=False)
    counts=pd.DataFrame({'year':years,'idx':idx}).groupby('year').size().rename('n_sessions').reset_index()
    counts.to_csv(OUT/'pre2018_calendar_counts.csv',index=False)
    note='''Pre-2018 follow-up: descriptive splits of already computed portfolios.
No model was refitted, no signal changed, and no strategy selected for these
subperiods. The2018cutoff was requested after looking at results; it is not a
prospectively specified structural-break hypothesis or new untouched holdout.
The shorter2015–2017period has only3years and much wider uncertainty.

Primary comparison is enhanced-minus-ITS-MATCHED-baseline net daily return,
not enhanced raw return or the market's return. Each period has36 models in
each of h1,h5,cost_guard. Holm72 corrects the two target-horizon families;
Holm108 additionally includes costguard. Corrections are per reported period
and net/gross statistic, not over all prior analyses or all imaginable cutoffs.
Joint profitability uses p_joint=max(two-sided p_increment,two-sided p_own),
then Holm adjustment and positive estimates for both. Daily HAC60 is used;
stock-level observations are not treated as independent evidence.

Gross refers to P&L on the actual5bp-scenario holdings before that day's
fee/borrow deductions, not a separately optimized zero-cost portfolio.
Portfolios remain continuous across2017/2018; no artificial exit/reentry cost
is inserted at the split. Only the original end-of2025terminal liquidation
appears. The change table estimates post-minus-pre incremental mean using
the SAME strategy and HAC60 regression on a fixed2018indicator. It does not
compare the best strategy chosen separately in each period.

Original-clock timing uses all eligible names, whereas factor baselines need
their own characteristics. Its moments are contextual, not a matched ablation.
All units in CSV are decimal returns unless explicitly named otherwise.

Reproduce: OPENBLAS_NUM_THREADS=1 .venv/bin/python factor_overlay/pre2018.py

Net incremental summary (each row contains the full36model family):
'''
    note+=summary.query("series=='net'").to_string(index=False)+'\n\nPlain baseline net annual means:\n'
    note+=baseline.query("series=='net'").pivot(index='factor',columns='period',values='annual_mean').to_string()+'\n'
    (OUT/'pre2018_notes.txt').write_text(note)
    print(summary.query("series=='net'").to_string(index=False))
    print(baseline.query("series=='net'").pivot(index='factor',columns='period',values='annual_mean').to_string())

if __name__=='__main__':main()
