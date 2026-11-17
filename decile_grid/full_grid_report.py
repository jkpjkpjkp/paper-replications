#!/usr/bin/env python3
"""Descriptive full-grid gross inference, preserving the entire search family."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
OUT = HERE/'results'
PERIODS = dict(development=(1985,2004),validation=(2005,2014),holdout=(2015,2025),
               post2018=(2019,2025),recent=(2020,2025),post2004=(2005,2025))


def hac(x,lags):
    n = len(x)
    mean = x.mean(axis=0)
    u = x-mean
    variance = np.sum(u*u,axis=0)
    for lag in range(1,min(lags,n-1)+1):
        variance += 2*(1-lag/(lags+1))*np.sum(u[lag:]*u[:-lag],axis=0)
    se = np.sqrt(np.maximum(variance,0))/n
    t = np.divide(mean,se,out=np.zeros_like(mean),where=se>1e-16)
    p = np.where(se>1e-16,2*norm.sf(abs(t)),1.)
    return dict(t=t,p=p,ci_low=(mean-norm.ppf(.975)*se)*1e4,
                ci_high=(mean+norm.ppf(.975)*se)*1e4)


def main():
    specs = json.loads((OUT/'specs.json').read_text())
    frame = pd.DataFrame(specs)
    index = {s['id']:j for j,s in enumerate(specs)}
    baseline = np.asarray([index[s['baseline_id']] if s['baseline_id'] else j for j,s in enumerate(specs)])
    candidates = (frame.id.ne(frame.baseline_id) & frame.family.ne('timing_only')).to_numpy()
    assert candidates.sum()==2532
    train = pd.read_csv(OUT/'training_gross_summary.csv').set_index('id').loc[frame.id]
    frame['training_selection_eligible'] = train.selection_eligible.to_numpy()
    frame['training_active_fraction'] = train.active_fraction.to_numpy()
    frame['training_baseline_active_fraction'] = train.baseline_active_fraction.to_numpy()
    frame['is_timing_candidate'] = candidates
    years = {}
    for year in range(1985,2026):
        with np.load(OUT/f'gross_{year}.npz') as z:
            years[year] = {k:z[k] for k in ['returns','active','n_long','n_short']}
    rows = []
    best = []
    counts = []
    for period,(start,end) in PERIODS.items():
        r = np.concatenate([years[y]['returns'] for y in range(start,end+1)])
        paired = r-r[:,baseline]
        f = frame.copy()
        f['period'] = period
        f['start_year'] = start
        f['end_year'] = end
        f['days'] = len(r)
        f['mean_bp'] = r.mean(axis=0)*1e4
        f['annual_mean'] = r.mean(axis=0)*252
        vol = r.std(axis=0,ddof=1)*np.sqrt(252)
        f['annual_vol'] = vol
        f['sharpe'] = np.divide(f.annual_mean,vol,out=np.full(len(f),np.nan),where=vol>1e-16)
        f['baseline_mean_bp'] = f.mean_bp.to_numpy()[baseline]
        f['uplift_bp'] = paired.mean(axis=0)*1e4
        f['uplift_annual'] = paired.mean(axis=0)*252
        for name in ['active','n_long','n_short']:
            value = np.concatenate([years[y][name] for y in range(start,end+1)]).mean(axis=0)
            f['active_fraction' if name=='active' else 'mean_'+name[2:]] = value
        f['baseline_active_fraction'] = f.active_fraction.to_numpy()[baseline]
        f['current_activity_eligible'] = candidates & f.active_fraction.ge(.5) & f.baseline_active_fraction.ge(.8)
        f['both_positive'] = f.mean_bp.gt(0)&f.uplift_bp.gt(0)
        for lag in [20,60]:
            for prefix,x in [('own',r),('paired',paired)]:
                result = hac(x,lag)
                for field,value in result.items():
                    f[f'{prefix}_{field}_hac{lag}'] = value
                # Empty/constant/inactive strategies remain tests with p=1.
                adjusted = np.full(len(f),np.nan)
                adjusted[candidates] = multipletests(result['p'][candidates],method='holm')[1]
                f[f'{prefix}_p_holm2532_hac{lag}'] = adjusted
        for (factor,family),g in f[candidates].groupby(['factor','family'],sort=False):
            eligible = g[g.current_activity_eligible]
            for criterion in ['uplift_bp','paired_t_hac20']:
                if len(eligible):
                    winner = eligible.sort_values([criterion,'id'],ascending=[False,True]).iloc[0].to_dict()
                    winner.update(expost_criterion=criterion,
                        expost_activity_eligible_count=len(eligible),
                        interpretation='Descriptive later/full-period maximum; selected using this period returns; training eligibility not required')
                    best.append(winner)
                else:
                    best.append(dict(period=period,factor=factor,family=family,id=None,
                        expost_criterion=criterion,expost_activity_eligible_count=0,
                        interpretation='No candidate satisfies current-period activity thresholds'))
        for name,g in [('all',f[candidates]),('D10',f[candidates&f.family.eq('D10')]),
                       ('S30',f[candidates&f.family.eq('S30')])]:
            for qualified in [False,True]:
                q = g[g.current_activity_eligible] if qualified else g
                entry = dict(period=period,family=name,activity_filtered=qualified,
                    tested_family_size=2532,count=len(q),both_positive=int(q.both_positive.sum()),
                    positive_uplift=int(q.uplift_bp.gt(0).sum()),positive_own=int(q.mean_bp.gt(0).sum()))
                for lag in [20,60]:
                    entry[f'positive_uplift_raw_hac{lag}'] = int((q.uplift_bp.gt(0)&q[f'paired_p_hac{lag}'].lt(.05)).sum())
                    entry[f'positive_uplift_holm_hac{lag}'] = int((q.uplift_bp.gt(0)&q[f'paired_p_holm2532_hac{lag}'].lt(.05)).sum())
                    entry[f'joint_positive_raw_hac{lag}'] = int((q.both_positive&q[f'paired_p_hac{lag}'].lt(.05)&q[f'own_p_hac{lag}'].lt(.05)).sum())
                    entry[f'joint_positive_holm_hac{lag}'] = int((q.both_positive&q[f'paired_p_holm2532_hac{lag}'].lt(.05)&q[f'own_p_holm2532_hac{lag}'].lt(.05)).sum())
                counts.append(entry)
        rows.append(f)
        print(time.strftime('%H:%M:%S'),period,len(r),'sessions',flush=True)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'grid_gross_summary.csv',index=False)
    pd.DataFrame(best).to_csv(OUT/'grid_expost_best.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/'grid_significance_counts.csv',index=False)
    audit=dict(total_models=len(specs),timing_candidates=int(candidates.sum()),
        baseline_models=int((~candidates & frame.family.ne('timing_only')).sum()),
        periods=PERIODS,adjustment='Holm across all 2532 timing candidates per period/statistic/lag; inactive and zero-variance strategies p=1',
        expost_selection='Current-period activity >=.5, baseline >=.8; do not require training eligibility; maxima are descriptive',
        confidence_intervals='Asymptotic normal 95% pointwise HAC intervals in daily basis points, not simultaneous intervals',
        trading_costs=0,annual_borrow=0,source='Complete gross daily arrays; no new portfolio selection or frozen-file changes')
    (OUT/'grid_report_audit.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__=='__main__':
    main()
