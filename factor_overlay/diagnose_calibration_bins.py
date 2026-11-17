#!/usr/bin/env python3
"""Descriptive out-of-sample calibration bins of the added clock forecast.

Uses existing direct-five-session ridge models and actual five-session fixed-
share labels. The bins use predictions only. No new model or strategy is fit.
"""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,pickle
import numpy as np
import pandas as pd
from data import HERE,FACTORS

OUT=HERE/'diagnostics'
PERIODS={'2005-2017':(2005,2017),'2018-2025':(2018,2025)}


def weighted_cuts(x,w):
    order=np.argsort(x,kind='stable');xx=x[order];ww=w[order]
    c=np.cumsum(ww)-.5*ww
    return np.interp(np.arange(1,10)/10*ww.sum(),c,xx)


def clustered_hac_ratio(y,w,idx,first_week,last_week):
    total=w.sum();mean=float(w@y/total)
    score=np.zeros(last_week-first_week+1)
    np.add.at(score,idx//5-first_week,w*(y-mean))
    # Ratio-mean influence, clustered by date: every stock and both legs on
    # the same date enter one score before serial covariance is calculated.
    meat=float(score@score)
    for lag in range(1,min(12,len(score)-1)+1):
        meat+=2*(1-lag/13)*float(score[lag:]@score[:-lag])
    se=float(np.sqrt(max(meat,0))/total)
    return mean,se,mean-1.95996398454*se,mean+1.95996398454*se


def main():
    OUT.mkdir(exist_ok=True)
    z=dict(np.load(HERE/'data/train5.npz'));records=[]
    for year in range(2005,2026):
        fitted=pickle.loads((HERE/f'fits_h5/{year}.pkl').read_bytes())
        for j,factor in enumerate(FACTORS):
            rows=np.flatnonzero((z['year']==year)&(z['factor']==j))
            xb=z['Xbase'][rows];xc=z['Xcycle'][rows]
            mb,_=fitted[factor+'__ridge__base'].predict(xb)
            me,_=fitted[factor+'__ridge__cycle'].predict(xc)
            # Match the execution's stored forecast precision. Actual labels
            # are already float32; never replace them with predicted outcomes.
            mb=mb.astype(np.float32).astype(float);me=me.astype(np.float32).astype(float)
            q=pd.DataFrame(dict(year=year,factor=factor,idx=z['idx'][rows],leg=xb[:,6],
                                weight=z['weight'][rows].astype(float),
                                x=me-mb,y=z['y'][rows].astype(float)-mb))
            group=q.groupby(['idx','leg'],sort=False)
            for name in ['x','y']:
                q['weighted']=q[name]*q.weight
                means=group.weighted.transform('sum')/group.weight.transform('sum')
                q[name]=(q[name]-means)*50000
            q=q.drop(columns='weighted')
            # Weighted within-date/leg zero sums check the centering itself.
            for name in ['x','y']:
                zero=(q[name]*q.weight).groupby([q.idx,q.leg]).sum()
                assert abs(zero).max()<1e-7
            records.append(q)
    frame=pd.concat(records,ignore_index=True)
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    out=[];edges=[]
    for period,(lo,hi) in PERIODS.items():
        calendar=cal[cal.date.dt.year.between(lo,hi)&cal.idx.mod(5).eq(0)]
        first_week=int(calendar.idx.min()//5);last_week=int(calendar.idx.max()//5)
        for factor in FACTORS:
            g=frame[frame.year.between(lo,hi)&frame.factor.eq(factor)].copy()
            x=g.x.to_numpy();y=g.y.to_numpy();w=g.weight.to_numpy();idx=g.idx.to_numpy(int)
            cuts=weighted_cuts(x,w);bins=np.searchsorted(cuts,x,side='left')+1
            edges.append(dict(period=period,factor=factor,cutpoints_bps=cuts.tolist()))
            assert np.all(np.diff(cuts)>=0)
            for b in range(1,11):
                keep=bins==b;ww=w[keep]
                if not keep.any():continue
                mean,se,low,high=clustered_hac_ratio(y[keep],ww,idx[keep],first_week,last_week)
                out.append(dict(factor=factor,period=period,bin=b,x_bps=float(ww@x[keep]/ww.sum()),
                    y_bps=mean,lo=low,hi=high,se=se,n=int(keep.sum()),dates=int(len(np.unique(idx[keep]))),
                    calendar_week_dates=last_week-first_week+1,weight_sum=float(ww.sum()),
                    realized_within_bin_sd_bps=float(np.sqrt(ww@((y[keep]-mean)**2)/ww.sum()))))
    result=pd.DataFrame(out)
    assert len(result)==120 and np.isfinite(result[['x_bps','y_bps','lo','hi']]).all().all()
    result.to_csv(OUT/'calibration_bins.csv',index=False)
    notes=dict(status='Descriptive post-result calibration; user-specified regime split; no fitting/tuning',
        method='Existing annual direct-five-day ridge fits_h5; prediction year uses preceding10years only',
        x='(cycle signed forecast minus baseline signed forecast), centered within each date/leg using original weights, multiplied5*10000 to five-day bp',
        y='(ACTUAL observed signed five-day fixed-share payoff/5 minus baseline forecast), centered within date/leg with same weights, multiplied5*10000',
        actual_outcomes='data/train5.npz labels: sum of fixed shares times observed CRSP total returns, retx share marking, distributions retained cash; missing listing forces exit, no survivor filtering; no fees/borrow',
        bins='Ten weighted quantile bins computed from x only separately within factor/period. Predictions determine bin membership; outcomes never do.',
        weights='Original30/n weights: each sampled date/leg equal total mass30, regardless of name count. Bin means are pooled weighted ratio means.',
        confidence_intervals='Pointwise normal95% intervals from date-clustered ratio-mean influences with BartlettHAC12 sampled-week lags. All stocks and both legs cluster jointly by date; absent-bin dates and purged year-end dates contribute zero scores on full idx%5 grid.',
        interpretation='45-degree line indicates correctly calibrated incremental conditional mean; flat nearzero indicates no realized within-leg residual gradient. Residualization removes baseline forecast and common date/leg return, not all potential confounding. Intervals are not simultaneous or causal.',
        rows=len(result),sample_records=len(frame),bin_edges=edges)
    (OUT/'calibration_bins_definitions.json').write_text(json.dumps(notes,indent=2))
    print(result.groupby(['factor','period']).agg(min_x=('x_bps','min'),max_x=('x_bps','max'),min_y=('y_bps','min'),max_y=('y_bps','max')).to_string())


if __name__=='__main__':main()
