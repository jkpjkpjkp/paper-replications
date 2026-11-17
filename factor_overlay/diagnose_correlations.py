#!/usr/bin/env python3
"""Post-result descriptive diagnostics; never selects a new trading rule."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import ndtr
from data import HERE, FACTORS, load_year

OUT = HERE/'diagnostics'
PERIODS = {'2005_2014': (2005, 2014), '2015_2025': (2015, 2025)}
PHASES = ['pre1_5', 'pre6_20', 'pre21_40', 'post0_10', 'post11plus']

def corr(x, y):
    x=np.asarray(x, float); y=np.asarray(y, float)
    x=x-x.mean(); y=y-y.mean()
    den=np.sqrt(x@x*(y@y))
    return float(x@y/den) if den>1e-15 else np.nan

def hac_mean(values, idx, lag=60):
    """Ratio-mean delta-method HAC on calendar-session distances, missing cells0."""
    values=np.asarray(values,float); idx=np.asarray(idx,int)
    ok=np.isfinite(values); values=values[ok]; idx=idx[ok]
    n=len(values)
    if n<2:return dict(n=n,mean=float(values[0]) if n else np.nan,se=np.nan,lo=np.nan,hi=np.nan,p=np.nan)
    assert len(np.unique(idx))==n
    mu=values.mean(); u=np.zeros(int(idx.max()-idx.min()+1))
    u[idx-idx.min()]=values-mu
    meat=u@u
    for k in range(1,min(lag,len(u)-1)+1):meat+=2*(1-k/(lag+1))*float(u[k:]@u[:-k])
    se=np.sqrt(max(meat,0))/n
    p=2*ndtr(-abs(mu/se)) if se else (1. if mu==0 else 0.)
    return dict(n=n,mean=float(mu),se=float(se),lo=float(mu-1.95996398454*se),hi=float(mu+1.95996398454*se),p=float(p))

def phase(clock):
    return np.select([(clock>=-5)&(clock<=-1),(clock>=-20)&(clock<=-6),
                      (clock>=-40)&(clock<=-21),(clock>=0)&(clock<=10),clock>10],
                     PHASES,default='outside')

def score_and_return_diagnostics():
    rows=[]
    for year in range(2005,2026):
        z=load_year(year); ii=z['idx']; starts=np.r_[0,np.flatnonzero(np.diff(ii))+1]
        ends=np.r_[starts[1:],len(ii)]
        for a,b in zip(starts,ends):
            cl=z['clock'][a:b]
            ts=np.where((cl>=-5)&(cl<=-1),1.,np.where((cl>=-40)&(cl<=-21),-1.,0.))
            for j,f in enumerate(FACTORS):
                valid=np.isfinite(z['rank'][a:b,j]); score=z['rank'][a:b,j][valid].astype(float)
                if f=='RDIPO':score=z['leg'][a:b,j][valid].astype(float)
                timing=ts[valid]; legs=z['leg'][a:b,j][valid]
                returns=z['ret'][a:b][valid]; missing=~np.isfinite(returns)
                returns=np.where(missing,0.,returns)
                fl=legs==1; fs=legs==-1; tl=timing==1; ss=timing==-1
                threshold=5 if f=='RDIPO' else 20
                common=(min(fl.sum(),fs.sum())>=threshold and min(tl.sum(),ss.sum())>=20)
                fr=float(returns[fl].mean()-returns[fs].mean()) if common else np.nan
                tr=float(returns[tl].mean()-returns[ss].mean()) if common else np.nan
                rows.append(dict(year=year,idx=int(ii[a]),date=str(z['date'][a]),factor=f,
                    n=len(score),pearson=corr(score,timing),spearman=corr(rankdata(score),rankdata(timing)),
                    factor_long=int(fl.sum()),factor_short=int(fs.sum()),timing_long=int(tl.sum()),
                    timing_short=int(ss.sum()),missing_return=int(missing.sum()),joint_active=common,
                    factor_gross=fr,timing_gross=tr))
        print(f'Correlation diagnostics {year}',flush=True)
    daily=pd.DataFrame(rows);daily.to_csv(OUT/'score_return_daily.csv',index=False)
    scores=[];rets=[]
    for period,(first,last) in PERIODS.items():
        sub=daily[daily.year.between(first,last)]
        for f,g in sub.groupby('factor',sort=False):
            for metric in ['pearson','spearman']:
                h=hac_mean(g[metric],g.idx)
                scores.append(dict(period=period,factor=f,metric=metric,mean_names=g.n.mean(),**h))
            a=g[g.joint_active]
            x=a.factor_gross.to_numpy(); y=a.timing_gross.to_numpy()
            rho=corr(x,y)
            if len(a)>2 and np.isfinite(rho):
                zx=(x-x.mean())/x.std();zy=(y-y.mean())/y.std()
                psi=zx*zy-rho*(zx*zx+zy*zy)/2
                se=hac_mean(psi,a.idx)['se']
            else:se=np.nan
            rets.append(dict(period=period,factor=f,n_joint=len(a),fraction_joint=len(a)/len(g),
                correlation=rho,corr_hac_se=se,corr_ci_low=max(-1,rho-1.95996398454*se),
                corr_ci_high=min(1,rho+1.95996398454*se),
                factor_mean_bps_day=float(x.mean()*1e4) if len(x) else np.nan,
                timing_mean_bps_day=float(y.mean()*1e4) if len(y) else np.nan,
                factor_vol_daily=float(x.std(ddof=1)) if len(x)>1 else np.nan,
                timing_vol_daily=float(y.std(ddof=1)) if len(y)>1 else np.nan,
                covariance_daily=float(np.cov(x,y,ddof=1)[0,1]) if len(x)>1 else np.nan))
    pd.DataFrame(scores).to_csv(OUT/'score_correlation_summary.csv',index=False)
    pd.DataFrame(rets).to_csv(OUT/'return_correlation_summary.csv',index=False)
    return daily,pd.DataFrame(scores),pd.DataFrame(rets)

def phase_interactions():
    z=dict(np.load(HERE/'data/train5.npz'))
    clock=np.rint(z['Xcycle'][:,11]*65).astype(int)
    df=pd.DataFrame(dict(factor=np.array(FACTORS)[z['factor']],year=z['year'],idx=z['idx'],
                        leg=z['Xbase'][:,6].astype(int),phase=phase(clock),signed=z['y']))
    df=df[df.year>=2005].copy();df['raw']=df.signed*df.leg
    # The sampler is uniform (deterministic hash) within each date/leg. Original
    # training weights are constant within that cell, so do not change its mean.
    cells=df.groupby(['factor','year','idx','phase','leg'],observed=True).agg(
        signed=('signed','mean'),raw=('raw','mean'),records=('signed','size')).reset_index()
    cells.to_csv(OUT/'phase_leg_daily_cells.csv',index=False)
    summaries=[];spreadrows=[];interactions=[]
    for period,(first,last) in PERIODS.items():
        g=cells[cells.year.between(first,last)]
        for (f,ph,leg),c in g.groupby(['factor','phase','leg'],observed=True):
            for target in ['raw','signed']:
                h=hac_mean(c[target],c.idx)
                summaries.append(dict(period=period,factor=f,phase=ph,leg=leg,target=target,
                    stock_records=int(c.records.sum()),mean_records_per_date=float(c.records.mean()),
                    **h,mean_bps_day=h['mean']*1e4,ci_low_bps_day=h['lo']*1e4,ci_high_bps_day=h['hi']*1e4))
        for f,fg in g.groupby('factor',sort=False):
            wide=fg.pivot(index='idx',columns=['phase','leg'],values='raw')
            for ph in PHASES:
                if (ph,1) not in wide or (ph,-1) not in wide:continue
                common=wide[[(ph,1),(ph,-1)]].dropna()
                spread=common[(ph,1)]-common[(ph,-1)]
                h=hac_mean(spread,common.index)
                spreadrows.append(dict(period=period,factor=f,phase=ph,**h,
                    mean_bps_day=h['mean']*1e4,ci_low_bps_day=h['lo']*1e4,ci_high_bps_day=h['hi']*1e4))
                ref='pre21_40'
                if ph==ref:continue
                cols=[(ph,1),(ph,-1),(ref,1),(ref,-1)]
                if not all(k in wide for k in cols):continue
                both=wide[cols].dropna()
                dl=both[(ph,1)]-both[(ref,1)]
                ds=both[(ph,-1)]-both[(ref,-1)]
                for label,value in [('long_raw_shift',dl),('short_raw_shift',ds),
                                    ('common_raw_shift',(dl+ds)/2),('factor_interaction',dl-ds)]:
                    h=hac_mean(value,both.index)
                    interactions.append(dict(period=period,factor=f,phase=ph,reference=ref,
                        contrast=label,**h,mean_bps_day=h['mean']*1e4,
                        ci_low_bps_day=h['lo']*1e4,ci_high_bps_day=h['hi']*1e4))
    pd.DataFrame(summaries).to_csv(OUT/'phase_leg_mean_summary.csv',index=False)
    pd.DataFrame(spreadrows).to_csv(OUT/'phase_factor_spread_summary.csv',index=False)
    pd.DataFrame(interactions).to_csv(OUT/'phase_interaction_summary.csv',index=False)
    return pd.DataFrame(interactions)

def main():
    OUT.mkdir(exist_ok=True)
    daily,scores,returns=score_and_return_diagnostics()
    interactions=phase_interactions()
    definitions=dict(status='Post-result descriptive mechanism checks; no new fitted or selected strategy',
        source='factor_overlay/data/YYYY.npz and data/train5.npz; existing audited feature/label lineage',
        score='Original fixed timing score: +1 clock[-5,-1], -1 clock[-40,-21], else0',
        factor_score='Signed source-month percentile rank across ALL finite factor names in eligible universe; RDIPO economic leg+1 raw0,-1 raw1',
        signal_correlation='Pearson and Spearman within each date/factor; average dates equally, not stock rows pooled',
        return_correlation='Daily reconstituted equal-weight gross factor spread and original timing spread on same finite-factor universe; both100%long/100%short; compare only shared active dates',
        minimum_names='Factor >=20 each leg, except RDIPO>=5; timing>=20each; thresholds follow baseline/sparse diagnostic conventions',
        return_limit='Diagnostic gross daily portfolios, not weekly holdings or executable net backtests; excludes transaction/borrow costs. Missing observed returns0 and counted. Common active support can differ by factor/period.',
        phase_bins=PHASES,
        phase_outcome='Five-session fixed-share gross underlying P&L divided5 from train5.npz, then raw or leg-signed; no fees/borrow; same original deterministic max30 per date/factor/leg sample',
        phase_weighting='Each date/phase/leg cell equal weight across dates; factor spread uses same dates with both legs; phase interaction requires all four phase×leg cells on same date',
        interaction='[raw long in phase - raw long in pre21_40] - [raw short in phase - raw short in pre21_40]; common shift is half the sum of those two leg shifts',
        uncertainty='Bartlett HAC60 in original trading-session distances; missing cell scores0 with observed-cell count normalization. Return correlation uses delta-method influence function. All95% intervals and p values descriptive/unadjusted; no causal or multiple-testing claims.',
        source_definition='factor_timing/research.py timing_only_LS; source methodology and limitations factor_overlay/sources.txt')
    (OUT/'correlation_definitions.json').write_text(json.dumps(definitions,indent=2))
    note='''These are post-result diagnostics, not another alpha search.

Correlation answers a different question from incremental profitability.
Negative factor/clock score correlation means their stock rankings oppose one
another; zero correlation means little alignment. Neither alone determines
whether a timing-conditioned long/short factor improves net returns. Likewise,
negative factor/timing portfolio return covariance can aid diversification but
does not show that within-factor timing tilts have a positive conditional mean.

Each score correlation is cross-sectional within one date. Summaries average
those correlations across dates with trading-date HAC uncertainty. Return
correlations use daily equal-weight GROSS diagnostic spreads on the same finite
factor universe and common active dates, not the weekly costed strategy ledger.

Phase comparisons use the audited five-session fixed-share labels. Each cell
first averages its sampled stocks, then dates are equally weighted. Raw short
returns describe underlying stocks; signed short payoffs have the opposite sign.
If preannouncement raw returns rise equally for both factor legs, the common
drift cancels in long minus short. The interaction table directly estimates the
difference between those leg-specific phase shifts on the SAME dates. It also
reports the common raw-return shift. Sparse cells, changing names and conditional
sample selection make these descriptive, not causal treatment effects.

All phase/interaction intervals are unadjusted descriptive intervals. The five-
session labels are sampled and not the full investment universe. These data
cannot establish an economic mechanism uniquely or repair prior sample search.

Reproduce: OPENBLAS_NUM_THREADS=1 .venv/bin/python factor_overlay/diagnose_correlations.py
Definitions: correlation_definitions.json. Primary source provenance:
../sources.txt and ../data/train5_audit.json.
'''
    signal_view=scores[scores.metric=='pearson'].pivot(index='factor',columns='period',values='mean').reindex(FACTORS)
    return_view=returns[returns.period=='2015_2025'][['factor','n_joint','fraction_joint','correlation','corr_ci_low','corr_ci_high']]
    interaction_view=interactions[(interactions.phase=='pre1_5')&(interactions.contrast=='factor_interaction')][['period','factor','n','mean_bps_day','ci_low_bps_day','ci_high_bps_day']]
    note+='''
Observed findings
The fixed timing score and factor stock rankings have correlations very close
to zero in both periods (absolute mean Pearson correlation below0.007).
The data do not support strong negative stock-score alignment as an explanation.
Gross diagnostic portfolio return correlations have mixed signs, a different
object from stock-score correlation; they are particularly support-sensitive
for RIO_Disp and MS. Negative covariance is not conditional return amplification.

The pre1..5 versus pre21..40 factor-strength interaction is imprecise for every
factor in both periods: all displayed95% intervals span zero. CompEquIss and
RIO_Disp change interaction sign, as does sparse RDIPO. Thus the observed
phase differences do not identify stable factor amplification. They also do
not establish equal cancellation as the unique explanation: common raw drift
and long-minus-short interaction are both estimated noisily.

Mean daily cross-sectional Pearson score correlation:
'''+signal_view.to_string()+'\n\n2015–2025 gross diagnostic return correlation:\n'+return_view.to_string(index=False)+'\n\nFive-day-label preannouncement factor interaction (basis points/day):\n'+interaction_view.to_string(index=False)+'\n'
    (OUT/'correlation_notes.txt').write_text(note)
    print(scores.query("period=='2015_2025' and metric=='pearson'")[['factor','mean','lo','hi']].to_string(index=False))
    print(returns.query("period=='2015_2025'")[['factor','fraction_joint','correlation','corr_ci_low','corr_ci_high']].to_string(index=False))
    print(interactions.query("period=='2015_2025' and phase=='pre1_5'")[['factor','contrast','n','mean_bps_day','ci_low_bps_day','ci_high_bps_day']].to_string(index=False))

if __name__=='__main__':main()
