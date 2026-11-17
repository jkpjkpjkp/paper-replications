#!/usr/bin/env python3
"""Post-result explanation of forecast strength, calibration and instability.

Uses saved annual primary models only. Inference is over sampled trading dates,
never individual stock rows. All results are diagnostic and not new selection.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,pickle,time
import numpy as np
import pandas as pd
from scipy.stats import rankdata,norm
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from data import HERE,FACTORS
from models import METHODS

OUT=HERE/'diagnostics'


def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)


def ic(a,b):
    if len(a)<5:return np.nan
    aa=rankdata(a);bb=rankdata(b);aa=aa-aa.mean();bb=bb-bb.mean()
    den=np.sqrt((aa@aa)*(bb@bb))
    return float(aa@bb/den) if den>1e-18 else np.nan


def mean_hac(a,lag=12):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if len(a)<3:return (np.nan,)*5
    f=sm.OLS(a,np.ones((len(a),1))).fit(cov_type='HAC',cov_kwds={'maxlags':min(lag,len(a)-1)})
    return float(f.params[0]),float(f.bse[0]),float(f.pvalues[0]),float(f.conf_int()[0,0]),float(f.conf_int()[0,1])


def joint_calibration(f):
    # Every row is one date, averaging equally across available long/short legs.
    aa=np.stack([f.bb.to_numpy(),f.bd.to_numpy(),f.bd.to_numpy(),f.dd.to_numpy()],axis=1).reshape(-1,2,2)
    yy=f[['by','dy']].to_numpy();amat=aa.mean(axis=0);inv=np.linalg.pinv(amat,rcond=1e-10)
    beta=inv@yy.mean(axis=0);score=yy-np.einsum('nij,j->ni',aa,beta)
    score-=score.mean(axis=0);n=len(score)
    omega=score.T@score/n
    for lag in range(1,min(12,n-1)+1):
        cov=score[lag:].T@score[:-lag]/n
        omega+=(1-lag/13)*(cov+cov.T)
    vcov=inv@omega@inv/n;se=np.sqrt(np.maximum(np.diag(vcov),0))
    return dict(calibration_base=float(beta[0]),calibration_increment=float(beta[1]),
                calibration_increment_se=float(se[1]),
                calibration_increment_p=float(2*norm.sf(abs(beta[1]/se[1]))) if se[1]>0 else np.nan,
                calibration_increment_ci_low=float(beta[1]-1.96*se[1]),
                calibration_increment_ci_high=float(beta[1]+1.96*se[1]),
                calibration_moment_condition=float(np.linalg.cond(amat)))


def make_daily():
    z=dict(np.load(HERE/'data/train.npz'));z5=dict(np.load(HERE/'data/train5.npz'))
    def key(q):return q['factor'].astype(np.int64)*10**12+q['idx'].astype(np.int64)*1000000+q['permno']
    k=key(z);k5=key(z5);order=np.argsort(k5);k5=k5[order]
    pos=np.searchsorted(k5,k);ok=(pos<len(k5))&(k5[np.minimum(pos,len(k5)-1)]==k)
    y5=np.full(len(k),np.nan);y5[ok]=z5['y'][order[pos[ok]]]
    rows=[]
    for year in range(2004,2026):
        modelyear=max(year,2005)
        fitted=pickle.loads((HERE/f'fits/{modelyear}.pkl').read_bytes())
        for j,factor in enumerate(FACTORS):
            mask=(z['year']==year)&(z['factor']==j)
            xb=z['Xbase'][mask];xc=z['Xcycle'][mask];idx=z['idx'][mask]
            leg=xb[:,6];y1=z['y'][mask].astype(float);yy5=y5[mask]
            groups={}
            for t in np.unique(idx):
                for sign in [-1,1]:
                    loc=np.flatnonzero((idx==t)&(leg==sign))
                    if len(loc):groups[(int(t),sign)]=loc
            for method in METHODS:
                prefix=f'{factor}__{method}'
                mb,sb=fitted[prefix+'__base'].predict(xb)
                mc,sc=fitted[prefix+'__cycle'].predict(xc)
                delta=mc-mb
                for horizon,target in [(1,y1),(5,yy5)]:
                    rec=[]
                    for (t,sign),loc in groups.items():
                        if not np.isfinite(target[loc]).all():continue
                        b=mb[loc];cy=mc[loc];d=delta[loc];y=target[loc]
                        b0=b-b.mean();c0=cy-cy.mean();d0=d-d.mean();y0=y-y.mean()
                        eb=y-b;ec=y-cy
                        vb=sb[loc]**2/horizon;vc=sc[loc]**2/horizon
                        n=len(loc)
                        rec.append(dict(idx=t,leg=sign,n_stocks=n,
                            base_ic=ic(b,y),cycle_ic=ic(cy,y),increment_ic=ic(d,y),
                            base_mse=np.mean(eb**2),cycle_mse=np.mean(ec**2),mse_change=np.mean(ec**2-eb**2),
                            base_gaussian_score=np.mean(np.log(vb)+eb**2/vb),
                            cycle_gaussian_score=np.mean(np.log(vc)+ec**2/vc),
                            gaussian_score_change=np.mean(np.log(vc)+ec**2/vc-np.log(vb)-eb**2/vb),
                            variance_only_score_change=np.mean(np.log(vc)+eb**2/vc-np.log(vb)-eb**2/vb),
                            base_forecast_mean=b.mean(),cycle_forecast_mean=cy.mean(),increment_mean=d.mean(),
                            base_forecast_ms=np.mean(b0**2),cycle_forecast_ms=np.mean(c0**2),increment_ms=np.mean(d0**2),realized_ms=np.mean(y0**2),
                            bb=np.mean(b0*b0),bd=np.mean(b0*d0),dd=np.mean(d0*d0),by=np.mean(b0*y0),dy=np.mean(d0*y0),
                            increment_pnl=np.mean(d0*y0),increment_residual_pnl=np.mean(d0*(y0-b0)),
                            base_pred_variance=np.mean(vb),cycle_pred_variance=np.mean(vc)))
                    if not rec:continue
                    # Equal stock weighting within leg, equal leg weighting within
                    # date; those are the original sample-weight conventions.
                    f=pd.DataFrame(rec).groupby('idx').mean().reset_index().drop(columns='leg')
                    f['factor']=factor;f['method']=method;f['year']=year;f['model_year']=modelyear;f['horizon']=horizon
                    rows.extend(f.to_dict('records'))
        log(f'{year}: primary saved forecasts diagnosed')
    out=pd.DataFrame(rows)
    out.to_csv(OUT/'forecast_daily_diagnostics.csv',index=False)
    return out


def summarize(f):
    periods={'in_sample_2004':(2004,2004),'validation':(2005,2014),'holdout':(2015,2025),'all_oos':(2005,2025)}
    annual=[];summaries=[]
    def one(g):
        out={'dates':len(g),'mean_stocks_per_leg':float(g.n_stocks.mean())}
        for x in ['base_mse','cycle_mse','base_gaussian_score','cycle_gaussian_score','base_forecast_mean','cycle_forecast_mean','increment_mean','base_pred_variance','cycle_pred_variance']:
            out[x]=float(g[x].mean())
        for name in ['base_ic','cycle_ic','increment_ic','mse_change','gaussian_score_change','variance_only_score_change']:
            mu,se,p,lo,hi=mean_hac(g[name])
            out.update({name:mu,name+'_se':se,name+'_p':p,name+'_ci_low':lo,name+'_ci_high':hi})
        for name in ['base_forecast','cycle_forecast','increment','realized']:
            out[name+'_crosssection_sd_bps']=float(np.sqrt(g[name+'_ms'].mean())*10000)
        out['mse_relative_change']=out['cycle_mse']/out['base_mse']-1
        out.update(joint_calibration(g))
        return out
    for (factor,method,horizon,year),g in f.groupby(['factor','method','horizon','year']):
        annual.append(dict(factor=factor,method=method,horizon=horizon,year=year,**one(g.sort_values('idx'))))
    for period,(lo,hi) in periods.items():
        for (factor,method,horizon),g in f[f.year.between(lo,hi)].groupby(['factor','method','horizon']):
            summaries.append(dict(period=period,factor=factor,method=method,horizon=horizon,**one(g.sort_values('idx'))))
    pd.DataFrame(annual).to_csv(OUT/'forecast_annual_diagnostics.csv',index=False)
    s=pd.DataFrame(summaries)
    for _,ix in s.groupby(['period','horizon']).groups.items():
        for col in ['variance_only_score_change_p','increment_ic_p','calibration_increment_p']:
            s.loc[ix,col+'_holm36']=multipletests(s.loc[ix,col],method='holm')[1]
    s.to_csv(OUT/'forecast_period_diagnostics.csv',index=False)
    # All quantitative cross-model summaries below are descriptive, with no
    # claim that36 correlated candidates are independent draws.
    lines=[]
    for horizon in [1,5]:
        for period in periods:
            q=s[(s.horizon==horizon)&(s.period==period)]
            lines.append(dict(period=period,horizon=horizon,comparisons=len(q),
                cycle_mse_worse=int((q.mse_change>0).sum()),
                median_mse_relative_change=float(q.mse_relative_change.median()),
                median_increment_ic=float(q.increment_ic.median()),
                positive_increment_ic=int((q.increment_ic>0).sum()),
                median_increment_calibration=float(q.calibration_increment.median()),
                median_forecast_increment_sd_bps=float(q.increment_crosssection_sd_bps.median()),
                median_realized_sd_bps=float(q.realized_crosssection_sd_bps.median()),
                worse_variance_score=int((q.gaussian_score_change>0).sum()),
                worse_variance_only_score=int((q.variance_only_score_change>0).sum()),
                median_variance_only_score_change=float(q.variance_only_score_change.median()),
                improved_variance_only_holm36=int(((q.variance_only_score_change<0)&(q.variance_only_score_change_p_holm36<.05)).sum())))
    aggregate=pd.DataFrame(lines);aggregate.to_csv(OUT/'forecast_diagnostic_summary.csv',index=False)
    audit=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),status='post-result explanation; no new fitted models or trading selection',
        model_source='primary fits/2005.pkl..2025.pkl;2004 diagnostic reuses2005 model and is in-sample',
        one_day='sampled signed next-day payoff',five_day='same primary one-day predictions checked against exact signed5-session fixed-share payoff divided by5; not the separately trained horizon sensitivity',
        grouping='Within each trading date and long/short leg, demean predictions and outcomes; equal stocks within leg, equal available legs within date',
        ic='Spearman cross-sectional correlations require at least5 sampled names per date-leg; correlations averaged across available legs; other moments include smaller legs',
        significance='HAC12 on weekly sampled date-level statistics; no stock-IID t statistics. Suffix_holm36 adjusts within36 factor/model comparisons for each period/horizon/diagnostic; other p-values unadjusted. These diagnostic tests are not a new claim of trading alpha.',
        calibration='Joint centered projection y=beta_base*forecast_base + beta_increment*(forecast_cycle-forecast_base); sandwich covariance built from date-level moments with Bartlett HAC12',
        interpretation='Incremental calibration1 means forecast changes correctly scaled conditional on baseline;0 means no added linear prediction;negative means wrong direction. These are predictive diagnostics, not causal mechanisms.',
        mse='Positive cycle-minus-base MSE means adding cycle worsens forecast error; raw untrimmed outcomes used',
        gaussian_score='Lower log(predicted variance)+residual_squared/predicted variance is better; for5-session average outcome, daily predicted variance divided by5 as approximation',
        variance_only_score='Same scoring rule, but residuals from baseline mean prediction used for both baseline and cycle variances; isolates variance-forecast change',
        independence='36 factor/model comparisons are correlated; aggregate counts and medians are descriptive',
        rows=len(f),summary_rows=len(s))
    (OUT/'forecast_diagnostic_audit.json').write_text(json.dumps(audit,indent=2))
    print(aggregate.to_string(index=False))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    f=make_daily();summarize(f)
