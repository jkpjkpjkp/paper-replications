"""Month-clustered forecast comparisons; frozen hypotheses and block bootstrap."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import norm,spearmanr,t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'

def inference(d):
    d=np.asarray(d);n=len(d);x=d-d.mean();lag=min(6,n-1)
    lr=x@x/n
    for k in range(1,lag+1):lr+=2*(1-k/(lag+1))*(x[k:]@x[:-k]/n)
    se=np.sqrt(max(lr,0)/n);t=d.mean()/se if se else 0
    rng=np.random.default_rng(20261106);boot=[]
    for _ in range(2000):
        starts=rng.integers(n,size=int(np.ceil(n/12)));ix=((starts[:,None]+np.arange(12))%n).ravel()[:n]
        boot.append(d[ix].mean())
    return {'delta':d.mean(),'hac_t':t,'p_two_sided':2*norm.sf(abs(t)),
            'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),'months':n}

def main():
    monthly=[];returns=[]
    files=sorted(OUT.glob('predictions-*.parquet'))
    assert len(files)==16,'All 2010–2025 folds must finish first'
    for f in files:
        df=pd.read_parquet(f)
        sup=OUT/f.name.replace('predictions','supplement')
        if sup.exists():df=pd.concat([df,pd.read_parquet(sup)],ignore_index=True)
        controls=OUT/f.name.replace('predictions','controls')
        if controls.exists():
            c=pd.read_parquet(controls)
            # Return-only control has no trained variance head; use the independent risk forecast.
            hv=df[df.model=='nn_ind'].set_index(['permno','target_month']).variance
            ix=c.model=='nn_noaux_matched'
            c.loc[ix,'variance']=hv.reindex(pd.MultiIndex.from_frame(c.loc[ix,['permno','target_month']])).to_numpy()
            df=pd.concat([df,c],ignore_index=True)
        assert np.isfinite(df.mu).all() and np.isfinite(df.variance).all() and (df.variance>0).all()
        assert not df.duplicated(['model','permno','target_month']).any()
        for (model,month),a in df.groupby(['model','target_month']):
            r=a.yret.to_numpy();m=a.mu.to_numpy();h=a.variance.to_numpy();v=a.yrv.to_numpy();ix=a.variance_observed.to_numpy()
            mse=(r-m)**2
            qlike=np.log(h[ix])+v[ix]/h[ix]
            # Deviance form has the same rankings and supports interpretable percent changes.
            qdev=v[ix]/h[ix]-np.log(v[ix]/h[ix])-1
            nll=.5*(np.log(2*np.pi*h)+mse/h)
            sd=np.sqrt(h)
            density_nll=(-student_t.logpdf((r-m)/np.sqrt(h*3/5),df=5)+np.log(np.sqrt(h*3/5))) if model=='nn_student' else a.exact_nll.to_numpy() if model in ['garch','sharv'] else nll
            row={'model':model,'month':month,'year':a.year.iloc[0],'n':len(a),'n_joint':ix.sum(),
                'mse':mse.mean(),'mse_joint':mse[ix].mean(),'zero_mse':np.mean(r*r),'qlike':qlike.mean(),
                'qdev':qdev.mean(),'rv_mse':np.mean((v[ix]-h[ix])**2),'nll':nll.mean(),
                'cover90':np.mean(abs(r-m)<=1.644853626951*sd),'std_resid2':np.mean(mse/h),
                'rv_ratio':v[ix].mean()/h[ix].mean(),'ic':spearmanr(r,m).statistic if np.std(m)>0 else np.nan,
                'density_nll':density_nll.mean()}
            monthly.append(row)
            if '_seed' not in model and np.std(m)>0:
                # Gross equal-weight decile spread; turnover calculated across successive months below.
                order=np.argsort(m,kind='stable');k=max(1,len(a)//10);weights=np.zeros(len(a));weights[order[:k]]=-1/k;weights[order[-k:]]=1/k
                returns.append({'model':model,'month':month,'gross':weights@r,
                    'holdings':dict(zip(a.permno.astype(int),weights)), 'realized':dict(zip(a.permno.astype(int),r))})
    month=pd.DataFrame(monthly);month.to_csv(OUT/'monthly_scores.csv',index=False)
    avg=month.groupby('model').mean(numeric_only=True)
    avg['return_R2_zero_pct']=100*(1-avg.mse/avg.zero_mse)
    ref=avg.loc['ind_ridge'];avg['mse_gain_vs_ridge_pct']=100*(1-avg.mse/ref.mse)
    avg['qlike_gain_vs_ridge_pct']=100*(1-avg.qdev/ref.qdev)
    avg.sort_values('mse').to_csv(OUT/'summary.csv')
    # Predeclared ablations plus comparisons with the strong independent ridge baseline.
    pairs=[('nn_joint','nn_ind'),('trees_joint','trees_ind'),('half_joint_half_hgb','half_ind_half_hgb'),
           ('ngboost','ind_hgb'),('sharv','garch'),('cmk','factor_ind'),('cmk_onestep','cmk'),
           ('nn_joint','ind_ridge'),('ngboost','ind_ridge'),('nn_gaussian','nn_ind'),('nn_student','nn_ind'),
           ('cmk_scaled','factor_scaled_ind'),('cmk_scaled','ind_ridge')]
    pairs += [('nn_joint','nn_noaux_matched'),('nn_joint','nn_shuffled_aux')]
    tests=[]
    for a,b in pairs:
        aa=month[month.model==a].set_index('month');bb=month[month.model==b].set_index('month')
        for metric in ['mse','qdev','nll','density_nll']:
            d=aa[metric]-bb[metric];s=inference(d)
            s.update(model=a,baseline=b,metric=metric,improvement_pct=-100*s['delta']/bb[metric].mean() if metric in ['mse','qdev'] else np.nan)
            tests.append(s)
    tt=pd.DataFrame(tests);order=np.argsort(tt.p_two_sided.to_numpy());adj=np.maximum.accumulate(np.minimum(1,tt.p_two_sided.to_numpy()[order]*(len(tt)-np.arange(len(tt)))))
    tt['p_holm']=np.nan;tt.loc[order,'p_holm']=adj;tt.to_csv(OUT/'paired_tests.csv',index=False)
    sub=[]
    for label,mask in [('2010-2017',month.year<=2017),('2018-2025',month.year>=2018),('2020',month.year==2020),('ex2020',month.year!=2020)]:
        z=month[mask].groupby('model').mean(numeric_only=True);z['period']=label;sub.append(z.reset_index())
    pd.concat(sub).to_csv(OUT/'subperiods.csv',index=False)
    cov=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('covariance-*.csv'))]);cov.to_csv(OUT/'covariance_scores.csv',index=False)
    ct=[]
    for a,b in [('cmk','factor_ind'),('cmk_onestep','cmk'),('cmk','cmk_no_floor'),('cmk_scaled','factor_scaled_ind')]:
        aa=cov[cov.model==a].set_index('month');bb=cov[cov.model==b].set_index('month')
        for metric in ['nll','nll_hgb_mean']:
            delta=(aa[metric]-bb[metric]).dropna()
            if len(delta):ct.append(dict(model=a,baseline=b,metric=metric,**inference(delta)))
    pd.DataFrame(ct).to_csv(OUT/'covariance_tests.csv',index=False)
    ports=[]
    for model in sorted(set(r['model'] for r in returns)):
        prev={};prevret={};prevpnl=0.
        for row in sorted([r for r in returns if r['model']==model],key=lambda r:r['month']):
            # Two $1 long/short sleeves; drift weights using observed prior-month returns.
            drift={k:v*(1+prevret.get(k,0))/max(1+prevpnl,1e-8) for k,v in prev.items()}
            w=row['holdings'];turn=sum(abs(w.get(k,0)-drift.get(k,0)) for k in set(w)|set(drift))
            ports.append({'model':model,'month':row['month'],'gross':row['gross'],'turnover':turn,'net_10bp_target_turnover':row['gross']-.001*turn})
            prev=w;prevret=row['realized'];prevpnl=row['gross']
    pp=pd.DataFrame(ports);pp.to_csv(OUT/'portfolio_monthly.csv',index=False)
    pr=[]
    for name,g in pp.groupby('model'):
        pr.append({'model':name,'annual_mean_gross':12*g.gross.mean(),'annual_vol':np.sqrt(12)*g.gross.std(),
                   'sharpe_gross':np.sqrt(12)*g.gross.mean()/g.gross.std(),'sharpe_net_target_turnover':np.sqrt(12)*g.net_10bp_target_turnover.mean()/g.net_10bp_target_turnover.std(),
                   'mean_target_turnover':g.turnover.mean()})
    pd.DataFrame(pr).to_csv(OUT/'portfolio_summary.csv',index=False)
    # Scientific plot: all displayed gains use the same 192 monthly observations.
    names=['ind_ridge','ind_hgb','trees_ind','trees_joint','nn_ind','nn_joint','ngboost','nn_gaussian','nn_student','garch','sharv','factor_ind','cmk','cmk_scaled']
    z=avg.loc[names];fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,col,title in zip(axes,['mse_gain_vs_ridge_pct','qlike_gain_vs_ridge_pct'],['Return MSE improvement vs independent ridge','Variance QLIKE improvement vs independent ridge']):
        values=z[col];ax.barh(names,values,color=['#247a68' if v>0 else '#b85b51' for v in values]);ax.axvline(0,color='black',lw=.7);ax.set_xlabel('Percent improvement (higher is better)');ax.set_title(title,fontsize=11);ax.invert_yaxis()
    fig.suptitle('Monthly CRSP top-1000 forecasts · 2010–2025 · annual refits',fontsize=14)
    fig.savefig(OUT/'comparison.png',dpi=180);plt.close(fig)
    print(avg.loc[names,['return_R2_zero_pct','mse_gain_vs_ridge_pct','qlike_gain_vs_ridge_pct','nll','cover90','ic']].to_string())
if __name__=='__main__':main()
