"""Exploratory volatility-scaled CMK and its matched separate-factor control.

Scaling is known at formation: sqrt(trailing three-month average realized variance).
This relaxes the constant idiosyncratic variance restriction in raw return units.
All extension hyperparameters are chosen on 2005-2009, never on test outcomes.
"""
import json
import numpy as np
import pandas as pd
from run import ROOT,OUT,phi
from models import CMK
full=pd.read_parquet(ROOT/'data/panel.parquet');df=full[full.variance_observed]
def risk(d):return np.sqrt(np.exp(d.log_rv3.fillna(d.log_rv).to_numpy())).clip(.005,1)
tr=df[df.year.between(1995,2004)];va=df[df.year.between(2005,2009)]
ps=phi(tr);pv=phi(va);ss=risk(tr);sv=risk(va);opts=[]
for eta in [0.,1.,10.]:
    for penalty in [0.,.1,1.]:
        mod=CMK(eta,penalty).fit(ps,tr.yret.to_numpy()/ss,tr.target_month.to_numpy())
        scores=[]
        for month in va.target_month.unique():
            ix=(va.target_month==month).to_numpy()
            scores.append(mod.density_score(pv[ix],va.yret.to_numpy()[ix]/sv[ix])+np.log(sv[ix]).mean())
        opts.append({'eta':eta,'penalty':penalty,'validation_nll':float(np.mean(scores))})
best=min(opts,key=lambda o:o['validation_nll'])
(OUT/'scaled_cmk_selection.json').write_text(json.dumps({'selected':best,'candidates':opts,'classification':'exploratory extension; validation-only tuning'},indent=2))
print('SELECTED',best,flush=True)
for year in range(2010,2026):
    tr=df[df.year.between(year-10,year-1)];te=full[(full.year==year)&full.return_observed]
    pt=phi(tr);pz=phi(te);s=risk(tr);sz=risk(te);preds=[];cov=[]
    for name,control in [('factor_scaled_ind',True),('cmk_scaled',False)]:
        mod=CMK(best['eta'],best['penalty'],control).fit(pt,tr.yret.to_numpy()/s,tr.target_month.to_numpy())
        mu,h=mod.predict(pz);mu*=sz;h*=sz**2
        p=te[['permno','target_month','date','year','yret','yrv','variance_observed','size_rank','cap','ydelist']].copy()
        p['model']=name;p['mu']=mu;p['variance']=h;preds.append(p)
        for month in te.target_month.unique():
            ix=(te.target_month==month).to_numpy()
            cov.append({'year':year,'month':month,'model':name,'nll':mod.density_score(pz[ix],te.yret.to_numpy()[ix]/sz[ix])+np.log(sz[ix]).mean(),'n':sum(ix)})
    pd.concat(preds,ignore_index=True).to_parquet(OUT/f'supplement-{year}.parquet',index=False)
    pd.DataFrame(cov).to_csv(OUT/f'covariance-scaled-{year}.csv',index=False)
    print('COMPLETE',year,flush=True)
