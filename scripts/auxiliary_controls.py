"""Exploratory capacity/duration-matched auxiliary-task controls; no test-based tuning."""
import json
import numpy as np
import pandas as pd
from run import ROOT,OUT,FEATURES
from models import Transform,Net
selection=json.loads((OUT/'selection.json').read_text())
full=pd.read_parquet(ROOT/'data/panel.parquet');df=full[full.variance_observed]
for year in range(2010,2026):
    tr=df[df.year.between(year-10,year-1)];te=full[(full.year==year)&full.return_observed]
    tf=Transform().fit(tr[FEATURES]);x=tf(tr[FEATURES]);z=tf(te[FEATURES]);r=tr.yret.to_numpy();v=tr.yrv.to_numpy()
    rng=np.random.default_rng(301);shuffled=v.copy()
    for month in tr.target_month.unique():
        ix=np.flatnonzero(tr.target_month.to_numpy()==month);shuffled[ix]=rng.permutation(v[ix])
    preds=[]
    for name,weight,target in [('nn_noaux_matched',0.,v),('nn_shuffled_aux',selection['nn_weight'],shuffled)]:
        mod=Net('joint',weight,epochs=selection['nn_joint_epochs']).fit(x,r,target);m,h=mod.predict(z)
        p=te[['permno','target_month','date','year','yret','yrv','variance_observed','size_rank','cap','ydelist']].copy()
        p['model']=name;p['mu']=m;p['variance']=h;preds.append(p)
    pd.concat(preds,ignore_index=True).to_parquet(OUT/f'controls-{year}.parquet',index=False)
    print('COMPLETE',year,flush=True)
