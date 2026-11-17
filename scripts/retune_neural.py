"""Expand neural regularization search before any test fold is run."""
import json
import numpy as np
import pandas as pd
from run import ROOT,OUT,FEATURES,scores
from models import Transform,Net
df=pd.read_parquet(ROOT/'data/panel.parquet');df=df[df.variance_observed]
tr=df[df.year.between(1995,2004)];va=df[df.year.between(2005,2009)]
tf=Transform().fit(tr[FEATURES]);x=tf(tr[FEATURES]);z=tf(va[FEATURES]);r=tr.yret.to_numpy();v=tr.yrv.to_numpy()
selection=json.loads((OUT/'selection.json').read_text())
old=pd.read_csv(OUT/'validation_scores.csv');records=old[~old.model.isin(['nn_ind','nn_joint'])].to_dict('records')
opts=[]
for ep in [5,15,35]:
    mod=Net('independent',epochs=ep).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
    s.update(model='nn_ind',param=ep);records.append(s);opts.append(s);print(s,flush=True)
selection['nn_ind_epochs']=[min(opts,key=lambda s:s['mse'])['param'],min(opts,key=lambda s:s['qlike'])['param']]
opts=[]
for ep in [10,35]:
    for w in [.1,1.,10.]:
        mod=Net('joint',w,epochs=ep).fit(x,r,v);m,h=mod.predict(z);s=scores(va,m,h)
        s.update(model='nn_joint',param=w,epochs=ep,objective=s['mse']/np.mean(r*r)+s['qlike'])
        records.append(s);opts.append(s);print(s,flush=True)
chosen=min(opts,key=lambda s:s['objective']);selection['nn_weight']=chosen['param'];selection['nn_joint_epochs']=chosen['epochs']
selection['protocol']['neural_epochs']='independent head-specific 5/15/35; joint 10/35, weight .1/1/10; Gaussian/Student 35'
(OUT/'selection.json').write_text(json.dumps(selection,indent=2))
pd.DataFrame(records).to_csv(OUT/'validation_scores.csv',index=False)
print('SELECTED',selection,flush=True)
