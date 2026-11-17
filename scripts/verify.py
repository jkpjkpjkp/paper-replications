"""Numerical checks for timing, the SHARV density, and covariance scoring."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import quad
from models import recurs,CMK
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_parquet(ROOT/'data/panel.parquet')
assert not df.duplicated(['permno','target_month']).any()
assert (df.target_month==df.month+1).all()
assert df.groupby('month').size().max()<=1000
assert (df.loc[df.variance_observed,'yrv']>0).all()
a,b,psi,state=.04,.8,.08,.3
def density(r):
    s=(b*state+np.sqrt((b*state)**2+4*(a+psi*state)*r*r))/2
    return np.exp(-.5*r*r/s)/np.sqrt(2*np.pi)*np.sqrt(s)/(2*s-b*state)
mass=quad(density,-np.inf,np.inf)[0]
moment=quad(lambda r:r*r*density(r),-np.inf,np.inf)[0]
expected=b*state+3*(a+psi*state)
assert abs(mass-1)<1e-8 and abs(moment-expected)<1e-8
rng=np.random.default_rng(1);p=6;n=25
model=CMK();model.b=rng.normal(size=p)*.01;A=rng.normal(size=(p,p));model.C=A@A.T*.001;model.u=.5
x=rng.normal(size=(n,p));r=rng.normal(size=n)*.1
cov=(x@model.C@x.T+model.u*np.eye(n))*.01;e=r-x@model.b*.1
direct=.5*(np.linalg.slogdet(cov)[1]+e@np.linalg.solve(cov,e)+n*np.log(2*np.pi))/n
woodbury=model.density_score(x,r)
assert abs(direct-woodbury)<1e-10
# Modifying future returns cannot change any earlier recursive forecast.
y=rng.normal(size=100);ids=np.zeros(100,dtype=np.int64);months=np.arange(100,dtype=np.int64)
old=recurs(y,ids,months,a,b,psi,state,True)[0];y[70:]=100
new=recurs(y,ids,months,a,b,psi,state,True)[0]
assert np.array_equal(old[:71],new[:71])
result={'passed':True,'rows':len(df),'sharv_density_integral':mass,'sharv_second_moment':moment,
        'sharv_expected_second_moment':expected,'woodbury_absolute_error':abs(direct-woodbury),
        'temporal_alignment':True,'recursive_future_perturbation':True}
(ROOT/'results/verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
