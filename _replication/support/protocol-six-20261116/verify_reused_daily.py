"""Recompute the saved daily experiments' HAC inference directly from ledgers."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
ROOT=Path('/home/jkp/data');HERE=Path(__file__).resolve().parent
f=pd.read_parquet(ROOT/'factors-daily/ff5_momentum_daily.parquet');f.date=pd.to_datetime(f.date)
checks=[]
for name,rel,template in [('LiMT','limt-us-reproduction/experiments/standard_characteristics','value_cost{}_ledger.csv'),
                          ('PRISM-VQ','prism-vq-reproduction/experiments/cz_vw','cost{}_ledger.csv')]:
    path=ROOT/rel/'analysis';alpha=pd.read_csv(path/'alpha.csv')
    if name=='LiMT':alpha=alpha.loc[alpha.weighting.eq('value')&alpha.window.eq('full-2017-2025')&alpha.factor_model.eq('FF5+UMD')]
    for cost in [0,10]:
        p=path/template.format(cost);r=pd.read_csv(p,parse_dates=['date'])
        j=r.merge(f,on='date',how='left',validate='one_to_one');assert len(j)==len(r)
        np.testing.assert_allclose(j.return_net,j.return_gross-j.fees/j.start_nav,atol=1e-12)
        np.testing.assert_allclose(j.fees,(j.traded_dollars+j.final_liquidation_dollars)*cost/10000,atol=1e-7)
        x=np.column_stack([np.ones(len(j)),j[['mkt_rf','smb','hml','rmw','cma','mom']]])
        y=j.return_net.to_numpy();assert np.isfinite(x).all() and np.isfinite(y).all()
        n,k=x.shape;beta=np.linalg.lstsq(x,y,rcond=None)[0];z=x*(y-x@beta)[:,None]
        meat=z.T@z
        for lag in range(1,21):
            cross=z[lag:].T@z[:-lag];meat+=(1-lag/21)*(cross+cross.T)
        bread=np.linalg.inv(x.T@x);cov=bread@meat@bread*n/(n-k);se=np.sqrt(cov[0,0]);t=beta[0]/se
        ci=(beta[0]+np.array([-1,1])*student_t.ppf(.975,n-k)*se)*252
        actual=np.r_[beta[0]*252,t,2*student_t.sf(abs(t),n-k),ci]
        v=alpha.loc[alpha.cost_bp.eq(cost)].iloc[0]
        expected=np.array([v.annual_alpha,v.get('hac_t',v.get('t')),v.p,v.ci_low,v.ci_high])
        np.testing.assert_allclose(actual,expected,rtol=1e-8,atol=1e-10)
        checks.append(dict(paper=name,cost_bps=cost,days=n,ledger_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),max_inference_error=float(abs(actual-expected).max())))
(HERE/'reused-daily-verification.json').write_text(json.dumps(dict(passed=True,checks=checks),indent=2))
print(json.dumps(checks,indent=2))
