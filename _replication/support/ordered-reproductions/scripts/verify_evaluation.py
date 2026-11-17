"""Independent numerical checks, not a model result or an original-paper experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from evaluate_cz import portfolios, regress, FACTORS

ROOT=Path(__file__).resolve().parents[1]
# Twenty names: verify cap-weighted deciles, drift, entry/exit and trading-cost signs.
frame=pd.DataFrame(dict(permno=np.tile(np.arange(20),2),
    return_month=np.repeat([202201,202202],20),lagged_market_cap=np.tile([1.,3.],20),
    nyse20=.5,target_return=0.,price_return=0.,delisting_month='N'))
scores=frame[['permno','return_month']].copy();scores['score']=scores.permno
frame.loc[frame.permno.eq(19),'target_return']=.1
frame.loc[frame.permno.eq(19),'price_return']=.08 # dividend component stays cash
r,h=portfolios(frame,scores,10)
assert np.allclose(r.gross,.075)
assert np.isclose(r.turnover.iloc[0],2)
prev=pd.Series({0:-.25/1.075,1:-.75/1.075,18:.25/1.075,19:.75*1.08/1.075})
target=pd.Series({0:-.25,1:-.75,18:.25,19:.75})
assert np.isclose(r.turnover.iloc[1],abs(target-prev).sum())
assert np.isclose(r.terminal_turnover.iloc[1],2.06)
assert np.allclose(r.net,r.gross-.001*(r.turnover+r.terminal_turnover))
assert np.allclose(h.groupby(['return_month','decile']).weight.sum(),1)
failed=False
try:portfolios(frame,scores.iloc[1:],10)
except ValueError:failed=True
assert failed,'Must reject missing scores rather than filter universe'

# Independently assemble Newey-West sandwich, finite-sample correction, t and CI.
from scipy.stats import t as student_t
rng=np.random.RandomState(119)
n,k,L=48,7,6
x=np.column_stack([np.ones(n),rng.normal(0,.04,(n,6))])
noise=rng.normal(0,.02,n)
for j in range(1,n):noise[j]+=.4*noise[j-1]
y=x@np.array([.003,.5,-.2,.1,.05,.2,-.1])+noise
f=pd.DataFrame(x[:,1:],columns=FACTORS);f['return_month']=np.arange(n)
rr=pd.DataFrame(dict(return_month=np.arange(n),gross=y,net=y-.001))
result=regress(rr,f,L)
b=np.linalg.lstsq(x,y,rcond=None)[0];u=y-x@b;z=x*u[:,None]
meat=z.T@z
for lag in range(1,L+1):
    cross=z[lag:].T@z[:-lag]
    meat+=(1-lag/(L+1))*(cross+cross.T)
bread=np.linalg.inv(x.T@x)
cov=bread@meat@bread*n/(n-k)
se=np.sqrt(cov[0,0]);t=b[0]/se
ci=(b[0]+np.array([-1,1])*student_t.ppf(.975,n-k)*se)*12
assert np.isclose(result['gross']['annual_alpha'],b[0]*12)
assert np.isclose(result['gross']['hac_t'],t)
assert np.allclose(result['gross']['annual_alpha_ci95'],ci)
assert np.isclose(result['gross']['p_value'],2*student_t.sf(abs(t),n-k))
assert np.isclose(result['net']['annual_alpha'],result['gross']['annual_alpha']-.012)
report=dict(passed=True,checks=['Market cap weights','Price drift and dividend cash',
    'Initial and terminal costs','Reject incomplete score universe','Independent Newey-West covariance',
    'Finite-sample t, p and confidence intervals','Zero-net portfolio: no second RF subtraction'],
    warning='Synthetic verification only; these numbers are not model performance.')
(ROOT/'results/evaluator-verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
