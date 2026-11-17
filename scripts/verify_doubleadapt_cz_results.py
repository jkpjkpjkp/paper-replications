"""Independent checks on actual saved scores, holdings, returns and HAC inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/doubleadapt-cz';EVAL=OUT/'evaluation'
report=json.loads((EVAL/'report.json').read_text())
scores=pd.read_csv(OUT/'scores.csv')
assert hashlib.sha256((OUT/'scores.csv').read_bytes()).hexdigest()==report['scores_sha256']
h=pd.read_parquet(EVAL/'holdings.parquet')
r=pd.read_csv(EVAL/'monthly-returns.csv')
p=pd.read_parquet(ROOT/'data/cz-monthly/panel.parquet',columns=['return_month','permno','target_return','lagged_market_cap','nyse20'])
p=p.loc[p.return_month.between(202201,202512)]
keys=['return_month','permno']
assert not scores.duplicated(keys).any() and not h.duplicated(keys).any()
assert set(map(tuple,scores[keys].to_numpy()))==set(map(tuple,p[keys].to_numpy()))
j=h.merge(p,on=keys,validate='one_to_one',suffixes=('','_source'))
assert len(j)==len(p)==len(scores)
assert np.allclose(j.lagged_market_cap,j.lagged_market_cap_source)
assert j.lagged_market_cap.ge(j.nyse20).all()
expected_weight=j.lagged_market_cap/j.groupby(['return_month','decile']).lagged_market_cap.transform('sum')
assert np.allclose(j.weight,expected_weight)
matched=h.merge(scores,on=keys,validate='one_to_one',suffixes=('_holding','_input'))
assert np.allclose(matched.score_holding,matched.score_input)
assert np.allclose(j.groupby(keys[0]).ls_weight.sum(),0,atol=1e-12)
assert np.allclose(j.ls_weight.abs().groupby(j.return_month).sum(),2)
actual=(j.ls_weight*j.target_return).groupby(j.return_month).sum()
assert np.allclose(r.gross,actual.reindex(r.return_month))
assert np.allclose(r.net,r.gross-.001*(r.turnover+r.terminal_turnover))
assert list(r.return_month)==[p.year*100+p.month for p in pd.period_range('2022-01','2025-12',freq='M')]
f=pd.read_csv(ROOT/'data/factors-monthly/ff5-umd.csv').set_index('return_month').loc[r.return_month]
x=np.column_stack([np.ones(len(r)),f[['Mkt-RF','SMB','HML','RMW','CMA','UMD']]])
n,k=x.shape;L=6
for name in ['gross','net']:
    y=r[name].to_numpy();b=np.linalg.lstsq(x,y,rcond=None)[0];z=x*(y-x@b)[:,None]
    middle=z.T@z
    for lag in range(1,L+1):
        cross=z[lag:].T@z[:-lag];middle+=(1-lag/(L+1))*(cross+cross.T)
    bread=np.linalg.inv(x.T@x);cov=bread@middle@bread*n/(n-k)
    se=np.sqrt(cov[0,0]);t=b[0]/se
    ci=12*(b[0]+np.array([-1,1])*student_t.ppf(.975,n-k)*se)
    expected=report['results'][name]
    assert np.isclose(expected['annual_alpha'],12*b[0])
    assert np.isclose(expected['hac_t'],t)
    assert np.isclose(expected['p_value'],2*student_t.sf(abs(t),n-k))
    assert np.allclose(expected['annual_alpha_ci95'],ci)
verified=dict(passed=True,test_months=n,stock_months=len(scores),
    checks=['Exact eligible test-universe score coverage','Recorded score hash','Lagged size filter and cap weights',
            'Zero net / two gross exposure','Holdings times original realized returns',
            'Recorded transaction-cost charges','Complete contiguous 48-month test',
            'Independent OLS/Newey-West alpha, t, p and CI on real returns'],
    scope='Numerical consistency of this adaptation; does not establish historical-vintage integrity or population alpha.')
(EVAL/'verification.json').write_text(json.dumps(verified,indent=2));print(json.dumps(verified,indent=2))
