"""Independent saved-score IC, turnover and HAC regression checks."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t as student_t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from transalpha.data import sha, save_json
BASE=Path('/home/jkp/data/.replication-local/artifacts/ordered-reproductions')
OUT=ROOT/'results/cz202'
protocol=json.loads((OUT/'protocol.json').read_text())
data_audit=json.loads((OUT/'data_audit.json').read_text())
assert sha(BASE/'data/cz-monthly/panel.parquet')==data_audit['source_sha256']
for name,digest in protocol['code_hashes'].items():assert sha(ROOT/name)==digest
assert sha(BASE/'scripts/evaluate_cz.py')==protocol['evaluator_sha256']
assert sha(BASE/'data/factors-monthly/ff5-umd.csv')==protocol['factor_sha256']
panel=pd.read_parquet(BASE/'data/cz-monthly/panel.parquet',columns=['return_month','permno','target_return','price_return','delisting_month','lagged_market_cap'])
panel=panel.loc[panel.return_month.between(202201,202512)]
factors=pd.read_csv(BASE/'data/factors-monthly/ff5-umd.csv')
fields=['Mkt-RF','SMB','HML','RMW','CMA','UMD']
receipt=[]
completion=json.loads((OUT/'completion.json').read_text())
expected=[OUT/f'{v}-seed{s}/metrics.json' for v in completion['variants'].split(',') for s in completion['seeds'].split(',')]
for path in expected:
    m=json.loads(path.read_text());folder=path.parent
    assert sha(folder/'model.pt')==m['checkpoint_sha256']
    assert sha(folder/'scores.csv')==m['scores_sha256']
    assert sha(OUT/'protocol.json')==m['protocol_sha256']
    scores=pd.read_csv(folder/'scores.csv')
    joined=panel.merge(scores,on=['return_month','permno'],how='outer',indicator=True,validate='one_to_one')
    assert joined._merge.eq('both').all() and np.isfinite(joined.score).all()
    returns=pd.read_csv(folder/'monthly_returns.csv')
    old={};ledger=[];ics=[]
    for month,g in joined.groupby('return_month',sort=True):
        ics.append(pearsonr(g.score,g.target_return).statistic)
        g=g.sort_values(['score','permno'])
        decile=np.arange(len(g))*10//len(g)+1
        long=g.loc[decile==10];short=g.loc[decile==1]
        weights={int(r.permno):float(r.lagged_market_cap/long.lagged_market_cap.sum()) for r in long.itertuples()}
        weights.update({int(r.permno):float(-r.lagged_market_cap/short.lagged_market_cap.sum()) for r in short.itertuples()})
        g=g.set_index('permno')
        gross=sum(w*g.at[s,'target_return'] for s,w in weights.items())
        turnover=sum(abs(weights.get(s,0)-old.get(s,0)) for s in set(weights)|set(old))
        end={s:(0 if g.at[s,'delisting_month']=='Y' else w*(1+g.at[s,'price_return'])) for s,w in weights.items()}
        old={s:d/(1+gross) for s,d in end.items()}
        ledger.append([gross,turnover,gross-.001*turnover])
    ledger[-1][2]-=.001*sum(abs(d) for d in end.values())
    np.testing.assert_allclose(ledger,returns[['gross','turnover','net']],atol=1e-11,rtol=0)
    np.testing.assert_allclose(ics,returns.ic,atol=1e-8,rtol=0) # Training labels were float32.
    data=returns.merge(factors,on='return_month',validate='one_to_one')
    x=np.column_stack([np.ones(len(data)),data[fields].to_numpy()]);n,k=x.shape
    inverse=np.linalg.inv(x.T@x)
    for kind in ['gross','net']:
        y=data[kind].to_numpy();beta=np.linalg.lstsq(x,y,rcond=None)[0];u=y-x@beta
        xu=x*u[:,None];meat=xu.T@xu
        for lag in range(1,7):
            cross=xu[lag:].T@xu[:-lag]
            meat+=(1-lag/7)*(cross+cross.T)
        cov=inverse@meat@inverse*n/(n-k)
        se=np.sqrt(cov[0,0]);t=beta[0]/se
        ci=(beta[0]+np.array([-1,1])*student_t.ppf(.975,n-k)*se)*12
        a=m['regression'][kind]
        np.testing.assert_allclose([beta[0]*12,t,2*student_t.sf(abs(t),n-k)],
            [a['annual_alpha'],a['hac_t'],a['p_value']],atol=1e-9,rtol=0)
        np.testing.assert_allclose(ci,a['annual_alpha_ci95'],atol=1e-9,rtol=0)
    receipt.append(dict(run=folder.name,months=n,passed=True))
assert len(receipt)==len(expected)
save_json(OUT/'audit.json',dict(passed=True,runs=receipt,checks=['Code/data/evaluator provenance','Complete stock-month prediction keys','Independent Pearson IC','Independent market-cap decile construction and price-drift turnover ledger','Independent numpy OLS and Bartlett HAC covariance; t, CI and p'],tolerances=dict(portfolio=1e-11,ic=1e-8,hac=1e-9)))
print(f'All {len(receipt)} CZ202 runs independently audited')
