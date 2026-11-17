"""Reconstruct monthly eligibility and next-month outcomes independently of the builder."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT=Path('/home/jkp/data/.replication-local/artifacts/ordered-reproductions');BASE=Path('/home/jkp/data')
manifest=json.loads((ROOT/'data/cz-monthly/manifest.json').read_text())
p=pd.read_parquet(ROOT/'data/cz-monthly/panel.parquet')
names=manifest['features']
assert len(names)==202 and np.isfinite(p[names]).all().all()
assert p[names].min().min()>=-1 and p[names].max().max()<=1
assert not p.duplicated(['permno','feature_month']).any()
columns=['permno','yyyymm','mthcap','mthret','mthretx','primaryexch','securitytype',
    'securitysubtype','sharetype','issuertype','usincflg','conditionaltype','tradingstatusflg','mthdelflg']
c=ds.dataset(BASE/'crsp-monthly/parquet').to_table(columns=columns,
    filter=(ds.field('yyyymm')>=200412)&(ds.field('yyyymm')<=202512)).to_pandas()
for n in ['mthcap','mthret','mthretx']:c[n]=c[n].astype(float)
outcomes=c[['permno','yyyymm','mthret','mthretx']].rename(columns={'yyyymm':'return_month'})
for n,v in dict(securitytype='EQTY',securitysubtype='COM',sharetype='NS',issuertype='CORP',
    usincflg='Y',conditionaltype='RW',tradingstatusflg='A',mthdelflg='N').items():
    c=c.loc[c[n].eq(v)]
c=c.loc[c.mthcap.gt(0)&c.yyyymm.le(202511)].copy()
breakpoints=c.loc[c.primaryexch.eq('N')].groupby('yyyymm').mthcap.quantile(.2)
c=c.loc[c.mthcap.ge(c.yyyymm.map(breakpoints))]
expected=pd.MultiIndex.from_frame(c[['permno','yyyymm']])
actual=pd.MultiIndex.from_frame(p[['permno','feature_month']])
assert set(expected)==set(actual),'CZ coverage differs from CRSP-defined universe'
merged=p.merge(c[['permno','yyyymm','mthcap']],left_on=['permno','feature_month'],
    right_on=['permno','yyyymm'],validate='one_to_one')
assert np.allclose(merged.lagged_market_cap,merged.mthcap)
assert np.allclose(p.nyse20,p.feature_month.map(breakpoints))
period=pd.PeriodIndex(p.feature_month.astype(str),freq='M')+1
assert (period.year*100+period.month==p.return_month).all()
merged=p.merge(outcomes,on=['permno','return_month'],validate='one_to_one',how='left')
assert np.allclose(merged.target_return,merged.mthret,equal_nan=True)
assert np.allclose(merged.price_return,merged.mthretx,equal_nan=True)
report=dict(passed=True,rows=len(p),features=202,months=p.return_month.nunique(),
    first_return_month=int(p.return_month.min()),last_return_month=int(p.return_month.max()),
    checks=['Exact full-universe coverage from independent CRSP reconstruction',
            'Lagged capitalization and NYSE20 quantile','One-calendar-month feature/return timing',
            'Total and price returns joined from correct outcome month','Unique keys and finite normalized features'],
    limitation=manifest['limitation'])
(Path(__file__).resolve().parents[1]/'results/cz202/source_verification.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
