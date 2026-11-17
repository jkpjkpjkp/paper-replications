"""Check the requested 202-column rule on the full eligible US training universe."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
signals = BASE/'Chen_Zimmermann_Factor_Zoo_2026_10/Firm Level Characteristics/Predictors.parquet'
columns = ['permno','yyyymm','mthcap','primaryexch','securitytype','securitysubtype',
           'sharetype','issuertype','usincflg','conditionaltype','tradingstatusflg','mthdelflg']
f = ds.dataset(BASE/'crsp-monthly/parquet').to_table(columns=columns,
        filter=(ds.field('yyyymm') >= 200812) & (ds.field('yyyymm') <= 201911)).to_pandas()
f['mthcap'] = f.mthcap.astype(float)
assert not f.duplicated(['permno','yyyymm']).any()
good = np.ones(len(f), dtype=bool)
for name, value in dict(securitytype='EQTY',securitysubtype='COM',sharetype='NS',
                        issuertype='CORP',usincflg='Y',conditionaltype='RW',tradingstatusflg='A').items():
    good &= f[name].eq(value)
good &= f.mthdelflg.eq('N') & f.mthcap.gt(0)
f = f.loc[good]
nyse20 = f.loc[f.primaryexch.eq('N')].groupby('yyyymm').mthcap.quantile(.2)
f = f.loc[f.mthcap.ge(f.yyyymm.map(nyse20))]
keys = pd.MultiIndex.from_frame(f[['permno','yyyymm']])
del f
pf = pq.ParquetFile(signals)
names = [n for n in pf.schema_arrow.names if n not in ['permno','yyyymm']]
count = np.zeros(len(names), dtype=np.int64)
lo = np.full(len(names), np.inf)
hi = np.full(len(names), -np.inf)
rows = 0
for i in range(pf.num_row_groups):
    block = pf.read_row_group(i).to_pandas()
    block = block.loc[block.yyyymm.between(200812,201911)]
    keep = pd.MultiIndex.from_frame(block[['permno','yyyymm']]).isin(keys)
    x = block.loc[keep, names].to_numpy(float)
    rows += len(x)
    finite = np.isfinite(x)
    count += finite.sum(0)
    if len(x):
        lo = np.minimum(lo, np.where(finite,x,np.inf).min(0))
        hi = np.maximum(hi, np.where(finite,x,-np.inf).max(0))
    if i%10 == 0:
        print(json.dumps(dict(groups=i+1,total_groups=pf.num_row_groups,eligible_training_rows=rows)), flush=True)
vary = (count > 0) & (hi > lo)
result = dict(universe='All eligible US common stocks, prior-month NYSE20 screen',
              forecast_training_months=['200901','201912'],characteristic_months=['200812','201911'],
              eligible_training_rows=rows,source_columns=len(names),
              nonconstant_observed_training_columns=int(vary.sum()),
              dropped=[n for n,k in zip(names,vary) if not k],
              requested_columns=202, feature_count_pass=bool(vary.sum()==202),
              note='Raw-column eligibility check only. Full monthly normalization and timing verification remain required. No features dropped to force 202.',
              observed_counts=dict(zip(names,map(int,count))))
out = ROOT/'results/cz-preflight.json'
out.write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='observed_counts'},indent=2))
