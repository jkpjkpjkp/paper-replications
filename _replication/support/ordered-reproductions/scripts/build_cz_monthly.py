"""Prepare shared monthly inputs; this does not train a model or bypass its original gate."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OUT=ROOT/'data/cz-monthly'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'manifest.json').exists():raise SystemExit('Existing completed panel is preserved')
started=time.monotonic()
source=BASE/'Chen_Zimmermann_Factor_Zoo_2026_10/Firm Level Characteristics/Predictors.parquet'
check=json.loads((ROOT/'results/cz-preflight.json').read_text())
assert check['feature_count_pass']
pf=pq.ParquetFile(source)
names=[n for n in pf.schema_arrow.names if n not in ['permno','yyyymm']+check['dropped']]
assert len(names)==202
cols=['permno','yyyymm','mthcap','mthret','mthretx','primaryexch','securitytype','securitysubtype',
      'sharetype','issuertype','usincflg','conditionaltype','tradingstatusflg','mthdelflg']
crsp=ds.dataset(BASE/'crsp-monthly/parquet').to_table(columns=cols,
        filter=(ds.field('yyyymm')>=200412)&(ds.field('yyyymm')<=202512)).to_pandas()
assert not crsp.duplicated(['permno','yyyymm']).any()
for col in ['mthcap','mthret','mthretx']:crsp[col]=crsp[col].astype(float)
outcome=crsp[['permno','yyyymm','mthret','mthretx','mthdelflg']].rename(columns={
    'yyyymm':'return_month','mthret':'target_return','mthretx':'price_return','mthdelflg':'delisting_month'})
good=np.ones(len(crsp),dtype=bool)
for n,v in dict(securitytype='EQTY',securitysubtype='COM',sharetype='NS',issuertype='CORP',
                usincflg='Y',conditionaltype='RW',tradingstatusflg='A').items():good &= crsp[n].eq(v)
good &= crsp.mthdelflg.eq('N') & crsp.mthcap.gt(0)
eligible=crsp.loc[good & crsp.yyyymm.le(202511)].copy()
cut=eligible.loc[eligible.primaryexch.eq('N')].groupby('yyyymm').mthcap.quantile(.2)
eligible['nyse20']=eligible.yyyymm.map(cut)
assert eligible.nyse20.notna().all()
eligible=eligible.loc[eligible.mthcap.ge(eligible.nyse20),['permno','yyyymm','mthcap','nyse20']]
del crsp
keys=pd.MultiIndex.from_frame(eligible[['permno','yyyymm']])
blocks=[]
for i in range(pf.num_row_groups):
    f=pf.read_row_group(i,columns=['permno','yyyymm']+names).to_pandas()
    f=f.loc[f.yyyymm.between(200412,202511)]
    f=f.loc[pd.MultiIndex.from_frame(f[['permno','yyyymm']]).isin(keys)].copy()
    f[names]=f[names].replace([np.inf,-np.inf],np.nan).astype(np.float32)
    blocks.append(f)
    if i%10==0:print(json.dumps(dict(stage='read_cz',groups=i+1,total=pf.num_row_groups)),flush=True)
panel=pd.concat(blocks,ignore_index=True)
del blocks
assert not panel.duplicated(['permno','yyyymm']).any()
panel=panel.merge(eligible,on=['permno','yyyymm'],how='inner',validate='one_to_one')
panel['observed_feature_count']=panel[names].notna().sum(axis=1).astype(np.int16)
panel[names]=(panel.groupby('yyyymm')[names].rank(method='average',pct=True)*2-1).fillna(0).astype(np.float32)
panel=panel.rename(columns={'yyyymm':'feature_month','mthcap':'lagged_market_cap'})
month=panel.feature_month%100;year=panel.feature_month//100
panel['return_month']=np.where(month==12,(year+1)*100+1,panel.feature_month+1).astype(np.int32)
panel=panel.merge(outcome,on=['permno','return_month'],how='left',validate='one_to_one')
assert panel.lagged_market_cap.ge(panel.nyse20).all()
assert panel.return_month.le(202512).all()
panel=panel.sort_values(['return_month','permno']).reset_index(drop=True)
panel.to_parquet(OUT/'panel.parquet',index=False)
cut.rename('nyse20').to_csv(OUT/'nyse20-breakpoints.csv')
coverage=panel.groupby('return_month').agg(stocks=('permno','size'),observed_labels=('target_return','count'))
coverage.to_csv(OUT/'coverage.csv')
h=hashlib.sha256()
with open(OUT/'panel.parquet','rb') as f:
    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
manifest=dict(status='data_prepared_not_model_trained',features=names,feature_count=len(names),rows=len(panel),
              return_month_range=[int(panel.return_month.min()),int(panel.return_month.max())],
              training=[200901,201912],validation=[202001,202112],test=[202201,202512],
              input_timing='Each return month uses immediately previous-month CZ, capitalization and NYSE20.',
              normalization='Within-month eligible-universe average-tie percentile ranks mapped to [-1,1]; missing feature=0.',
              stock_selection='Current feature-month eligibility only; missing future returns never remove a test stock.',
              features_selected_on_training_only=True,missing_test_returns=int(panel.loc[panel.return_month.ge(202201),'target_return'].isna().sum()),
              source_cz=str(source),source_cz_receipt=str(source.parent/'download_receipt.json'),
              source_crsp_receipt=str(BASE/'crsp-monthly/manifest.json'),
              limitation='Current-vintage CZ/CRSP data with publication lags; not an as-published historical data vintage.',
              panel_sha256=h.hexdigest(),wall_seconds=time.monotonic()-started)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({k:v for k,v in manifest.items() if k!='features'},indent=2),flush=True)
