"""Independently reconstruct the saved paper inputs; do not modify source data."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,pickle,re,time
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/gap-resolution-20261113';OUT.mkdir(exist_ok=True)
start=time.monotonic()
with (ROOT/'results/doubleadapt-original/handler.pkl').open('rb') as f:h=pickle.load(f)
raw=h._data; learn=h._learn; inf=h._infer
cal=pd.DatetimeIndex(pd.to_datetime((ROOT/'data/crowd_data/calendars/day.txt').read_text().splitlines()))
def series(stock,field):
 b=np.fromfile(ROOT/'data/crowd_data/features'/stock.lower()/(field+'.day.bin'),dtype='<f4');x=np.full(len(cal),np.nan,dtype=np.float32);i=int(b[0]);x[i:i+len(b)-1]=b[1:];return x
mismatch={};maxerr={};label_bad=0;nextday_diff=0;nextday_pairs=0;groups=raw.groupby(level='instrument').indices
for number,(stock,rows) in enumerate(groups.items()):
 dates=raw.index.get_level_values('datetime')[rows];idx=cal.get_indexer(dates);assert (idx>=59).all()
 close=series(stock,'close');volume=series(stock,'volume')
 for family in ['CLOSE','OPEN','HIGH','LOW','VWAP','VOLUME']:
  data=series(stock,family.lower());den=volume if family=='VOLUME' else close
  if family=='VOLUME':den=den+np.float32(1e-12)
  for lag in range(60):
   name=family+str(lag);expected=data[idx-lag]/den[idx];actual=raw[('feature',name)].to_numpy()[rows]
   good=np.isclose(actual,expected,rtol=2e-6,atol=2e-7,equal_nan=True)
   mismatch[name]=mismatch.get(name,0)+int((~good).sum())
   finite=np.isfinite(actual)&np.isfinite(expected)
   if finite.any():maxerr[name]=max(maxerr.get(name,0),float(np.max(np.abs(actual[finite]-expected[finite]))))
 expected=close[idx+2]/close[idx+1]-1;actual=raw['label'].iloc[:,0].to_numpy()[rows]
 label_bad+=int((~np.isclose(actual,expected,rtol=2e-6,atol=2e-7,equal_nan=True)).sum())
 literal=close[idx+1]/close[idx]-1;finite=np.isfinite(actual)&np.isfinite(literal);nextday_pairs+=int(finite.sum());nextday_diff+=int((np.abs(actual[finite]-literal[finite])>1e-6).sum())
 if (number+1)%100==0:print('price reconstruction',number+1,len(groups),flush=True)
assert sum(mismatch.values())==0,(sum(mismatch.values()),sorted(mismatch.items(),key=lambda v:-v[1])[:5])
assert label_bad==0
norm=h.infer_processors[0];train=(raw.index.get_level_values('datetime')>='2008-01-01')&(raw.index.get_level_values('datetime')<='2014-12-31')
normalization_bad={};median_error=0;scale_error=0
for j,name in enumerate(raw['feature'].columns):
 a=raw[('feature',name)].to_numpy();f=a[train];mu=np.nanmedian(f);sd=(np.nanmedian(np.abs(f-mu))+1e-12)*1.4826
 median_error=max(median_error,float(abs(mu-norm.mean_train[j])));scale_error=max(scale_error,float(abs(sd-norm.std_train[j])))
 # Match the source's float32 stored scale rounding, independently derived above.
 sd=np.float32(sd);expected=np.clip((a-mu)/sd,-3,3);expected=np.where(np.isnan(expected),0,expected)
 actual=inf[('feature',name)].to_numpy();normalization_bad[name]=int((~np.isclose(actual,expected,rtol=2e-5,atol=2e-6)).sum())
 if (j+1)%60==0:print('normalization',j+1,flush=True)
assert sum(normalization_bad.values())==0
labels=raw['label'].iloc[:,0].dropna();expected=(labels-labels.groupby(level='datetime').transform('mean'))/(labels.groupby(level='datetime').transform('std')+1e-12)
actual=learn['label'].iloc[:,0];assert expected.index.equals(actual.index)
label_norm_error=float(np.max(np.abs(expected.to_numpy()-actual.to_numpy())))
assert label_norm_error<1e-4,label_norm_error
zero=(learn['feature']==0).all(axis=1);assert zero.sum()==5208
zero_idx=zero[zero].index
# Retain masks, not a rewritten handler. Any cleaned training variant must retain
# the original full forecast/evaluation universe so metrics stay comparable.
zero_idx.to_frame(index=False).to_parquet(OUT/'unavailable-input-keys.parquet',index=False)
result=dict(passed=True,raw_rows=len(raw),learning_rows=len(learn),features=360,calendar_lags_reconstructed=60,feature_cells_verified=len(raw)*360,feature_mismatches=sum(mismatch.values()),label_mismatches=label_bad,normalization_mismatches=sum(normalization_bad.values()),normalization_median_max_error=median_error,normalization_scale_max_error=scale_error,label_normalization_max_error=label_norm_error,unavailable_labeled_inputs=int(zero.sum()),training_unavailable=int((zero&(learn.index.get_level_values('datetime')<='2014-12-31')).sum()),source_contract={'features':'Alpha360, 6 fields x 60 calendar sessions; latest-close/latest-volume ratios','scaler':'Training-only median/MAD * 1.4826, clip [-3,3], fill feature NaN with zero','label':'Released executable: close[t+2]/close[t+1]-1; daily z-score after DropnaLabel','paper_definition_note':'Paper mathematical Definition 1 describes close[t+1]/close[t]-1. Release deliberately includes a one-day trading lag and corresponding task truncation. Do not silently switch label timing or claim either convention generated Table 1 without further evidence.'},alternative_literal_label_finite_pairs=nextday_pairs,alternative_literal_label_differs=nextday_diff,scope='Exact executable preprocessing reconstruction; not certification of historical vendor vintage',seconds=time.monotonic()-start)
(OUT/'doubleadapt-data-contract.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
