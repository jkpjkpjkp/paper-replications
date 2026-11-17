#!/usr/bin/env python3
"""Original three classical baselines, with declared per-window defaults.

Uses an existing pinned statsmodels environment read-only. No return inference.
ARIMA parameters are fit separately per channel/origin, then reused across H.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import warnings
from collections import Counter

import numpy as np
import statsmodels
from statsmodels.tsa.arima.model import ARIMA

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/tfb_classical'


def atomic(path,obj):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(dataset,layout):
    dest=OUT/layout/dataset;dest.mkdir(parents=True,exist_ok=True)
    source=ROOT/'data'/f'{dataset}.{layout}.csv'
    with source.open() as f:
        reader=csv.reader(f);names=next(reader);a=np.asarray(list(reader),dtype=np.float64)
    raw=a[:,1:];n=len(raw);train_end=int(n*.7);test_start=n-int(n*.2)
    mean=raw[:train_end].mean(0);scale=raw[:train_end].std(0);z=(raw-mean)/scale
    origins=np.arange(test_start,n-24+1);horizons=[24,36,48,60]
    signature={'source_sha256':sha(source),'frozen_config_sha256':sha(ROOT/'frozen_tfb_config.json'),'origins':origins.tolist()}
    result_path=dest/'metrics.json'
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['signature']!=signature:raise RuntimeError('Existing completed classical run has changed inputs; preserve')
        if not (dest/'predictions.npz').exists():raise RuntimeError('Missing completed-run prediction artifact')
        return old
    cache_path=dest/'arima-cache.npz';completed=0;counts=Counter();nonconverged=0
    arima=np.zeros((len(origins),60,5),dtype=np.float64)
    if cache_path.exists():
        c=np.load(cache_path,allow_pickle=False);arima=c['prediction'];completed=int(c['completed'])
        meta=json.loads((dest/'arima-cache.json').read_text())
        if meta['signature']!=signature:raise RuntimeError('Existing ARIMA partial cache inputs differ; preserve')
        counts.update(meta['warnings']);nonconverged=meta['nonconverged_fits']
    started=time.time()
    for i in range(completed,len(origins)):
        t=origins[i]
        for channel in range(5):
            with warnings.catch_warnings(record=True) as warning_records:
                warnings.simplefilter('always')
                model=ARIMA(z[t-96:t,channel],order=(1,1,1),trend='n',enforce_stationarity=True,enforce_invertibility=True)
                fitted=model.fit(method='statespace',method_kwargs={'maxiter':50,'disp':0})
                arima[i,:,channel]=fitted.forecast(60)
            nonconverged+=not fitted.mle_retvals.get('converged',True)
            counts.update(type(w.message).__name__+': '+str(w.message) for w in warning_records)
        if not np.isfinite(arima[i]).all():raise RuntimeError('Nonfinite ARIMA forecast, no substitution')
        if (i+1)%25==0 or i+1==len(origins):
            temp=dest/'arima-cache.tmp.npz'
            np.savez_compressed(temp,prediction=arima,completed=i+1)
            os.replace(temp,cache_path)
            atomic(dest/'arima-cache.json',{'signature':signature,'completed_origins':i+1,'nonconverged_fits':int(nonconverged),'warnings':dict(counts)})
            print(json.dumps({'ARIMA':dataset,'layout':layout,'completed_origins':i+1,'total_origins':len(origins),'nonconverged_fits':int(nonconverged)}),flush=True)
    persistence=np.repeat(z[origins-1,None,:],60,axis=1)
    drift=(z[origins-1]-z[origins-96])/95
    random_walk=persistence+np.arange(1,61)[None,:,None]*drift[:,None,:]
    forecasts={'Persistence':persistence,'RW_drift':random_walk,'ARIMA111':arima}
    metrics=[]
    arrays={'origins':origins,'ordinal':a[:,0],'mean':mean,'scale':scale}
    for H in horizons:
        keep=origins+H<=n
        truth=np.stack([z[t:t+H] for t in origins[keep]])
        arrays[f'target_H{H}']=truth;arrays[f'origins_H{H}']=origins[keep]
        for method,prediction in forecasts.items():
            p=prediction[keep,:H];error=p-truth
            metrics.append({'method':method,'horizon':H,'mse':float(np.mean(error**2)),'mae':float(np.mean(np.abs(error))),
                            'origins':len(truth),'forecast_steps':int(truth.size)})
            arrays[f'{method}_H{H}']=p
    # Independent persistence accounting: squared difference from last observed
    # z-score must equal error from raw values divided by training-only scales.
    H=24;t=origins[0];direct=(raw[t:t+H]-raw[t-1])/scale
    if not np.allclose(direct,z[t:t+H]-z[t-1],rtol=1e-12,atol=1e-12):raise RuntimeError('Global scaling metric reconciliation failed')
    np.savez_compressed(dest/'predictions.npz',**arrays)
    result={'dataset':dataset,'layout':layout,'signature':signature,'metrics':metrics,
            'reproduction_level':'partial_methodological_fixed_lookback96','parameters':{'lookback':96,'RW_drift':'per-window mean of96-point first differences','ARIMA_order':[1,1,1],'ARIMA_trend':'none','ARIMA_fitting':'statsmodels state-space MLE, maxiter50; no validation/test selection'},
            'split':{'train_end':train_end,'test_start':test_start,'samples':n},'arima_fits':len(origins)*5,
            'nonconverged_fits':int(nonconverged),'warning_counts':dict(counts),'elapsed_seconds':time.time()-started,
            'predictions_path':str(dest/'predictions.npz'),'genuine_calendar_dates':False,'trading_returns_inferred':False,
            'validation':{'labels_wholly_inside_test':True,'all_origins_retained':True,'train_only_scaler':True,'persistence_raw_standardized_metric_reconciliation':True,'ARIMA_uses_only_prior96_values':True}}
    atomic(result_path,result)
    return result


def main():
    config=json.loads((ROOT/'frozen_tfb_config.json').read_text());OUT.mkdir(parents=True,exist_ok=True)
    results=[]
    for layout in config['layouts']:
        for dataset in config['datasets']:
            results.append(run(dataset,layout));atomic(OUT/'progress.json',{'completed_dataset_layouts':len(results),'results':results})
    atomic(OUT/'summary.json',{'status':'complete_partial_methodological_scope','original_paper_reproduction_complete':False,
                            'results':results,'statsmodels_version':statsmodels.__version__,'numpy_version':np.__version__})


if __name__=='__main__':main()
