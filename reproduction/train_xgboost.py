"""CSI300 XGBoost baseline using the released notebook's 295 lagged features and parameters."""
import argparse
import json
from pathlib import Path
import time

from common import ROOT, LocalStockData, initialize, daily_correlations, metrics, fresh_output
from evaluate import portfolio_proxy
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType


def build_features(data, destination, target, low, high):
    """The notebook uses daily OHLCV percentage changes lagged 1..59 days (295 columns)."""
    values = data.data.numpy()[:, :5, :].transpose(0, 2, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        changes = values[1:]/values[:-1]-1
    dates = data._dates[data.max_backtrack_days:-data.max_future_days]
    rows, columns = np.where((dates >= pd.Timestamp(low))[:, None] &
                             (dates <= pd.Timestamp(high))[:, None] &
                             np.isfinite(values[data.max_backtrack_days:-data.max_future_days, :, 1]))
    # One observation per present member, preserving the source's absent-input filling convention.
    features = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float32, shape=(len(rows), 295))
    for lag in range(1,60):
        features[:, (lag-1)*5:lag*5] = changes[data.max_backtrack_days+rows-lag-1, columns]
    features.flush()
    return features, target[rows, columns].copy(), rows, columns, dates


def normalize_features(features, mean=None, std=None):
    if mean is None:
        sums = np.zeros(features.shape[1], dtype=np.float64)
        squares = np.zeros_like(sums)
        counts = np.zeros_like(sums)
        positive_inf = np.zeros_like(sums,dtype=bool)
        negative_inf = np.zeros_like(sums,dtype=bool)
        for offset in range(0,len(features),10000):
            chunk = np.asarray(features[offset:offset+10000], dtype=np.float64)
            finite = np.isfinite(chunk)
            clean = np.where(finite,chunk,0)
            counts += finite.sum(0)
            sums += clean.sum(0)
            squares += (clean*clean).sum(0)
            positive_inf |= np.isposinf(chunk).any(0)
            negative_inf |= np.isneginf(chunk).any(0)
        mean = np.divide(sums, counts, out=np.full_like(sums,np.nan), where=counts>0)
        variance = np.divide(squares-sums*mean, counts-1, out=np.full_like(sums,np.nan), where=counts>1)
        std = np.sqrt(np.maximum(variance,0.))
        # Match pandas mean/std skip-NaN semantics exactly, including infinities:
        # an infinite training input poisons that column's std, so fillna later
        # yields zero for the whole column. Do not silently treat inf as missing.
        mean[positive_inf & ~negative_inf] = np.inf
        mean[negative_inf & ~positive_inf] = -np.inf
        mean[positive_inf & negative_inf] = np.nan
        std[positive_inf | negative_inf] = np.nan
    for offset in range(0,len(features),10000):
        with np.errstate(divide='ignore',invalid='ignore'):
            chunk = (np.asarray(features[offset:offset+10000],dtype=np.float64)-mean)/std
        features[offset:offset+10000] = np.clip(np.nan_to_num(chunk,nan=0,posinf=4,neginf=-4),-4,4)
    features.flush()
    return mean, std


def normalize_train_labels(labels, rows):
    result = np.asarray(labels, dtype=np.float64).copy()
    result[~np.isfinite(result)] = np.nan
    for day in np.unique(rows):
        mask = (rows == day) & np.isfinite(result)
        section = result[mask]
        std = section.std(ddof=1) if len(section) > 1 else 0.
        result[mask] = (section-section.mean())/std if std>1e-12 else 0.
    return np.clip(result,-4,4).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--rounds',type=int,default=100)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--train-end', choices=['2020-11-30', '2020-12-31'], default='2020-11-30')
    parser.add_argument('--legacy-return-proxy', action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    out = fresh_output(args.output)
    torch.set_num_threads(args.threads)
    initialize()
    started=time.monotonic()
    data=LocalStockData('csi300','2010-01-01','2024-12-31')
    target=data.forward_returns(20).numpy()
    (out/'data-provenance.json').write_text(json.dumps(data.provenance, indent=2))
    ranges=dict(train=('2010-01-01',args.train_end), validation=('2021-01-01','2021-12-31'), test=('2022-01-01','2024-12-31'))
    arrays={}
    for label,(low,high) in ranges.items():
        arrays[label]=build_features(data,out/f'{label}_features.npy',target,low,high)
        print(json.dumps(dict(event='features_prepared',split=label,rows=len(arrays[label][0]),columns=295)),flush=True)
    mean,std=normalize_features(arrays['train'][0])
    for label in ['validation','test']:
        normalize_features(arrays[label][0],mean,std)
    np.savez(out/'normalization.npz',mean=mean,std=std)
    train_y=normalize_train_labels(arrays['train'][1],arrays['train'][2])
    valid_y=np.clip(arrays['validation'][1],-4,4)
    params=dict(objective='reg:squarederror',eval_metric='rmse',colsample_bytree=0.8879,
                learning_rate=0.01,subsample=0.8789,reg_lambda=205.6999,reg_alpha=580.9768,
                max_depth=4,seed=args.seed,nthread=args.threads,tree_method='hist',device='cpu')
    protocol=dict(method='XGBoost, upstream ML notebook features and parameters',params=params,
                  rounds=args.rounds,seed=args.seed,features='OHLCV one-day returns at lags 1..59 (295 columns); no VWAP',
                  preprocessing='Training-only feature mean/std, fill missing with zero, clip [-4,4]; training labels standardized by day',
                  nonfinite_preprocessing='Pandas-compatible: training infinities poison column std; zero variance is retained; normalization NaNs fill to zero and infinities clip to bounds',
                  label=20,ranges=ranges,checkpoint_selection='fixed round budget, validation diagnostics only',
                  source_bug='Notebook puts num_boost_round=100 inside params but does not pass it to xgb.train, which defaults to 10. Both first-10 and intended-100 predictions are saved.',
                  missing_outcomes='Excluded from training/evaluation loss, never filled with zero; predictions still emitted for all current members',
                  xgboost_version=xgb.__version__)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    train_good, valid_good = np.isfinite(train_y), np.isfinite(valid_y)
    dtrain=xgb.DMatrix(arrays['train'][0][train_good],label=train_y[train_good],nthread=args.threads)
    dvalid=xgb.DMatrix(arrays['validation'][0][valid_good],label=valid_y[valid_good],nthread=args.threads)
    history={}
    model=xgb.train(params,dtrain,num_boost_round=args.rounds,evals=[(dtrain,'train'),(dvalid,'validation')],evals_result=history,verbose_eval=10)
    model.save_model(out/'model.ubj')
    if args.rounds>=10:
        model[:10].save_model(out/'literal_notebook_10_rounds.ubj')
    results={}
    variants=['intended_100_rounds','literal_notebook_10_rounds'] if args.rounds>=10 else ['reduced_rounds']
    for variant in variants:
        num_rounds=10 if variant=='literal_notebook_10_rounds' else args.rounds
        result={}
        for split in ['validation','test']:
            features,labels,rows,columns,dates=arrays[split]
            block=xgb.DMatrix(features,nthread=args.threads)
            values=model.predict(block,iteration_range=(0,num_rounds))
            mask=(dates>=pd.Timestamp(ranges[split][0])) & (dates<=pd.Timestamp(ranges[split][1]))
            split_dates=dates[mask]
            prediction=np.full((len(split_dates),data.n_stocks),np.nan,dtype=np.float32)
            offsets=rows-np.flatnonzero(mask)[0]
            prediction[offsets,columns]=values
            truth=target[mask]
            result[split]=metrics(prediction,truth)
            proxy,returns=(portfolio_proxy(prediction,truth) if args.legacy_return_proxy else (None,np.array([])))
            result[split+'_return_proxy']=proxy
            np.savez_compressed(out/f'{variant}_{split}_predictions.npz',predictions=prediction,targets=truth,
                                dates=split_dates.to_numpy(),stocks=np.asarray(data._stock_ids,dtype=str),proxy_returns=returns)
        results[variant]=result
    result=dict(status='complete',rounds=args.rounds,variants=results,learning_history=history,
                samples={split:len(arrays[split][0]) for split in arrays},feature_count=295,
                wall_seconds=time.monotonic()-started,
                limitations=['Community data and one CSI300 seed',
                             'CPU histogram tree method specified for reproducibility',
                             'The released notebook uses standardized training labels but raw validation labels; retained here',
                             'No executable backtest; optional legacy return proxies are diagnostic only'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
