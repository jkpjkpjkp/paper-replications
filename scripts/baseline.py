"""A declared LightGBM comparator on identical samples and normalized targets."""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from scripts.train import corr
from scripts.build_data import write_json,sha
ROOT=Path(__file__).resolve().parents[1]


def samples(panel,split):
    x,y,metadata=[],[],[]
    for t in panel.split(split):
        ids,window,norm,raw=panel.day(t)
        x.append(window[:,-1]);y.append(norm[:,0])
        metadata.append(pd.DataFrame(dict(date=panel.dates[t],realized_date=panel.dates[t+2],
            symbol=panel.symbols[ids],label_return=raw[:,0],label_volume=raw[:,1],label_volatility=raw[:,2])))
    return np.concatenate(x),np.concatenate(y),pd.concat(metadata,ignore_index=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--universe',default='csi300',choices=['csi300','csi500'])
    a=p.parse_args();panel=Panel(ROOT/'data'/a.universe)
    tx,ty,_=samples(panel,'train');vx,vy,meta=samples(panel,'valid')
    train_mask=np.isfinite(ty);valid_mask=np.isfinite(vy)
    train=lgb.Dataset(tx[train_mask],label=ty[train_mask])
    valid=lgb.Dataset(vx[valid_mask],label=vy[valid_mask],reference=train)
    testx,_,testmeta=samples(panel,'test')
    for seed in range(3):
        out=ROOT/'results/baseline'/a.universe/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        params=dict(objective='regression',metric='l2',learning_rate=.05,num_leaves=31,max_depth=-1,
            min_data_in_leaf=100,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,seed=seed,
            num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True)
        model=lgb.train(params,train,num_boost_round=1000,valid_sets=[valid],
            callbacks=[lgb.early_stopping(50,verbose=False)])
        model.save_model(str(out/'model.txt'));metrics={}
        for split,features,frame in [('valid',vx,meta.copy()),('test',testx,testmeta.copy())]:
            frame['pred_return']=model.predict(features,num_threads=2)
            frame['pred_volume']=0.;frame['pred_volatility']=0.
            ic=frame.groupby('date').apply(lambda f:corr(f.pred_return,f.label_return),include_groups=False)
            ric=frame.groupby('date').apply(lambda f:f.pred_return.corr(f.label_return,method='spearman'),include_groups=False)
            metrics[split]=dict(return_metrics=dict(IC=float(ic.mean()),ICIR=float(ic.mean()/ic.std(ddof=1)),
                RankIC=float(ric.mean()),RankICIR=float(ric.mean()/ric.std(ddof=1)),days=len(ic)))
            frame.to_parquet(out/f'{split}_predictions.parquet',index=False)
        write_json(out/'metrics.json',metrics)
        write_json(out/'config.json',dict(params=params,max_rounds=1000,early_stopping=50,
            selection='validation MSE',best_iteration=model.best_iteration,lookback='last row of Alpha158 window',
            data_manifest_sha256=sha(panel.folder/'manifest.json'),paper_hyperparameters='not supplied; independent comparator'))
        print(json.dumps(dict(universe=a.universe,seed=seed,best_iteration=model.best_iteration,test=metrics['test'])),flush=True)


if __name__=='__main__':main()
