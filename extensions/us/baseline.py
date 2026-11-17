"""Declared LightGBM reference on the same panel and target dates."""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from usdata import ROOT,Panel,config,sha,write_json


def samples(panel,split):
    features=[];targets=[];frames=[]
    for t in panel.split(split):
        ids,x,norm,raw=panel.day(t)
        features.append(x[:,-1]);targets.append(norm[:,0])
        frames.append(pd.DataFrame(dict(date=panel.dates[t],realized_date=panel.dates[t+2],symbol=panel.symbols[ids],
            label_return=raw[:,0],label_volume=raw[:,1],label_volatility=raw[:,2])))
    return np.concatenate(features),np.concatenate(targets),pd.concat(frames,ignore_index=True)


def main():
    panel=Panel();p=config()['baseline']
    tx,ty,_=samples(panel,'train');vx,vy,validmeta=samples(panel,'valid');zx,_,testmeta=samples(panel,'test')
    train_mask=np.isfinite(ty);valid_mask=np.isfinite(vy)
    train=lgb.Dataset(tx[train_mask],label=ty[train_mask])
    valid=lgb.Dataset(vx[valid_mask],label=vy[valid_mask],reference=train)
    for seed in p['seeds']:
        out=ROOT/'baseline'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():continue
        params={k:p[k] for k in ['num_leaves','learning_rate','min_data_in_leaf','feature_fraction','bagging_fraction','bagging_freq']}
        params.update(objective='regression',metric='l2',seed=seed,num_threads=2,verbosity=-1,deterministic=True,force_col_wise=True)
        model=lgb.train(params,train,num_boost_round=p['max_rounds'],valid_sets=[valid],
                        callbacks=[lgb.early_stopping(p['early_stopping'],verbose=False)])
        model.save_model(str(out/'model.txt'))
        for split,x,meta in [('valid',vx,validmeta.copy()),('test',zx,testmeta.copy())]:
            meta['pred_return']=model.predict(x,num_threads=2)
            meta['pred_volume']=0.;meta['pred_volatility']=0.
            meta.to_parquet(out/f'{split}_predictions.parquet',index=False)
        write_json(out/'complete.json',dict(seed=seed,params=params,best_iteration=model.best_iteration,
            protocol_sha256=sha(ROOT/'protocol.json'),data_manifest_sha256=sha(ROOT/'data/manifest.json'),
            code_sha256=sha(__file__)))
        print('BASELINE COMPLETE',seed,model.best_iteration,flush=True)


if __name__=='__main__':
    import fcntl
    with (ROOT/'baseline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
