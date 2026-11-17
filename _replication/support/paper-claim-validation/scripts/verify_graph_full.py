#!/usr/bin/env python3
"""Verify every full-search CS-GAT ensemble, preserving the paper comparison."""
import json
import hashlib
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'financial-graph-reproduction'
OUT=ROOT/'paper-claim-validation/results/graph_full'
sys.path.insert(0,str(PROJECT/'scripts'))
from models import GraphModel
from run_experiment import split_indices


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=OUT)
    parser.add_argument('--released-rule',action='store_true')
    args=parser.parse_args();OUT=args.input.resolve()
    torch.set_num_threads(2)
    config=json.loads((ROOT/'paper-claim-validation/evidence/graph_full_config.json').read_text())
    for source in config['sources']:
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],source['path']
    panel=np.load(PROJECT/'data/panel.npz')
    graphs=np.load(PROJECT/'data/graphs.npz')
    dates=graphs['dates']
    x=torch.from_numpy(panel['X'][35:])
    y=panel['y'][35:]
    a=torch.from_numpy((graphs['CS']!=0).astype('float32'))
    a[:,np.arange(x.shape[1]),np.arange(x.shape[1])]=1
    checked=[]
    all_y=[];all_p=[]
    for year in range(1999,2023):
        path=OUT/f'CS_GAT_{year}.parquet'
        df=pd.read_parquet(path)
        meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['profile']=='full' and meta['ensemble']==10 and meta['candidates']==12 and meta['max_epochs']==500
        if args.released_rule:
            source=ROOT/'paper-claim-validation/results/graph_full'/f'CS_GAT_{year}.json'
            previous=json.loads(source.read_text())
            assert meta['source_search_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
            assert meta['retraining_rule']=='released_stopping_epoch'
            assert meta['selected_epochs']==min(500,previous['selected_epochs']+5)
        tr,va,te=split_indices(dates,year)
        assert meta['train_last']==int(dates[tr[-1]]) and meta['val_last']==int(dates[va[-1]])
        assert not df.duplicated(['yyyymm','permno']).any()
        np.testing.assert_array_equal(df.yyyymm,np.repeat(dates[te],len(panel['permnos'])))
        np.testing.assert_array_equal(df.permno,np.tile(panel['permnos'],len(te)))
        np.testing.assert_array_equal(df.target,y[te].ravel())
        checkpoint=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=True)
        states=checkpoint['ensemble_state_dicts']
        assert len(states)==10
        cfg=meta['parameters']
        model=GraphModel(units=cfg['units'],heads=cfg['heads'],architecture='GAT').eval()
        p=[]
        for state in states:
            model.load_state_dict(state)
            with torch.no_grad():p.append(model(x[te],a[te]).numpy())
        mean=np.mean(p,axis=0).ravel()
        np.testing.assert_allclose(mean,df.prediction,atol=2e-6,rtol=2e-5)
        all_y.extend(df.target.to_numpy());all_p.extend(df.prediction.to_numpy())
        checked.append(dict(year=year,ensemble_members_replayed=10,rows=len(df),selected_epochs=meta['selected_epochs'],parameters=cfg))
    target=np.array(all_y,dtype=float);prediction=np.array(all_p,dtype=float)
    r2=100*(1-np.sum((target-prediction)**2)/np.sum(target**2))
    reduced=pd.read_csv(PROJECT/'results/reconstruction/metrics.csv').set_index('model').loc['CS_GAT','R2_zero_pct']
    record=dict(passed=True,model='CS_GAT',checked_years=checked,observations=len(target),
                full_R2_zero_pct=r2,reduced_R2_zero_pct=float(reduced),paper_R2_zero_pct=1.1592,
                original_data_reproduced=False,released_stopping_epoch_rule=args.released_rule,
                limitation=('164-vs-165 current-vintage reconstructed panel; full search/seed protocol reconstructed' if args.released_rule
                            else '164-vs-165 current-vintage reconstructed panel; best-epoch vs released stopping-epoch retrain'))
    (OUT/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
