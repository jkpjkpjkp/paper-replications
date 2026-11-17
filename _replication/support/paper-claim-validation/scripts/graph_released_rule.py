#!/usr/bin/env python3
"""Reuse full validation searches; retrain with the released stop-epoch rule.

All data, graph equations, search choices and ensemble seeds are reused. Only
the best-epoch versus stopping-epoch retraining difference is isolated here.
This closes a code-protocol discrepancy, not the unavailable 165-firm snapshot.
"""
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'paper-claim-validation'
PROJECT=ROOT/'financial-graph-reproduction'
SOURCE=BASE/'results/graph_full'
OUT=BASE/'results/graph_released_rule'
sys.path.insert(0,str(PROJECT/'scripts'))
from models import GraphModel
from run_experiment import neural_fit,split_indices


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def status(phase,**kw):
    value=dict(status=phase,updated_utc=datetime.now(timezone.utc).isoformat(),
               completed_years=len(list(OUT.glob('CS_GAT_*.json'))),required_years=24,**kw)
    save(OUT/'STATUS.json',value);print(json.dumps(value),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'run.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen=json.loads((BASE/'evidence/graph_full_config.json').read_text())
    for source in frozen['sources']:assert sha(source['path'])==source['sha256']
    provenance=dict(purpose='Released stopping-epoch retraining, with prior full validation search reused',
        source_grid=str(SOURCE),scope='CS_GAT, 1999-2022, ten members',
        epoch_rule='min(500,best_validation_epoch+5); patience=5 and strict improvement',
        original_snapshot_reproduced=False,
        files_sha256={str(p):sha(p) for p in [Path(__file__),PROJECT/'sources/GAT_Model_.py']})
    save(BASE/'evidence/graph_released_rule_config.json',provenance)
    panel=np.load(PROJECT/'data/panel.npz');g=np.load(PROJECT/'data/graphs.npz')
    dates=g['dates'];x=torch.from_numpy(panel['X'][35:]);y=torch.from_numpy(panel['y'][35:])
    a=torch.from_numpy((g['CS']!=0).astype('float32'))
    a[:,np.arange(x.shape[1]),np.arange(x.shape[1])]=1
    for year in range(1999,2023):
        source=SOURCE/f'CS_GAT_{year}.json'
        while not source.exists():
            live=subprocess.run(['systemctl','--user','is-active','paper-claim-graph-full.service'],capture_output=True,text=True).stdout.strip()
            if live!='active':raise RuntimeError(f'Missing source search for {year}; full-search service is {live}')
            status('waiting_for_live_source_search',year=year);time.sleep(20)
        original=json.loads(source.read_text())
        assert original['profile']=='full' and original['candidates']==12 and original['ensemble']==10
        cfg=original['parameters'];epoch=min(original['max_epochs'],original['selected_epochs']+5)
        output=OUT/f'CS_GAT_{year}.parquet';meta_path=output.with_suffix('.json')
        if output.exists() and meta_path.exists() and output.with_suffix('.pt').exists():
            meta=json.loads(meta_path.read_text())
            assert meta['source_search_sha256']==sha(source) and meta['selected_epochs']==epoch
            continue
        start=time.monotonic();tr,va,te=split_indices(dates,year);combined=np.r_[tr,va]
        members=[];forecasts=[]
        member_dir=OUT/'members'/str(year);member_dir.mkdir(parents=True,exist_ok=True)
        for run in range(10):
            path=member_dir/f'member{run}.pt'
            if path.exists():
                record=torch.load(path,map_location='cpu',weights_only=True)
                assert record['epochs']==epoch and record['source_search_sha256']==sha(source)
                state=record['state_dict'];prediction=record['forecast'].numpy()
            else:
                factory=lambda:GraphModel(units=cfg['units'],heads=cfg['heads'],architecture='GAT')
                model,_,_=neural_fit(factory,x,y,a,combined,va,cfg,original['seed']+run*10000,
                    original['max_epochs'],5,retrain_epochs=epoch)
                model.eval()
                with torch.no_grad():prediction=model(x[te],a[te]).numpy()
                state=copy.deepcopy(model.state_dict())
                record=dict(state_dict=state,forecast=torch.from_numpy(prediction),epochs=epoch,
                            source_search_sha256=sha(source))
                temp=path.with_suffix('.pt.tmp');torch.save(record,temp);temp.replace(path)
            members.append(state);forecasts.append(prediction)
            status('training_released_rule',year=year,member=run+1,retrain_epochs=epoch)
        pred=np.mean(forecasts,axis=0)
        frame=pd.DataFrame(dict(yyyymm=np.repeat(dates[te],len(panel['permnos'])),
            permno=np.tile(panel['permnos'],len(te)),target=y[te].numpy().ravel(),prediction=pred.ravel(),model='CS_GAT'))
        meta=dict(original,retraining_rule='released_stopping_epoch',best_validation_epoch=original['selected_epochs'],
                  selected_epochs=epoch,source_search_sha256=sha(source),elapsed_seconds=time.monotonic()-start)
        checkpoint=dict(state_dict=members[-1],ensemble_state_dicts=members,metadata=meta)
        temp=output.with_suffix('.pt.tmp');torch.save(checkpoint,temp);temp.replace(output.with_suffix('.pt'))
        temp=output.with_suffix('.parquet.tmp');frame.to_parquet(temp,index=False);temp.replace(output)
        save(meta_path,meta);status('year_complete',year=year)
    subprocess.run([sys.executable,str(PROJECT/'scripts/summarize.py'),'--input',str(OUT)],check=True)
    subprocess.run([sys.executable,str(BASE/'scripts/verify_graph_full.py'),'--input',str(OUT),'--released-rule'],check=True)
    status('complete');subprocess.run([sys.executable,str(BASE/'scripts/compare_claims.py')],check=True)


if __name__=='__main__':main()
