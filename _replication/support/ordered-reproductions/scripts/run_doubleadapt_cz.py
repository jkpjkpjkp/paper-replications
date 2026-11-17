"""Monthly CZ adaptation of author DoubleAdapt; requires a reviewed original gate.

--preflight checks input timing and query-label causality using synthetic data only.
It does not train on the CZ panel. The original experiment remains separate.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import copy
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from qlib.contrib.meta.incremental.model import DoubleAdaptManager

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/doubleadapt-cz'
WINDOW=12


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str));tmp.replace(OUT/name)


def month_code(period):return period.year*100+period.month


class MonthlyData:
    def __init__(self):
        self.manifest=json.loads((ROOT/'data/cz-monthly/manifest.json').read_text())
        self.names=self.manifest['features'];assert len(self.names)==202
        self.panel=pd.read_parquet(ROOT/'data/cz-monthly/panel.parquet').set_index(['return_month','permno']).sort_index()
        self.features=self.panel[self.names]
        self.cache={}

    def month(self,period):
        code=month_code(period)
        if code in self.cache:return self.cache[code]
        current=self.panel.xs(code,level='return_month')
        instruments=current.index.to_numpy()
        periods=[period-i for i in range(WINDOW-1,-1,-1)]
        # A calendar grid prevents history gaps from being compressed into adjacent months.
        index=pd.MultiIndex.from_product([[month_code(p) for p in periods],instruments],names=['return_month','permno'])
        x=self.features.reindex(index).fillna(0).to_numpy(dtype=np.float32)
        x=x.reshape(WINDOW,len(instruments),202).transpose(1,0,2).copy()
        raw=current.target_return.to_numpy(dtype=np.float32)
        if not np.isfinite(raw).all():raise ValueError(f'Missing training/validation return: {code}')
        y=(raw-raw.mean())/(raw.std(ddof=0)+1e-12)
        idx=pd.MultiIndex.from_arrays([[period.to_timestamp('M')]*len(instruments),instruments],names=['datetime','instrument'])
        result=torch.from_numpy(x),torch.from_numpy(y),idx
        # Keep at most two monthly tensors; avoid materializing all sequence windows.
        if len(self.cache)>=2:self.cache.pop(next(iter(self.cache)))
        self.cache[code]=result
        return result


class Task:
    def __init__(self,data,query):self.data,self.query=data,query
    def get_meta_input(self):
        x,y,index=self.data.month(self.query-1)
        q,z,test_index=self.data.month(self.query)
        # Support outcome is realized by formation; query outcome is used only AFTER prediction.
        assert (self.query-1).end_time<self.query.start_time
        return dict(X_train=x,y_train=y,train_idx=index,X_test=q,y_test=z,test_idx=test_index,meta_end=len(z))


class TaskSet:
    def __init__(self,data,online=False):self.data,self.online=data,online
    def prepare_tasks(self,phases):
        if isinstance(phases,list):return [self.prepare_tasks(p) for p in phases]
        start,end=('2020-01','2025-12') if self.online else (('2009-01','2019-12') if phases=='train' else ('2020-01','2021-12'))
        return [Task(self.data,p) for p in pd.period_range(start,end,freq='M')]


def manager(device='cpu'):
    config=dict(model=dict(**{'class':'GRU','module_path':'qlib.contrib.model.pytorch_gru'},
        kwargs=dict(d_feat=202,hidden_size=64,num_layers=2,dropout=0,GPU=0 if device=='cuda' else -1,seed=43)))
    return DoubleAdaptManager(task_config=config,lr_model=.001,lr_da=.01,
        online_lr=dict(lr_model=.001,lr_da=.0001,lr_ma=.001),factor_num=202,x_dim=WINDOW*202,
        alpha=360,num_head=8,temperature=10,begin_valid_epoch=10,first_order=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true')
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=ap.parse_args()
    if args.device=='cuda':
        assert torch.cuda.is_available(),'CUDA requested but unavailable'
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if not args.preflight:
        gate=ROOT/'results/doubleadapt-original/gate.json'
        if not gate.exists():raise RuntimeError('Original-paper fidelity review pending: gate.json is absent')
        approval=json.loads(gate.read_text())
        assert approval['status']=='passed'
        assert approval['summary_sha256']==sha(ROOT/'results/doubleadapt-original/summary.json')
        assert approval['protocol_sha256']==sha(ROOT/'results/doubleadapt-original/protocol.json')
        if (OUT/'scores.csv').exists():raise RuntimeError('Existing CZ scores are preserved')
    torch.set_num_threads(2);torch.manual_seed(43);np.random.seed(0);random.seed(0)
    data=MonthlyData();model=manager(args.device)
    if args.preflight:
        task=Task(data,pd.Period('2009-01',freq='M')).get_meta_input()
        assert task['X_test'].shape[1:]==(12,202)
        assert task['train_idx'].get_level_values('datetime').max()<task['test_idx'].get_level_values('datetime').min()
        # Perturb unknown query labels on a tiny synthetic task; current forecast must be unchanged.
        # This exercises native higher/DoubleAdapt without fitting any real observations.
        a=dict(X_train=torch.randn(8,12,202),y_train=torch.randn(8),
               X_test=torch.randn(8,12,202),y_test=torch.randn(8),meta_end=8)
        b={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in a.items()}
        b['y_test']=b['y_test']+2
        before=copy.deepcopy(model.state_dict())
        pa=model.run_task(a,'online');after_a=copy.deepcopy(model.state_dict())
        model.load_state_dict(before);pb=model.run_task(b,'online')
        assert np.allclose(pa,pb,atol=1e-7),'Current query labels affected current prediction'
        assert any(not torch.equal(v,model.framework.state_dict()[k]) for k,v in after_a['framework'].items()),'Query feedback failed to update future state'
        model.load_state_dict(before)
        # Capacity check uses full monthly shapes but synthetic observations and labels.
        full=dict(X_train=torch.randn_like(task['X_train']),y_train=torch.randn_like(task['y_train']),
                  X_test=torch.randn_like(task['X_test']),y_test=torch.randn_like(task['y_test']),meta_end=len(task['y_test']))
        trial=model.run_task(full,'train')
        assert np.isfinite(trial).all()
        save('preflight.json',dict(passed=True,device=args.device,torch_version=torch.__version__,input_shape=list(task['X_test'].shape),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
            support_month=200812,query_month=200901,checks=['Calendar-preserving history','202 inputs',
            'Strictly past support outcomes','Native DoubleAdapt current forecast invariant to query-label perturbation',
            'Query feedback updates future model state','Full monthly synthetic task capacity and finite outputs'],cz_training_started=False))
        return
    spec=dict(method='Author DoubleAdapt, adapted to monthly CZ',seed=43,window_months=12,task_months=1,
        device=args.device,torch_version=torch.__version__,runtime_change='Original check used PyTorch 1.9 CPU; CZ uses prepared CUDA-compatible PyTorch 2.4 runtime when --device cuda.',
        train=[200901,201912],validation=[202001,202112],test=[202201,202512],
        updates='Support previous month, forecast current month, then reveal current returns for meta-update',
        label='Cross-sectional z-score of one-month total return, used only in realized training/update loss',
        history='Previous 12 CZ observations on calendar month grid, zero fill outside eligible historical coverage',
        architecture='GRU 202 inputs, 64 hidden, two layers; 8-head feature and label adapters',
        schedule='Native 100 epoch maximum, 8 validation checks without improvement; validation begins epoch 10',
        checkpoint='Validation 2020-2021 mean monthly IC; test does not select hyperparameters or checkpoint',
        test_warmup='Replay validation months once from chosen offline model, then chronological 2022-2025 test',
        source_panel_sha256=sha(ROOT/'data/cz-monthly/panel.parquet'),runner_sha256=sha(__file__),
        adaptation='Monthly CZ experiment, not a replication of original daily CSI300 metrics')
    save('protocol.json',spec)
    started=time.monotonic();native=model.run_epoch;epochs=[0]
    def logged(phase,tasks,*a,**kw):
        save('status.json',dict(stage=phase,epochs_completed=epochs[0],seconds=time.monotonic()-started))
        result=native(phase,tasks,*a,**kw)
        if phase=='train':epochs[0]+=1
        if phase=='test':print(json.dumps(dict(epoch=epochs[0],validation_ic=float(result[1]))),flush=True)
        return result
    model.run_epoch=logged
    model.fit(TaskSet(data))
    torch.save(model.state_dict(),OUT/'offline-model.pt')
    predictions,_=model.inference(TaskSet(data,online=True))
    predictions=predictions.reset_index()
    predictions['return_month']=predictions.datetime.dt.year*100+predictions.datetime.dt.month
    scores=predictions.loc[predictions.return_month.between(202201,202512)].rename(columns={'instrument':'permno','pred':'score'})
    scores[['permno','return_month','score']].to_csv(OUT/'scores.csv',index=False)
    torch.save(model.state_dict(),OUT/'online-model.pt')
    save('status.json',dict(stage='scores_complete_evaluation_pending',epochs_completed=epochs[0],seconds=time.monotonic()-started))


if __name__=='__main__':main()
