#!/usr/bin/env python3
"""Equation-based RISE reproduction on two explicitly declared TFB layouts.

Original RISE code and TFB tuning grid are unavailable. All unspecified choices
are frozen in frozen_tfb_config.json and results are methodological/partial.
Run from the staged workspace described in README.md. Completed runs are checked and reused.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/tfb_rise'
DATA_ROOT = ROOT / 'data'
CONFIG_PATH = ROOT / 'frozen_tfb_config.json'


def atomic(path, value):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n'); os.replace(tmp, path)


def filehash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_data(dataset, layout):
    p = DATA_ROOT / f'{dataset}.{layout}.csv'
    with p.open() as f:
        reader = csv.reader(f); columns = next(reader); values = np.asarray(list(reader), dtype=np.float64)
    return values[:, 1:], values[:, 0], columns[1:], filehash(p)


def prepare(dataset, layout, horizon, lookback=96, split_context='preceding'):
    if split_context not in ['preceding','within_split']:
        raise ValueError('Unknown split context')
    if lookback < 16 or lookback % 4 or horizon % 4:
        raise ValueError('Lookback >=16 and lookback/horizon divisible by four are required')
    raw, ordinal, names, data_hash = load_data(dataset, layout)
    n = len(raw); train_end = int(n * .7); test_start = n - int(n * .2)
    mean = raw[:train_end].mean(axis=0); scale = raw[:train_end].std(axis=0)
    if np.any(scale <= 0) or not np.isfinite(raw).all():
        raise ValueError('Nonfinite or constant channel; no implicit imputation')
    z = (raw - mean) / scale
    L = lookback; H = horizon
    starts = np.arange(n - L - H + 1)
    x = np.stack([z[i:i+L] for i in starts]).astype(np.float32)
    y = np.stack([z[i+L:i+L+H] for i in starts]).astype(np.float32)
    origins = starts + L
    train = np.flatnonzero(origins + H <= train_end)
    offset = L if split_context == 'within_split' else 0
    val = np.flatnonzero((origins >= train_end + offset) & (origins + H <= test_start))
    test = np.flatnonzero((origins >= test_start + offset) & (origins + H <= n))
    assert len(train) and len(val) and len(test)
    assert origins[train].max() + H <= train_end
    assert origins[val].min() >= train_end and origins[val].max()+H <= test_start
    assert origins[test].min() >= test_start and origins[test].max()+H <= n
    return {'x':x,'y':y,'starts':starts,'origins':origins,'train':train,'val':val,'test':test,
            'mean':mean,'scale':scale,'ordinal':ordinal,'names':names,'data_hash':data_hash,
            'n':n,'train_end':train_end,'test_start':test_start,
            'lookback':L,'split_context':split_context}


def causal_ma(x, k=25):
    # x [N,T,C]. Early observations are extended by first-value replication.
    a = x.transpose(1,2)
    return torch.nn.functional.avg_pool1d(torch.nn.functional.pad(a,(k-1,0),mode='replicate'),k,1).transpose(1,2)


def smooth(x, g):
    n,t,c=x.shape
    assert t % g == 0
    pooled=x.reshape(n,t//g,g,c).mean(2).repeat_interleave(g,1)
    return pooled-pooled[:,-1:,:]


@torch.no_grad()
def retrieve(data, device):
    x=torch.as_tensor(data['x'],device=device)
    y=torch.as_tensor(data['y'],device=device)
    train=torch.as_tensor(data['train'],device=device)
    D=x-causal_ma(x)
    Dy=y[train]-causal_ma(y[train])
    starts=torch.as_tensor(data['starts'],device=device)
    bank_starts=starts[train]
    length=x.shape[1]+y.shape[1]
    overlap=(starts[:,None]-bank_starts[None,:]).abs()<length
    # Enforce masking for training queries exactly as in Section3.4. Evaluation
    # only accesses the fixed training bank, regardless of overlap with inputs.
    mask=torch.zeros_like(overlap);mask[train]=overlap[train]
    outputs=[]; checks=[]
    for g in [4,2,1]:
        q=smooth(D,g).flatten(1)
        q=q-q.mean(1,keepdim=True)
        q=q/q.norm(dim=1,keepdim=True).clamp_min(1e-8)
        scores=q@q[train].T
        scores=scores.masked_fill(mask,-torch.inf)
        values,indices=scores.topk(20,dim=1,largest=True,sorted=True)
        assert torch.isfinite(values).all()
        assert not mask.gather(1,indices)[train].any()
        weights=torch.softmax(values/.1,dim=1)
        residual=smooth(Dy,g)
        retrieved=(residual[indices]*weights[:,:,None,None]).sum(1)
        assert torch.isfinite(retrieved).all()
        outputs.append(retrieved)
        checks.append({'granularity':g,'minimum_training_query_available_neighbors':int((~mask[train]).sum(1).min().item()),'no_training_interval_overlap_retrieved':True})
    result=torch.stack(outputs,1)
    # Chronology check: every bank future label is inside training partition.
    assert int((bank_starts+length).max().item()) <= data['train_end']
    return result,checks


class RISE(nn.Module):
    def __init__(self,H,L=96):
        super().__init__();self.H=H
        self.num_patches=(L+8-16)//8+1
        self.trend=nn.Linear(L,H)
        self.patch=nn.Linear(16,128)
        self.ctx=nn.ModuleList([nn.Linear(H,128) for _ in range(3)])
        self.granularity_embedding=nn.Parameter(torch.zeros(3,128))
        layer=nn.TransformerEncoderLayer(128,4,dim_feedforward=512,dropout=.1,activation='gelu',batch_first=True,norm_first=False)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.patch_head=nn.Linear(self.num_patches*128,H)
        self.ret=nn.ModuleList([nn.Linear(H//g,H) for g in [4,2,1]])
        self.fusion=nn.Parameter(torch.full((3,),1/3))
        pos=torch.arange(self.num_patches).float()[:,None]
        freq=torch.exp(torch.arange(0,128,2).float()*(-math.log(10000)/128))
        pe=torch.zeros(self.num_patches,128);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq)
        self.register_buffer('positional_encoding',pe)

    def forward(self,x,r):
        anchor=x[:,-1:,:]
        x=x-anchor
        T=causal_ma(x);D=x-T
        n,L,c=x.shape
        trend=self.trend(T.transpose(1,2)).transpose(1,2)
        sequence=D.transpose(1,2).reshape(n*c,L)
        patches=torch.nn.functional.pad(sequence,(0,8),mode='replicate').unfold(1,16,8)
        tokens=self.patch(patches)+self.positional_encoding
        contexts=[];ret=0
        for j,g in enumerate([4,2,1]):
            a=r[:,j].transpose(1,2)
            contexts.append(self.ctx[j](a).reshape(n*c,128)+self.granularity_embedding[j])
            ret=ret+self.ret[j](a[:,:,::g]).transpose(1,2)
        full=torch.cat([tokens,torch.stack(contexts,1)],dim=1)
        encoded=self.encoder(full)[:,:self.num_patches]
        patch=self.patch_head(encoded.flatten(1)).reshape(n,c,self.H).transpose(1,2)
        return (self.fusion[0]*patch+self.fusion[1]*trend+self.fusion[2]*ret)+anchor


@torch.no_grad()
def evaluate(model,x,r,y,ids):
    model.eval(); predictions=[]
    for lo in range(0,len(ids),32):
        k=ids[lo:lo+32]
        predictions.append(model(x[k],r[k]).cpu().numpy())
    p=np.concatenate(predictions); t=y[ids].cpu().numpy()
    error=p.astype(np.float64)-t.astype(np.float64)
    return p,{'mse':float(np.mean(error**2)),'mae':float(np.mean(np.abs(error))),'origins':len(ids),'forecast_steps':len(ids)*p.shape[1]*p.shape[2]}


def train_one(dataset,layout,H,seed,data,r,retrieval_checks,device):
    destination=OUT/f'{layout}/{dataset}/H{H}/seed{seed}'
    destination.mkdir(parents=True,exist_ok=True)
    result_path=destination/'metrics.json'
    frozen_hash=filehash(CONFIG_PATH)
    if result_path.exists():
        previous=json.loads(result_path.read_text())
        if previous['data_sha256'] != data['data_hash'] or previous['frozen_config_sha256']!=frozen_hash:
            raise RuntimeError('Existing run differs from current frozen input/config; preserve and stop')
        if not (destination/'predictions.npz').exists() or not (destination/'best.pt').exists():
            raise RuntimeError('Existing completed metrics missing required artifact')
        print(json.dumps({'reused':str(result_path)}),flush=True)
        return previous
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model=RISE(H,data['lookback']).to(device)
    x=torch.as_tensor(data['x'],device=device);y=torch.as_tensor(data['y'],device=device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0)
    generator=np.random.default_rng(seed)
    best=float('inf');best_state=None;history=[];started=time.time()
    for epoch in range(10):
        model.train();order=generator.permutation(data['train']); total=0.;seen=0
        for low in range(0,len(order),32):
            ids=order[low:low+32]
            optimizer.zero_grad(set_to_none=True)
            prediction=model(x[ids],r[ids]);loss=(prediction-y[ids]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
        _,vm=evaluate(model,x,r,y,data['val'])
        history.append({'epoch':epoch+1,'train_mse':total/seen,'validation_mse':vm['mse'],'learning_rate':optimizer.param_groups[0]['lr']})
        if vm['mse']<best:
            best=vm['mse'];best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
            best_epoch=epoch+1
        scheduler.step(epoch+1)
    model.load_state_dict(best_state)
    test_prediction,metric=evaluate(model,x,r,y,data['test'])
    second,metric_second=evaluate(model,x,r,y,data['test'])
    if not np.array_equal(test_prediction,second):raise RuntimeError('Deterministic checkpoint inference mismatch')
    vp,validation_metric=evaluate(model,x,r,y,data['val'])
    # Causal filter test changes only a suffix; prior trend outputs must remain identical.
    example=x[:1].clone();changed=example.clone();changed[:,48:]+=10
    if not torch.equal(causal_ma(example)[:,:48],causal_ma(changed)[:,:48]):raise RuntimeError('Causal moving average leaks future inputs')
    # Global level shift per channel cancels in the detrended key.
    if not torch.allclose((example-causal_ma(example)),(example+3)-causal_ma(example+3),atol=2e-6):raise RuntimeError('Decomposition not shift-invariant')
    target=data['y'][data['test']]
    np.savez_compressed(destination/'predictions.npz',test_prediction=test_prediction,test_target=target,
        test_origins=data['origins'][data['test']],validation_prediction=vp,validation_target=data['y'][data['val']],
        validation_origins=data['origins'][data['val']],ordinal=data['ordinal'],mean=data['mean'],scale=data['scale'])
    torch.save(best_state,destination/'best.pt')
    atomic(destination/'history.json',history)
    result={'dataset':dataset,'layout':layout,'method':'RISE_equation_reimplementation','horizon':H,'seed':seed,
        'reproduction_level':'partial_methodological_fixed_default_TFB_config','data_sha256':data['data_hash'],
        'frozen_config_sha256':frozen_hash,'metrics':metric,'validation_metrics':validation_metric,
        'best_epoch':best_epoch,'selection':'minimum validation MSE among ten epochs; no test selection',
        'elapsed_seconds':time.time()-started,'n_samples':data['n'],'train_end_exclusive':data['train_end'],
        'test_start_inclusive':data['test_start'],'train_windows':len(data['train']),'validation_windows':len(data['val']),
        'lookback':data['lookback'],'split_context':data['split_context'],
        'test_windows':len(data['test']),'parameters':sum(p.numel() for p in model.parameters()),
        'retrieval_checks':retrieval_checks,'validation':{'deterministic_checkpoint_rerun':True,'causal_MA_prefix_invariance':True,
        'detrended_key_shift_invariance':True,'all_labels_within_split':True,'training_bank_only':True,'drop_last':False},
        'predictions_path':str(destination/'predictions.npz'),'checkpoint_path':str(destination/'best.pt'),
        'calendar_dates_available':False,'trading_returns_inferred':False}
    atomic(result_path,result)
    print(json.dumps({'finished':dataset,'layout':layout,'H':H,'seed':seed,'mse':metric['mse'],'seconds':result['elapsed_seconds']}),flush=True)
    del model,optimizer,x,y
    return result


def main():
    global OUT, DATA_ROOT, CONFIG_PATH
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--config',type=Path,default=CONFIG_PATH)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    CONFIG_PATH=args.config.resolve();DATA_ROOT=args.data_root.resolve();OUT=args.output.resolve()
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested GPU unavailable; do not change device silently')
    OUT.mkdir(parents=True,exist_ok=True);results=[]
    config=json.loads(CONFIG_PATH.read_text())
    lookback=config.get('lookback',96)
    split_context=config.get('split_context','preceding')
    required_runs=len(config['layouts'])*len(config['datasets'])*len(config['horizons'])*len(config['seeds'])
    for layout in config['layouts']:
        for dataset in config['datasets']:
            for H in config['horizons']:
                data=prepare(dataset,layout,H,lookback,split_context);r,checks=retrieve(data,args.device)
                for seed in config['seeds']:
                    results.append(train_one(dataset,layout,H,seed,data,r,checks,args.device))
                    atomic(OUT/'progress.json',{'completed_runs':len(results),'required_runs':required_runs,'runs':results})
                del r
                if args.device=='cuda':torch.cuda.empty_cache()
    aggregates=[]
    for layout in config['layouts']:
        for dataset in config['datasets']:
            for H in config['horizons']:
                a=[r['metrics'] for r in results if r['layout']==layout and r['dataset']==dataset and r['horizon']==H]
                aggregates.append({'layout':layout,'dataset':dataset,'horizon':H,'seeds':len(config['seeds']),
                    'mse_mean':float(np.mean([x['mse'] for x in a])),'mae_mean':float(np.mean([x['mae'] for x in a])),
                    'mse_std_across_seeds':float(np.std([x['mse'] for x in a],ddof=1)) if len(a)>1 else None,
                    'mae_std_across_seeds':float(np.std([x['mae'] for x in a],ddof=1)) if len(a)>1 else None})
    atomic(OUT/'summary.json',{'status':'complete_partial_methodological_scope','original_paper_reproduction_complete':False,
        'completed_runs':len(results),'aggregates':aggregates,'all_runs':results,'torch_version':torch.__version__,'numpy_version':np.__version__,
        'max_torch_gpu_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0})


if __name__=='__main__':main()
