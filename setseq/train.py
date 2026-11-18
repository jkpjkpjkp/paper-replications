"""Rolling equity fits, fixed last-epoch checkpoint, dated weight exports."""
import argparse
import contextlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch

from .features import digest
from .model import PortfolioModel,optimizer,sharpe_loss

ROOT=Path(__file__).resolve().parents[1]


def initialize(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(data,out,cfg,year,seed,kind,device,benchmark=False):
    directory=out/kind/f'seed-{seed}'/str(year)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        saved=json.loads((directory/'complete.json').read_text())
        if saved['config']!=cfg:
            raise ValueError('Refusing to reuse a run with a different configuration')
        if saved['model_code_sha256']!=digest(ROOT/'setseq/model.py') or saved['train_code_sha256']!=digest(Path(__file__)):
            raise ValueError('Refusing to mix completed runs from different source versions')
        if saved['panel_manifest_sha256']!=digest(data/'panels'/str(year)/'manifest.json'):
            raise ValueError('Refusing to reuse a fit after changing its input panel')
        print(f'reuse {kind} {seed} {year}',flush=True);return saved
    if (directory/'history.jsonl').exists():
        (directory/'history.jsonl').rename(directory/f'history-incomplete-{time.time_ns()}.jsonl')
    derived_seed=10000*seed+year
    initialize(derived_seed)
    rng=np.random.default_rng(derived_seed)
    panel=data/'panels'/str(year)
    x=np.load(panel/'x.npy',mmap_mode='r')
    labels=np.load(panel/'labels.npz')
    y=labels['y'];n_train=int(labels['train'].sum())
    assert labels['train'][:n_train].all() and not labels['train'][n_train:].any()
    assert (labels['input_dates']<labels['dates']).all()
    model=PortfolioModel(cfg,kind).to(device)
    opt=optimizer(model,cfg)
    amp=lambda: torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    count=sum(p.numel() for p in model.parameters())
    length=cfg['sequence_length']
    steps=int(np.ceil(n_train/length))
    history=[];start=time.monotonic()
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg['epochs']):
        model.train();losses=[];norms=[]
        # One sampled start in each equal stratum; all labels remain in train.
        edges=np.linspace(0,n_train-length+1,steps+1).astype(int)
        starts=[int(rng.integers(a,max(a+1,b))) for a,b in zip(edges[:-1],edges[1:])]
        rng.shuffle(starts)
        for offset in starts:
            xb=torch.tensor(np.asarray(x[offset:offset+length]),device=device)
            yb=torch.tensor(y[offset:offset+length],device=device)
            opt.zero_grad(set_to_none=True)
            with amp():
                w=model.weights(xb)
                loss=sharpe_loss(w,yb)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite train loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()));norms.append(float(norm))
            if benchmark and len(losses)==4:
                torch.cuda.synchronize() if device.type=='cuda' else None
                report={'seconds':time.monotonic()-start,'steps':4,'parameters':count,'max_gpu_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None}
                (directory/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(report),flush=True);return report
        row={'epoch':epoch+1,'mean_loss':float(np.mean(losses)),'max_gradient_norm':float(max(norms)),'seconds':time.monotonic()-start}
        history.append(row)
        with (directory/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if epoch==0 or (epoch+1)%5==0:
            print(f'{kind} seed={seed} year={year} epoch={epoch+1}/{cfg["epochs"]} loss={row["mean_loss"]:.5f} seconds={row["seconds"]:.1f}',flush=True)
    # Save before any test evaluation. No test-driven selection or refitting.
    checkpoint={'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':cfg,'kind':kind,'seed':seed,'derived_seed':derived_seed,'year':year}
    torch.save(checkpoint,directory/'model.pt')
    model.eval();all_weights=[]
    with torch.no_grad():
        for offset in range(n_train,len(x),length):
            begin=max(0,offset-model.receptive_history)
            end=min(offset+length,len(x))
            xb=torch.tensor(np.asarray(x[begin:end]),device=device)
            with amp():
                w=model.weights(xb)[offset-begin:]
            all_weights.append(w.cpu().numpy())
    weights=np.concatenate(all_weights)
    if not np.isfinite(weights).all() or not np.allclose(abs(weights).sum(1),1,atol=1e-5):
        raise ValueError('Invalid exported portfolio')
    np.savez_compressed(directory/'predictions.npz',weights=weights,returns=y[n_train:],dates=labels['dates'][n_train:],input_dates=labels['input_dates'][n_train:],permnos=labels['permnos'],market=labels['market'][n_train:],rf=labels['rf'][n_train:])
    report={'kind':kind,'year':year,'seed':seed,'derived_seed':derived_seed,'epochs':cfg['epochs'],'steps':cfg['epochs']*steps,'parameters':count,
        'elapsed_seconds':time.monotonic()-start,'max_gpu_gib':torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else None,
        'n_train':n_train,'n_test':len(weights),'config':cfg,'panel_manifest_sha256':digest(panel/'manifest.json'),
        'model_code_sha256':digest(ROOT/'setseq/model.py'),'train_code_sha256':digest(Path(__file__)),
        'prediction_sha256':digest(directory/'predictions.npz'),'checkpoint_sha256':digest(directory/'model.pt'),
        'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(device) if device.type=='cuda' else str(device),'python':platform.python_version()}}
    (directory/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'complete {kind} seed={seed} year={year}: {report["elapsed_seconds"]:.1f}s',flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/reconstruction.json')
    p.add_argument('--years',type=int,nargs='+');p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--models',nargs='+')
    p.add_argument('--device',default='cuda');p.add_argument('--benchmark',action='store_true')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    device=torch.device(a.device)
    years=a.years or list(range(cfg['test_years'][0],cfg['test_years'][1]+1))
    # Entire chronological OOS track per seed/model, so each completed track is usable.
    for seed in (a.seeds or cfg['seeds']):
        for kind in (a.models or cfg['models']):
            for year in years:
                fit(a.data,a.out,cfg,year,seed,kind,device,a.benchmark)


if __name__=='__main__':main()
