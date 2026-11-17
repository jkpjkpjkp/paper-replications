#!/usr/bin/env python3
"""Four official compact TFB model implementations, common declared protocol."""
import csv
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
import einops
sys.path.insert(0,str(ROOT/'official-tfb'))
import numpy as np
import torch
from torch import nn
from run_rise_tfb import prepare

OUT=ROOT/'results/tfb_neural_baselines'
CONFIG=ROOT/'frozen_tfb_baseline_config.json'


def atomic(path,obj):
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(temp,path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


class Model(nn.Module):
    def __init__(self,name,H,kwargs):
        super().__init__()
        module=importlib.import_module('ts_benchmark.baselines.time_series_library.models.'+name)
        c=dict(kwargs);c['pred_len']=H
        self.native=getattr(module,name)(SimpleNamespace(**c))
    def forward(self,x):
        anchor=x[:,-1:,:]
        return self.native(x-anchor,None,None,None)+anchor


@torch.no_grad()
def evaluate(model,x,y,ids):
    model.eval();pred=[]
    for i in range(0,len(ids),32):pred.append(model(x[ids[i:i+32]]).cpu().numpy())
    p=np.concatenate(pred);truth=y[ids].cpu().numpy();e=p.astype(np.float64)-truth.astype(np.float64)
    if not np.isfinite(p).all():raise RuntimeError('Nonfinite neural forecast')
    return p,{'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'origins':len(ids),'forecast_steps':int(p.size)}


def fit(method,dataset,layout,H,seed,data,config):
    dest=OUT/method/layout/dataset/f'H{H}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    result_path=dest/'metrics.json';signature={'data_sha256':data['data_hash'],'frozen_config_sha256':sha(CONFIG)}
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old['signature']!=signature:raise RuntimeError('Completed neural run inputs changed; preserve')
        if not (dest/'best.pt').exists() or not (dest/'predictions.npz').exists():raise RuntimeError('Missing completed neural artifacts')
        return old
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=Model(method,H,config['model_kwargs']).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0)
    x=torch.as_tensor(data['x'],device='cuda');y=torch.as_tensor(data['y'],device='cuda')
    best=float('inf');history=[];generator=np.random.default_rng(seed);started=time.time()
    for epoch in range(10):
        model.train();order=generator.permutation(data['train']);total=0;seen=0
        for i in range(0,len(order),32):
            k=order[i:i+32];optimizer.zero_grad(set_to_none=True)
            p=model(x[k]);loss=(p-y[k]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite baseline loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(k);seen+=len(k)
        _,val=evaluate(model,x,y,data['val'])
        history.append({'epoch':epoch+1,'train_mse':total/seen,'validation_mse':val['mse'],'lr':optimizer.param_groups[0]['lr']})
        if val['mse']<best:
            best=val['mse'];best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        scheduler.step(epoch+1)
    model.load_state_dict(best_state)
    p,m=evaluate(model,x,y,data['test']);rerun,_=evaluate(model,x,y,data['test'])
    if not np.array_equal(p,rerun):raise RuntimeError('Baseline checkpoint deterministic inference mismatch')
    vp,vm=evaluate(model,x,y,data['val'])
    np.savez_compressed(dest/'predictions.npz',test_prediction=p,test_target=data['y'][data['test']],
        test_origins=data['origins'][data['test']],validation_prediction=vp,validation_target=data['y'][data['val']],
        validation_origins=data['origins'][data['val']],ordinal=data['ordinal'],mean=data['mean'],scale=data['scale'])
    torch.save(best_state,dest/'best.pt');atomic(dest/'history.json',history)
    r={'method':method,'dataset':dataset,'layout':layout,'horizon':H,'seed':seed,'signature':signature,'metrics':m,
       'validation_metrics':vm,'best_epoch':best_epoch,'selection':'minimum validation MSE of ten epochs; no test selection',
       'parameters':sum(v.numel() for v in model.parameters()),'elapsed_seconds':time.time()-started,
       'reproduction_level':'partial_methodological_official_TFB_model_common_fixed_defaults','predictions_path':str(dest/'predictions.npz'),
       'checkpoint_path':str(dest/'best.pt'),'validation':{'training_only_scaler':True,'labels_inside_split':True,'drop_last':False,'deterministic_inference_rerun':True},
       'calendar_dates_available':False,'trading_returns_inferred':False}
    atomic(result_path,r);print(json.dumps({'finished':method,'dataset':dataset,'layout':layout,'H':H,'seed':seed,'mse':m['mse'],'seconds':r['elapsed_seconds']}),flush=True)
    return r


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    config=json.loads(CONFIG.read_text());OUT.mkdir(parents=True,exist_ok=True)
    # Validate every native model interface on training-only input before fitting
    # or examining its first test result; no architecture selected by test error.
    example=prepare('NASDAQ','tfb_native_blocks',24)['x'][:2]
    for method in config['models']:
        model=Model(method,24,config['model_kwargs']).cuda().eval()
        with torch.no_grad():output=model(torch.as_tensor(example,device='cuda'))
        assert output.shape==(2,24,5) and torch.isfinite(output).all()
        del model
    results=[]
    for method in config['models']:
        for layout in config['layouts']:
            for dataset in config['datasets']:
                for H in config['horizons']:
                    data=prepare(dataset,layout,H)
                    for seed in config['seeds']:
                        results.append(fit(method,dataset,layout,H,seed,data,config))
                        atomic(OUT/'progress.json',{'completed_runs':len(results),'required_runs':320,'results':results})
                    torch.cuda.empty_cache()
    aggregates=[]
    for method in config['models']:
        for layout in config['layouts']:
            for dataset in config['datasets']:
                for H in config['horizons']:
                    a=[r['metrics'] for r in results if r['method']==method and r['layout']==layout and r['dataset']==dataset and r['horizon']==H]
                    aggregates.append({'method':method,'layout':layout,'dataset':dataset,'horizon':H,'seeds':5,
                        'mse_mean':float(np.mean([v['mse'] for v in a])),'mae_mean':float(np.mean([v['mae'] for v in a])),
                        'mse_std_across_seeds':float(np.std([v['mse'] for v in a],ddof=1))})
    atomic(OUT/'summary.json',{'status':'complete_partial_methodological_scope','completed_runs':len(results),'aggregates':aggregates,
        'all_runs':results,'torch_version':torch.__version__,'einops_version':einops.__version__,
        'max_torch_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'original_paper_reproduction_complete':False})


if __name__=='__main__':main()
