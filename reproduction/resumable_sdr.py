"""Checkpoint the unchanged per-day SDR solves; no stride/interpolation."""
import hashlib,json,math,time
from pathlib import Path
import torch
from native_runtime import ROOT,atomic


def evaluate_sdr(problem,split,mu_hat_all,k,selector,cfg,identity):
    from utils import summarize_returns
    assert selector=='sdr'and cfg.sdr_stride==1
    key=hashlib.sha256(json.dumps({'identity':identity,'split':split.name,'k':k},sort_keys=True).encode()).hexdigest()
    dest=ROOT/'runtime/sdr';dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'{key}.pt';rows={};weights=[];last={};start=0
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        rows=saved['rows'];weights=saved['weights'];last=saved['last'];start=saved['next']
    began=time.monotonic()
    with torch.no_grad():
        for i in range(start,len(split)):
            sample=split[i];mu=mu_hat_all[i]
            w,info=problem.get_decision(mu,sample,k,'sdr')
            w64=w.detach().cpu().double().view(-1);cov=sample.cov.detach().cpu().double();raw=sample.y_raw.detach().cpu().double().view(-1)
            ret=float(torch.dot(raw,w64));den=torch.sqrt((w64@(cov@w64)).clamp_min(1e-16))
            step=float(torch.dot(raw,w64)/(den+1e-12))
            rows[i]={'t':i,'ret_true_raw':ret,'step_sr_raw':step,'weight_nnz_1e-8':int((w64.abs()>1e-8).sum()),
                     'weight_abs_sum':float(w64.abs().sum()),'weight_sum':float(w64.sum()),'gate_sum':math.nan,'gate_nnz_1e-3':math.nan,'interpolated':False}
            weights.append(w64.clone())
            last={'w':w64.clone(),'pred':mu.detach().cpu().double().view(-1).clone(),'true_raw':raw.clone(),'gate':None}
            if(i+1)%10==0 or i+1==len(split):
                tmp=path.with_suffix('.pt.tmp');torch.save({'next':i+1,'rows':rows,'weights':weights,'last':last},tmp);tmp.replace(path)
                atomic(ROOT/'runtime/sdr_status.json',{'dataset':problem.dataset,'k':k,'solved':i+1,'total':len(split),'elapsed_seconds':time.monotonic()-began,'stride':1,'interpolation':False})
                print(f'[SDR {problem.dataset} k={k}] {i+1}/{len(split)} elapsed={time.monotonic()-began:.1f}s',flush=True)
    trace={split[i].date:rows[i]for i in range(len(split))}
    metrics=summarize_returns([rows[i]['ret_true_raw']for i in range(len(split))],split.name,invalid_policy=cfg.mdd_invalid_policy)
    metrics[f'{split.name}_avg_pfl_loss']=torch.nn.MSELoss()(mu_hat_all.detach().cpu().float(),split.y_norm.detach().cpu().float()).item()if cfg.compute_pfl_loss else math.nan
    metrics[f'{split.name}_avg_step_sr_raw']=sum(rows[i]['step_sr_raw']for i in range(len(split)))/len(split)
    return metrics,trace,last
