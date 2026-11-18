#!/usr/bin/env python3
"""Five-seed US LSTM comparison, with a training-only horizon meta-objective.

The full run requires the complete input audit and tensor build. --smoke uses
only pre-July-2024 cached dates for engineering checks, and emits no financial
test statistics. The held-out test is evaluated only after all fits finish.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import copy
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent

def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def log(message):print(dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')+' '+message,flush=True)

def stage_signature(data,seed,horizon,kind,protocol):
    value={'data':data.signature,'seed':seed,'horizon':horizon,'kind':kind,'protocol':protocol}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def stage_valid(path,signature):
    receipt=path.with_suffix(path.suffix+'.receipt.json')
    if not path.exists() or not receipt.exists():return False
    saved=json.loads(receipt.read_text())
    return saved.get('signature')==signature and sha(path)==saved.get('sha256')

def stage_receipt(path,signature):
    save_json(path.with_suffix(path.suffix+'.receipt.json'),{'signature':signature,'sha256':sha(path),'bytes':path.stat().st_size})

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

class Model(nn.Module):
    def __init__(self,features=72,hidden=64):
        super().__init__();self.lstm=nn.LSTM(features,hidden,batch_first=True)
        self.head=nn.Linear(hidden,1)
    def forward(self,x):
        output,_=self.lstm(x)
        return self.head(output[:,-1,:]).squeeze(-1)

def zlabels(y):
    valid=torch.isfinite(y)
    values=torch.where(valid,y,torch.zeros_like(y))
    count=valid.sum(0).clamp_min(1)
    mean=values.sum(0)/count
    delta=torch.where(valid,y-mean,torch.zeros_like(y))
    variance=delta.square().sum(0)/count
    z=delta/variance.clamp_min(1e-12).sqrt()
    eligible=(count>=20)&(variance>1e-12)
    return z,valid,count,eligible

def daily_loss(pred,y):
    """Per-horizon MSE, with prediction and label standardized on same mask."""
    z,mask,count,eligible=zlabels(y)
    p=pred[:,None]
    mean=(p*mask).sum(0)/count
    delta=torch.where(mask,p-mean,torch.zeros_like(z))
    variance=delta.square().sum(0)/count
    normalized=delta/variance.clamp_min(1e-12).sqrt()
    # Equal to standardized MSE when both variances are nonzero. Defining a
    # constant prediction's IC as zero avoids rewarding its undefined z-score
    # with MSE=1 rather than the uninformative IC loss=2.
    loss=2-2*(normalized*z*mask).sum(0)/count
    return loss,eligible

def batch_loss(pred,days,weights=None,horizon=None,warmup=False):
    losses=[];offset=0
    for data in days:
        n=len(data['Y']);p=pred[offset:offset+n];offset+=n;y=data['Y']
        if warmup:
            z,mask,_,available=zlabels(y)
            mask=mask&available[None,:]
            mean=(z*mask).sum(1)/mask.sum(1).clamp_min(1)
            mean=torch.where(mask.any(1),mean,torch.full_like(mean,float('nan')))
            values,valid=daily_loss(p,mean[:,None]);loss=values[valid].mean()
        elif horizon is not None:
            values,valid=daily_loss(p,y[:,horizon:horizon+1]);loss=values[valid].mean()
        else:
            values,valid=daily_loss(p,y)
            w=weights*valid
            loss=(values*w).sum()/w.sum().clamp_min(1e-12)
        losses.append(loss)
    return torch.stack(losses).mean()

def fit_scaler(records,cache):
    count=np.zeros(72,dtype=np.int64);total=np.zeros(72);square=np.zeros(72)
    for r in records:
        with np.load(cache/(r['date']+'.npz')) as data:x=data['X'].reshape(-1,72).astype(np.float64)
        valid=np.isfinite(x);values=np.where(valid,x,0)
        count+=valid.sum(0);total+=values.sum(0);square+=(values*values).sum(0)
    mean=np.divide(total,count,out=np.zeros(72),where=count>0)
    variance=np.maximum(np.divide(square,count,out=np.zeros(72),where=count>0)-mean*mean,0)
    scale=np.sqrt(variance);scale[(scale<1e-8)|(count==0)]=1
    return {'mean':mean.tolist(),'std':scale.tolist(),'count':count.tolist(),
            'fit_dates':[r['date'] for r in records], 'clip':10.0}

class Days:
    def __init__(self,cache,scaler,max_cached=1400):
        self.cache=cache;self.scaler=scaler;self.loaded=OrderedDict();self.max_cached=max_cached
        self.mean=np.array(scaler['mean'],dtype=np.float32)
        self.std=np.array(scaler['std'],dtype=np.float32)
    def get(self,date):
        if date not in self.loaded:
            with np.load(self.cache/(date+'.npz')) as data:
                x=(data['X']-self.mean)/self.std
                x=np.clip(np.nan_to_num(x,nan=0,posinf=0,neginf=0),-10,10)
                self.loaded[date]={'X':x,'Y':data['Y'].copy(),'permno':data['permno'].copy()}
            while len(self.loaded)>self.max_cached:self.loaded.popitem(last=False)
        else:self.loaded.move_to_end(date)
        return self.loaded[date]
    def batch(self,dates,device):
        entries=[self.get(date) for date in dates]
        x=torch.as_tensor(np.concatenate([e['X'] for e in entries]),device=device)
        return x,[{'Y':torch.as_tensor(e['Y'],device=device)} for e in entries]

def windows(dates,size,rng):
    groups=[dates[i:i+size] for i in range(0,len(dates),size)]
    return [groups[i] for i in rng.permutation(len(groups))]

@torch.no_grad()
def validation_loss(model,dates,data,device):
    model.eval();values=[]
    for i in range(0,len(dates),10):
        x,days=data.batch(dates[i:i+10],device)
        # Mean of daily losses, with each session equally weighted.
        pred=model(x);offset=0
        for day in days:
            n=len(day['Y']);loss,_=daily_loss(pred[offset:offset+n],day['Y'][:,-1:])
            values.append(float(loss[0]));offset+=n
    return float(np.mean(values))

def fit_supervised(seed,horizon,train_dates,val_dates,data,device,protocol,out):
    signature=stage_signature(data,seed,horizon,'supervised',protocol)
    if stage_valid(out,signature):
        log(f'Reusing verified supervised checkpoint: {out.name}, seed {seed}')
        return torch.load(out,map_location='cpu')
    seed_all(seed);rng=np.random.default_rng(seed)
    model=Model(hidden=protocol['model']['hidden_size']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['adamw_learning_rate'],weight_decay=protocol['weight_decay'])
    best_loss=math.inf;best_state=None;history=[];bad=0
    for epoch in range(protocol['maximum_epochs']):
        model.train();training=[];started=time.monotonic()
        for block in windows(train_dates,protocol['batch_days'],rng):
            x,days=data.batch(block,device)
            loss=batch_loss(model(x),days,horizon=horizon)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite supervised training loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            training.append(float(loss.detach()))
        val=validation_loss(model,val_dates,data,device)
        history.append({'epoch':epoch+1,'train_loss':float(np.mean(training)),'validation_loss':val,
                        'seconds':time.monotonic()-started})
        log(f'seed {seed} horizon {horizon+1}, epoch {epoch+1}: validation loss {val:.6f}')
        if val<best_loss:
            best_loss=val;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bad=0
        else:bad+=1
        if bad>=protocol['patience']:break
    model.load_state_dict(best_state)
    checkpoint={'model':best_state,'seed':seed,'horizon':horizon+1,'best_validation_loss':best_loss,
                'history':history,'protocol':protocol}
    temporary=out.with_suffix('.pt.tmp');torch.save(checkpoint,temporary);temporary.replace(out)
    stage_receipt(out,signature)
    return checkpoint

def fit_bilevel(seed,train_dates,val_dates,data,device,protocol,out):
    signature=stage_signature(data,seed,None,'bilevel',protocol)
    if stage_valid(out,signature):
        log(f'Reusing verified horizon selection, seed {seed}')
        return json.loads(out.read_text())
    seed_all(seed);rng=np.random.default_rng(seed)
    model=Model(hidden=protocol['model']['hidden_size']).to(device)
    warm_optimizer=torch.optim.AdamW(model.parameters(),lr=protocol['adamw_learning_rate'],weight_decay=protocol['weight_decay'])
    for epoch in range(protocol['warmup_epochs']):
        model.train()
        for block in windows(train_dates,protocol['batch_days'],rng):
            x,days=data.batch(block,device);loss=batch_loss(model(x),days,warmup=True)
            warm_optimizer.zero_grad(set_to_none=True);loss.backward();warm_optimizer.step()
        log(f'seed {seed} warmup {epoch+1}/{protocol["warmup_epochs"]}')
    logits=nn.Parameter(torch.zeros(390,device=device))
    outer_optimizer=torch.optim.Adam([logits],lr=protocol['outer_learning_rate'])
    best_loss=validation_loss(model,val_dates,data,device)
    best_logits=logits.detach().cpu().clone();best_epoch=0;bad=0;history=[]
    # cuDNN LSTM has no double backward in this installed Torch runtime.
    with torch.backends.cudnn.flags(enabled=False):
        for epoch in range(protocol['maximum_epochs']):
            model.train();norms=[];inner_losses=[];query_losses=[];started=time.monotonic()
            for block in windows(train_dates,protocol['batch_days'],rng):
                if len(block)<4:continue
                shuffle=rng.permutation(len(block));half=len(block)//2
                support=[block[i] for i in shuffle[:half]];query=[block[i] for i in shuffle[half:]]
                xs,ys=data.batch(support,device);xq,yq=data.batch(query,device)
                params=dict(model.named_parameters());weights=torch.softmax(logits,dim=0)
                inner=batch_loss(model(xs),ys,weights=weights)
                gradients=torch.autograd.grad(inner,tuple(params.values()),create_graph=True)
                fast={name:param-protocol['inner_learning_rate']*grad for (name,param),grad in zip(params.items(),gradients)}
                query_prediction=functional_call(model,fast,(xq,))
                query_loss=batch_loss(query_prediction,yq,horizon=389)
                entropy=-(weights*torch.log_softmax(logits,dim=0)).sum()
                outer=query_loss-protocol['entropy_weight']*entropy
                hypergradient=torch.autograd.grad(outer,logits)[0]
                if not torch.isfinite(hypergradient).all():raise RuntimeError('Nonfinite horizon hypergradient')
                outer_optimizer.zero_grad(set_to_none=True);logits.grad=hypergradient;outer_optimizer.step()
                # The student accumulates the look-ahead support updates (Eq.13).
                with torch.no_grad():
                    for name,param in model.named_parameters():param.copy_(fast[name].detach())
                norms.append(float(hypergradient.norm()));inner_losses.append(float(inner.detach()));query_losses.append(float(query_loss.detach()))
            val=validation_loss(model,val_dates,data,device)
            weights=torch.softmax(logits,dim=0).detach().cpu().numpy()
            history.append({'epoch':epoch+1,'validation_loss':val,'inner_loss':float(np.mean(inner_losses)),
                            'query_loss':float(np.mean(query_losses)),'hypergradient_norm':float(np.mean(norms)),
                            'argmax_horizon':int(np.flatnonzero(weights==weights.max())[-1]+1),
                            'entropy':float(-np.sum(weights*np.log(np.maximum(weights,1e-30)))),
                            'seconds':time.monotonic()-started})
            log(f'seed {seed} BLO epoch {epoch+1}: validation {val:.6f}, argmax horizon {history[-1]["argmax_horizon"]}')
            if val<best_loss:
                best_loss=val;best_logits=logits.detach().cpu().clone();best_epoch=epoch+1;bad=0
            else:bad+=1
            if bad>=protocol['patience']:break
    chosen=torch.softmax(best_logits,dim=0).numpy()
    # Uniform weights carry no learned preference. Break exact ties at target.
    horizon=int(np.flatnonzero(chosen==chosen.max())[-1]+1)
    result={'seed':seed,'selected_horizon':horizon,'weights':chosen.tolist(),'best_epoch':best_epoch,
            'best_validation_loss':best_loss,'history':history,'protocol':protocol}
    save_json(out,result)
    stage_receipt(out,signature)
    return result

@torch.no_grad()
def evaluate(checkpoint,dates,data,device,protocol):
    model=Model(hidden=protocol['model']['hidden_size']).to(device);model.load_state_dict(checkpoint['model']);model.eval()
    rows=[];predictions=[]
    for date in dates:
        day=data.get(date);x=torch.as_tensor(day['X'],device=device);pred=model(x).cpu().numpy();y=day['Y'][:,-1]
        finite=np.isfinite(pred)&np.isfinite(y);p=pred[finite];r=y[finite];ids=day['permno'][finite]
        if len(r)<20:raise RuntimeError('Insufficient held-out observations '+date)
        ic=float(np.corrcoef(p,r)[0,1]);rank=float(spearmanr(p,r).statistic)
        top=max(1,int(math.ceil(len(r)*0.1)));selected=np.argsort(p,kind='stable')[-top:]
        rows.append({'date':date,'ic':ic,'rank_ic':rank,'top_return':float(r[selected].mean()),
                     'universe_return':float(r.mean()),'stocks':len(r)})
        predictions.extend({'date':date,'permno':int(i),'score':float(a),'target_return':float(b)} for i,a,b in zip(ids,p,r))
    return rows,predictions

def summarize(rows):
    ic=np.array([r['ic'] for r in rows]);rank=np.array([r['rank_ic'] for r in rows]);returns=np.array([r['top_return'] for r in rows])
    return {'sessions':len(rows),'IC':float(ic.mean()),'ICIR':float(ic.mean()/ic.std(ddof=1)),
            'RankIC':float(rank.mean()),'RankICIR':float(rank.mean()/rank.std(ddof=1)),
            'TopReturn_percent':float(100*returns.mean()),'Sharpe':float(math.sqrt(252)*returns.mean()/returns.std(ddof=1))}

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args();protocol=json.loads((HERE/'protocol.json').read_text())
    torch.set_num_threads(4)
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(0.40)
    status=json.loads((HERE/'tensor_status.json').read_text());records=sorted(status['cached'],key=lambda r:r['date'])
    primary=HERE/'results';out=HERE/'smoke' if args.smoke else primary;out.mkdir(exist_ok=True)
    lock=(out/'.train.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.smoke:
        # Engineering fixtures use historical validation dates only, never the
        # published July-2024 onward test interval.
        records=[r for r in records if r['date']<'2024-07-01'][:36]
        if len(records)<28:raise RuntimeError('Need 28 pre-test days for an engineering smoke run')
        train=records[:24];val=records[24:];test=[]
        protocol=copy.deepcopy(protocol);protocol.update(seeds=[0],maximum_epochs=2,patience=2,warmup_epochs=1,batch_days=4)
    else:
        audit=json.loads((ROOT/'validation.json').read_text())
        if not audit.get('complete') or not status.get('complete'):raise RuntimeError('Full source and tensor audits must complete before primary training')
        train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='validation'];test=[r for r in records if r['split']=='test']
        if not train or not val or not test:raise RuntimeError('Missing a primary chronological split')
    # Check cache bytes and configuration before any model reads their arrays.
    for r in train+val+test:
        if r['config_sha256']!=status['config_sha256'] or sha(HERE/'cache'/(r['date']+'.npz'))!=r['sha256']:
            raise RuntimeError('Tensor cache audit failed '+r['date'])
    scaler=fit_scaler(train,HERE/'cache');save_json(out/'scaler.json',scaler)
    data=Days(HERE/'cache',scaler)
    data.signature=hashlib.sha256(json.dumps({'tensor_config':status['config_sha256'],'scaler':scaler,
       'source_sha256':sha(__file__),'train_dates':[r['date'] for r in train],
       'validation_dates':[r['date'] for r in val],
       'training_validation_cache_sha256':[{r['date']:r['sha256']} for r in train+val],
       'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':args.device}},sort_keys=True).encode()).hexdigest()
    run={'state':'training','engineering_smoke':args.smoke,'protocol':protocol,'tensor_config_sha256':status['config_sha256'],
         'source_sha256':sha(__file__),'scaler_sha256':sha(out/'scaler.json'),
         'dates':{'train':[r['date'] for r in train],'validation':[r['date'] for r in val],'test':[r['date'] for r in test]},
         'runtime':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'device':args.device,
                    'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pyarrow','matplotlib']}},'finished_seeds':[]}
    save_json(out/'status.json',run)
    train_dates=run['dates']['train'];val_dates=run['dates']['validation'];fitted=[]
    for seed in protocol['seeds']:
        folder=out/f'seed_{seed}';folder.mkdir(exist_ok=True)
        standard=fit_supervised(seed,389,train_dates,val_dates,data,args.device,protocol,folder/'standard.pt')
        blo=fit_bilevel(seed,train_dates,val_dates,data,args.device,protocol,folder/'horizon_selection.json')
        selected=fit_supervised(seed,blo['selected_horizon']-1,train_dates,val_dates,data,args.device,protocol,folder/'selected.pt')
        fitted.append((seed,standard,selected,blo));run['finished_seeds'].append(seed);save_json(out/'status.json',run)
    if args.smoke:
        run['state']='engineering_smoke_complete';run['financial_test_metrics_computed']=False
        run['finite_hypergradients']=all(np.isfinite([x['hypergradient_norm'] for x in f[3]['history']]).all() for f in fitted)
        save_json(out/'status.json',run);log('Engineering smoke complete; no financial test metrics computed');return
    # Test access is deliberately after every training and selection fit above.
    summaries=[];paired=[];daily_all=[]
    for seed,standard,selected,blo in fitted:
        for name,checkpoint in [('standard',standard),('selected',selected)]:
            daily,pred=evaluate(checkpoint,run['dates']['test'],data,args.device,protocol)
            write_csv(out/f'seed_{seed}/{name}_daily.csv',daily);write_csv(out/f'seed_{seed}/{name}_predictions.csv',pred)
            result={'seed':seed,'method':name,'training_horizon':checkpoint['horizon'],**summarize(daily)};summaries.append(result)
            daily_all.extend({'seed':seed,'method':name,**row} for row in daily)
            if name=='standard':baseline=daily
            else:paired.append(np.array([r['ic']-s['ic'] for r,s in zip(daily,baseline)]))
    write_csv(out/'seed_metrics.csv',summaries);write_csv(out/'daily_metrics.csv',daily_all)
    difference=np.mean(paired,axis=0);rng=np.random.default_rng(20261103);n=len(difference);replicates=[]
    for _ in range(10000):
        starts=rng.integers(n,size=math.ceil(n/20));idx=np.concatenate([(s+np.arange(20))%n for s in starts])[:n]
        replicates.append(float(difference[idx].mean()))
    metric_names=['IC','ICIR','RankIC','RankICIR','TopReturn_percent','Sharpe']
    averaged={name:{metric:float(np.mean([r[metric] for r in summaries if r['method']==name]))
                   for metric in metric_names} for name in ['standard','selected']}
    result={'per_seed':summaries,'five_seed_mean_by_method':averaged,'paired_IC_difference':{'estimate':float(difference.mean()),
             'date_block_bootstrap_95pct_CI':np.quantile(replicates,[.025,.975]).tolist(),'block_sessions':20,'replicates':10000},
            'interpretation':'Independent disclosed-feature test; gross signal metrics, not executable net investment returns'}
    save_json(out/'metrics.json',result);run['state']='complete';save_json(out/'status.json',run)
    log('Primary five-seed comparison complete')

if __name__=='__main__':main()
