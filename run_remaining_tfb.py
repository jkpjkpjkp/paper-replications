#!/usr/bin/env python3
"""Official GTR/RAFT/DUET modules, declared TFB methodological scope."""
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
import tqdm
sys.path.insert(0,str(ROOT/'official-extra/RAFT'))
sys.path.insert(0,str(ROOT/'official-extra/DUET'))
import numpy as np
import torch
from torch import nn
from run_rise_tfb import prepare,smooth

OUT=ROOT/'results/tfb_remaining'
CONFIG=ROOT/'frozen_remaining_config.json'


def atomic(path,value):
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,path)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def direct_module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class WindowBank:
 def __init__(self,data,H):self.data=data;self.pred_len=H;self.label_len=0
 def __len__(self):return len(self.data['train'])
 def __getitem__(self,i):
  k=self.data['train'][i]
  return i,self.data['x'][k],self.data['y'][k],None,None


@torch.no_grad()
def raft_retrieval(data,H):
 from layers.Retrieval import RetrievalTool
 rt=RetrievalTool(96,H,5,n_period=3,temperature=.1,topm=20)
 rt.prepare_dataset(WindowBank(data,H))
 x=torch.as_tensor(data['x'],device='cuda');train=torch.as_tensor(data['train'],device='cuda')
 train_count=len(train);parts=[]
 for start in range(0,train_count,256):
  ids=torch.arange(start,min(start+256,train_count),device='cuda')
  parts.append(rt.retrieve(x[ids],ids,train=True))
 for start in range(train_count,len(x),256):
  ids=torch.arange(start,min(start+256,len(x)),device='cuda')
  parts.append(rt.retrieve(x[ids],ids,train=False))
 result=torch.cat(parts,dim=1).detach()
 assert result.shape==(3,len(x),H,5)
 # Independent raw-window retrieval reference verifies native mask, similarity,
 # recentering and aggregation on one training and one test query.
 for query in [train_count//2,int(data['test'][0])]:
  for j,g in enumerate([4,2,1]):
   keys=smooth(x[train],g).flatten(1);q=smooth(x[query:query+1],g).flatten(1)
   keys=keys-keys.mean(1,keepdim=True);q=q-q.mean(1,keepdim=True)
   scores=torch.nn.functional.normalize(q,dim=1)@torch.nn.functional.normalize(keys,dim=1).T
   if query<train_count:
    overlap=(data['starts'][query]-data['starts'][data['train']])
    overlap=torch.as_tensor(np.abs(overlap)<96+H,device='cuda')
    scores[:,overlap]=-torch.inf
   values,indices=scores.topk(20,dim=1)
   future=smooth(torch.as_tensor(data['y'][data['train']],device='cuda'),g)
   reference=(future[indices[0]]*torch.softmax(values[0]/.1,0)[:,None,None]).sum(0)
   if not torch.allclose(result[j,query],reference,atol=3e-5,rtol=3e-5):raise RuntimeError('Native RAFT retrieval differs from independent reference')
 return result,{'periods':[4,2,1],'native_retrieval_matches_independent_reference':True,'train_only_bank':True,
                'whole_interval_training_overlap_mask':True,'framework_8_worker_loader_not_called':True,'retrieval_batches':256}


class Model(nn.Module):
 def __init__(self,method,H,config,retrieval=None):
  super().__init__();self.method=method
  cfg=dict(config['method_kwargs'][method]);cfg.update(seq_len=96,pred_len=H,enc_in=5)
  if method=='GTR':
   module=direct_module('official_gtr',ROOT/'official-extra/GTR/models/GTR.py');self.native=module.Model(SimpleNamespace(**cfg))
  elif method=='RAFT':
   module=direct_module('official_raft',ROOT/'official-extra/RAFT/models/RAFT.py');self.native=module.Model(SimpleNamespace(**cfg))
   self.native.retrieval_dict={'all':retrieval}
  elif method=='DUET':
   module=importlib.import_module('ts_benchmark.baselines.duet.models.duet_model');self.native=module.DUETModel(SimpleNamespace(**cfg))
  else:raise ValueError(method)
 def forward(self,x,ids):
  anchor=x[:,-1:,:];centered=x-anchor
  if self.method=='GTR':p=self.native(centered,(ids+96)%24);extra=x.new_zeros(())
  elif self.method=='RAFT':p=self.native(centered,ids,mode='all');extra=x.new_zeros(())
  else:p,extra=self.native(centered)
  return p+anchor,extra


@torch.no_grad()
def evaluate(model,x,y,ids,eval_seed):
 model.eval();predictions=[]
 # DUET samples a Gumbel channel mask even in eval. Freeze this evaluation
 # random stream without modifying native model logic or training RNG state.
 with torch.random.fork_rng(devices=[0]):
  torch.manual_seed(eval_seed);torch.cuda.manual_seed_all(eval_seed)
  for start in range(0,len(ids),32):
   k=torch.as_tensor(ids[start:start+32],device='cuda');p,_=model(x[k],k);predictions.append(p.cpu().numpy())
 p=np.concatenate(predictions);target=y[ids].cpu().numpy();e=p.astype(np.float64)-target.astype(np.float64)
 if not np.isfinite(p).all():raise RuntimeError('Nonfinite remaining-baseline predictions')
 return p,{'mse':float(np.mean(e*e)),'mae':float(np.mean(np.abs(e))),'origins':len(ids),'forecast_steps':int(p.size)}


def fit(method,dataset,layout,H,seed,data,config,retrieval,checks):
 dest=OUT/method/layout/dataset/f'H{H}/seed{seed}';dest.mkdir(parents=True,exist_ok=True)
 metric_path=dest/'metrics.json';signature={'data_sha256':data['data_hash'],'frozen_config_sha256':sha(CONFIG)}
 if metric_path.exists():
  old=json.loads(metric_path.read_text())
  if old['signature']!=signature:raise RuntimeError('Completed remaining-baseline inputs differ; preserve')
  if not (dest/'best.pt').exists() or not (dest/'predictions.npz').exists():raise RuntimeError('Completed run missing artifacts')
  return old
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 model=Model(method,H,config,retrieval).cuda();x=torch.as_tensor(data['x'],device='cuda');y=torch.as_tensor(data['y'],device='cuda')
 optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
 scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=1,eta_min=0)
 best=float('inf');history=[];generator=np.random.default_rng(seed);started=time.time()
 for epoch in range(10):
  model.train();order=generator.permutation(data['train']);total=0.;seen=0
  for start in range(0,len(order),32):
   ids=torch.as_tensor(order[start:start+32],device='cuda');optimizer.zero_grad(set_to_none=True)
   p,extra=model(x[ids],ids)
   if method=='DUET':loss=torch.nn.functional.huber_loss(p,y[ids])+extra
   else:loss=(p-y[ids]).square().mean()
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite original-baseline loss')
   loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
  _,val=evaluate(model,x,y,data['val'],10000+seed)
  history.append({'epoch':epoch+1,'train_objective':total/seen,'validation_mse':val['mse'],'lr':optimizer.param_groups[0]['lr']})
  if val['mse']<best:
   best=val['mse'];best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
  scheduler.step(epoch+1)
 model.load_state_dict(best_state)
 p,metrics=evaluate(model,x,y,data['test'],20000+seed);again,_=evaluate(model,x,y,data['test'],20000+seed)
 if not np.array_equal(p,again):raise RuntimeError('Fixed-evaluation-stream rerun mismatch')
 vp,vm=evaluate(model,x,y,data['val'],10000+seed)
 np.savez_compressed(dest/'predictions.npz',test_prediction=p,test_target=data['y'][data['test']],test_origins=data['origins'][data['test']],
  validation_prediction=vp,validation_target=data['y'][data['val']],validation_origins=data['origins'][data['val']],ordinal=data['ordinal'],mean=data['mean'],scale=data['scale'])
 torch.save(best_state,dest/'best.pt');atomic(dest/'history.json',history)
 record={'method':method,'dataset':dataset,'layout':layout,'horizon':H,'seed':seed,'signature':signature,'metrics':metrics,'validation_metrics':vm,
  'best_epoch':best_epoch,'selection':'minimum validation MSE of ten epochs; no test tuning','elapsed_seconds':time.time()-started,
  'parameters':sum(p.numel() for p in model.parameters()),'reproduction_level':'partial_methodological_official_baseline_with_declared_defaults',
  'validation':{'fixed_random_stream_checkpoint_rerun':True,'all_labels_wholly_in_split':True,'train_only_scaler':True,'no_drop_last':True},
  'retrieval_checks':checks,'predictions_path':str(dest/'predictions.npz'),'checkpoint_path':str(dest/'best.pt'),
  'calendar_dates_available':False,'trading_returns_inferred':False}
 atomic(metric_path,record);print(json.dumps({'finished':method,'dataset':dataset,'layout':layout,'H':H,'seed':seed,'mse':metrics['mse'],'seconds':record['elapsed_seconds']}),flush=True)
 return record


def main():
 torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 config=json.loads(CONFIG.read_text());OUT.mkdir(parents=True,exist_ok=True);results=[]
 for method in config['models']:
  for layout in config['layouts']:
   for dataset in config['datasets']:
    for H in config['horizons']:
     data=prepare(dataset,layout,H);retrieval=None;checks={}
     if method=='RAFT':retrieval,checks=raft_retrieval(data,H)
     for seed in config['seeds']:
      results.append(fit(method,dataset,layout,H,seed,data,config,retrieval,checks))
      atomic(OUT/'progress.json',{'completed_runs':len(results),'required_runs':240,'results':results})
     del retrieval;torch.cuda.empty_cache()
 aggregates=[]
 for method in config['models']:
  for layout in config['layouts']:
   for dataset in config['datasets']:
    for H in config['horizons']:
     a=[r['metrics'] for r in results if r['method']==method and r['layout']==layout and r['dataset']==dataset and r['horizon']==H]
     aggregates.append({'method':method,'layout':layout,'dataset':dataset,'horizon':H,'seeds':5,'mse_mean':float(np.mean([x['mse'] for x in a])),
                        'mae_mean':float(np.mean([x['mae'] for x in a])),'mse_std_across_seeds':float(np.std([x['mse'] for x in a],ddof=1))})
 atomic(OUT/'summary.json',{'status':'complete_partial_methodological_scope','original_paper_reproduction_complete':False,'completed_runs':len(results),
                           'aggregates':aggregates,'all_runs':results,'max_torch_gpu_allocated_bytes':torch.cuda.max_memory_allocated()})


if __name__=='__main__':main()
