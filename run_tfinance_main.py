#!/usr/bin/env python3
"""Guard-only faithful author-code T-Finance main seeds, with observational saves."""
import contextlib,copy,hashlib,json,os,random,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent;OFF=R/'official';OUT=R/'results/local-main';OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(OFF));os.environ['DGLBACKEND']='pytorch'
import torch,numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from utils import Dataset
from models.fraudgnam_detector import FraudGNAMDetector
from refined_detector import RefinedFraudGNAMDetector
THREADS=int(os.environ.get('OMP_NUM_THREADS','2'));torch.set_num_threads(THREADS);torch.set_num_interop_threads(1)
def atomic(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def metric(y,p):
 k=int(y.sum());return {'AUROC':float(roc_auc_score(y,p)),'AUPRC':float(average_precision_score(y,p)),'RecK':float(y[p.argsort()[-k:]].sum()/k)}
def sha(p):
 with p.open('rb') as h:return hashlib.file_digest(h,'sha256').hexdigest()
audit=json.loads((R/'results/data-audit.json').read_text());assert audit['fully_supervised_masks_disjoint_exhaustive']
assert audit['nodes']==39357 and audit['edges']==42484443
results=[]
for trial in range(10):
 folder=OUT/f'trial{trial}';folder.mkdir(exist_ok=True);done=folder/'result.json'
 if done.exists():
  prior=json.loads(done.read_text());assert prior['code_commit']=='6f13f4f7363a5a39448fd759a14963ed7e7b8cb9' and prior['seed']==3407+10*trial
  results.append(prior);continue
 seed=3407+10*trial;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 data=Dataset('tfinance',prefix=str(R/'data')+'/');data.split(0,trial)
 tc={'device':'cpu','epochs':200,'patience':50,'metric':'AUPRC','inductive':False,'seed':seed};mc={'model':'FraudGNAM','lr':0.01,'drop_rate':0}
 start=time.time()
 try:
  with (folder/'train.log').open('w') as log,contextlib.redirect_stdout(log):
   det=FraudGNAMDetector(tc,mc,data)
   original_eval=det.eval;val_n=int(det.val_mask.sum());test_n=int(det.test_mask.sum());obs={'val_calls':0,'test_saves':0}
   def recorded_eval(labels,probs):
    score=original_eval(labels,probs)
    if len(labels)==val_n:obs['val_calls']+=1
    elif len(labels)==test_n:
     obs['test_saves']+=1
     obs['best_epoch']=obs['val_calls']-1
     obs['test_y']=labels.detach().cpu().numpy().copy();obs['test_p']=probs.detach().cpu().numpy().copy()
     obs['best_state']={k:v.detach().cpu().clone() for k,v in det.model.state_dict().items()}
    return score
   det.eval=recorded_eval
   score=det.train()
  p=obs.pop('test_p');y=obs.pop('test_y');best_state=obs.pop('best_state')
  recomputed=metric(y,p)
  assert all(abs(recomputed[k]-score[k])<1e-12 for k in score)
  np.savez_compressed(folder/'best-test-predictions.npz',node_ids=np.flatnonzero(det.test_mask.cpu().numpy()),labels=y,probabilities=p)
  torch.save({'model':best_state,'model_config':mc,'train_config':tc,'best_epoch':obs['best_epoch']},folder/'best-validation-model.pt')
  result={'paper_id':429,'code_commit':'6f13f4f7363a5a39448fd759a14963ed7e7b8cb9','trial':trial,'seed':seed,'metrics':{k:float(v) for k,v in score.items()},'recomputed_metrics':recomputed,'model_config':mc,'train_config':tc,'observation':obs,'elapsed_seconds':time.time()-start,'graph_sha256':audit['sha256'],'source_behavior':'unchanged author implementation; all-label homophily/fraud_rate retained and disclosed','adaptations':['CPU PyTorch2.4.0+cpu/DGL1.1.3 vs published GPU/DGL2.4;twoCPUthreads','Only observational eval wrapper to save best-epoch weights/predictions; no training/selection changes'],'best_model_sha256':sha(folder/'best-validation-model.pt'),'prediction_sha256':sha(folder/'best-test-predictions.npz'),'submitted_refined_equivalence':'ForT-Finance h=.9708,d=10,nf=721, order2,HP0:refined R1/R2/R3 do not change configuration. Refined class does not rebuild when unchanged; seed-equivalent main fit reused.','created_at':datetime.now(timezone.utc).isoformat()}
  atomic(done,result);results.append(result)
  print(json.dumps({'trial':trial,'seed':seed,'metrics':recomputed,'seconds':result['elapsed_seconds']}),flush=True)
  del det,data,best_state;import gc;gc.collect()
 except BaseException as e:
  atomic(folder/'failure.json',{'trial':trial,'seed':seed,'type':type(e).__name__,'message':str(e),'traceback':traceback.format_exc()});raise
 atomic(OUT/'progress.json',{'completed_trials':len(results),'total_trials':10,'results':[str(OUT/f"trial{x['trial']}/result.json") for x in results]})
summary={'paper_id':429,'scope':'author-code T-Finance main,10seeds;CPUenvironment adaptation; no clean-label/chronological claim','seeds':[r['seed'] for r in results],'trials':len(results),'metrics':{k:{'mean':float(np.mean([r['metrics'][k] for r in results])),'std_population':float(np.std([r['metrics'][k] for r in results]))} for k in ['AUROC','AUPRC','RecK']},'released_paper_comparison':{k:{'local_mean':float(np.mean([r['metrics'][k] for r in results])),'released_mean':json.loads((OFF/'results/fraudgnam_refined_tfinance.json').read_text())[k+'_mean']} for k in ['AUROC','AUPRC','RecK']},'limitations':['Required baselines,ablations,sensitivity,faithfulness,perturbation,runtime and other auxiliary datasets remain outside this phase.','Retains author-code label-dependent configuration and different paper class-weight definition; not certified leak-free.'],'results':results}
atomic(OUT/'summary.json',summary);print(json.dumps(summary['metrics'],indent=2),flush=True)
