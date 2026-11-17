"""Native full-task capacity and runtime probe. No experiment result is selected."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import importlib.util,json,sys,time,copy
from pathlib import Path
import torch,numpy as np,qlib,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gap-resolution-20261113';REPO=ROOT/'sources/DoubleAdapt-qlib'
sys.path.insert(0,str(REPO));torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
orig=qlib.init
def init(*a,**kw):kw.update(kernels=2,joblib_backend='threading',expression_cache=None,dataset_cache=None);return orig(*a,**kw)
qlib.init=init
spec=importlib.util.spec_from_file_location('native_main',REPO/'examples/benchmarks_dynamic/incremental/main.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
np.random.seed(0);torch.manual_seed(43)
cfg=dict(data_dir='crowd_data',root_path=str(ROOT/'data')+'/',market='csi300',forecast_model='GRU',alpha=360,step=20,rank_label=False,num_head=8,tau=10,lr=.001,lr_da=.01,online_lr=None,begin_valid_epoch=30,preprocess_tensor=True,save=False,tag='cuda-probe',h_path=str(ROOT/'results/doubleadapt-original/handler.pkl'))
exp=mod.Incremental(**cfg);exp.meta_dataset_offline,exp.meta_dataset_online=exp.dump_data()

calendar=pd.DatetimeIndex(pd.to_datetime((ROOT/'data/crowd_data/calendars/day.txt').read_text().splitlines()))
def end_label(index):
 dates=index.get_level_values('datetime')
 pos=calendar.get_indexer(dates)
 assert (pos>=0).all()
 return calendar[pos+2].max()
def dates(index):
 d=index.get_level_values('datetime');return str(d.min().date()),str(d.max().date())
summary={};train_last=[]
for phase, tasks in [('offline_train',exp.meta_dataset_offline.prepare_tasks('train')),('offline_validation',exp.meta_dataset_offline.prepare_tasks('test')),('online',exp.meta_dataset_online.prepare_tasks('test'))]:
 bounds=[]
 for i, task in enumerate(tasks):
  mi=task.get_meta_input();support=mi['train_idx'];query=mi['test_idx']
  support_label_end=end_label(support);first_forecast=query.get_level_values('datetime').min()
  assert support_label_end <= first_forecast, (phase,i,support_label_end,first_forecast)
  meta_index=query[:mi['meta_end']];meta_label_end=end_label(meta_index)
  if phase!='offline_train' and i+1<len(tasks):
   next_first=tasks[i+1].get_meta_input()['test_idx'].get_level_values('datetime').min()
   assert meta_label_end <= next_first, (phase,i,meta_label_end,next_first)
  bounds.append(dict(task=i,support=dates(support),query=dates(query),support_last_outcome=str(support_label_end.date()),meta_last_outcome=str(meta_label_end.date())))
  if phase=='offline_train':train_last.append(end_label(query))
 summary[phase]=dict(tasks=len(tasks),first=bounds[0],last=bounds[-1])
 (OUT/(phase+'-temporal-boundaries.json')).write_text(json.dumps(bounds,indent=2))
last_training_outcome=max(train_last)
summary.update(passed=True,last_offline_training_outcome=str(last_training_outcome.date()),native_first_validation_forecast=summary['offline_validation']['first']['query'][0],strict_first_validation_forecast='2015-01-05',interpretation='Native validation includes one 2014-12-31 forecast before the last offline training outcome (2015-01-05) is known. Strict validation-rate scoring excludes it. Later validation and online support/meta updates satisfy endpoint availability at or before the next forecast date, consistent with the released next-session execution lag.')
assert last_training_outcome <= pd.Timestamp('2015-01-05')
(OUT/'doubleadapt-temporal-contract.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
