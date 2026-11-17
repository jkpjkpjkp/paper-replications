"""Native full-task capacity and runtime probe. No experiment result is selected."""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
import importlib.util,json,sys,time,copy,argparse
from pathlib import Path
import torch,numpy as np,qlib
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gap-resolution-20261113';REPO=ROOT/'sources/DoubleAdapt-qlib'
sys.path.insert(0,str(REPO));torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
orig=qlib.init
def init(*a,**kw):kw.update(kernels=2,joblib_backend='threading',expression_cache=None,dataset_cache=None);return orig(*a,**kw)
qlib.init=init
spec=importlib.util.spec_from_file_location('native_main',REPO/'examples/benchmarks_dynamic/incremental/main.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
np.random.seed(0);torch.manual_seed(43)
cfg=dict(data_dir='crowd_data',root_path=str(ROOT/'data')+'/',market='csi300',forecast_model='GRU',alpha=360,step=5,rank_label=False,num_head=8,tau=10,lr=.001,lr_da=.01,online_lr=None,begin_valid_epoch=30,preprocess_tensor=True,save=False,tag='cuda-step5-probe',h_path=str(ROOT/'results/doubleadapt-original/handler.pkl'))
exp=mod.Incremental(**cfg);exp.meta_dataset_offline,exp.meta_dataset_online=exp.dump_data()
tasks=exp.meta_dataset_offline.prepare_tasks('train');task=max(tasks,key=lambda t:len(t.get_meta_input()['y_train'])+len(t.get_meta_input()['y_test']));mi=task.get_meta_input()
manager=mod.DoubleAdaptManager(exp.basic_task,x_dim=exp.x_dim,lr_model=exp.lr,weight_decay=exp.weight_decay,first_order=exp.first_order,alpha=exp.alpha,begin_valid_epoch=exp.begin_valid_epoch,factor_num=exp.factor_num,lr_da=exp.lr_da,lr_ma=exp.lr_ma,online_lr=exp.online_lr,lr_x=exp.lr_x,lr_y=exp.lr_y,adapt_x=exp.adapt_x,adapt_y=exp.adapt_y,reg=exp.reg,num_head=exp.num_head,temperature=exp.temperature)
assert manager.framework.device.type=='cuda'
parser=argparse.ArgumentParser();parser.add_argument('--after-tra',action='store_true');args=parser.parse_args()
if args.after_tra:
 assert (OUT/'tra-calendar-corrected/summary.json').exists()
 previous=OUT/'cuda-step5-probe.json'
 if previous.exists(): (OUT/'cuda-step5-first-probe.json').write_bytes(previous.read_bytes())
torch.cuda.set_per_process_memory_fraction(0.30 if args.after_tra else 0.15)
manager.framework.train();torch.cuda.reset_peak_memory_stats();t=time.monotonic()
try:
 for i in range(3):
  output=manager.run_task(mi,'train');assert np.isfinite(output).all();print('native full-task pass',i+1,flush=True)
 torch.cuda.synchronize();result=dict(passed=True,torch=torch.__version__,qlib=qlib.__version__,device=str(manager.framework.device),support=list(mi['X_train'].shape),query=list(mi['X_test'].shape),seconds_per_task=(time.monotonic()-t)/3,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),native_full_tasks=True)
except Exception as e:
 result=dict(passed=False,error=repr(e),torch=torch.__version__,peak_allocated_bytes=torch.cuda.max_memory_allocated())
(OUT/'cuda-step5-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
