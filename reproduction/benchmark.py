"""Real-input model timing and causality preflight; not a financial result."""
import argparse
import json
import resource
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_provider.data_loader import Dataset_Sig
from model.SIT import Model
from exp.exp_main import decision_focused_cvar_loss

parser=argparse.ArgumentParser()
parser.add_argument('--device',default='cpu')
parser.add_argument('--batch',type=int,default=8)
parser.add_argument('--pool',type=int,default=40)
parser.add_argument('--activation-offload',action='store_true')
parser.add_argument('--state-output',type=Path)
args=parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(2023)
p=args.pool
cfg=SimpleNamespace(data_pool=p,d_model=8,dropout=.1,ff_dim=32 if p==40 else 8,
                    n_heads=4 if p==40 else 8,num_layers=2,sig_input_dim=2,
                    cross_sig_dim=1,hidden_c=8 if p==40 else 64,max_position=5,
                    time_feat_dim=3)
df=pd.read_csv(ROOT/'asset_data/full_dataset.csv',parse_dates=['Date']).set_index('Date').iloc[:,:p]
ds=Dataset_Sig(df.loc['2000':'2016'],df.loc['2000':'2016'],60,20)
started=time.perf_counter()
sample=ds[100]
construction=time.perf_counter()-started
batch={k:torch.tensor(sample[k]).unsqueeze(0).repeat(args.batch,*([1]*np.ndim(sample[k]))).to(args.device)
       for k in ['x_sigs','cross_sigs','date_feats','future_return_unscaled']}
model=Model(cfg).to(args.device)
optimizer=torch.optim.Adam(model.parameters(),lr=.001)
durations=[]
for i in range(4):
    tick=time.perf_counter()
    optimizer.zero_grad()
    with torch.autograd.graph.save_on_cpu(pin_memory=False) if args.activation_offload else nullcontext():
        loss,_=decision_focused_cvar_loss(model,batch['x_sigs'],batch['cross_sigs'],
                                         batch['date_feats'],batch['future_return_unscaled'],temperature=1.3)
    loss.backward();optimizer.step()
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    durations.append(time.perf_counter()-tick)
model.eval()
with torch.no_grad():
    base=model(batch['x_sigs'],batch['cross_sigs'],batch['date_feats'])[1]
    x=batch['x_sigs'].clone();c=batch['cross_sigs'].clone()
    x[:,1:]*=3;c[:,1:]*=2
    changed=model(x,c,batch['date_feats'])[1]
    error=float((base[:,0]-changed[:,0]).abs().max())
assert error==0
result={'device':args.device,'pool':p,'batch':args.batch,'torch':torch.__version__,
        'activation_offload':args.activation_offload,
        'cpu_threads':torch.get_num_threads(),'sample_build_seconds':construction,
        'train_step_seconds':durations,'median_warm_train_step_seconds':float(np.median(durations[1:])),
        'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'prefix_causality_max_error':error,'train_samples':len(ds),
        'note':'Untrained real-input timing/engineering check. Four steps are not a financial replication; no performance is claimed.'}
if args.device.startswith('cuda'):
    try:
        result['peak_cuda_mib']=torch.cuda.max_memory_allocated()/2**20
    except RuntimeError:
        result['peak_cuda_mib']=None  # Non-caching managed allocator reports its own peak.
if args.state_output:
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()}, args.state_output)
suffix='_offload' if args.activation_offload else ''
(ROOT/'reproduction'/f'benchmark_{args.device.replace(":","_")}_{p}{suffix}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
