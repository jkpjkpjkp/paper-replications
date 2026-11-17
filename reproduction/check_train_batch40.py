"""Check the first actual shuffled training batch against the original loader."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from data_provider.data_loader import Dataset_Sig, PrecomputedSigDataset
from model.SIT import Model

torch.set_num_threads(2)
torch.manual_seed(2023)
cfg=SimpleNamespace(data_pool=40,d_model=8,dropout=.1,ff_dim=32,n_heads=4,
    num_layers=2,sig_input_dim=2,cross_sig_dim=1,hidden_c=8,max_position=5,time_feat_dim=3)
model=Model(cfg)
data=pd.read_csv(ROOT/'asset_data/full_dataset.csv',parse_dates=['Date']).set_index('Date').iloc[:,:40].loc['2000':'2016']
original=Dataset_Sig(data,data,60,20,sig_scaler=joblib.load(ROOT/'signature_cache_6020/pool_40/signature_scaler.pkl'))
cached=PrecomputedSigDataset(str(ROOT/'signature_cache_6020/pool_40'),'train')
# Same default generator and sampler setup as the released training loader.
loader=DataLoader(TensorDataset(torch.arange(len(cached))),batch_size=64,
                  shuffle=True,num_workers=10,drop_last=True)
iterator=iter(loader)
indices=next(iterator)[0].tolist()
del iterator
errors={k:0. for k in ['x_sigs','cross_sigs','date_feats','future_return_unscaled']}
tick=time.perf_counter()
for i,index in enumerate(indices):
    a,b=original[index],cached[index]
    assert a['dates_horizon']==b['dates_horizon']
    for k in errors:
        errors[k]=max(errors[k],float(np.max(np.abs(a[k]-b[k]))))
        assert np.array_equal(a[k],b[k]),(index,k)
    if (i+1)%16==0:print('Training samples checked',i+1,flush=True)
state_hash=hashlib.sha256()
for k,v in model.state_dict().items():
    state_hash.update(k.encode());state_hash.update(v.numpy().tobytes())
result=dict(indices=indices,initial_model_state_sha256=state_hash.hexdigest(),
            max_array_errors=errors,seconds=time.perf_counter()-tick,
            scope='First shuffled batch under released 40-asset setup and local torch2.4.1; not an exhaustive full-cache verification.')
(ROOT/'reproduction/discrepancy40/first_training_batch_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
