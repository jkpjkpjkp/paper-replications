#!/usr/bin/env python3
"""Training-only full-size validation; temporary scaler and no fitted outputs."""
import datetime as dt
import json
from pathlib import Path
import resource
import tempfile
import time
import numpy as np
import torch
from core import supervised_step,bilevel_step
from data import DailyStore,save_json
from train import new_model,adamw

HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2)
    store=DailyStore(HERE.parent,'cpu',require_complete=False,max_cached_days=20)
    dates=store.splits['train'][:20]
    if len(dates)!=20:raise RuntimeError('Twenty rebuilt training dates required')
    store.splits={'train':dates,'validation':[],'test':[]}
    with tempfile.TemporaryDirectory() as tmp:store.fit_scaler(Path(tmp)/'temporary_scaler.json')
    days=store.days(dates)
    assert all(x.dtype==y.dtype==torch.float64 and torch.isfinite(x).all() and torch.isfinite(y).all() for x,y in days)
    config=json.loads((HERE/'protocol.json').read_text());checks={}
    for name in ['LSTM','GRU','TCN']:
        config['model']['name']=name
        model=new_model(config,0,'cpu');start=time.monotonic()
        supervised=supervised_step(model,days,adamw(model,config),389)
        logits=torch.nn.Parameter(torch.zeros(390,dtype=torch.double))
        meta=bilevel_step(model,days[:10],days[10:],logits,torch.optim.SGD([logits],lr=.001),1e-6,.001,389)
        assert all(np.isfinite(v) for v in meta.values()) and torch.isfinite(logits).all()
        assert meta['hypergradient_norm']>0 and float(logits.abs().max())>0
        checks[name]={'passed':True,'supervised_loss':supervised,**meta,'seconds':time.monotonic()-start}
        print(name,json.dumps(checks[name]),flush=True)
    save_json(HERE/'real_batch_checks.json',{'status':'passed','checked_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'scope':'twenty training dates only; temporary scaler; no validation/test or primary fitted checkpoints',
        'device':'cpu','dtype':'float64','threads':2,'dates':dates,'stocks_per_date':[len(y) for x,y in days],
        'shape_per_stock':[26,66],'horizons':390,'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'backbones':checks})

if __name__=='__main__':main()
