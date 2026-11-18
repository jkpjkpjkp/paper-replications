#!/usr/bin/env python3
"""Independent checks for adjustment, split isolation, and recovery/autodiff."""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def check_features():
    from build_tensors import FEATURES,features_from_grid,eligible_date
    grid=np.full((3,390,7),12.,dtype=float)
    grid[:,:,4]=1200;grid[:,:,5]=100;grid[:,:,6]=10
    x=features_from_grid(grid)
    assert x.shape==(3,26,72) and np.isfinite(x).all()
    for field in ['open','high','low','close','amount','volume','trade_count']:
        for name in ['change_rate','normalized_range','std','skew','excess_kurtosis']:
            np.testing.assert_allclose(x[:,:,FEATURES.index(field+'_'+name)],0,atol=1e-7)
    np.testing.assert_allclose(x[:,:,FEATURES.index('vwap_to_mean_close')],0,atol=1e-7)
    grid[:,7,0:4]=np.nan
    missing=features_from_grid(grid)
    np.testing.assert_allclose(missing[:,0,FEATURES.index('missing_price_fraction')],1/15,atol=1e-7)
    cal={str(r['session_date']):r for r in pq.read_table(ROOT/'support/trading_calendar.parquet').to_pylist()}
    # June 30 also precedes the July 3 early close, so either exclusion applies.
    assert not eligible_date('2023-06-30',cal)[0]
    assert eligible_date('2024-06-28',cal)==(False,'split_boundary_purge')
    assert eligible_date('2025-07-31',cal)==(True,'2025-08-01')
    assert eligible_date('2024-07-03',cal)==(False,'short_input_or_label_session')
    return '72 feature descriptors, missingness flags, and date-boundary purge verified'

def check_actual_split():
    # Nvidia's June 2024 split gives a strong check against a spurious -90% label.
    aux=pq.read_table(ROOT/'support/crsp_daily_auxiliary.parquet',columns=['dlycaldt','permno','ticker','dlycumfacpr']).to_pydict()
    day='2024-06-07';nextday='2024-06-10';factors={};permno=None
    for d,p,t,f in zip(aux['dlycaldt'],aux['permno'],aux['ticker'],aux['dlycumfacpr']):
        if str(d) in [day,nextday] and t=='NVDA':
            factors[str(d)]=float(f);permno=p
    assert set(factors)=={day,nextday} and factors[day]/factors[nextday]>2
    closes=[]
    for date in [day,nextday]:
        bar=pq.read_table(ROOT/f'bars/{date[:4]}/{date}.parquet',columns=['permno','minute_index','close']).to_pydict()
        selected=[c for p,m,c in zip(bar['permno'],bar['minute_index'],bar['close']) if p==permno and m==389]
        assert len(selected)==1 and selected[0]>0;closes.append(selected[0])
    expected=(closes[1]/factors[nextday])/(closes[0]/factors[day])-1
    with np.load(HERE/'cache'/f'{day}.npz') as data:
        index=np.flatnonzero(data['permno']==permno);assert len(index)==1
        actual=float(data['Y'][index[0],389])
    np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=1e-6)
    assert abs(actual-(closes[1]/closes[0]-1))>.5
    return {'real_split_adjusted_label':actual,'naive_unadjusted_label':closes[1]/closes[0]-1}

def check_reconnect():
    spec=importlib.util.spec_from_file_location('tested_download',ROOT/'download_minutes.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);source=root/'fixture.parquet'
        pq.write_table(pa.table({'permno':[1],'minute_index':[0],'amount':[10.0],'volume':[1],'trade_count':[1]}),source)
        info={'state':'complete','name':'fixture.parquet','path':'fixture.parquet','bytes':source.stat().st_size,
              'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'copy_rows':1,
              'totals':{'rows':1,'volume':1,'trades':1,'amount':10.0},'elapsed_seconds':0}
        instances=[]
        class FakeBridge:
            def __init__(self,*args):self.token='fixture-token';self.number=len(instances);instances.append(self)
            def connect(self):self.username='fixture';return self
            def execute(self,*args,**kwargs):return ''
        def call(bridge,function,*args):
            if function=='status' and bridge.number==0:raise ConnectionError('simulated lost socket')
            return info if function=='status' else {'ok':True}
        tasks=queue.Queue();tasks.put((dt.date(2020,1,2),'16:00:00',[{'permno':1,'sym_root':'FIX','sym_suffix':None}]))
        progress={'completed':0};worker=root/'worker.py';worker.write_text('# fixture\n')
        with patch.object(m,'ROOT',root),patch.object(m,'Bridge',FakeBridge),patch.object(m,'remote',call),\
             patch.object(m,'close_bridge',lambda b:None),patch.object(m,'download',lambda b,i,d:shutil.copyfile(source,d)),\
             patch.object(m.stop_event,'wait',return_value=False):
            m.run_worker(1,tasks,progress,1,'fixture-hash',worker)
        assert len(instances)==2 and progress['completed']==1 and tasks.empty()
        receipt=json.loads((root/'bars/2020/2020-01-02.receipt.json').read_text())
        assert receipt['sha256']==info['sha256'] and not (root/'bars/2020/2020-01-02.error.json').exists()
    return 'Lost socket causes a fresh kernel and exactly one checksum-verified completion'

def check_hypergradient():
    import torch
    from torch.func import functional_call
    from train_lstm import batch_loss,daily_loss
    torch.manual_seed(7)
    model=torch.nn.Linear(3,1).double();xs=torch.randn(30,3,dtype=torch.float64);xq=torch.randn(31,3,dtype=torch.float64)
    ys=[{'Y':torch.randn(30,2,dtype=torch.float64)}];yq=[{'Y':torch.randn(31,1,dtype=torch.float64)}]
    def objective(logits):
        parameters=dict(model.named_parameters());weights=torch.softmax(logits,dim=0)
        inner=batch_loss(model(xs).squeeze(-1),ys,weights=weights)
        gradients=torch.autograd.grad(inner,tuple(parameters.values()),create_graph=True)
        fast={n:p-0.01*g for (n,p),g in zip(parameters.items(),gradients)}
        return batch_loss(functional_call(model,fast,(xq,)).squeeze(-1),yq,horizon=0)
    logits=torch.tensor([.2,-.1],dtype=torch.float64,requires_grad=True)
    analytic=torch.autograd.grad(objective(logits),logits)[0].detach().numpy();numeric=[]
    for i in range(2):
        delta=torch.zeros(2,dtype=torch.float64);delta[i]=1e-4
        numeric.append(float((objective(logits+delta)-objective(logits-delta))/(2e-4)))
    np.testing.assert_allclose(analytic,numeric,rtol=1e-4,atol=1e-8)
    assert np.linalg.norm(analytic)>0
    constant_loss,_=daily_loss(torch.ones(31,dtype=torch.float64),yq[0]['Y'])
    np.testing.assert_allclose(constant_loss.detach().numpy(),[2.0],atol=1e-12)
    return {'analytic':analytic.tolist(),'finite_difference':numeric,'constant_prediction_loss':float(constant_loss[0])}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--features',action='store_true')
    parser.add_argument('--download',action='store_true');parser.add_argument('--model',action='store_true')
    args=parser.parse_args();path=HERE/'engineering_checks.json'
    result=json.loads(path.read_text()) if path.exists() else {}
    if args.features:
        result['features_and_boundaries']=check_features();result['actual_split_label']=check_actual_split()
    if args.download:result['connection_recovery']=check_reconnect()
    if args.model:result['hypergradient']=check_hypergradient()
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
