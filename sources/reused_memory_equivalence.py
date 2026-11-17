"""Validate a streamed implementation of the same finite-sample DFL update.

Only diagnostic updates, no checkpoints. Source train_model is the reference.
The same sampled indices, dropout RNG state, loss, solver settings, optimizer,
and effective batch are retained. Nonfinite losses stop the wrapper rather
than silently changing the source failure policy.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import resource
import sys
import time

HERE=Path(__file__).resolve().parent
UPSTREAM=Path('/home/jkp/data/sparse-tangent-reproduction')


def atomic(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');temp.replace(path)


def rss():
    fields=dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
    return int(fields['VmHWM'].split()[0])*1024


def streamed_update(model,problem,loss_fn,batchsize,seed):
    import torch
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    model.train()
    indices=random.Random(seed).sample(range(len(problem.train)),min(batchsize,len(problem.train)))
    optimizer.zero_grad()
    # Preserve source forward/dropout order while retaining only the small MLP
    # graphs. Release each large conic derivative context after its backward.
    # Reverse order follows source autograd's branch accumulation order.
    predictions=[model(problem.train[i].x).squeeze() for i in indices]
    losses=[None]*len(indices)
    for j in reversed(range(len(indices))):
        i=indices[j]
        sample=problem.train[i]
        prediction=predictions[j]
        loss,_=loss_fn(prediction,sample.y_norm,sample)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite loss: source nansum/failure semantics require separate validation; no sample omitted.')
        losses[j]=loss.detach()
        (loss/len(indices)).backward()
        # Source differentiable contexts are released before the next sample.
        del loss,prediction
        predictions[j]=None
    loss_value=float(torch.stack(losses).nansum()/len(indices))
    optimizer.step()
    model.eval()
    return loss_value,indices


def compare(market):
    sys.path.insert(0,str(UPSTREAM/'src'))
    import numpy as np
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    from main import build_argparser,make_problem,new_model
    from losses import get_loss_fn
    from train import train_model
    from utils import set_seed,resolve_k_from_percent
    args=build_argparser().parse_args(['--mode','dfl','--dataset',market,'--hidden_dims','512','256','--batchsize','64','--cuda','-1'])
    problem=make_problem(args,UPSTREAM,torch.device('cpu'))
    # Reject an unreviewed cross-sample architecture.
    sample_model=new_model(problem,args,torch.device('cpu'))
    allowed=(torch.nn.Sequential,torch.nn.Linear,torch.nn.Tanh,torch.nn.Dropout)
    assert all(isinstance(module,allowed) for module in sample_model.modules())
    del sample_model
    result={'market':market,'purpose':'implementation_equivalence_only','n_assets':problem.n_assets,
            'batch_dependent_operations':False,'source_architecture':'Linear/Tanh/Dropout only',
            'source_loss':'Sum of independent per-sample mixed losses, divided by finite sample count',
            'implementation':'Source forward/dropout order; reverse per-sample solve/backward accumulation; same single Adam step',
            'nonfinite_policy':'Stop and report; no claim of equivalence for source NaN paths',
            'reference':'Unchanged src/train.py:train_model, one update',
            'same_source_solver_options':True,'equivalence_checks':[],'full_batch_runs':[],
            'started_at':datetime.now(timezone.utc).isoformat()}
    alphas=(0.,.1,.5,.9,1.) if market=='euro50' else (.5,)
    reference_batch=8 if market=='nikkei225' else 64
    for alpha in alphas:
        for percent in (10,15,20):
            k=resolve_k_from_percent(percent,problem.n_assets)
            set_seed(0)
            base=new_model(problem,args,torch.device('cpu'))
            initial=copy.deepcopy(base.state_dict())
            rng=torch.random.get_rng_state().clone()
            np_rng=copy.deepcopy(np.random.get_state())
            loss_fn=get_loss_fn('dfl',problem,k=k,alpha=alpha)
            Path('/proc/self/clear_refs').write_text('5\n')
            start=time.monotonic()
            reference,history=train_model(base,problem,loss_fn,1,reference_batch,.001,.0001,0,'EQUIVALENCE REFERENCE')
            reference_seconds=time.monotonic()-start
            reference_peak=rss()
            reference_state={n:p.detach().clone() for n,p in reference.named_parameters()}
            reference_grad={n:p.grad.detach().clone() for n,p in reference.named_parameters()}
            reference_rng=torch.random.get_rng_state().clone()
            reference_np=copy.deepcopy(np.random.get_state())
            del reference,base
            gc.collect()
            streamed=new_model(problem,args,torch.device('cpu'))
            streamed.load_state_dict(initial)
            torch.random.set_rng_state(rng);np.random.set_state(np_rng)
            Path('/proc/self/clear_refs').write_text('5\n')
            start=time.monotonic()
            stream_loss,indices=streamed_update(streamed,problem,loss_fn,reference_batch,0)
            stream_seconds=time.monotonic()-start
            stream_peak=rss()
            errors={}
            passed=True
            for kind,expected in [('gradient',reference_grad),('updated_parameter',reference_state)]:
                absolute=0.;squared_error=0.;squared_reference=0.
                for name,param in streamed.named_parameters():
                    observed=param.grad if kind=='gradient' else param
                    delta=(observed.detach()-expected[name]).double()
                    absolute=max(absolute,float(delta.abs().max()))
                    squared_error+=float(delta.square().sum())
                    squared_reference+=float(expected[name].double().square().sum())
                    passed &= bool(torch.allclose(observed.detach(),expected[name],rtol=2e-4,atol=2e-6))
                errors[kind]={'max_absolute_error':absolute,'relative_l2_error':(squared_error/max(squared_reference,1e-30))**.5}
            same_rng=bool(torch.equal(torch.random.get_rng_state(),reference_rng))
            current_np=np.random.get_state()
            same_np=(current_np[0]==reference_np[0] and np.array_equal(current_np[1],reference_np[1]) and current_np[2:]==reference_np[2:])
            loss_error=abs(stream_loss-history[0]['loss'])
            passed &= same_rng and same_np and loss_error <= 2e-6*max(1.,abs(history[0]['loss']))
            check={'alpha':alpha,'k_percent':percent,'k':k,'effective_batch':reference_batch,
                   'passed':bool(passed),'same_dropout_rng_after_update':same_rng,'same_numpy_rng_after_update':bool(same_np),
                   'reference_seconds':reference_seconds,'stream_seconds':stream_seconds,
                   'reference_peak_rss_bytes':reference_peak,'stream_peak_rss_bytes':stream_peak,
                   'reference_loss':history[0]['loss'],'stream_loss':stream_loss,'loss_absolute_error':loss_error,
                   'parameter_and_gradient_errors':errors,'sample_indices':indices,
                   'tolerance':{'rtol':2e-4,'atol':2e-6}}
            result['equivalence_checks'].append(check)
            atomic(HERE/f'accumulation_{market}.json',result)
            del streamed,initial,reference_grad,reference_state
            gc.collect()
            if not passed:
                raise RuntimeError('Equivalence failed; full-batch continuation refused')
    if market=='nikkei225':
        for percent in (10,15,20):
            k=resolve_k_from_percent(percent,problem.n_assets)
            set_seed(0);model=new_model(problem,args,torch.device('cpu'))
            Path('/proc/self/clear_refs').write_text('5\n');start=time.monotonic()
            loss,indices=streamed_update(model,problem,get_loss_fn('dfl',problem,k=k,alpha=.5),64,0)
            result['full_batch_runs'].append({'k_percent':percent,'k':k,'alpha':.5,'effective_batch':64,
                                              'seconds':time.monotonic()-start,'peak_rss_bytes':rss(),
                                              'finite_loss':bool(np.isfinite(loss)),'loss':loss,
                                              'sample_indices':indices,'optimizer_steps':1,
                                              'direct_full64_reference_comparison':False,
                                              'qualification':'Gradient linearity plus full64 equivalence on Euro/FTSE and batch8 equivalence on this panel; full64 reference stalled under guard.'})
            atomic(HERE/f'accumulation_{market}.json',result)
            del model;gc.collect()
    result.update(passed=all(x['passed'] for x in result['equivalence_checks']),
                  finished_at=datetime.now(timezone.utc).isoformat(),peak_process_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    atomic(HERE/f'accumulation_{market}.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--market',required=True,choices=('euro50','ftse100','nikkei225'))
    compare(parser.parse_args().market)
