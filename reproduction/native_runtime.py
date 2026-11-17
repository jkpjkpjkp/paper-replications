"""Memory-bounded execution of the released per-sample training objective.

Upstream modules remain byte-identical. Same sampled indices, forward/dropout
order, per-sample losses, reverse branch accumulation and Adam step. Stops on
nonfinite loss: equivalence is not claimed for upstream NaN paths.
"""
import ctypes,gc,hashlib,json,os,random,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
_libc=ctypes.CDLL(None)


def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str));tmp.replace(path)


def release(problem,full=True):
    if not problem._oscar_layers:
        return
    for layer in problem._oscar_layers.values():
        layer.layer_O.info.clear();layer.layer_AR.info.clear()
    gc.collect(2 if full else 0)


def streamed_step(model,problem,loss_fn,optimizer,indices):
    model.train();optimizer.zero_grad()
    predictions=[model(problem.train[i].x).squeeze()for i in indices]
    losses=[None]*len(indices);parts=[None]*len(indices)
    for j in reversed(range(len(indices))):
        s=problem.train[indices[j]]
        loss,part=loss_fn(predictions[j],s.y_norm,s)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'Nonfinite released loss at {s.date}; source NaN path not certified, stopping instead of silently omitting samples')
        losses[j]=loss.detach();parts[j]=part
        (loss/len(indices)).backward()
        del loss
        predictions[j]=None
        release(problem,full=False)
    value=float(torch.stack(losses).nansum()/len(indices))
    optimizer.step();model.eval();release(problem,full=True)
    keys={key for part in parts for key in part}
    return value,{key:sum(part[key]for part in parts if key in part)/sum(key in part for part in parts)for key in keys}


def bounded_train(model,problem,loss_fn,epochs,batchsize,lr,weight_decay,seed,log_prefix='',identity=None):
    optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)
    rng=random.Random(seed)
    identity=identity or {'log_prefix':log_prefix,'seed':seed,'epochs':epochs,'batchsize':batchsize}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    folder=ROOT/'runtime/checkpoints';folder.mkdir(parents=True,exist_ok=True)
    ckpt=folder/f'{fingerprint}.pt';history=[];start=0
    if ckpt.exists():
        saved=torch.load(ckpt,map_location='cpu',weights_only=False)
        assert saved['identity']==identity
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        rng.setstate(saved['sample_rng']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng'])
        history=saved['history'];start=saved['next_epoch']
        print('Resumed',log_prefix,start,flush=True)
    began=time.monotonic()
    for epoch in range(start,epochs):
        indices=rng.sample(range(len(problem.train)),min(batchsize,len(problem.train)))
        loss,parts=streamed_step(model,problem,loss_fn,optimizer,indices)
        history.append({'epoch':epoch+1,'loss':loss,**parts})
        atomic(ROOT/'runtime/training_status.json',{'identity':identity,'epoch':epoch+1,'epochs':epochs,'loss':loss,'elapsed_seconds':time.monotonic()-began})
        if(epoch+1)%10==0 or epoch==start:
            print(f'[{log_prefix}] epoch={epoch+1}/{epochs} loss={loss:.8f} elapsed={time.monotonic()-began:.1f}s',flush=True)
        if(epoch+1)%25==0 or epoch+1==epochs:
            tmp=ckpt.with_suffix('.pt.tmp')
            torch.save({'identity':identity,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                        'sample_rng':rng.getstate(),'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                        'history':history,'next_epoch':epoch+1},tmp);tmp.replace(ckpt)
            atomic(folder/f'{fingerprint}.history.json',history)
        _libc.malloc_trim(0)
    return model.eval(),history
