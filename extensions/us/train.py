"""Train the pinned LiMT core on the locked US panel, with resumable seeds."""
from pathlib import Path
import argparse
import json
import random
import time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch

from limt.model import LiMT
from usdata import ROOT,Panel,config,sha,write_json


def corr(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    x,y=x[mask],y[mask]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and x.std()>0 and y.std()>0 else np.nan


def loss_fn(pred,target,weights):
    valid=torch.isfinite(target)
    return (((pred-torch.nan_to_num(target)).square()*valid).sum(0)/valid.sum(0).clamp_min(1)*weights).sum()


@torch.no_grad()
def evaluate(model,panel,days,device,save=False):
    model.eval()
    rows=[];scores=[]
    for t in days:
        ids,x,_,raw=panel.day(t)
        pred=model(torch.from_numpy(x).to(device)).cpu().numpy()
        scores.append([corr(pred[:,k],raw[:,k]) for k in range(3)])
        if save:
            rows.append(pd.DataFrame(dict(date=panel.dates[t],realized_date=panel.dates[t+2],
                symbol=panel.symbols[ids],pred_return=pred[:,0],pred_volume=pred[:,1],pred_volatility=pred[:,2],
                label_return=raw[:,0],label_volume=raw[:,1],label_volatility=raw[:,2])))
    a=np.asarray(scores)
    metrics={name:dict(IC=float(np.nanmean(a[:,k])),ICIR=float(np.nanmean(a[:,k])/np.nanstd(a[:,k],ddof=1)),
                      days=int(np.isfinite(a[:,k]).sum())) for k,name in enumerate(['return','volume','volatility'])}
    return metrics,pd.concat(rows,ignore_index=True) if save else None


def run(seed):
    p=config();settings=p['training']
    out=ROOT/'results'/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    lock=dict(protocol_sha256=sha(ROOT/'protocol.json'),data_manifest_sha256=sha(ROOT/'data/manifest.json'),
        model_sha256=sha(ROOT/'limt/model.py'),author_code_sha256=sha(ROOT/'upstream/limt.py'),
        trainer_sha256=sha(__file__),usdata_sha256=sha(ROOT/'usdata.py'),seed=seed)
    if (out/'lock.json').exists():
        assert json.loads((out/'lock.json').read_text())==lock,'Run inputs changed; refusing to reuse checkpoint'
    write_json(out/'lock.json',lock)
    if (out/'status.json').exists() and json.loads((out/'status.json').read_text()).get('phase')=='complete':
        print('Already complete',seed,flush=True);return
    panel=Panel();days={s:panel.split(s) for s in ['train','valid','test']}
    summary={s:dict(days=len(ix),first_signal=str(panel.dates[ix[0]]),last_signal=str(panel.dates[ix[-1]]),
        last_realization=str(panel.dates[ix[-1]+2]),stocks_per_day=[int(panel.eligible[:,ix].sum(0).min()),int(panel.eligible[:,ix].sum(0).max())]) for s,ix in days.items()}
    write_json(out/'data_summary.json',summary)
    assert summary['train']['last_realization']<=p['train'][1]<summary['valid']['first_signal']
    assert summary['valid']['last_realization']<=p['valid'][1]<summary['test']['first_signal']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    torch.cuda.set_per_process_memory_fraction(.4)
    device=torch.device('cuda')
    m=p['model']
    model=LiMT(d_model=m['d_model'],cross_heads=m['cross_heads'],temporal_heads=m['temporal_heads'],
               experts=m['experts'],dropout=m['dropout'],variant=m['variant']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
    weights=torch.tensor(settings['weights'],device=device)
    history=[];best=-np.inf;stale=0;start=0;best_epoch=0
    latest=out/'latest.pt'
    if latest.exists():
        ck=torch.load(latest,map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer'])
        best,stale,start,history,best_epoch=[ck[k] for k in ['best','stale','epoch','history','best_epoch']]
        torch.set_rng_state(ck['rng'].cpu());torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),device)
    write_json(out/'status.json',dict(phase='training',seed=seed,start_epoch=start,updated_utc=datetime.now(timezone.utc).isoformat()))
    for epoch in range(start,settings['epochs']):
        if stale>=settings['patience']:break
        begin=time.perf_counter();model.train();losses=[]
        for t in np.random.default_rng(seed+epoch).permutation(days['train']):
            _,x,y,_=panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            pred=model(torch.from_numpy(x).to(device))
            loss=loss_fn(pred,torch.from_numpy(y).to(device),weights)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_value_(model.parameters(),settings['gradient_value_clip'])
            optimizer.step();losses.append(loss.item())
        metrics,_=evaluate(model,panel,days['valid'],device)
        valid=metrics['return']['IC']
        assert np.isfinite(valid)
        if valid>best:
            best=valid;stale=0;best_epoch=epoch+1
            torch.save(model.state_dict(),out/'best.pt')
        else:stale+=1
        rec=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=metrics,
                 seconds=time.perf_counter()-begin,best_epoch=best_epoch,stale=stale)
        history.append(rec)
        write_json(out/'history.json',history)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),best=best,stale=stale,
            epoch=epoch+1,history=history,best_epoch=best_epoch,rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state()),latest)
        write_json(out/'status.json',dict(phase='training',seed=seed,**rec))
        print(json.dumps(dict(seed=seed,**rec)),flush=True)
    model.load_state_dict(torch.load(out/'best.pt',map_location=device,weights_only=True))
    results={}
    for split in ['valid','test']:
        metrics,pred=evaluate(model,panel,days[split],device,save=True)
        ric=pred.groupby('date',sort=True).apply(lambda a:a.pred_return.corr(a.label_return,method='spearman'),include_groups=False)
        metrics['return'].update(RankIC=float(ric.mean()),RankICIR=float(ric.mean()/ric.std(ddof=1)))
        pred.to_parquet(out/f'{split}_predictions.parquet',index=False)
        results[split]=metrics
    write_json(out/'metrics.json',results)
    write_json(out/'provenance.json',dict(**lock,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        parameters=sum(a.numel() for a in model.parameters()),best_sha256=sha(out/'best.pt')))
    write_json(out/'status.json',dict(phase='complete',seed=seed,epochs=len(history),best_epoch=best_epoch,
        stop_reason='early_stopping' if stale>=settings['patience'] else 'epoch_limit',best_validation_ic=best,
        updated_utc=datetime.now(timezone.utc).isoformat()))
    print('COMPLETE',seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True,choices=[0,1,2])
    run(parser.parse_args().seed)
