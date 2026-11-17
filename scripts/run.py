"""Run six matched fits on real local prices, writing every daily prediction."""
from pathlib import Path
import copy, hashlib, json, time
import numpy as np
import torch
from model import Model
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2)
    cfg=json.loads((ROOT/'protocol.json').read_text())
    z=np.load(ROOT/'data/price_only.npz')
    origins={k:z[k] for k in ['train','valid','test']}
    x=np.stack([z['x'][t-6:t+1].transpose(1,0,2) for t in range(6,len(z['dates']))])
    X=torch.from_numpy(x)
    Y=torch.from_numpy(np.stack((z['yret'],z['yrisk']),axis=-1))
    MASK=torch.from_numpy(z['mask'])
    result=ROOT/'results';result.mkdir(exist_ok=True)
    lock={'protocol_sha256':hashlib.sha256((ROOT/'protocol.json').read_bytes()).hexdigest(),
          'data_sha256':hashlib.sha256((ROOT/'data/price_only.npz').read_bytes()).hexdigest(),
          'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/model.py',ROOT/'scripts/prepare.py']}}
    lp=result/'run_lock.json'
    if lp.exists():assert json.loads(lp.read_text())==lock,'Inputs changed; archive results before rerun'
    else:lp.write_text(json.dumps(lock,indent=2)+'\n')
    def batch(ids):return X[ids-6],Y[ids],MASK[ids]
    for seed in cfg['seeds']:
        for variant in ['return_only','multitask']:
            dest=result/f'{variant}_seed{seed}';dest.mkdir(exist_ok=True)
            if (dest/'predictions.npz').exists():continue
            started=time.time();torch.manual_seed(seed)
            model=Model(X.shape[-1],hidden=cfg['hidden_per_direction'],head=cfg['heads'],dropout=cfg['dropout'])
            optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])
            best=np.inf;best_epoch=0;history=[];best_weights=None
            for epoch in range(cfg['max_epochs']):
                # Reset RNG each matched seed/epoch so both variants have matching shuffles/dropout.
                torch.manual_seed(seed*1000+epoch)
                perm=np.random.default_rng(seed*1000+epoch).permutation(origins['train'])
                model.train();losses=[]
                for start in range(0,len(perm),cfg['batch_dates']):
                    ids=perm[start:start+cfg['batch_dates']];xx,yy,mm=batch(ids)
                    p,h=model(xx);lr=(p[...,0][mm]-yy[...,0][mm]).abs().mean()
                    if variant=='multitask':
                        lk=(p[...,1][mm]-yy[...,1][mm]).abs().mean()
                        loss=(1-cfg['alpha'])*lr+cfg['alpha']*lk
                    else:loss=lr
                    loss=loss+.001*h.square().mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
                model.eval();val=[]
                with torch.no_grad():
                    for ids in np.array_split(origins['valid'],4):
                        xx,yy,mm=batch(ids);p,_=model(xx)
                        val.extend((p[...,0][mm]-yy[...,0][mm]).abs().tolist())
                score=float(np.mean(val));history.append({'epoch':epoch+1,'train_objective':float(np.mean(losses)),'valid_return_mae':score})
                if score<best:
                    best=score;best_epoch=epoch+1;best_weights=copy.deepcopy(model.state_dict())
                if epoch==0 or (epoch+1)%10==0:
                    print(f'{variant} seed={seed} epoch={epoch+1} val={score:.6f} elapsed={time.time()-started:.1f}s',flush=True)
                (dest/'progress.json').write_text(json.dumps({'epoch':epoch+1,'best_epoch':best_epoch,'best_valid_mae':best,'elapsed':time.time()-started})+'\n')
                if epoch+1-best_epoch>=cfg['early_stop_patience']:break
            model.load_state_dict(best_weights);model.eval();pp=[]
            with torch.no_grad():
                for ids in np.array_split(origins['test'],8):
                    p,_=model(batch(ids)[0]);pp.append(p.numpy())
            pred=np.concatenate(pp)
            assert np.isfinite(pred).all()
            np.savez_compressed(dest/'predictions.npz',pred=pred,y=Y[origins['test']].numpy(),mask=MASK[origins['test']].numpy(),dates=z['target_dates'][origins['test']],symbols=z['symbols'])
            torch.save(best_weights,dest/'best.pt')
            (dest/'history.json').write_text(json.dumps(history,indent=2)+'\n')
            (dest/'fit.json').write_text(json.dumps({'variant':variant,'seed':seed,'best_epoch':best_epoch,'epochs':len(history),'best_validation_return_mae':best,'seconds':time.time()-started,'parameters':sum(p.numel() for p in model.parameters())},indent=2)+'\n')
            print(f'FINISHED {variant} {seed} best_epoch={best_epoch} seconds={time.time()-started:.1f}',flush=True)
    (result/'training_complete.json').write_text(json.dumps({'completed_fits':6,'exact_replication':False},indent=2)+'\n')

if __name__=='__main__':main()
