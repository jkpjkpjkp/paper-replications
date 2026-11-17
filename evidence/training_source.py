"""Train reconstructed STABLE models and backtest strictly held-out market data."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from diffusion_model import ConditionalUNet, cosine_schedule, sample
from stable_core import bl_posterior, ridge_covariance, view_moments

ROOT=Path(__file__).resolve().parent
P=json.loads((ROOT/'protocol.json').read_text())
OUT=ROOT/'results/best_effort'


def fit_scale(x,mask):
    y=x[mask]
    axes=tuple(range(y.ndim-1))
    mu=np.nanmean(y,axis=axes)
    sd=np.nanstd(y,axis=axes)
    return mu,np.maximum(sd,1e-6)


def features(prices,global_macro):
    """All rolling/filter operations are causal. Fitted scalers see fit data only."""
    dates=prices.index
    returns=np.log(prices).diff().to_numpy()
    fit=(dates>=P['train_start'])&(dates<=P['fit_end'])
    mean=np.nanmean(returns[fit],axis=0)
    scale=np.maximum(np.nanstd(returns[fit],axis=0),1e-4)
    lag_dates=dates-pd.Timedelta(days=1)
    gm=global_macro.reindex(global_macro.index.union(lag_dates)).sort_index().ffill().reindex(lag_dates)
    gm.index=dates
    market=(1+pd.DataFrame(np.expm1(returns),index=dates).mean(axis=1).fillna(0)).cumprod()
    raw=pd.DataFrame({'market':market,'dollar':gm.DTWEXBGS,'spread':gm.T10Y3M,
                      'vix':gm.vix,'gold':gm.GLD},index=dates)
    roll=(raw-raw.rolling(120,min_periods=60).mean())/raw.rolling(120,min_periods=60).std().clip(lower=1e-6)
    change=np.log(raw.drop(columns='spread')).diff()
    change['spread']=raw.spread.diff()
    m=np.concatenate([raw.to_numpy(),roll.to_numpy(),change.to_numpy()],axis=1)
    mm,ms=fit_scale(m,fit)
    m=np.nan_to_num((m-mm)/ms,nan=0,posinf=0,neginf=0).clip(-10,10)
    scaled=np.nan_to_num((returns-mean)/scale,nan=0)
    n,s=returns.shape;d=m.shape[1]
    beta=np.zeros((s,d));cov=np.eye(d);identity=np.eye(d)
    betas=np.empty((n,s,d),dtype=np.float32)
    for t in range(n):
        cov=cov+P['kalman_Q']*identity
        v=m[t];gain=cov@v/(v@cov@v+P['kalman_R'])
        beta=beta+(scaled[t]-beta@v)[:,None]*gain
        a=identity-np.outer(gain,v)
        cov=a@cov@a.T+P['kalman_R']*np.outer(gain,gain)
        betas[t]=beta
    logp=np.log(prices)
    zp=(logp-logp.rolling(120,min_periods=60).mean())/logp.rolling(120,min_periods=60).std().clip(lower=1e-6)
    stock=np.concatenate([betas,np.nan_to_num(zp.to_numpy())[:,:,None],scaled[:,:,None]],axis=2)
    cm,cs=fit_scale(stock,fit)
    stock=np.nan_to_num((stock-cm)/cs).clip(-10,10)
    return {'macro':m.astype('float32'),'stock':stock.astype('float32'),
            'returns':returns,'mean':mean,'scale':scale,'dates':dates,
            'scalers':{'macro_mean':mm.tolist(),'macro_std':ms.tolist(),
                       'stock_mean':cm.tolist(),'stock_std':cs.tolist(),
                       'return_mean':mean.tolist(),'return_std':scale.tolist()}}


def origins(dates,start,end,full=True):
    first=int(dates.searchsorted(pd.Timestamp(start)))
    last=int(dates.searchsorted(pd.Timestamp(end),side='right'))
    return [j for j in range(first,last,20) if not full or j+20<=last]


def make_examples(f,idx):
    s=f['stock'].shape[1]
    target=np.concatenate([((f['returns'][j:j+20]-f['mean'])/f['scale']).T for j in idx])
    macro=np.concatenate([np.repeat(f['macro'][j-2][None,:],s,axis=0) for j in idx])
    stock=np.concatenate([f['stock'][j-2] for j in idx])
    if not all(np.isfinite(q).all() for q in [target,macro,stock]): raise ValueError('nonfinite features or labels')
    return [torch.from_numpy(q.astype('float32')) for q in (target,macro,stock)]


def train_model(market,seed,f,device,epochs=None):
    torch.manual_seed(seed);np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainidx=origins(f['dates'],P['train_start'],P['fit_end'])
    validx=origins(f['dates'],P['validation_start'],P['validation_end'])
    tr=[q.to(device) for q in make_examples(f,trainidx)]
    va=[q.to(device) for q in make_examples(f,validx)]
    model=ConditionalUNet(tr[1].shape[1],tr[2].shape[1],width=P['unet_width']).to(device)
    schedule=cosine_schedule(P['forward_steps']).to(device)
    checkpoint=ROOT/f'checkpoints/{market}_{seed}.pt'
    fingerprint=hashlib.sha256((ROOT/'protocol.json').read_bytes()+
        (ROOT/'diffusion_model.py').read_bytes()+(ROOT/'run_empirical.py').read_bytes()+
        (ROOT/f'data/{market}_prices.parquet').read_bytes()+
        (ROOT/'data/global_macro.parquet').read_bytes()).hexdigest()
    if checkpoint.exists() and epochs is None:
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        if state['fingerprint']==fingerprint and state.get('training_complete'):
            model.load_state_dict(state['state_dict']);print('reuse completed',market,seed,flush=True)
            return model,state['training'],schedule
    opt=torch.optim.Adam(model.parameters(),lr=P['learning_rate'],weight_decay=P['adam_weight_decay'])
    vg=torch.Generator(device=device).manual_seed(4024)
    ve=torch.randn(va[0].shape,device=device,generator=vg)
    vt=torch.randint(len(schedule),(len(ve),),device=device,generator=vg)
    aa=schedule[vt,None];vx=aa.sqrt()*va[0]+(1-aa).sqrt()*ve
    hist=[];best=math.inf;beststate=None;best_epoch=0;tic=time.perf_counter()
    total_epochs=epochs or P['epochs']
    for epoch in range(1,total_epochs+1):
        model.train();order=torch.randperm(len(tr[0]),device=device);loss_sum=0.
        for ids in order.split(P['batch_size']):
            y,m,c=[q[ids] for q in tr]
            t=torch.randint(len(schedule),(len(y),),device=device)
            noise=torch.randn_like(y);a=schedule[t,None]
            pred=model(a.sqrt()*y+(1-a).sqrt()*noise,t,m,c)
            loss=F.mse_loss(pred,noise)
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),P['gradient_clip_norm'])
            opt.step();loss_sum+=loss.detach().item()*len(ids)
        model.eval();vsum=0.
        with torch.no_grad():
            for begin in range(0,len(ve),P['batch_size']):
                sl=slice(begin,begin+P['batch_size'])
                vl=F.mse_loss(model(vx[sl],vt[sl],va[1][sl],va[2][sl]),ve[sl],reduction='sum')
                vsum+=vl.item()
        val=vsum/ve.numel()
        row={'epoch':epoch,'train_noise_mse':loss_sum/len(tr[0]),'validation_noise_mse':val,
             'elapsed_seconds':time.perf_counter()-tic}
        hist.append(row)
        if val<best:
            best=val;best_epoch=epoch
            beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%10==0:
            print(market,seed,json.dumps(row),flush=True)
            pd.DataFrame(hist).to_csv(OUT/f'{market}_{seed}_training.csv',index=False)
    training={'seed':seed,'epochs':total_epochs,'best_epoch':best_epoch,'validation_noise_mse':best,
              'first_validation_noise_mse':hist[0]['validation_noise_mse'],
              'train_examples':len(tr[0]),'validation_examples':len(va[0]),
              'training_seconds':time.perf_counter()-tic,
              'parameters':sum(p.numel() for p in model.parameters()),
              'train_last_label':str(f['dates'][trainidx[-1]+19].date()),
              'validation_last_label':str(f['dates'][validx[-1]+19].date())}
    model.load_state_dict(beststate)
    torch.save({'state_dict':beststate,'fingerprint':fingerprint,'training':training,
                'scalers':f['scalers'],'protocol':P,'training_complete':total_epochs==P['epochs']},checkpoint)
    pd.DataFrame(hist).to_csv(OUT/f'{market}_{seed}_training.csv',index=False)
    return model,training,schedule


def repair_weights(mean,cov,gross_limit=2.):
    equal=np.full(len(mean),1/len(mean))
    direction=np.linalg.solve(cov,mean);den=float(direction.sum())
    diag={'denominator':den,'fallback':False,'gross_cap_applied':False}
    if not np.isfinite(direction).all() or den<=1e-10:
        diag['fallback']=True;return equal,diag
    w=direction/den
    if np.abs(w).sum()>gross_limit:
        lo,hi=0.,1.
        for _ in range(60):
            a=(lo+hi)/2
            if np.abs(equal+a*(w-equal)).sum()>gross_limit:hi=a
            else:lo=a
        w=equal+lo*(w-equal);diag['gross_cap_applied']=True
    if not np.isclose(w.sum(),1) or np.abs(w).sum()>gross_limit+1e-8:raise AssertionError('invalid weights')
    return w,diag


def allocate(paths,historical,ridge):
    view_mean,view_cov=view_moments(paths)
    cv=ridge_covariance(view_cov,ridge)
    prior=ridge_covariance(np.cov(historical,rowvar=False)*P['horizon'],ridge)
    mu0=prior@np.full(prior.shape[0],1/prior.shape[0])
    mu,cov=bl_posterior(mu0,prior,view_mean,cv)
    w,diag=repair_weights(mu,cov,P['maximum_gross_exposure'])
    diag.update({'view_cov_rank':int(np.linalg.matrix_rank(view_cov)),
                 'min_repaired_eigenvalue':float(np.linalg.eigvalsh(cv)[0])})
    return w,diag


def dtw(a,b):
    z=np.full((len(a)+1,len(b)+1),np.inf);z[0,0]=0.
    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            z[i,j]=abs(a[i-1]-b[j-1])+min(z[i-1,j],z[i,j-1],z[i-1,j-1])
    return z[-1,-1]/max(len(a),len(b))


def simulate(returns,dates,weights,cost=0.):
    """Self-financing signed buy-and-hold positions with explicit rebalance costs."""
    start=min(weights);end=len(returns);s=returns.shape[1]
    w=np.zeros(s);wealth=1.;rows=[]
    for j in range(start,end):
        turnover=0.
        if j in weights:
            new=weights[j];turnover=float(np.abs(new-w).sum());w=new.copy()
        gross=float(np.abs(w).sum())
        rp=float(w@np.expm1(returns[j]))
        net=(1-cost*turnover)*(1+rp)-1
        wealth*=1+net
        rows.append({'date':str(dates[j].date()),'return':net,'gross_return':rp,'wealth':wealth,
                     'turnover':turnover,'gross_exposure':gross,'rebalanced':j in weights})
        if 1+rp<=0 or wealth<=0:raise ValueError('portfolio insolvent')
        w=w*(1+np.expm1(returns[j]))/(1+rp)
    return pd.DataFrame(rows)


def metrics(ledger):
    r=ledger['return'].to_numpy();nav=np.r_[1.,ledger.wealth.to_numpy()]
    sd=r.std(ddof=1)
    return {'ASR':float(np.sqrt(252)*r.mean()/sd) if sd>0 else None,
            'RMDD_percent':float(100*np.max(1-nav/np.maximum.accumulate(nav))),
            'AVol_percent':float(100*np.sqrt(252)*sd),'total_return_percent':float(100*(nav[-1]-1)),
            'test_days':len(r),'total_turnover':float(ledger.turnover.sum()),
            'max_gross_exposure_including_drift':float(ledger.gross_exposure.max())}


def evaluate(market,seed,f,model,schedule,device,sensitivities):
    idx=origins(f['dates'],P['test_start'],P['test_end'],full=False)
    s=f['stock'].shape[1];k=P['sample_paths'];h=P['horizon']
    names=['STABLE_repaired','CRP','MOM','MVO']
    if sensitivities:names+=['STABLE_ridge_0.0001','STABLE_ridge_0.1','STABLE_correlated_shocks']
    weights={name:{} for name in names};diagnostics=[];forecasts=[];saved=[]
    model.eval();tic=time.perf_counter()
    for j in idx:
        signal=j-2;hist=f['returns'][signal-119:signal+1]
        if not np.isfinite(hist).all():raise ValueError('nonfinite trailing returns')
        macro=torch.from_numpy(np.tile(f['macro'][signal],(k*s,1))).to(device)
        stock=torch.from_numpy(np.tile(f['stock'][signal],(k,1))).to(device)
        gen=torch.Generator(device=device).manual_seed(seed+j*997)
        eps=torch.randn((k,s,h),device=device,generator=gen)
        norm,clip=sample(model,macro,stock,eps.reshape(k*s,h),schedule,P['ddim_steps'],
                          P['ddim_eta'][market],gen)
        paths=norm.cpu().numpy().reshape(k,s,h).transpose(0,2,1)*f['scale']+f['mean']
        w,diag=allocate(paths,hist,P['relative_ridge']);weights['STABLE_repaired'][j]=w
        diag.update({'first_return_date':str(f['dates'][j].date()),
                     'signal_date':str(f['dates'][signal].date()),
                     'execution_date':str(f['dates'][j-1].date()),'sampler_clip_fraction':clip,
                     'method':'STABLE_repaired'})
        diagnostics.append(diag)
        weights['CRP'][j]=np.full(s,1/s)
        mom=np.zeros(s);top=np.argsort(hist.sum(axis=0),kind='stable')[-max(1,s//5):]
        mom[top]=1/len(top);weights['MOM'][j]=mom
        weights['MVO'][j],d=repair_weights(hist.mean(axis=0)*h,ridge_covariance(np.cov(hist,rowvar=False)*h,P['relative_ridge']))
        diagnostics.append(dict(d,method='MVO',first_return_date=str(f['dates'][j].date())))
        if sensitivities:
            for ridge in [.0001,.1]:
                key=f'STABLE_ridge_{ridge}'
                weights[key][j],d=allocate(paths,hist,ridge)
                diagnostics.append(dict(d,method=key,first_return_date=str(f['dates'][j].date())))
            correlation=np.corrcoef(hist,rowvar=False)
            correlation=np.nan_to_num(correlation);np.fill_diagonal(correlation,1)
            chol=np.linalg.cholesky(.9*correlation+.1*np.eye(s))
            eps2=torch.einsum('ab,kbh->kah',torch.tensor(chol,dtype=torch.float32,device=device),eps)
            gen2=torch.Generator(device=device).manual_seed(seed+j*997+1)
            alt,_=sample(model,macro,stock,eps2.reshape(k*s,h),schedule,P['ddim_steps'],P['ddim_eta'][market],gen2)
            paths2=alt.cpu().numpy().reshape(k,s,h).transpose(0,2,1)*f['scale']+f['mean']
            weights['STABLE_correlated_shocks'][j],d=allocate(paths2,hist,P['relative_ridge'])
            diagnostics.append(dict(d,method='STABLE_correlated_shocks',first_return_date=str(f['dates'][j].date())))
        # Forecast metrics use complete 20-day horizons only, never a padded truth.
        if j+h<=len(f['dates']):
            truth=f['returns'][j:j+h]
            prediction=paths.mean(axis=0)
            for name,pred in [('STABLE',prediction),('zero',np.zeros_like(truth)),
                              ('trailing_mean',np.tile(hist.mean(axis=0),(h,1)))]:
                forecasts.append({'market':market,'seed':seed,'method':name,
                    'first_return_date':str(f['dates'][j].date()),
                    'MSE':float(np.square(pred-truth).mean()),
                    'normalized_DTW':float(np.mean([dtw(pred[:,a],truth[:,a]) for a in range(s)]))})
        saved.append(paths)
    rows=[]
    for name,ws in weights.items():
        for cost in [0.,.001]:
            ledger=simulate(f['returns'],f['dates'],ws,cost)
            bps=round(cost*1e4)
            ledger.to_csv(OUT/f'{market}_{seed}_{name}_{bps}bps_daily.csv',index=False)
            rows.append(dict(market=market,seed=seed,method=name,cost_bps=bps,**metrics(ledger)))
        table=pd.DataFrame(np.stack(list(ws.values())),columns=[str(q) for q in range(s)])
        table.insert(0,'first_return_date',[str(f['dates'][j].date()) for j in ws])
        table.to_csv(OUT/f'{market}_{seed}_{name}_weights.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/f'{market}_{seed}_allocation_diagnostics.csv',index=False)
    np.savez_compressed(OUT/f'{market}_{seed}_generated_paths.npz',paths=np.stack(saved),
                        first_return_dates=np.array([str(f['dates'][j].date()) for j in idx]))
    return rows,forecasts,{'inference_seconds':time.perf_counter()-tic,'rebalances':len(idx),
                          'fallback_count':sum(q['fallback'] for q in diagnostics if q['method']=='STABLE_repaired'),
                          'gross_cap_count':sum(q['gross_cap_applied'] for q in diagnostics if q['method']=='STABLE_repaired')}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--markets',nargs='+',default=P['markets'])
    parser.add_argument('--seeds',nargs='+',type=int,default=P['seeds'])
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--train-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True,exist_ok=True)
    gm=pd.read_parquet(ROOT/'data/global_macro.parquet')
    for market in args.markets:
        prices=pd.read_parquet(ROOT/f'data/{market}_prices.parquet')
        f=features(prices,gm)
        (OUT/f'{market}_feature_scalers.json').write_text(json.dumps(f['scalers'],indent=2))
        for seed in args.seeds:
            model,training,schedule=train_model(market,seed,f,device,args.epochs)
            if args.train_only:continue
            rows,forecasts,diagnostics=evaluate(market,seed,f,model,schedule,device,seed==P['seeds'][0])
            result={'market':market,'seed':seed,'training':training,'evaluation':diagnostics,
                    'metrics':rows,'forecast_metrics':forecasts,'device':device,'torch':torch.__version__,
                    'author_reproduction':False,'test_end':P['test_end']}
            (OUT/f'{market}_{seed}_result.json').write_text(json.dumps(result,indent=2))
            print('COMPLETED',market,seed,json.dumps(training),json.dumps(diagnostics),flush=True)
            del model
            if device=='cuda':torch.cuda.empty_cache()


if __name__=='__main__': main()
