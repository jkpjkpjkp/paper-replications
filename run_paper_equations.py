#!/usr/bin/env python3
"""All-slice artifact-based reproduction of the final paper's stated objectives.

This deliberately does NOT claim that the released token/entropy solver equals
the published quadratic program. Baseline details and endpoint interpretations
are frozen in equation-config.json before execution. No return-based selection.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.stats import norm

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'release/data'
OUT=ROOT/'results/paper-equations'
pa.set_cpu_count(1);pa.set_io_thread_count(1)
METHODS=['upstream']+['rura_lambda%.1f'%(i/10) for i in range(1,10)]+['cm1','cm2','cm3']
K=10


def atomic_json(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,default=str)+'\n');t.replace(p)


def diagonal_qp(a,d,mask,equality=False):
    """Max a.w - .5*d.w^2, simplex; 64 bisections of KKT multiplier."""
    d=np.maximum(d,1e-12)
    unconstrained=np.where(mask,np.maximum(a,0)/d,0)
    inactive=(unconstrained.sum(axis=1)<=1)&(not equality)
    # A common linear offset preserves a fixed unit-budget optimum. Centering
    # avoids subtracting two large monetary scores near a zero-risk optimum.
    offset=np.max(np.where(mask,a,-np.inf),axis=1)
    a=a-offset[:,None]
    if equality:
        lo=np.max(np.where(mask,a-d,-np.inf),axis=1)
    else:
        lo=-offset
    hi=np.max(np.where(mask,a,-np.inf),axis=1)
    hi=np.maximum(hi,lo)
    for _ in range(64):
        mid=(lo+hi)/2
        weights=np.where(mask,np.maximum(a-mid[:,None],0)/d,0)
        too_large=weights.sum(axis=1)>1
        lo=np.where(too_large,mid,lo);hi=np.where(too_large,hi,mid)
    weights=np.where(mask,np.maximum(a-hi[:,None],0)/d,0)
    weights[inactive]=unconstrained[inactive]
    if equality:weights=weights/weights.sum(axis=1,keepdims=True)
    return weights


def gaussian_cvar(mu,var,mask,alpha=.95):
    """Exact diagonal GaussianCVaR minimizer via one-dimensional KKT root.

    Objective kappa*sqrt(sum(var*w^2))-mu.w, nonnegative unit simplex.
    Positive B rescales the entire objective, leaving its argmin unchanged.
    """
    var=np.maximum(var,1e-12)
    kappa=norm.pdf(norm.ppf(alpha))/(1-alpha)
    lo=np.min(np.where(mask,mu,np.inf),axis=1)-kappa*np.sqrt(np.max(np.where(mask,var,0),axis=1))-1.
    hi=np.max(np.where(mask,mu,-np.inf),axis=1)
    for _ in range(64):
        threshold=(lo+hi)/2
        residual=np.sum(np.where(mask,np.maximum(mu-threshold[:,None],0)**2/var,0),axis=1)
        too_large=residual>kappa*kappa
        lo=np.where(too_large,threshold,lo);hi=np.where(too_large,hi,threshold)
    raw=np.where(mask,np.maximum(mu-hi[:,None],0)/var,0)
    return raw/raw.sum(axis=1,keepdims=True)


def compute_scores(mu,p,var,rho,B,mask):
    result={'upstream':np.where(mask,mu,-np.inf)}
    proposed=[]
    for i in range(1,10):
        lam=i/10
        a=p*B*mu;d=rho*p*B**2*var+2*lam
        weights=diagonal_qp(a,d,mask)
        if np.any(weights.sum(axis=1)>1+1e-8) or np.any(weights<0):raise RuntimeError('RURA budget feasibility')
        proposed.append(weights)
        result['rura_lambda%.1f'%lam]=np.where(mask,weights,-np.inf)
    result['cm1']=np.where(mask,p*mu/np.sqrt(np.maximum(var,1e-12)),-np.inf)
    result['cm2']=np.where(mask,diagonal_qp(B*mu,rho*B**2*var,mask,equality=True),-np.inf)
    result['cm3']=np.where(mask,gaussian_cvar(mu,var,mask),-np.inf)
    return result


def numerical_validation(mu,p,var,rho,B,mask):
    checks=[]
    # Real released profiles; spread deterministic indices across the user panel.
    indices=np.unique(np.linspace(0,len(mu)-1,8,dtype=int))
    for j in indices:
        ok=mask[j];m=mu[j,ok];pr=p[j,ok];v=var[j,ok];r=float(rho[j,0]);b=float(B[j,0]);n=len(m)
        for lam in [.1,.5,.9,None]:
            eq=lam is None
            a=b*m if eq else pr*b*m
            d=r*b*b*v if eq else r*pr*b*b*v+2*lam
            ours=diagonal_qp(a[None,:],d[None,:],np.ones((1,n),bool),equality=eq)[0]
            scale=max(1.,float(np.max(np.abs(a))),float(np.max(np.abs(d))))
            f=lambda x:float((.5*np.sum(d*x*x)-np.dot(a,x))/scale)
            jac=lambda x:(d*x-a)/scale
            fit=minimize(f,np.full(n,1/n),jac=jac,bounds=[(0,1)]*n,
                         constraints=[{'type':'eq' if eq else 'ineq','fun':lambda x:1-x.sum(),'jac':lambda x:-np.ones_like(x)}],
                         method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
            gap=f(ours)-fit.fun
            checks.append({'row_index':int(j),'lambda':lam,'method':'cm2' if eq else 'rura','slsqp_success':bool(fit.success),
                           'scaled_objective_gap_ours_minus_slsqp':float(gap),'budget':float(ours.sum())})
            if fit.success and gap>1e-7:raise RuntimeError('Water-filling fails independent optimizer objective check')
        ours=gaussian_cvar(m[None,:],v[None,:],np.ones((1,n),bool))[0]
        kap=norm.pdf(norm.ppf(.95))/.05;v=np.maximum(v,1e-12)
        f=lambda x:float(kap*np.sqrt(np.dot(v,x*x))-np.dot(m,x))
        jac=lambda x:kap*v*x/max(np.sqrt(np.dot(v,x*x)),1e-20)-m
        fit=minimize(f,np.full(n,1/n),jac=jac,bounds=[(0,1)]*n,
                     constraints=[{'type':'eq','fun':lambda x:x.sum()-1,'jac':lambda x:np.ones_like(x)}],
                     method='SLSQP',options={'ftol':1e-12,'maxiter':2000})
        gap=f(ours)-fit.fun
        checks.append({'row_index':int(j),'method':'cm3','slsqp_success':bool(fit.success),'objective_gap_ours_minus_slsqp':float(gap)})
        if fit.success and gap>1e-7:raise RuntimeError('GaussianCVaR fails independent optimizer objective check')
    return checks


def build_labels(tx,meta,user_index,asset_index):
    start=pd.Timestamp(meta['test_start']);end=pd.Timestamp(meta['test_end'])
    before=tx[tx.date<start].copy()
    before['signed']=np.where(before.side.str.upper().eq('BUY'),before.quantity,-before.quantity)
    positions=before.groupby(['customer_id','asset_id'],sort=False).signed.sum()
    test=tx[(tx.date>=start)&(tx.date<end)].sort_values(['customer_id','asset_id','date'],kind='stable')
    first=test.drop_duplicates(['customer_id','asset_id'],keep='first')
    first=first[first.side.str.upper().eq('BUY')].copy()
    prior=positions.reindex(pd.MultiIndex.from_frame(first[['customer_id','asset_id']])).fillna(0).to_numpy()
    first=first[prior<=0]
    labels=np.zeros((len(user_index),len(asset_index)),dtype=bool)
    u=first.customer_id.map(user_index);a=first.asset_id.map(asset_index);valid=u.notna()&a.notna()
    labels[u[valid].to_numpy(dtype=int),a[valid].to_numpy(dtype=int)]=True
    return labels


def load_slice(sid,asset_index):
    frame=pq.read_table(ROOT/'inputs'/(sid+'.parquet')).to_pandas()
    frame=frame[frame.in_paper30].copy()
    sizes=frame.groupby('user_id',sort=False).size()
    good=set(sizes[sizes>=K].index)
    excluded=int((sizes<K).sum())
    frame=frame[frame.user_id.isin(good)]
    if frame.empty:raise RuntimeError('No valid paper users in '+sid)
    row,userids=pd.factorize(frame.user_id,sort=False)
    col=frame.groupby('user_id',sort=False).cumcount().to_numpy()
    n=len(userids);width=30
    mask=np.zeros((n,width),bool);mask[row,col]=True
    def arr(name,default=0.):
        out=np.full((n,width),default,dtype=float);out[row,col]=frame[name].to_numpy(float);return out
    aid=np.full((n,width),-1,dtype=np.int16);aid[row,col]=frame.item_id.map(asset_index).to_numpy(np.int16)
    profiles=frame.drop_duplicates('user_id').set_index('user_id').loc[userids]
    return {'userids':np.asarray(userids,dtype='U32'),'aid':aid,'mu':arr('score_blend'),
            'p':arr('p_i'),'var':arr('sigma2'),'rho':profiles.rho.to_numpy()[:,None],
            'B':profiles.B.to_numpy()[:,None],'mask':mask,
            'category':profiles.category.to_numpy(),'excluded_less_than_k':excluded}


def plot_sensitivity(aggregate):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    curve=aggregate[(aggregate.category=='Overall')&aggregate.method.str.startswith('rura_lambda')].copy()
    curve['lambda']=curve.method.str.replace('rura_lambda','',regex=False).astype(float)
    curve=curve.sort_values('lambda')
    curve.to_csv(OUT/'figure1-numerical-data.csv',index=False)
    fig,axes=plt.subplots(3,1,figsize=(8,9),sharex=True,layout='constrained')
    table_x=[.1,.5,.9]
    reported={'roi_percent':[.2924,.3878,.5166],'ndcg_percent':[.0955,.1053,.1155],'utility_x1e4':[3.602,3.593,3.583]}
    for ax,col,label in zip(axes,['roi_percent','ndcg_percent','utility_x1e4'],['Period ROI (%)','nDCG@10 (%)','Proxy utility (x 1e-4)']):
        ax.plot(curve['lambda'],curve[col],marker='o',label='Published equations on released inputs')
        ax.scatter(table_x,reported[col],marker='x',s=60,color='#b54b32',label='Reported table cells')
        ax.set_ylabel(label);ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel('Diversification lambda');axes[-1].set_xticks([i/10 for i in range(1,10)])
    fig.suptitle('RURA sensitivity: artifact-based methodological reproduction\nEndpoint and baseline interpretations disclosed; no daily trading inference',fontsize=11)
    fig.savefig(OUT/'figure1-reproduction.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'equation-config.json').read_text())
    config_hash=hashlib.sha256((ROOT/'equation-config.json').read_bytes()).hexdigest()
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    slices=[json.loads(x) for x in (DATA/'splits/slices.jsonl').read_text().splitlines()]
    assets=sorted(pq.read_table(DATA/'clean/assets.parquet',columns=['asset_id']).column('asset_id').to_pylist())
    asset_index={a:i for i,a in enumerate(assets)}
    tx=pq.read_table(DATA/'clean/transactions.parquet').to_pandas()
    groups=[];per_slice=[];validation=[]
    discounts=1/np.log2(np.arange(2,K+2));ideal=np.r_[0,np.cumsum(discounts)]
    for sidx,meta in enumerate(slices):
        sid=meta['slice_id'];directory=OUT/sid;directory.mkdir(exist_ok=True)
        summarypath=directory/'summary.json'
        if summarypath.exists():
            prior=json.loads(summarypath.read_text())
            if prior.get('status')=='complete':
                if prior.get('config_sha256')!=config_hash or prior.get('script_sha256')!=script_hash:
                    raise RuntimeError('Completed output has a different code/config hash; preserve it and use a separately identified run')
                groups.extend(prior['groups']);per_slice.extend(prior['overall']);validation.extend(prior.get('validation',[]))
                print(json.dumps({'slice':sid,'status':'reuse_complete_equation_evaluation'}),flush=True);continue
        started=time.time();x=load_slice(sid,asset_index)
        for name in ['mu','p','var']:
            if not np.isfinite(x[name][x['mask']]).all():raise RuntimeError('Nonfinite released input: '+name)
        if np.any((x['p'][x['mask']]<0)|(x['p'][x['mask']]>1)) or np.any(x['var'][x['mask']]<0):raise RuntimeError('Invalid probability/variance input range')
        if not np.isfinite(x['rho']).all() or not np.isfinite(x['B']).all() or np.any(x['rho']<=0) or np.any(x['B']<=0):raise RuntimeError('Invalid profile risk/scale')
        labels=build_labels(tx,meta,{u:i for i,u in enumerate(x['userids'])},asset_index)
        endpoint=pq.read_table(ROOT/'inputs'/(sid+'-endpoints.parquet')).to_pandas()
        if 'asset_id' in endpoint.columns:endpoint=endpoint.set_index('asset_id')
        if endpoint.index.name!='asset_id':raise RuntimeError('Endpoint asset identifier/index is missing')
        returns=np.full(len(assets),np.nan)
        for a,j in asset_index.items():
            if a in endpoint.index and endpoint.loc[a,'valid']:
                returns[j]=endpoint.loc[a,'end_close']/endpoint.loc[a,'start_close']-1
        scores=compute_scores(x['mu'],x['p'],x['var'],x['rho'],x['B'],x['mask'])
        rank_assets=[];slice_groups=[];slice_overall=[];metric_frames=[]
        upstream_utility=None
        numlabels=labels.sum(axis=1);idcg=ideal[np.minimum(K,numlabels)]
        for method in METHODS:
            # Stable weights sort retains frozen upstream tie order.
            order=np.argsort(-scores[method],axis=1,kind='stable')[:,:K]
            ids=np.take_along_axis(x['aid'],order,axis=1)
            if np.any(ids<0):raise RuntimeError('Padding entered displayed slate')
            relevant=np.take_along_axis(labels,ids,axis=1)
            dcg=(relevant*discounts).sum(axis=1)
            ndcg=np.divide(dcg,idcg,out=np.zeros_like(dcg),where=idcg>0)
            gross=returns[ids].mean(axis=1)
            if not np.isfinite(gross).all():raise RuntimeError('Invalid common endpoint price in evaluated slate')
            roi=gross-.003
            selected_mu=np.take_along_axis(x['mu'],order,axis=1)
            utility=(-np.expm1(-x['rho']*selected_mu)).mean(axis=1)
            if method=='upstream':upstream_utility=utility.copy()
            elif np.any(utility>upstream_utility+1e-12):raise RuntimeError('Proxy utility exceeds monotone top-mu upper bound; likely input/index/weight error')
            all_groups=['Overall']+['Aggressive','Income','Balanced','Conservative','Unknown']
            for cat in all_groups:
                keep=np.ones(len(gross),dtype=bool) if cat=='Overall' else x['category']==cat
                if not keep.any():continue
                record={'slice_id':sid,'method':method,'category':cat,'users':int(keep.sum()),
                        'ndcg_percent':float(ndcg[keep].mean()*100),'roi_percent':float(roi[keep].mean()*100),
                        'gross_period_roi_percent':float(gross[keep].mean()*100),
                        'utility_x1e4':float(utility[keep].mean()*1e4)}
                if cat=='Overall':slice_overall.append(record)
                slice_groups.append(record)
            rank_assets.append(ids.astype(np.int16))
            metric_frames.append(pd.DataFrame({'user_id':x['userids'],'method':method,'category':x['category'],
                                               'ndcg':ndcg,'period_roi':roi,'gross_period_roi':gross,'utility':utility}))
        prediction=np.stack(rank_assets,axis=1)
        digest=hashlib.sha256(prediction.tobytes()).hexdigest()
        np.savez_compressed(directory/'rankings.npz',user_ids=x['userids'],asset_ids=np.asarray(assets),
                            methods=np.asarray(METHODS),ranked_asset_indices=prediction)
        pd.concat(metric_frames,ignore_index=True).to_parquet(directory/'user-metrics.parquet',index=False,compression='zstd')
        checks=[]
        if sidx in [0,len(slices)-1]:
            checks=numerical_validation(x['mu'],x['p'],x['var'],x['rho'],x['B'],x['mask'])
            rerun=compute_scores(x['mu'],x['p'],x['var'],x['rho'],x['B'],x['mask'])
            reranked=np.stack([np.take_along_axis(x['aid'],np.argsort(-rerun[m],axis=1,kind='stable')[:,:K],axis=1).astype(np.int16) for m in METHODS],axis=1)
            if not np.array_equal(prediction,reranked):raise RuntimeError('Full-slice deterministic rerun changed rankings')
            checks.append({'type':'full_slice_all13variants_deterministic_rerun','equal':True,'rankings_sha256':digest})
        summary={'status':'complete','slice_id':sid,'users':len(x['userids']),
                 'config_sha256':config_hash,'script_sha256':script_hash,
                 'input_sha256':hashlib.sha256((ROOT/'inputs'/(sid+'.parquet')).read_bytes()).hexdigest(),
                 'endpoint_sha256':hashlib.sha256((ROOT/'inputs'/(sid+'-endpoints.parquet')).read_bytes()).hexdigest(),
                 'excluded_less_than_k':x['excluded_less_than_k'],'methods':METHODS,
                 'groups':slice_groups,'overall':slice_overall,'validation':checks,
                 'rankings_sha256':digest,'seconds':time.time()-started,
                 'monotone_proxy_utility_upper_bound_check':'all methods <= frozen upstream top-mu equal-slate utility per user',
                 'evaluation_scope':'Original period ranking metrics under explicit artifact/endpoint/baseline interpretations;not a daily trading backtest.'}
        atomic_json(summarypath,summary);groups.extend(slice_groups);per_slice.extend(slice_overall);validation.extend(checks)
        print(json.dumps({'slice':sid,'status':'complete_equation_evaluation','users':len(x['userids']),'variants':len(METHODS),'seconds':time.time()-started}),flush=True)
    full=pd.DataFrame(groups)
    rows=[]
    for (method,cat),g in full.groupby(['method','category'],sort=False):
        weights=g.users.to_numpy(float)
        record={'method':method,'category':cat,'user_slices':int(weights.sum()),'slices':len(g)}
        for col in ['ndcg_percent','roi_percent','gross_period_roi_percent','utility_x1e4']:
            record[col]=float(np.average(g[col],weights=weights))
        rows.append(record)
    aggregate=pd.DataFrame(rows);aggregate.to_csv(OUT/'aggregate.csv',index=False)
    pd.DataFrame(per_slice).to_csv(OUT/'all-slice-metrics.csv',index=False)
    inventory=json.loads((ROOT/'experiment-inventory.json').read_text())
    comparisons=[]
    for exp in inventory['original_paper_experiments']:
        if 'reported' not in exp:continue
        found=aggregate[(aggregate.method==exp['id'])&(aggregate.category=='Overall')]
        if len(found)!=1:continue
        got=found.iloc[0].to_dict()
        for metric in ['ndcg_percent','roi_percent']:
            comparisons.append({'method':exp['id'],'metric':metric,'paper':exp['reported'][metric],
                                'reproduced':got[metric],'difference':got[metric]-exp['reported'][metric]})
        if 'utility_x1e4' in exp['reported']:
            for cat,paper in zip(['Aggressive','Income','Balanced','Conservative','Unknown','Overall'],exp['reported']['utility_x1e4']):
                value=aggregate[(aggregate.method==exp['id'])&(aggregate.category==cat)].iloc[0].utility_x1e4
                comparisons.append({'method':exp['id'],'metric':'utility_x1e4_'+cat,'paper':paper,'reproduced':value,'difference':value-paper})
    pd.DataFrame(comparisons).to_csv(OUT/'paper-comparison.csv',index=False)
    plot_sensitivity(aggregate)
    atomic_json(OUT/'status.json',{'status':'complete_equation_artifact_scope','original_paper_status':'partial',
                                  'slices':len(slices),'methods':METHODS,'method_slice_runs':len(slices)*len(METHODS),
                                  'config_sha256':config_hash,'script_sha256':script_hash,
                                  'numerical_validation':validation,'config':config,
                                  'market_neutral_nocost':'blocked_missing_cashflows_and_point_in_time_provenance',
                                  'significance_or_profitability_claim':False})
    print(json.dumps({'status':'complete_equation_artifact_scope','method_slice_runs':len(slices)*len(METHODS),
                      'original_paper_status':'partial','results_path':str(OUT)}),flush=True)


if __name__=='__main__':main()
