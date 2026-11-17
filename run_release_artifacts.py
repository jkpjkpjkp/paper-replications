#!/usr/bin/env python3
"""Guarded parity and full evaluation of the released diagonal RURA algorithm."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from prepare_inputs import user_groups
from release_solver import solve

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'release/data';OUT=ROOT/'results/released-algorithm'
sys.path.insert(0,str(ROOT/'upstream'))
from src.rura_core_fast import run_rura_for_user,ruraConfig
from src.groundtruth_utils import build_newbuy_labels
pa.set_cpu_count(1);pa.set_io_thread_count(1)
SCALES={'Conservative':1.2,'Income':1.1,'Balanced':1.,'Aggressive':.85,'Unknown':1.}


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n');tmp.replace(path)


def context(sid,assets):
    prof=pq.read_table(DATA/'profiles_riskmap_cat'/sid/'user_params.parquet').to_pandas().set_index('user_id')
    if not prof.k.eq(10).all() or not prof.risk_model.eq('CARA').all():raise RuntimeError('Unexpected released profile k/risk model; do not silently change it')
    stats=pq.read_table(DATA/'features'/sid/'market_stats.parquet').to_pandas().set_index('asset_id')
    pop=pq.read_table(DATA/'features'/sid/'popularity.parquet').to_pandas().set_index('asset_id')
    default=max(1e-8,float(np.nanmedian(stats.sigma2)))
    return prof,stats.reindex(assets).r.fillna(0).to_numpy(),stats.reindex(assets).sigma2.fillna(default).to_numpy(),pop.reindex(assets).s.fillna(0).to_numpy()


def batches(sid,prof,asset_index,limit=None):
    batch=[];total=0
    for uid,g in user_groups(DATA/'candidates'/(sid+'_all.parquet')):
        if uid not in prof.index:continue
        g=g[g.eligible&~g.in_holdings_flag].copy()
        if len(g)==0:continue
        batch.append((uid,g));total+=1
        if len(batch)==128:
            yield batch;batch=[]
        if limit is not None and total>=limit:break
    if batch:yield batch


def pack(batch,prof,asset_index,r,var,pop):
    n=len(batch);width=max(len(g) for _,g in batch)
    codes=np.full((n,width),32767,dtype=np.int16);valid=np.zeros((n,width),bool)
    p=np.zeros((n,width));mu=np.zeros((n,width));rho=np.zeros((n,1));B=np.zeros((n,1));cats=[]
    for j,(uid,g) in enumerate(batch):
        k=len(g);valid[j,:k]=True;codes[j,:k]=g.item_id.map(asset_index).to_numpy(np.int16)
        p[j,:k]=g.p_i.to_numpy();mu[j,:k]=g.score_blend.to_numpy()
        rho[j,0]=float(prof.loc[uid,'gamma_or_eta']);B[j,0]=float(prof.loc[uid,'B']);cats.append(str(prof.loc[uid,'category']))
    safe=np.where(valid,codes,0)
    return {'p':p,'r':r[safe],'var':var[safe],'pop':pop[safe],'rho':rho,'B':B,'valid':valid,'asset_codes':codes,'mu':mu,'cats':np.array(cats),'userids':[u for u,_ in batch]}


def reference(x,j,lam,assets):
    valid=x['valid'][j];codes=x['asset_codes'][j,valid]
    rows=run_rura_for_user('parity',x['userids'][j],np.asarray(assets)[codes],x['p'][j,valid],x['r'][j,valid],x['var'][j,valid],x['pop'][j,valid],float(x['rho'][j,0]),float(x['B'][j,0]),10,
                           ruraConfig(method='rura_la',lambda_div=lam,beta_pop=.1,L=30,w_max=.4,device='cpu'),None,'pinned_author',2025)
    index={a:i for i,a in enumerate(assets)}
    ids=np.full(10,-1,dtype=np.int16);weights=np.zeros(10)
    for k,row in enumerate(rows):ids[k]=index[row.item_id];weights[k]=row.w
    return ids,weights


def calculate(x,lam,assets):
    keys=['p','r','var','pop','rho','B','valid','asset_codes']
    ids,w,ambiguous=solve(**{k:x[k] for k in keys},lam=lam)
    for j in np.flatnonzero(ambiguous):ids[j],w[j]=reference(x,j,lam,assets)
    return ids,w,int(ambiguous.sum())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    slices=[json.loads(x) for x in (DATA/'splits/slices.jsonl').read_text().splitlines()]
    assets=sorted(pq.read_table(DATA/'clean/assets.parquet',columns=['asset_id']).column('asset_id').to_pylist());asset_index={a:i for i,a in enumerate(assets)}
    hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['release_solver.py','run_release_artifacts.py']}
    if args.probe:
        checks=[];mismatch=0
        for meta in [slices[0],slices[-1]]:
            sid=meta['slice_id'];prof,r,v,pop=context(sid,assets)
            for batch in batches(sid,prof,asset_index,limit=64):
                x=pack(batch,prof,asset_index,r,v,pop)
                for i in range(1,10):
                    ids,w,fallback=calculate(x,i/10,assets)
                    for j in range(len(batch)):
                        expected,ew=reference(x,j,i/10,assets)
                        same=np.array_equal(ids[j],expected);gap=float(np.max(np.abs(w[j]-ew)))
                        checks.append({'slice':sid,'user':x['userids'][j],'lambda':i/10,'same_ranked_ids':same,'max_abs_weight_difference':gap})
                        mismatch+=int(not same or gap>1e-12)
        result={'status':'passed' if mismatch==0 else 'failed','scope':'128real full-candidate user-slices across first/last slices x9lambdas; compare unchanged author code','cases':len(checks),'mismatch_cases':mismatch,'checks':checks,'code_sha256':hashes}
        dump(OUT/'parity.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
        if mismatch:raise RuntimeError('Do not execute full release run: parity mismatch')
        return
    parity=json.loads((OUT/'parity.json').read_text())
    if parity['status']!='passed' or parity['code_sha256']!=hashes:raise RuntimeError('Current implementation has no passing exact-author parity test')
    price=pq.read_table(DATA/'clean/prices.parquet').to_pandas();px=price.pivot(index='date',columns='asset_id',values='close_eur')
    summaries=[];allgroups=[];discount=1/np.log2(np.arange(2,12));ideal=np.r_[0,np.cumsum(discount)]
    for meta in slices:
        sid=meta['slice_id'];sd=OUT/sid;sd.mkdir(exist_ok=True);summarypath=sd/'summary.json'
        if summarypath.exists():
            old=json.loads(summarypath.read_text())
            if old.get('status')=='complete' and old.get('code_sha256')==hashes:
                summaries.append(old);allgroups+=old['groups'];print(json.dumps({'slice':sid,'status':'reuse_completed_native_run'}),flush=True);continue
        began=time.time();prof,r,v,pop=context(sid,assets);users=list(prof.index);uindex={u:i for i,u in enumerate(users)}
        n=len(users);rankings=np.full((n,9,10),-1,dtype=np.int16);weights=np.zeros((n,9,10));mu_selected=np.zeros((n,9,10));seen=np.zeros(n,bool);fallbacks=0
        for batch in batches(sid,prof,asset_index):
            x=pack(batch,prof,asset_index,r,v,pop);rows=np.array([uindex[u] for u in x['userids']]);seen[rows]=True
            mu_lookup=np.zeros((len(batch),len(assets)))
            rr,cc=np.where(x['valid']);mu_lookup[rr,x['asset_codes'][rr,cc]]=x['mu'][rr,cc]
            for i in range(1,10):
                ids,ww,fallback=calculate(x,i/10,assets)
                rankings[rows,i-1]=ids;weights[rows,i-1]=ww;fallbacks+=fallback
                mu_selected[rows,i-1]=np.take_along_axis(mu_lookup,np.maximum(ids,0),axis=1)
            if seen.sum()%5120<128:print(json.dumps({'slice':sid,'native_users':int(seen.sum()),'fallbacks':fallbacks}),flush=True)
        labels=build_newbuy_labels(DATA/'clean/transactions.parquet',pd.Timestamp(meta['test_start']).date(),pd.Timestamp(meta['test_end']).date(),chunk_rows=200000)
        y=np.zeros((n,len(assets)),bool)
        for u,items in labels.items():
            if u in uindex:
                for a in items:
                    if a in asset_index:y[uindex[u],asset_index[a]]=True
        start=pd.Timestamp(meta['test_start']);end=pd.Timestamp(meta['test_end'])
        native=np.full(len(assets),np.nan)
        if start in px.index and end in px.index:
            p0=px.loc[start].reindex(assets).to_numpy();p1=px.loc[end].reindex(assets).to_numpy()
            good=(p0>0)&np.isfinite(p0)&np.isfinite(p1);native[good]=p1[good]/p0[good]-1
        categories=prof.category.to_numpy();rho=prof.gamma_or_eta.to_numpy();extra=np.array([SCALES.get(c,1.) for c in categories])
        groups=[];metricrows=[]
        for a in range(9):
            ids=rankings[:,a];w=weights[:,a];valid=(ids>=0)&seen[:,None];safe=np.maximum(ids,0);count=valid.sum(axis=1);active=seen&(count>0)
            relevant=np.take_along_axis(y,safe,axis=1)&valid;dcg=(relevant*discount).sum(axis=1);idcg=ideal[np.minimum(count,y.sum(axis=1))]
            ndcg=np.divide(dcg,idcg,out=np.zeros(n),where=idcg>0)
            rr=native[safe];price_valid=np.isfinite(rr)&valid
            gross=np.sum(np.where(price_valid,w*rr,0),axis=1);roi=gross-.003*w.sum(axis=1)
            single=np.divide(np.sum(np.where(valid,-np.expm1(-rho[:,None]*mu_selected[:,a]),0),axis=1),count,out=np.zeros(n),where=count>0)
            double=np.divide(np.sum(np.where(valid,-np.expm1(-rho[:,None]*extra[:,None]*mu_selected[:,a]),0),axis=1),count,out=np.zeros(n),where=count>0)
            for cat in ['Overall','Aggressive','Income','Balanced','Conservative','Unknown']:
                keep=active if cat=='Overall' else active&(categories==cat)
                if not keep.any():continue
                groups.append({'slice_id':sid,'lambda':(a+1)/10,'category':cat,'users':int(keep.sum()),'ndcg_percent':float(ndcg[keep].mean()*100),'roi_percent':float(roi[keep].mean()*100),'gross_period_roi_percent':float(gross[keep].mean()*100),'utility_single_scale_x1e4':float(single[keep].mean()*1e4),'utility_readme_double_scale_x1e4':float(double[keep].mean()*1e4),'mean_slate_size':float(count[keep].mean()),'missing_exact_endpoint_slot_fraction':float((valid&~price_valid)[keep].sum()/valid[keep].sum())})
            metricrows.append(pd.DataFrame({'user_id':users,'lambda':(a+1)/10,'ndcg':ndcg,'native_period_roi':roi,'proxy_utility_single':single,'proxy_utility_double':double,'active':active}))
        np.savez_compressed(sd/'rankings.npz',user_ids=np.array(users),asset_ids=np.array(assets),ranked_asset_indices=rankings,learned_weights=weights)
        pd.concat(metricrows,ignore_index=True).to_parquet(sd/'user-metrics.parquet',index=False,compression='zstd')
        result={'status':'complete','slice':sid,'users':int(seen.sum()),'lambdas':9,'groups':groups,'scalar_fallbacks':fallbacks,'code_sha256':hashes,'seconds':time.time()-began}
        dump(summarypath,result);summaries.append(result);allgroups+=groups;print(json.dumps({k:v for k,v in result.items() if k not in ['groups','code_sha256']}),flush=True)
    frame=pd.DataFrame(allgroups);out=[]
    for (lam,cat),g in frame.groupby(['lambda','category']):
        record={'lambda':lam,'category':cat,'user_slices':int(g.users.sum()),'slices':len(g)}
        for col in ['ndcg_percent','roi_percent','gross_period_roi_percent','utility_single_scale_x1e4','utility_readme_double_scale_x1e4','mean_slate_size','missing_exact_endpoint_slot_fraction']:record[col]=float(np.average(g[col],weights=g.users))
        out.append(record)
    pd.DataFrame(out).to_csv(OUT/'aggregate.csv',index=False)
    dump(OUT/'status.json',{'status':'complete_released_algorithm_artifact_scope','slices':16,'lambda_variants':9,'method_slice_runs':144,'parity_cases':parity['cases'],'code_sha256':hashes,'scope':'Native full-candidate entropy/token routine,learned-weight ROI,exact-date prices,and separate single/double risk scaling utilities. Explicitly differs from published equations and equal-slate ROI. Not a daily market-neutral ledger.'})
    print(json.dumps({'status':'complete_released_algorithm_artifact_scope','method_slice_runs':144}),flush=True)


if __name__=='__main__':main()
