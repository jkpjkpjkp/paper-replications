#!/usr/bin/env python3
"""Run reproducible rolling, paired regressions and share-level portfolios."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse,json,pickle,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from data import FACTORS,HERE,ROOT,load_year,features
from models import METHODS,Regression
from portfolio import Ledger,allocate_leg

OUT=HERE/'results'; FIT=HERE/'fits';COSTS=[0.,.0005,.001]
HORIZON=1
SPECS=[]
for j,f in enumerate(FACTORS):
    SPECS.append(dict(id=f+'__plain',factor=f,factor_index=j,method='plain',cycle=False))
    for method in METHODS:
        for cycle in [False,True]:
            SPECS.append(dict(id=f+'__'+method+('__cycle' if cycle else '__base'),factor=f,factor_index=j,method=method,cycle=cycle))
SPECS.append(dict(id='Mom12m__control',factor='Mom12m',factor_index=-1,method='plain',cycle=False))
NM=len(SPECS)
def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def write(p,obj):p.write_text(json.dumps(obj,indent=2,default=str))

def fit_all(start=2005,end=2025):
    FIT.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
    trainpath=HERE/('data/train.npz' if HORIZON==1 else 'data/train5.npz')
    z=dict(np.load(trainpath))
    signature={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'models.py',HERE/'data.py',HERE/'design.json',trainpath]}
    signature['horizon']=HORIZON
    cache_valid=(FIT/'signature.json').exists() and json.loads((FIT/'signature.json').read_text())==signature
    audit=[]
    for year in range(start,end+1):
        path=FIT/f'{year}.pkl'
        if path.exists() and cache_valid:
            audit.extend(json.loads((FIT/f'{year}.json').read_text()));continue
        fitted={};records=[]
        for j,factor in enumerate(FACTORS):
            mask=(z['year']>=year-10)&(z['year']<year)&(z['factor']==j)
            assert np.max(z['year'][mask])<year
            for method in METHODS:
                for cycle in [False,True]:
                    key=factor+'__'+method+('__cycle' if cycle else '__base')
                    x=z['Xcycle' if cycle else 'Xbase'][mask].astype(float)
                    model=Regression(method).fit(x,z['y'][mask],z['weight'][mask],variance_horizon=HORIZON)
                    fitted[key]=model
                    records.append(dict(id=key,prediction_year=year,training_first_year=int(z['year'][mask].min()),training_last_year=int(z['year'][mask].max()),**model.audit()))
            log(f'fit {year} {factor}: {mask.sum():,} training rows')
        path.write_bytes(pickle.dumps(fitted,protocol=5));write(FIT/f'{year}.json',records);audit.extend(records)
    write(OUT/'fit_audit.json',audit)
    write(FIT/'signature.json',signature)

def evaluate(start=2005,end=2025):
    OUT.mkdir(exist_ok=True);write(OUT/'models.json',SPECS)
    calendar=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])
    calendar=calendar[calendar.date.dt.year.between(start,end)]
    ledgers=[Ledger(NM,c) for c in COSTS];outputs=[];days=[];predictive=[]
    c=duckdb.connect();c.execute('SET threads=4')
    train=dict(np.load(HERE/('data/train.npz' if HORIZON==1 else 'data/train5.npz')))
    for year in range(start,end+1):
        log(f'Evaluation {year}')
        fitted=pickle.loads((FIT/f'{year}.pkl').read_bytes())
        z=load_year(year)
        rr=np.flatnonzero(z['idx']%5==0)
        reb={k:v[rr] for k,v in z.items()}
        mu=np.zeros((NM,len(rr)),np.float32);sigma=np.full_like(mu,.03)
        for j,f in enumerate(FACTORS):
            rows=np.flatnonzero(reb['leg'][:,j]!=0)
            xb=features(reb,j,rows);xc=features(reb,j,rows,cycle=True)
            for m,spec in enumerate(SPECS):
                if spec['factor']!=f or spec['method']=='plain':continue
                model=fitted[spec['id']]
                mu[m,rows],sigma[m,rows]=model.predict(xc if spec['cycle'] else xb)
                ts=(train['year']==year)&(train['factor']==j)
                tm,tv=model.predict(train['Xcycle' if spec['cycle'] else 'Xbase'][ts])
                weight=train['weight'][ts];weight=weight/weight.sum()
                err=train['y'][ts]-tm
                predictive.append(dict(year=year,id=spec['id'],mse=float(weight@(err**2)),gaussian_score=float(weight@(np.log(tv**2)+HORIZON*err**2/tv**2)),n=int(ts.sum())))
        rix=reb['idx'];starts=np.r_[0,np.flatnonzero(np.diff(rix))+1];ends=np.r_[starts[1:],len(rix)]
        rmap={int(rix[a]):(a,b) for a,b in zip(starts,ends)}
        market=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        # Masked Arrow/duckdb numeric arrays retain nulls as masked values.
        for key in ['ret','retx']:
            market[key]=np.ma.filled(market[key],np.nan)
        mi=market['idx'];ss=np.r_[0,np.flatnonzero(np.diff(mi))+1];ee=np.r_[ss[1:],len(mi)]
        mmap={int(mi[a]):(a,b) for a,b in zip(ss,ee)}
        for day in calendar[calendar.date.dt.year.eq(year)].itertuples():
            idx=int(day.idx);a,b=mmap[idx]
            terminal=idx==int(calendar.idx.iloc[-1]);result=[]
            for ci,ledger in enumerate(ledgers):
                ids=None;w=None
                if idx%5==0:
                    ra,rb=rmap.get(idx,(0,0));ids=reb['permno'][ra:rb]
                    old=ledger.lookup(ids);w=np.zeros((NM,len(ids)))
                    for j,f in enumerate(FACTORS):
                        leg=reb['leg'][ra:rb,j];long=np.flatnonzero(leg==1);short=np.flatnonzero(leg==-1)
                        minimum=5 if f=='RDIPO' else 20
                        if min(len(long),len(short))<minimum:continue
                        plain=j*13;w[plain,long]=1/len(long);w[plain,short]=-1/len(short)
                        mods=np.arange(plain+1,plain+13)
                        for sign,rows in [(1,long),(-1,short)]:
                            aa=allocate_leg(mu[np.ix_(mods,ra+rows)],sigma[np.ix_(mods,ra+rows)],sign*old[np.ix_(mods,rows)],COSTS[ci])
                            w[np.ix_(mods,rows)]=sign*aa
                    mom=reb['mom'][ra:rb];lo=np.flatnonzero(mom>2/3);sh=np.flatnonzero(mom<1/3)
                    if min(len(lo),len(sh))>=20:w[-1,lo]=1/len(lo);w[-1,sh]=-1/len(sh)
                    assert np.max(abs(w.sum(axis=1)))<1e-9
                    assert np.max(abs(w))<=.2+1e-9
                result.append(ledger.step(market['permno'][a:b],market['ret'][a:b],market['retx'][a:b],ids,w,terminal=terminal))
            outputs.append({k:np.stack([r[k] for r in result]) for k in result[0]});days.append(day.date)
        # Annual checkpoints only; full output created after completing series.
        log(f'{year} complete; max accounting residual {max(l.max_accounting_error for l in ledgers):.2g}')
    arrays={k:np.stack([r[k] for r in outputs]) for k in outputs[0]}
    arrays.update(dates=np.array(days,dtype='datetime64[D]'),costs=COSTS)
    np.savez_compressed(OUT/'daily.npz',**arrays)
    pd.DataFrame(predictive).to_csv(OUT/'predictive_scores.csv',index=False)
    write(OUT/'execution_audit.json',dict(max_accounting_error=max(l.max_accounting_error for l in ledgers),bankrupt_models=[dict(cost=COSTS[i],models=[SPECS[k]['id'] for k in np.flatnonzero(l.dead)]) for i,l in enumerate(ledgers)],terminal_holdings=[int(l.hold.size) for l in ledgers],days=len(days),strategies=NM,costs=COSTS,design_sha256=hashlib.sha256((HERE/'design.json').read_bytes()).hexdigest()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','evaluate','all']);p.add_argument('--start',type=int,default=2005);p.add_argument('--end',type=int,default=2025);p.add_argument('--horizon',type=int,choices=[1,5],default=1);a=p.parse_args()
    HORIZON=a.horizon
    if HORIZON==5:OUT=HERE/'results_h5';FIT=HERE/'fits_h5'
    if a.stage in ['fit','all']:fit_all(a.start,a.end)
    if a.stage in ['evaluate','all']:evaluate(a.start,a.end)
