#!/usr/bin/env python3
"""Factor x entry-close x exit-close exhaustive regressions and exact neutral books."""
from __future__ import annotations
import argparse,json,math,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mpl-cache'))
import numpy as np
import pandas as pd
import duckdb
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import research as base

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DATA=HERE/'horizon_data';OUT=HERE/'horizon_results'
DESIGN={
 'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
 'request':'Exhaustive factor209 x buy N sessions before forecast announcement x sell M sessions after known actual announcement; same-factor N=M=1 baseline.',
 'grid':{'N':list(range(1,31)),'M':list(range(1,31)),'factors':209,'total':188100},
 'entry':'Forecast F comes only from known regular historical releases. Buy close C=F-N; first return C+1. Entry eligibility is inherited lagged >=5-analyst, US common stock, regular65 profile at C. Actual next E/K labels NEVER filter entries. Factor previous completed calendar-month signed percentile at entry, frozen for the entire spell.',
 'exit':'Sell close min(F+40,max(next economic event_idx+M,next known_idx)). Next release becomes tradable only after known_idx close. If no next record, timeout F+40. Disappearance closes at prior mark. Keep trades through subsequent analyst/regularity changes. Desired exit and availability delay are audited. E/F are distinct; actual-date prior entries cannot be reconstructed without point-in-time calendars.',
 'portfolio':'Primary at least ONE stock each leg; every day target +100%/-100% after-cost NAV (200% gross, zero dollar net). Initiate stocks ONLY on scheduled entry close, never late. Freeze sign until scheduled exit; daily equal-dollar rescale within leg. When either leg empties force whole book flat and abandon remaining spells, preventing late reentry. New scheduled entries can restart only on their original day. Report forced exits, concentration and active days. >=5 each-leg sensitivity.',
 'costs':'5bp every bought/sold dollar; exact cost-dependent NAV/share drift, total-return P&L, retx share marks, 1% annual short borrow. Also gross and10bp. Charge terminal liquidation at every simulation endpoint. Cash zero; no impact/financing/rebate or measured borrow availability.',
 'regression':'All188100 training event regressions y=a+b*(signed entry rank-.5), forecast-date-clustered errors; tail-LS event payoff/cost proxy. Labels purged with common F in training_start+30..training_end-40. Event regressions are an exhaustive shortlist screen, not calendar-time strategy returns.',
 'selection':'Training1985-2004 only. Per factor retain three distinct screen candidates: best net annual proxy Sharpe, highest mean proxy, largest regression t; additionally top2 per N/pool (published<=2004 versus all). Exact calendar execution on this deduplicated shortlist plus every same-factor -1/+1 baseline. Freeze per-factor best exact net Sharpe (active>=20%), global primary published<=2004, global unrestricted, per-N published<=2004 and best means before later evaluation. No 2005+ return is used to choose.',
 'inference':'Calendar-time paired differences to identical-factor -1/+1 baseline; HAC20/60, 95% CI, Holm across frozen candidates by period; common-active-day paired sensitivity; pooled post2004 and separate validation2005-2014/holdout2015-2025. Supplement long-short same-factor baseline performance and max drawdown. Retrospective study after earlier related holdouts were examined.',
 'oracle':'Observed future actual announcement dates are never used to schedule primary entries. Actual versus forecast entry offsets are diagnostic only; an exact realized-date pre-entry strategy is not identified from these data.',
}

def log(s):base.log(s)
def save(p,x):base.save(p,x)
def freeze():
 DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 p=OUT/'design.json'
 if p.exists():
  a=json.loads(p.read_text());b=DESIGN.copy();a.pop('created_utc');b.pop('created_utc');assert a==b
 else:save(p,DESIGN)
def connection():
 c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='10GB'");c.execute(f"SET temp_directory='{DATA}/tmp'");return c

def prepare():
 freeze();c=connection()
 source=str(base.PANEL/'[12]*.parquet')
 # Future columns are labels for the stopping rule, not entry criteria.
 c.execute(f"""CREATE TEMP TABLE ev AS SELECT permno,known_idx,event_idx,
 row_number() OVER(ORDER BY permno,known_idx) AS event_id,
 lead(event_idx) OVER(PARTITION BY permno ORDER BY known_idx) AS next_event_LABEL,
 lead(known_idx) OVER(PARTITION BY permno ORDER BY known_idx) AS next_known_LABEL
 FROM read_parquet('{ROOT}/results/linked_events.parquet')""")
 if not (DATA/'entries.parquet').exists():
  log('Build all ex-ante entry opportunities')
  c.execute(f"""COPY(SELECT p.permno,p.date,p.idx AS entry_idx,p.forecast_idx,
   CAST(p.forecast_idx-p.idx+1 AS INTEGER) AS n,e.event_id,e.next_event_LABEL,e.next_known_LABEL
   FROM read_parquet('{source}') p JOIN ev e USING(permno,known_idx,event_idx)
   WHERE p.eligible AND p.forecast_idx-p.idx+1 BETWEEN 1 AND 30
   ORDER BY p.permno,p.idx) TO '{DATA}/entries.parquet' (FORMAT PARQUET)""")
 if not (DATA/'marks.parquet').exists():
  log('Build total-return prefixes and contiguous listing spells')
  c.execute(f"""COPY(WITH r AS (
   SELECT p.permno,p.idx,coalesce(p.ret,0) AS ret,coalesce(p.retx,p.ret,0) AS retx,
    p.ret IS NULL AS missing,lag(p.idx) OVER(PARTITION BY p.permno ORDER BY p.idx) AS prior_idx
   FROM read_parquet('{source}') p SEMI JOIN (SELECT DISTINCT permno FROM read_parquet('{DATA}/entries.parquet')) e USING(permno)),
   s AS(SELECT *,sum(CASE WHEN prior_idx IS NULL OR idx-prior_idx>1 THEN 1 ELSE 0 END)
     OVER(PARTITION BY permno ORDER BY idx) AS span FROM r),
   z AS(SELECT *,sum(CASE WHEN ret>-1 THEN ln(1+ret) ELSE 0 END)
      OVER(PARTITION BY permno ORDER BY idx) AS logtotal,
     sum((ret<=-1)::INTEGER) OVER(PARTITION BY permno ORDER BY idx) AS zeros,
     max(idx) OVER(PARTITION BY permno,span) AS span_end FROM s)
   SELECT permno,idx,ret,retx,missing,logtotal,zeros,span_end FROM z ORDER BY permno,idx)
   TO '{DATA}/marks.parquet' (FORMAT PARQUET)""")
 log('Load return-prefix lookup')
 marks=c.execute(f"SELECT * FROM read_parquet('{DATA}/marks.parquet') ORDER BY permno,idx").df()
 key=marks.permno.to_numpy(np.int64)*100000+marks.idx.to_numpy(np.int64)
 cum=marks.logtotal.to_numpy(float);zeros=marks.zeros.to_numpy(np.int32);span=marks.span_end.to_numpy(np.int32)
 r=marks.ret.to_numpy(float)
 # Small annual return caches support later exact calendar portfolios.
 cal=base.calendar();year_of=dict(zip(cal.idx.to_numpy(int),cal.date.dt.year.to_numpy(int)))
 years=marks.idx.map(year_of)
 for y in range(1985,2026):
  dest=DATA/f'market_{y}.npz'
  if dest.exists():continue
  q=marks.loc[years.eq(y)].sort_values(['idx','permno'])
  np.savez_compressed(dest,permno=q.permno.to_numpy(np.int32),idx=q.idx.to_numpy(np.int32),ret=q.ret.to_numpy(float),retx=q.retx.to_numpy(float),missing=q.missing.to_numpy(bool))
 del years
 # Load only rank lookups once per entry year; aggregate by N after lookup.
 for y in range(1985,2026):
  dest=DATA/f'entries_{y}.npz'
  if dest.exists():continue
  log(f'Entry snapshot and stopping labels {y}')
  f=c.execute(f"SELECT * FROM read_parquet('{DATA}/entries.parquet') WHERE year(date)={y} ORDER BY entry_idx,permno").df()
  f['date']=pd.to_datetime(f.date)
  ym=f.date.dt.to_period('M')-1
  ym=(ym.dt.year*100+ym.dt.month).to_numpy(np.int64)
  zs=[np.load(base.DATA/f'ranks_finite_{z}.npz') for z in sorted(set(ym//100))]
  rk=np.concatenate([z['key'] for z in zs]);rr=np.concatenate([z['ranks'] for z in zs]);[z.close() for z in zs]
  wanted=ym*1000000+f.permno.to_numpy(np.int64);loc=np.searchsorted(rk,wanted)
  valid=(loc<len(rk))&(rk[np.minimum(loc,len(rk)-1)]==wanted)
  rank=np.full((len(f),len(base.metadata()['factors'])),np.nan,np.float32);rank[valid]=rr[loc[valid]]
  mloc=np.searchsorted(key,f.permno.to_numpy(np.int64)*100000+f.entry_idx.to_numpy(np.int64))
  assert np.all(key[mloc]==f.permno.to_numpy(np.int64)*100000+f.entry_idx.to_numpy(np.int64))
  fcast=f.forecast_idx.to_numpy(np.int32)
  e=f.next_event_LABEL.to_numpy(float);k=f.next_known_LABEL.to_numpy(float)
  scheduled=np.minimum(fcast[:,None]+40,np.maximum(e[:,None]+np.arange(1,31),k[:,None]))
  scheduled=np.where(np.isfinite(scheduled),scheduled,fcast[:,None]+40).astype(np.int32)
  exit_idx=np.minimum(scheduled,span[mloc,None])
  assert np.all(exit_idx>=f.entry_idx.to_numpy()[:,None]-1)
  wanted_exit=f.permno.to_numpy(np.int64)[:,None]*100000+exit_idx
  eloc=np.searchsorted(key,wanted_exit);assert np.all(key[eloc]==wanted_exit)
  before=cum[mloc]-np.where(r[mloc]>-1,np.log1p(np.maximum(r[mloc],-.999999999999)),0)
  zbefore=zeros[mloc]-(r[mloc]<=-1)
  ret=np.expm1(cum[eloc]-before[:,None]);ret=np.where(zeros[eloc]>zbefore[:,None],-1.,ret)
  duration=exit_idx-f.entry_idx.to_numpy(np.int32)[:,None]+1
  np.savez_compressed(dest,permno=f.permno.to_numpy(np.int32),entry_idx=f.entry_idx.to_numpy(np.int32),forecast_idx=fcast,n=f.n.to_numpy(np.int8),event_id=f.event_id.to_numpy(np.int64),next_event_LABEL=e,next_known_LABEL=k,rank=rank,returns=ret,duration=duration,exit_idx=exit_idx,scheduled_exit_idx=scheduled)
 # Training-only consolidated matrices; annual caches retain all later data.
 combined={}
 for y in range(1985,2005):
  with np.load(DATA/f'entries_{y}.npz') as z:
   for field in ['n','rank','returns','duration','entry_idx','forecast_idx','exit_idx','event_id','permno']:
    combined.setdefault(field,[]).append(z[field])
 combined={k:np.concatenate(v) for k,v in combined.items()}
 for n in range(1,31):
  dest=DATA/f'opportunities_n{n:02d}.npz'
  if dest.exists():continue
  log(f'Consolidate exhaustive regression cache N={n}')
  keep=combined['n']==n
  np.savez_compressed(dest,**{k:v[keep] for k,v in combined.items() if k!='n'})
 save(OUT/'input_audit.json',dict(factor_source=base.metadata()['sha256'],prior_event_validation=str(ROOT/'coverage5/results_regular65/validation_checks.json'),entries_sha256=base.sha(DATA/'entries.parquet'),marks_sha256=base.sha(DATA/'marks.parquet')))

def model(factor,n,m):return dict(id=f'{factor}__N{int(n):02d}_M{int(m):02d}',factor=factor,n=int(n),m=int(m))
def shortlist():
 s=pd.read_csv(OUT/'regression_screen.csv').rename(columns={'n':'reg_n','buy_before':'n','sell_after':'m'});s=s[s.eligible].copy();chosen=set()
 for metric in ['sharpe_proxy','annual_mean_proxy','cluster_t']:
  for row in s.sort_values([metric,'factor','n','m'],ascending=[False,True,True,True]).groupby('factor').head(1).itertuples():chosen.add((row.factor,int(row.n),int(row.m)))
 old=set(base.metadata()['published_by2004'])
 for pool in [s,s[s.factor.isin(old)]]:
  for row in pool.sort_values(['sharpe_proxy','factor','n','m'],ascending=[False,True,True,True]).groupby('n').head(2).itertuples():chosen.add((row.factor,int(row.n),int(row.m)))
 # Every factor's -1/+1 baseline is an exact portfolio, not a screen proxy.
 chosen.update((f,1,1) for f in base.metadata()['factors'])
 models=[model(*x) for x in sorted(chosen)]
 save(OUT/'shortlist.json',models);return models

def summary(a,dates,models,costs,name):
 rows=[]
 for period,(lo,hi) in base.PERIODS.items():
  mask=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
  if not mask.any():continue
  for ci,cost in enumerate(costs):
   ret=a['returns'][mask,ci];mean=ret.mean(axis=0)*252;vol=ret.std(axis=0,ddof=1)*np.sqrt(252)
   nav=np.cumprod(1+ret,axis=0);peak=np.maximum.accumulate(np.vstack([np.ones((1,len(models))),nav]),axis=0)[1:]
   for j,m in enumerate(models):
    rows.append(dict(**m,period=period,cost_bps=cost*10000,annual_mean=mean[j],cagr=nav[-1,j]**(252/len(ret))-1,sharpe=mean[j]/vol[j] if vol[j]>1e-12 else np.nan,vol=vol[j],max_drawdown=(nav[:,j]/peak[:,j]-1).min(),active=a['active'][mask,ci,j].mean(),annual_turnover=a['turnover'][mask,ci,j].mean()*252,annual_cost=a['fees'][mask,ci,j].mean()*252,mean_longs=a['n_long'][mask,ci,j].mean(),mean_shorts=a['n_short'][mask,ci,j].mean(),bankrupt=bool(np.any(a['returns'][:,ci,j]<=-1))))
 df=pd.DataFrame(rows);df.to_csv(OUT/f'{name}_statistics.csv',index=False);return df

def simulate(models,start,end,name,costs=(.0005,),min_names=1):
 freeze();factors=base.metadata()['factors'];fi={f:j for j,f in enumerate(factors)}
 cal=base.calendar();cal=cal[cal.date.dt.year.between(start,end)].reset_index(drop=True)
 # Dense stock axis permits efficient hundreds-model exact neutrality.
 stocks=np.array([],int)
 nm=len(models);nc=len(costs);ns=0;shape=(len(cal),nc,nm)
 arr={k:np.zeros(shape) for k in ['returns','turnover','fees','active','n_long','n_short']}
 direction=np.zeros((nm,ns),np.int8);expiry=np.zeros((nm,ns),np.int32);v=np.zeros((nc,nm,ns));dead=np.zeros((nc,nm),bool)
 max_net=0.;accounting_error=0.;forced=0;attempted=0;opened=0;overlap=0;missingheld=0;di=0
 byn={n:np.array([j for j,m in enumerate(models) if m['n']==n],int) for n in sorted({m['n'] for m in models})}
 cost=np.asarray(costs)[:,None]
 for year in range(start,end+1):
  log(f'{name}: {year}, {nm} models, {nc} costs, minleg={min_names}')
  z=np.load(DATA/f'entries_{year}.npz');e={k:z[k] for k in z.files};z.close()
  carried=(direction!=0).any(axis=0)|(np.abs(v).sum(axis=(0,1))>0)
  newstocks=np.union1d(np.unique(e['permno']),stocks[carried]);dest=np.searchsorted(newstocks,stocks[carried])
  newdir=np.zeros((nm,len(newstocks)),np.int8);newdir[:,dest]=direction[:,carried]
  newexp=np.zeros((nm,len(newstocks)),np.int32);newexp[:,dest]=expiry[:,carried]
  newv=np.zeros((nc,nm,len(newstocks)));newv[:,:,dest]=v[:,:,carried]
  stocks=newstocks;direction=newdir;expiry=newexp;v=newv;ns=len(stocks)
  ez={int(k):g.to_numpy(int) for k,g in pd.Series(np.arange(len(e['entry_idx']))).groupby(e['entry_idx'])}
  z=np.load(DATA/f'market_{year}.npz');mkt={k:z[k] for k in z.files};z.close()
  groups={int(k):g.to_numpy(int) for k,g in pd.Series(np.arange(len(mkt['idx']))).groupby(mkt['idx'])}
  for idx in cal.loc[cal.date.dt.year.eq(year),'idx']:
   rr=groups.get(int(idx),np.array([],int));loc=np.searchsorted(stocks,mkt['permno'][rr]);keep=(loc<ns)&(stocks[np.minimum(loc,ns-1)]==mkt['permno'][rr]);rr=rr[keep];loc=loc[keep];exists=np.zeros(ns,bool);exists[loc]=True
   direction[(expiry<int(idx))|(~exists)[None,:]]=0
   starts=ez.get(int(idx),np.array([],int))
   for n,jj in byn.items():
    kk=starts[e['n'][starts]==n]
    if not len(kk):continue
    ii=np.searchsorted(stocks,e['permno'][kk]);ranks=e['rank'][kk][:,[fi[models[j]['factor']] for j in jj]].T
    signs=np.where(ranks>2/3,1,np.where(ranks<1/3,-1,0)).astype(np.int8)
    prev=direction[np.ix_(jj,ii)];overlap+=int(((prev!=0)&(signs!=0)).sum())
    signs=np.where(prev==0,signs,0);attempted+=int((signs!=0).sum())
    direction[np.ix_(jj,ii)]=np.where(signs!=0,signs,prev)
    exits=e['exit_idx'][kk][:,[models[j]['m']-1 for j in jj]].T
    expiry[np.ix_(jj,ii)]=np.where(signs!=0,exits,expiry[np.ix_(jj,ii)])
   nl=(direction>0).sum(axis=1);nt=(direction<0).sum(axis=1);active=(nl>=min_names)&(nt>=min_names)
   forced+=int(((~active)&(np.abs(v).sum(axis=(0,2))>0)).sum())
   direction[~active]=0
   wl=(direction>0)/np.maximum(nl,1)[:,None];ws=(direction<0)/np.maximum(nt,1)[:,None]
   w=np.where(active[:,None],wl-ws,0.)
   r=np.zeros(ns);rx=np.zeros(ns);r[loc]=mkt['ret'][rr];rx[loc]=mkt['retx'][rr]
   missingheld+=int(((np.abs(v[:,:,loc])>0)&mkt['missing'][rr]).sum())
   ww=w[None,:,:]*(~dead[:,:,None]);x=np.ones((nc,nm))
   for _ in range(5):x=1-cost*np.abs(x[:,:,None]*ww-v).sum(axis=2)
   pos=x[:,:,None]*ww;turn=np.abs(pos-v).sum(axis=2);fees=cost*turn
   long=np.maximum(pos,0).sum(axis=2);short=-np.minimum(pos,0).sum(axis=2)
   max_net=max(max_net,float(np.abs(long-short).max()));accounting_error=max(accounting_error,float(np.abs(x-(1-fees)).max()))
   borrow=np.where(cost>0,.01/252,0)*short
   pnl=(pos*r).sum(axis=2)-fees-borrow
   insolvent=1+pnl<=0;dead|=insolvent;pnl=np.maximum(pnl,-1.)
   after=np.where(dead[:,:,None],0.,pos*(1+rx)/np.maximum(1+pnl[:,:,None],1e-300))
   if di==len(cal)-1:
    terminal_turn=np.abs(after).sum(axis=2);terminal_fee=cost*terminal_turn
    fees+=(1+pnl)*terminal_fee;turn+=(1+pnl)*terminal_turn
    pnl=(1+pnl)*(1-terminal_fee)-1;after[:]=0
   arr['returns'][di]=pnl;arr['turnover'][di]=turn;arr['fees'][di]=fees;arr['active'][di]=long>0;arr['n_long'][di]=(pos>0).sum(axis=2);arr['n_short'][di]=(pos<0).sum(axis=2)
   v=after;di+=1
 assert max_net<1e-10 and accounting_error<1e-10
 np.savez_compressed(OUT/f'{name}.npz',**arr,dates=cal.date.to_numpy(dtype='datetime64[D]'),costs=costs)
 save(OUT/f'{name}_models.json',models)
 save(OUT/f'{name}_audit.json',dict(max_net_dollar_error=max_net,accounting_error=accounting_error,forced_flat_modeldays=forced,attempted_entries=attempted,overlaps_skipped=overlap,missing_held_marks=missingheld,terminal_liquidation_charged=True,strict_scheduled_entries=True,min_names_each_leg=min_names,insolvent_models=[models[j]['id'] for j in np.flatnonzero(dead.any(axis=0))]))
 return summary(arr,cal.date.to_numpy(dtype='datetime64[D]'),models,costs,name)

def train():
 freeze();models=shortlist();s=simulate(models,1985,2004,'exact_train')
 z=s[(s.period=='development')&(s.active>=.20)&(~s.bankrupt)].copy();old=set(base.metadata()['published_by2004'])
 # Preserve a frozen candidate for every factor, not just one global optimum.
 per=z.sort_values(['sharpe','id'],ascending=[False,True]).groupby('factor').head(1)
 chosen=set(per.id)
 for pool in [z,z[z.factor.isin(old)]]:
  chosen.update(pool.sort_values(['sharpe','id'],ascending=[False,True]).groupby('n').head(1).id)
  chosen.update(pool.nlargest(1,'annual_mean').id)
 eligible=z[z.factor.isin(old)].sort_values(['sharpe','id'],ascending=[False,True]);primary=eligible.iloc[0].id
 lookup={m['id']:m for m in models};selected=[lookup[x] for x in sorted(chosen)]
 # Always include exact -1/+1 counterpart for every frozen candidate.
 selected_ids={m['id'] for m in selected}
 for f in sorted({m['factor'] for m in selected}):
  b=model(f,1,1)
  if b['id'] not in selected_ids:selected.append(b);selected_ids.add(b['id'])
 save(OUT/'frozen_choices.json',dict(frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),primary=primary,models=selected,candidates=sorted(chosen),per_factor_winners=per.id.tolist(),development_end='2004-12-31',screen_count=188100,exact_shortlist_count=len(models),minimum_training_active=.2,no_2005_plus_selection=True))
 log(f'Frozen primary {primary}; {len(chosen)} frozen candidates')

def evaluate():
 f=json.loads((OUT/'frozen_choices.json').read_text());models=f['models']
 simulate(models,2005,2025,'final',costs=(.0005,))
 p=next(m for m in models if m['id']==f['primary']);pair=[p,model(p['factor'],1,1)]
 pair=list({m['id']:m for m in pair}.values())
 simulate(pair,2005,2025,'cost_sensitivity',costs=(0.,.0005,.001))
 simulate(pair,2005,2025,'min5_sensitivity',min_names=5)
 simulate(pair,2015,2025,'holdout_reset')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);a=p.parse_args();globals()[a.stage]()
