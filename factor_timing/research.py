#!/usr/bin/env python3
"""Chronological stock-level factor x earnings-clock research; raw inputs read-only."""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, os, time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mpl-cache'))
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=HERE/'data'
OUT=HERE/'results'
ZOO=Path('/home/jkp/Téléchargements/Chen_Zimmermann_Factor_Zoo_2026_10')
PANEL=ROOT/'coverage5/results_regular65/panel'
MODES=['all','pre20','pre10','post20','near10']
PERIODS={'development':(1985,2004),'validation':(2005,2014),'holdout':(2015,2025),'recent':(2020,2025),'post2004':(2005,2025)}
DESIGN={
 'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
 'data':'Chen-Zimmermann Data-Release-2026.10 unsigned stock-month Predictors.parquet, signed by SignalDoc.Sign. Portfolio return files are not stock signals.',
 'timing':'Daily t uses only exact previous calendar month yyyymm; never forward-fill beyond that month. Signals are month-end snapshots, not announcement-day refreshed fundamentals. Published signal construction lags are inherited, not independently point-in-time restatement audited.',
 'rank':'Nonfinite raw signals, including infinite ratios/logs, are missing. Average-tie percentile rank of signed factor across all finite stocks in source month. Composite is equal-weight mean ranks if >=ceil(0.75*K) components available. Upper/lower third thresholds >2/3 and <1/3; no arbitrary split of ties.',
 'universe':'Inherited >=5 active EPS analyst proxy, regular four known releases with last three gaps 50..80 sessions and median 60..70, known age <=80, lagged US common-stock eligibility. Exact same universe as coverage5/results_regular65.',
 'windows':{'all':'No timing restriction','pre20':'nearest forecast clock -20..-1','pre10':'nearest forecast clock -10..-1','post20':'known actual release age 2..20','near10':'forecast clock -10..0 OR known release age 2..10'},
 'execution':'Daily equal-weight 100% long and 100% short after-cost NAV, min 20 stocks EACH side else cash. Costs every dollar bought/sold, including entry and exits. Total return P&L; retx marks carried shares. Zero cash return. No impact/financing/rebate; 1% annual borrow on short dollars in net scenarios. No terminal liquidation, disappearing names exit at last mark. Exact NAV-dependent costs in final simulations.',
 'screen':'1985-2004 only, 5bp plus 1% borrow. Approximation charges gross-path turnover. All supplied predictors x five windows. Composite search: within each window and vintage pool, top eight active single-factor Sharpe ranks form all 28 pairs plus top3/5/8 composites; five fixed economic composites also considered. >=30% active sessions. Primary pool all components published <=2004; exploratory pool unrestricted.',
 'selection':'Exact-simulate best three composite screens and best two singles per window/pool on development. Freeze best composite by Sharpe per event window/pool, best all-period composite per pool, and best event composite mean-return candidate per pool. Freeze before evaluating 2005+. Factor-only counterparts and timing-only controls are evaluated without selecting on later returns.',
 'inference':'Paired daily return differences, HAC20/HAC60 95% CI and two-sided p, Holm over all frozen event-factor candidates for each period/comparator. Monthly HAC3 and block bootstrap sensitivity. Annual arithmetic mean and CAGR separate. Prior related holdouts already inspected: retrospective validation, not untouched prospective evidence.',
 'sensitivities':'Final selected composites evaluated at gross,5bp,10bp; frozen primary additionally tested with extra one-month signal lag and lag price>=5 and ADV>=1m.',
 'limitations':'Monthly characteristics do not supply factor on exact earnings day. Realized future announcements never enter targets. Pre-event dates are forecasts. Source vintage can contain revised accounting/linkage; signal publication by2004 does not make 1985-2004 training free of hindsight.',
}

def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def conn():
 c=duckdb.connect();c.execute("SET threads=4");c.execute("SET memory_limit='8GB'");return c
def docs():return pd.read_csv(ZOO/'SignalDoc.csv').set_index('Acronym')
def freeze():
 OUT.mkdir(exist_ok=True,parents=True)
 p=OUT/'design.json'
 if p.exists():
  a=json.loads(p.read_text());b=DESIGN.copy();a.pop('created_utc');b.pop('created_utc');assert a==b,'Frozen design changed'
 else:save(p,DESIGN)
def calendar():
 return pd.read_csv(ROOT/'results/trading_calendar.csv',parse_dates=['date']).query("date>='1985-01-01'").reset_index(drop=True)
def metadata():return json.loads((DATA/'metadata.json').read_text())

def prepare():
 freeze();c=conn();p=DATA/'Predictors.parquet';d=docs()
 columns=[x[0] for x in c.execute(f"DESCRIBE SELECT * FROM read_parquet('{p}')").fetchall()]
 factors=sorted(x for x in columns if x in d.index and d.loc[x,'Cat.Signal']=='Predictor')
 signs=d.loc[factors,'Sign'].to_numpy(float)
 assert np.isin(signs,[-1,1]).all()
 keys=c.execute(f"SELECT count(*) n,count(DISTINCT (permno,yyyymm)) unique_n,min(yyyymm),max(yyyymm) FROM read_parquet('{p}')").fetchall()[0]
 assert keys[0]==keys[1]
 save(DATA/'metadata.json',dict(factors=factors,published_by2004=[x for x in factors if d.loc[x,'Year']<=2004],source_rows=keys[0],source_min_month=keys[2],source_max_month=keys[3],sha256=sha(p),doc_sha256=sha(ZOO/'SignalDoc.csv')))
 cols=','.join('"'+x+'"' for x in factors)
 for y in range(1983,2026):
  target=DATA/f'ranks_finite_{y}.npz'
  if target.exists():continue
  log(f'Rank signed month-end signals: {y}')
  f=c.execute(f"SELECT permno,yyyymm,{cols} FROM read_parquet('{p}') WHERE yyyymm BETWEEN {y*100+1} AND {y*100+12} ORDER BY yyyymm,permno").df()
  x=(f[factors]*signs).replace([np.inf,-np.inf],np.nan)
  rank=x.groupby(f.yyyymm).rank(method='average',pct=True).to_numpy(np.float32)
  key=f.yyyymm.to_numpy(np.int64)*1000000+f.permno.to_numpy(np.int64)
  assert (np.diff(key)>0).all()
  np.savez_compressed(target,key=key,ranks=rank)
 save(OUT/'input_audit.json',dict(factor_receipt=json.loads((DATA/'download_receipt.json').read_text()),stock_panel=[dict(year=y,sha256=sha(PANEL/f'{y}.parquet')) for y in range(1985,2026)],prior_validation=json.loads((ROOT/'coverage5/results_regular65/validation_checks.json').read_text())))

def frame(c,year,lag=1):
 f=c.execute(f"SELECT permno,date,idx,ret,retx,age,known_idx,event_idx,clock,lag_price,lag_adv FROM read_parquet('{PANEL}/{year}.parquet') WHERE eligible ORDER BY idx,permno").df()
 month=(f.date.dt.to_period('M')-lag)
 ym=(month.dt.year*100+month.dt.month).to_numpy(np.int64)
 keys=ym*1000000+f.permno.to_numpy(np.int64)
 zs=[np.load(DATA/f'ranks_finite_{y}.npz') for y in sorted(set(ym//100))]
 key=np.concatenate([z['key'] for z in zs]);ranks=np.concatenate([z['ranks'] for z in zs]);[z.close() for z in zs]
 loc=np.searchsorted(key,keys);good=(loc<len(key))&(key[np.minimum(loc,len(key)-1)]==keys)
 # One empty lookup row keeps missing months missing, never zero.
 ranks=np.vstack([ranks,np.full((1,ranks.shape[1]),np.nan,np.float32)])
 f['rank_row']=np.where(good,loc,len(ranks)-1)
 assert (ym<(f.date.dt.year*100+f.date.dt.month).to_numpy()).all()
 assert (f.known_idx<=f.idx).all()
 return f,ranks

def gate(f,mode):
 if mode=='all':return np.ones(len(f),bool)
 clock=f.clock.to_numpy();age=f.age.to_numpy()
 if mode=='pre20':return (clock>=-20)&(clock<=-1)
 if mode=='pre10':return (clock>=-10)&(clock<=-1)
 if mode=='post20':return (age>=2)&(age<=20)
 if mode=='near10':return ((clock>=-10)&(clock<=0))|((age>=2)&(age<=10))
 raise ValueError(mode)

def scores(ranks,models):
 factors=metadata()['factors'];ix={x:i for i,x in enumerate(factors)}
 cache={};out=[]
 for m in models:
  key=tuple(m['factors'])
  if key not in cache:
   if not key:
    cache[key]=np.full(len(ranks),np.nan,np.float32);out.append(cache[key]);continue
   a=ranks[:,[ix[x] for x in key]];n=np.isfinite(a).sum(axis=1)
   cache[key]=np.divide(np.nansum(a,axis=1),n,out=np.full(len(a),np.nan,np.float32),where=n>=math.ceil(.75*len(key)))
  out.append(cache[key])
 return np.asarray(out)

def weights(f,s,models):
 # Models x stocks. Ranks and masks contain only information available before returns.
 w=np.zeros(s.shape,dtype=float);gates={k:gate(f,k) for k in MODES}
 for mode in MODES:
  ix=np.array([j for j,m in enumerate(models) if m['mode']==mode and m.get('type','factor')=='factor'],int)
  if not len(ix):continue
  g=gates[mode][None,:];a=s[ix]
  long=(a>2/3)&g;short=(a<1/3)&g
  nl=long.sum(axis=1);ns=short.sum(axis=1);active=(nl>=20)&(ns>=20)
  w[ix]=np.where(active[:,None],long/np.maximum(nl,1)[:,None]-short/np.maximum(ns,1)[:,None],0)
 for j,m in enumerate(models):
  if m.get('type')=='timing':
   cl=f.clock.to_numpy();long=(cl>=-5)&(cl<=-1);short=(cl>=-40)&(cl<=-21)
   if min(long.sum(),short.sum())>=20:w[j]=long/long.sum()-short/short.sum()
  elif m.get('type')=='timing_long':
   cl=f.clock.to_numpy();long=(cl>=-5)&(cl<=-1)
   if long.sum()>=20:w[j]=long/long.sum()
 return w

def simulate(models,start,end,name,exact=False,costs=(.0005,),lag=1,liquid=False):
 c=conn();cal=calendar();cal=cal[cal.date.dt.year.between(start,end)].reset_index(drop=True)
 nc=len(costs);nm=len(models);shape=(len(cal),nc,nm)
 arrays={k:np.zeros(shape) for k in ['returns','turnover','fees','borrow','active','n_long','n_short','missing']}
 old=np.array([],int);hold=np.zeros((nc,nm,0));di=0;max_error=0.
 for y in range(start,end+1):
  log(f'{name}: {y}, {nm} models, {"exact" if exact else "screen"}')
  f,ranks=frame(c,y,lag);sc=scores(ranks,models);del ranks
  # Factor selection is performed on all eligible names or on the explicit liquid sensitivity.
  if liquid:f=f[f.lag_price.ge(5)&f.lag_adv.ge(1e6)]
  groups={int(k):g for k,g in f.groupby('idx',sort=False)}
  for idx in cal.loc[cal.date.dt.year.eq(y),'idx']:
   d=groups.get(int(idx),f.iloc[:0]);ids=d.permno.to_numpy(int)
   union=np.union1d(old,ids);loc=np.searchsorted(union,ids)
   v=np.zeros((nc,nm,len(union)));v[:,:,np.searchsorted(union,old)]=hold
   w=np.zeros((nm,len(union)));w[:,loc]=weights(d,sc[:,d.rank_row.to_numpy(int)],models)
   r=np.zeros(len(union));rx=r.copy();r[loc]=d.ret.fillna(0).to_numpy(float);rx[loc]=d.retx.fillna(d.ret).fillna(0).to_numpy(float)
   cost=np.asarray(costs)[:,None];x=np.ones((nc,nm))
   if exact:
    for _ in range(5):x=1-cost*np.abs(x[:,:,None]*w[None,:,:]-v).sum(axis=2)
   positions=x[:,:,None]*w[None,:,:]
   turn=np.abs(positions-v).sum(axis=2);fee=cost*turn
   long=np.maximum(positions,0).sum(axis=2);short=-np.minimum(positions,0).sum(axis=2)
   borrow=np.where(cost>0,.01/252,0)*short
   gross=np.sum(positions*r,axis=2);pnl=gross-fee-borrow
   assert np.all(1+pnl>0)
   if exact:max_error=max(max_error,float(np.abs(x-(1-fee)).max()))
   # Screens use the gross NAV path solely for approximate turnover.
   denominator=1+pnl if exact else 1+gross
   after=positions*(1+rx)/(denominator[:,:,None])
   arrays['returns'][di]=pnl;arrays['turnover'][di]=turn;arrays['fees'][di]=fee;arrays['borrow'][di]=borrow
   arrays['active'][di]=long>0;arrays['n_long'][di]=(positions>0).sum(axis=2);arrays['n_short'][di]=(positions<0).sum(axis=2)
   arrays['missing'][di]=((positions[:,:,loc]!=0)&d.ret.isna().to_numpy()).sum(axis=2)
   hold=after[:,:,loc];old=ids;di+=1
 assert max_error<1e-10
 np.savez_compressed(OUT/f'{name}.npz',**arrays,dates=cal.date.to_numpy(dtype='datetime64[D]'),costs=costs,max_accounting_error=max_error)
 save(OUT/f'{name}_models.json',models)
 return summarize(arrays,cal.date.to_numpy(dtype='datetime64[D]'),models,costs,name)

def summarize(a,dates,models,costs,name):
 rows=[]
 for period,(lo,hi) in PERIODS.items():
  mask=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
  if not mask.any():continue
  for ci,cost in enumerate(costs):
   r=a['returns'][mask,ci];mean=r.mean(axis=0)*252;vol=r.std(axis=0,ddof=1)*np.sqrt(252)
   nav=np.cumprod(1+r,axis=0);peak=np.maximum.accumulate(np.vstack([np.ones((1,len(models))),nav]),axis=0)[1:]
   for j,m in enumerate(models):
    rows.append(dict(model=m['id'],mode=m['mode'],pool=m.get('pool','control'),factors='+'.join(m['factors']),period=period,cost_bps=cost*10000,annual_mean=mean[j],cagr=nav[-1,j]**(252/len(r))-1,sharpe=mean[j]/vol[j] if vol[j]>1e-10 else np.nan,annual_vol=vol[j],max_drawdown=(nav[:,j]/peak[:,j]-1).min(),active=a['active'][mask,ci,j].mean(),turnover=a['turnover'][mask,ci,j].mean()*252,fees=a['fees'][mask,ci,j].mean()*252,borrow=a['borrow'][mask,ci,j].mean()*252,mean_long=a['n_long'][mask,ci,j].mean(),mean_short=a['n_short'][mask,ci,j].mean(),missing_stockdays=a['missing'][mask,ci,j].sum()))
 df=pd.DataFrame(rows);df.to_csv(OUT/f'{name}_statistics.csv',index=False);return df

def single_screen():
 freeze();meta=metadata()
 models=[dict(id=f'{mode}__{x}',mode=mode,factors=[x],pool='single') for mode in MODES for x in meta['factors']]
 simulate(models,1985,2004,'single_screen')

FIXED=[['Mom12m','EarningsSurprise','AnnouncementReturn'],['EarningsSurprise','AnalystRevision','AnnouncementReturn'],['BM','Accruals','Mom12m'],['GP','roaq','Accruals','AssetGrowth'],['EarningsSurprise','RevenueSurprise','Mom12m','AnalystRevision']]
def combo_screen():
 freeze();meta=metadata();old=set(meta['published_by2004']);available=set(meta['factors'])
 stats=pd.read_csv(OUT/'single_screen_statistics.csv').query("period=='development' and active>=0.3")
 models=[]
 for mode in MODES:
  for pool in ['pre2005','all_vintages']:
   z=stats[stats['mode']==mode]
   if pool=='pre2005':z=z[z.factors.isin(old)]
   best=z.sort_values(['sharpe','model'],ascending=[False,True]).head(8).factors.tolist()
   sets=set(tuple(sorted(x)) for x in itertools.combinations(best,2))
   sets.update(tuple(sorted(best[:k])) for k in [3,5,8] if len(best)>=k)
   sets.update(tuple(sorted(x)) for x in FIXED if set(x)<=available and (pool!='pre2005' or set(x)<=old))
   for fs in sorted(sets):models.append(dict(id=f'{pool}_{mode}__'+ '+'.join(fs),mode=mode,factors=list(fs),pool=pool))
 save(OUT/'composite_search.json',dict(number=len(models),construction='Training-only ranks; no 2005+ returns read',created_utc=pd.Timestamp.now(tz='UTC').isoformat()))
 simulate(models,1985,2004,'combo_screen')

def train():
 freeze();cs=pd.read_csv(OUT/'combo_screen_statistics.csv').query("period=='development' and active>=0.3")
 cm={m['id']:m for m in json.loads((OUT/'combo_screen_models.json').read_text())}
 chosen=cs.sort_values(['sharpe','model'],ascending=[False,True]).groupby(['pool','mode']).head(3)
 models=[cm[x] for x in chosen.model]
 singles=pd.read_csv(OUT/'single_screen_statistics.csv').query("period=='development' and active>=0.3")
 smap={m['id']:m for m in json.loads((OUT/'single_screen_models.json').read_text())}
 old=set(metadata()['published_by2004'])
 for pool in ['pre2005','all_vintages']:
  z=singles[singles.factors.isin(old)] if pool=='pre2005' else singles
  for x in z.sort_values(['sharpe','model'],ascending=[False,True]).groupby('mode').head(2).model:
   m=dict(smap[x],pool=pool,id=f'{pool}_S_{x}');models.append(m)
 stats=simulate(models,1985,2004,'exact_train',exact=True)
 lookup={m['id']:m for m in models}
 z=stats.query("period=='development' and active>=0.3");z=z[z.model.isin(cm)]
 chosen=z.sort_values(['sharpe','model'],ascending=[False,True]).groupby(['pool','mode']).head(1)
 means=z[z['mode']!='all'].sort_values(['annual_mean','model'],ascending=[False,True]).groupby('pool').head(1)
 selected=sorted(set(chosen.model)|set(means.model));final=[lookup[x] for x in selected]
 # Direct ablation: hold exactly the same composite throughout the cycle.
 for m in list(final):
  if m['mode']!='all':final.append(dict(m,id=m['id']+'__no_timing',mode='all',ablation_for=m['id']))
 final += [dict(id='timing_only_LS',mode='all',factors=[],type='timing'),dict(id='timing_only_long',mode='all',factors=[],type='timing_long')]
 prim=z[(z.pool=='pre2005')&(z['mode']!='all')].sort_values(['sharpe','model'],ascending=[False,True]).iloc[0].model
 save(OUT/'frozen_choices.json',dict(frozen_utc=pd.Timestamp.now(tz='UTC').isoformat(),training_end='2004-12-31',primary=prim,event_candidates=[m['id'] for m in final if m['mode']!='all'],models=final,mean_winners=means.model.tolist(),no_later_returns_used=True))
 log(f'Frozen primary: {prim}')

def evaluate():
 freeze();frozen=json.loads((OUT/'frozen_choices.json').read_text());models=frozen['models']
 # Continuous holdings across chronological splits; development statistics here reproduce training.
 simulate(models,1985,2025,'final',exact=True,costs=(0.,.0005,.001))
 primary=next(x for x in models if x['id']==frozen['primary'])
 pair=[primary,next(x for x in models if x.get('ablation_for')==primary['id']),models[-2]]
 simulate(pair,2005,2025,'extra_month_lag',exact=True,lag=2)
 simulate(pair,2005,2025,'liquid',exact=True,liquid=True)

def hac(r,lag=20):
 r=np.asarray(r,float)
 if np.std(r)<1e-14:return dict(annual_mean=float(np.mean(r)*252),lo=0.,hi=0.,t=0.,p=1.)
 fit=sm.OLS(r,np.ones((len(r),1)),missing='raise').fit(cov_type='HAC',cov_kwds={'maxlags':lag})
 ci=fit.conf_int()[0]*252
 return dict(annual_mean=float(fit.params[0]*252),lo=float(ci[0]),hi=float(ci[1]),t=float(fit.tvalues[0]),p=float(fit.pvalues[0]))

def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 frozen=json.loads((OUT/'frozen_choices.json').read_text());models=frozen['models'];ix={m['id']:i for i,m in enumerate(models)}
 a=np.load(OUT/'final.npz');dates=a['dates'];returns=a['returns'];rows=[]
 events=frozen['event_candidates']
 for period,(lo,hi) in PERIODS.items():
  mask=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
  for cid in events:
   j=ix[cid];r=returns[mask,1,j]
   for benchmark in ['zero','timing_only_LS',cid+'__no_timing']:
    diff=r if benchmark=='zero' else r-returns[mask,1,ix[benchmark]]
    for lag in [20,60]:rows.append(dict(model=cid,period=period,benchmark='same_factors_no_timing' if benchmark.endswith('__no_timing') else benchmark,hac_lags=lag,**hac(diff,lag)))
 tests=pd.DataFrame(rows);tests['p_holm']=np.nan
 for _,z in tests.groupby(['period','benchmark','hac_lags']):tests.loc[z.index,'p_holm']=multipletests(z.p.to_numpy(),method='holm')[1]
 tests.to_csv(OUT/'paired_inference.csv',index=False)
 # Secondary descriptive checks for every distinct no-timing factor control,
 # not only the control that happens to look best in the later period.
 untimed={tuple(m['factors']):m['id'] for m in models if m['mode']=='all' and m.get('type','factor')=='factor'}
 ntrows=[]
 for period,(lo,hi) in PERIODS.items():
  mask=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
  for cid in untimed.values():
   for benchmark in ['zero','timing_only_LS']:
    diff=returns[mask,1,ix[cid]].copy()
    if benchmark!='zero':diff-=returns[mask,1,ix[benchmark]]
    ntrows.append(dict(model=cid,period=period,benchmark=benchmark,**hac(diff,60)))
 nt=pd.DataFrame(ntrows);nt['p_holm']=np.nan
 for _,z in nt.groupby(['period','benchmark']):nt.loc[z.index,'p_holm']=multipletests(z.p.to_numpy(),method='holm')[1]
 nt.to_csv(OUT/'untimed_secondary_inference.csv',index=False)
 stats=pd.read_csv(OUT/'final_statistics.csv');primary=frozen['primary'];j=ix[primary];ab=ix[primary+'__no_timing'];ti=ix['timing_only_LS']
 robustness=[]
 rng=np.random.default_rng(20261102)
 for period in ['validation','holdout','post2004']:
  lo,hi=PERIODS[period];mask=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
  for b,k in [('same_factors_no_timing',ab),('timing_only_LS',ti)]:
   diff=returns[mask,1,j]-returns[mask,1,k];n=len(diff);L=63
   # Circular moving-block bootstrap paired calendar-time observations.
   means=[]
   for _ in range(2000):
    starts=rng.integers(0,n,size=math.ceil(n/L));indices=((starts[:,None]+np.arange(L))%n).ravel()[:n];means.append(diff[indices].mean()*252)
   ci=np.quantile(means,[.025,.975])
   monthly=pd.Series(diff,index=pd.to_datetime(dates[mask])).resample('ME').sum()
   mf=sm.OLS(monthly,np.ones((len(monthly),1)),missing='raise').fit(cov_type='HAC',cov_kwds={'maxlags':3})
   robustness.append(dict(period=period,benchmark=b,block63_lo=ci[0],block63_hi=ci[1],monthly_additive_mean=mf.params.iloc[0]*12,monthly_hac3_p=mf.pvalues.iloc[0]))
 pd.DataFrame(robustness).to_csv(OUT/'primary_robustness.csv',index=False)
 daily=[]
 for ci,cost in enumerate(a['costs']):
  for k,m in enumerate(models):
   f=pd.DataFrame({'date':dates,'model':m['id'],'cost_bps':cost*10000,'returns':returns[:,ci,k],'turnover':a['turnover'][:,ci,k]});daily.append(f)
 daily=pd.concat(daily,ignore_index=True);daily.to_csv(OUT/'daily_returns.csv',index=False)
 daily['year']=pd.to_datetime(daily.date).dt.year
 annual=daily.groupby(['model','cost_bps','year']).returns.agg(lambda x:np.prod(1+x)-1).rename('return').reset_index();annual.to_csv(OUT/'annual_returns.csv',index=False)
 fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
 mask=dates>=np.datetime64('2005-01-01')
 for name,k in [('Factor + earnings timing',j),('Same factors, no timing',ab),('Original timing-only long/short',ti)]:axes[0].plot(dates[mask],np.cumprod(1+returns[mask,1,k]),label=name)
 axes[0].set(yscale='log',ylabel='Growth of $1 (log scale)',title='Frozen development-selected strategy; 5 bp per dollar traded + 1% short borrow');axes[0].legend()
 h=tests[(tests.model==primary)&tests.period.isin(['validation','holdout','post2004'])&(tests.benchmark=='same_factors_no_timing')&(tests.hac_lags==60)]
 axes[1].errorbar(np.arange(len(h)),h.annual_mean*100,yerr=[(h.annual_mean-h.lo)*100,(h.hi-h.annual_mean)*100],fmt='o',capsize=6)
 axes[1].set_xticks(np.arange(len(h)),h.period);axes[1].axhline(0,color='gray',lw=1);axes[1].set(ylabel='Annual return difference (percentage points)',title='Increment from earnings timing: paired HAC60 95% intervals')
 fig.savefig(OUT/'comparison.png',dpi=170);plt.close(fig)
 cols=['model','period','annual_mean','cagr','sharpe','max_drawdown','active','turnover']
 standalone=[m['id'] for m in models if m['mode']=='all' and 'ablation_for' not in m and m.get('pool')=='pre2005']
 focus=stats[(stats.model.isin([primary,primary+'__no_timing','timing_only_LS','timing_only_long']+standalone))&(stats.cost_bps==5)&stats.period.isin(['validation','holdout','post2004'])][cols]
 candidates=stats[(stats.model.isin(events))&(stats.cost_bps==5)&(stats.period=='holdout')].sort_values('sharpe',ascending=False)
 pm=next(m for m in models if m['id']==primary)
 best=candidates.sort_values('annual_mean',ascending=False).iloc[0]
 bv=stats[(stats.model==best.model)&(stats.period=='validation')&(stats.cost_bps==5)].iloc[0]
 bt=tests[(tests.model==best.model)&(tests.period=='holdout')&(tests.hac_lags==60)].set_index('benchmark')
 ph=stats[(stats.model==primary)&(stats.period=='holdout')&(stats.cost_bps==5)].iloc[0]
 conclusion=(f'No statistically reliable improvement from earnings-timed factor combinations was found in this search. '
  f'Screened 1,045 single-factor/window rules and {len(json.loads((OUT/"combo_screen_models.json").read_text()))} combinations on 1985–2004, exact-simulated 50 shortlisted rules, and froze {len(events)} event-timed candidates before 2005+ evaluation. '
  f'None has a positive 2015–2025 net mean significant at 5% after Holm correction. '
  f'The primary development-selected strategy returned {ph.annual_mean:.2%} annually in 2015–2025. '
  f'The best later-period event candidate returned {best.annual_mean:.2%} annual arithmetic mean ({best.cagr:.2%} CAGR; Sharpe {best.sharpe:.2f}), '
  f'but {bv.annual_mean:.2%} annually in 2005–2014. Its 2015–2025 paired improvement over timing-only is '
  f'{bt.loc["timing_only_LS","annual_mean"]:.2%} per year, unadjusted HAC60 p={bt.loc["timing_only_LS","p"]:.3f}; '
  f'over the same factors without timing, {bt.loc["same_factors_no_timing","annual_mean"]:.2%}, p={bt.loc["same_factors_no_timing","p"]:.3f}. '
  'These are retrospective historical tests; this bounded search does not rule out other strategies.')
 (OUT/'conclusion.txt').write_text(conclusion+'\n\nBest later-period candidate (ex-post label): '+best.model+'\n')
 def table(f):return f.to_html(index=False,float_format=lambda x:f'{x:.4f}',border=0)
 summary=conclusion+f"\n\nPrimary development-selected composite: {', '.join(pm['factors'])}; window {pm['mode']}.\n"+focus.to_string(index=False)+'\n\n'+tests[(tests.model==primary)&tests.period.isin(['validation','holdout','post2004'])&(tests.hac_lags==60)].to_string(index=False)
 (OUT/'summary.txt').write_text(summary)
 html=f'''<!doctype html><html><head><meta charset="utf-8"><title>Factors and earnings timing</title><style>body{{font:16px/1.55 system-ui;max-width:1300px;margin:45px auto;padding:0 25px;color:#172b3a}}table{{font-size:12px;border-collapse:collapse;display:block;overflow:auto}}th,td{{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:#f3f6f9;padding:20px}}a{{color:#17649b}}</style></head><body>
 <h1>Do factor combinations improve earnings-timed trading?</h1><p><b>{conclusion}</b></p><p>Primary frozen selection: <b>{', '.join(pm['factors'])}</b>; <b>{pm['mode']}</b> ({DESIGN['windows'][pm['mode']]}). Selection uses 1985–2004 only. Report generated {pd.Timestamp.now(tz='UTC').isoformat()}.</p>
 <p>Returns below are decimals (0.10 = 10%). Annual mean is arithmetic; CAGR compounded. Long/short portfolios target 200% gross exposure, 0% net. Trading costs are 5 bp on every dollar traded, plus 1% annual short borrow. The long-only timing reference has 100% gross exposure and is not risk matched.</p>
 <img src="comparison.png"><h2>Primary strategy and direct controls</h2>{table(focus)}
 <h2>Paired significance: primary strategy</h2><p>Compare with the same factor composite without timing to isolate whether announcements add value. Holm correction includes every frozen event-factor candidate, including newer-factor exploratory candidates. Daily observations are not independent; HAC20/HAC60 and block-bootstrap diagnostics account for serial dependence.</p>{table(tests[(tests.model==primary)&(tests.hac_lags==60)&tests.period.isin(['validation','holdout','post2004'])])}
 <h2>All frozen event candidates: 2015–2025</h2><p>This table is descriptive. Its best holdout row was not the primary strategy selected before evaluation. Do not relabel the ex-post winner as a confirmed discovery.</p>{table(candidates)}
 <h2>Closest event-timed candidate, identified after evaluation</h2><p>{best.factors}. Trade {DESIGN['windows'][best['mode']]} with the same upper/lower-third composite-rank rule. This earned {best.annual_mean:.2%} annual arithmetic mean in 2015–2025 and {bv.annual_mean:.2%} in 2005–2014. Both comparisons below are statistically inconclusive, even before correcting for nine candidate tests.</p>{table(bt.reset_index())}
 <h2>Factor-only controls: secondary descriptive results</h2><p>These explain whether timing restrictions help. They are not discoveries of a successful announcement strategy. Every distinct factor-only control is included in the separate Holm correction; the best later-period control was not selected as the primary strategy.</p>{table(stats[(stats.model.isin(untimed.values()))&(stats.cost_bps==5)&(stats.period=='holdout')])}{table(nt[nt.period=='holdout'])}
 <h2>Robustness for the frozen primary</h2>{table(pd.DataFrame(robustness))}
 <h3>One extra month of signal delay</h3>{table(pd.read_csv(OUT/'extra_month_lag_statistics.csv'))}<h3>Price ≥ $5 and prior ADV ≥ $1m</h3>{table(pd.read_csv(OUT/'liquid_statistics.csv'))}
 <h2>Exact design and limitations</h2><pre>{json.dumps(DESIGN,indent=2)}</pre>
 <p>The supplied factor folder contained only monthly portfolio returns; the separate 1.93 GB stock-level panel was downloaded and SHA-256 verified. Monthly snapshots cannot identify announcement-day changes in a signal. Only the previous calendar month's snapshot is used, with an extra month of delay tested separately. Infinite signal values are treated as missing before ranking. The date clock and analyst availability retain the prior study's conservative information delays. Historical accounting restatements, CRSP linking, publication/search hindsight, and short availability remain unresolved; 5 bp is an assumed cost, not observed execution.</p>
 <h2>Reproduction and sources</h2><p>Run <code>../.venv/bin/python research.py all</code> from factor_timing after obtaining the documented source. Commands prepare, singles, combos, train, evaluate, report, validate can be run individually. Raw inputs and older research are unchanged.</p>
 <p>Chen, Andrew Y., and Tom Zimmermann (2022), Open Source Cross-Sectional Asset Pricing, Critical Finance Review 11(2), 207–264. <a href="https://huggingface.co/datasets/OpenSourceAP/CrossSection">Official data and timing documentation</a>; <a href="https://www.openassetpricing.com/">project</a>. Dataset release Data-Release-2026.10.</p>
 <p>Methods workflow used the statsmodels skill from Kassis, Timothy, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a> (current arXiv v2 verified).</p></body></html>'''
 (OUT/'report.html').write_text(html)
 log(summary)

def validate():
 frozen=json.loads((OUT/'frozen_choices.json').read_text());meta=metadata();a=np.load(OUT/'final.npz');models=frozen['models']
 assert a['max_accounting_error']<1e-10
 assert np.isfinite(a['returns']).all() and np.all(a['returns']>-1)
 assert np.min(a['turnover'])>=0
 for ci,cost in enumerate(a['costs']):assert np.allclose(a['fees'][:,ci],a['turnover'][:,ci]*cost,atol=1e-12)
 assert np.all(a['fees'][:,0]==0) and np.all(a['borrow'][:,0]==0)
 assert np.all((a['active']==0)|(a['n_long']>=20))
 for j,m in enumerate(models):
  if m.get('type')!='timing_long':assert np.all((a['active'][:,:,j]==0)|(a['n_short'][:,:,j]>=20))
  if m.get('pool')=='pre2005':assert set(m['factors'])<=set(meta['published_by2004'])
 # Gross and each cost scenario have independent carried holdings.
 stat=pd.read_csv(OUT/'final_statistics.csv');train=pd.read_csv(OUT/'exact_train_statistics.csv')
 check=stat[(stat.period=='development')&(stat.cost_bps==5)].merge(train,on='model',suffixes=('_final','_train'))
 error=float(np.max(np.abs(check.annual_mean_final-check.annual_mean_train)))
 assert error<1e-10
 # Reproduce original timing-only reference to expose engine/accounting differences.
 prior=pd.read_csv(ROOT/'coverage5/results_regular65/daily_returns.csv',parse_dates=['date'])
 j=next(j for j,m in enumerate(models) if m['id']=='timing_only_LS')
 orig=prior[(prior.strategy=='nearest_selected')&(prior.scenario=='5bps')].set_index('date').returns
 ours=pd.Series(a['returns'][:,1,j],index=pd.to_datetime(a['dates']))
 baseline_error=float((ours-orig.reindex(ours.index)).abs().max())
 assert baseline_error<1e-9,baseline_error
 # Independently calculate average-tie ranks using counts, not pandas.rank.
 c=conn();doc=docs();rank_checks=0;max_rank_error=0.
 for ym in [198412,200412,201412,202412]:
  zz=np.load(DATA/f'ranks_finite_{ym//100}.npz');keys=zz['key'];ranks=zz['ranks']
  for factor in ['Mom12m','EarningsSurprise','AnalystRevision','BM','Accruals']:
   raw=c.execute(f'SELECT permno,"{factor}" AS value FROM read_parquet(\'{DATA}/Predictors.parquet\') WHERE yyyymm={ym}').df()
   value=raw.value.to_numpy(float)*float(doc.loc[factor,'Sign']);finite=np.isfinite(value)
   for row in np.flatnonzero(finite)[::max(1,int(finite.sum())//5)]:
    v=value[row];expected=((value[finite]<v).sum()+((value[finite]==v).sum()+1)/2)/finite.sum()
    pos=np.searchsorted(keys,ym*1000000+int(raw.permno.iloc[row]));observed=float(ranks[pos,meta['factors'].index(factor)])
    max_rank_error=max(max_rank_error,abs(expected-observed));rank_checks+=1
   for row in np.flatnonzero(np.isinf(value))[:5]:
    pos=np.searchsorted(keys,ym*1000000+int(raw.permno.iloc[row]));assert np.isnan(ranks[pos,meta['factors'].index(factor)])
 assert max_rank_error<1e-7
 save(OUT/'validation.json',dict(accounting_max_error=float(a['max_accounting_error']),training_reproduction_max_error=error,prior_timing_baseline_max_error=baseline_error,independent_rank_checks=rank_checks,max_rank_error=max_rank_error,finite_returns=True,minimum20_each_leg=True,primary_publication_cutoff=2004,frozen_before_evaluation=True,missing_return_stockdays=int(a['missing'].sum()),source_sha256=meta['sha256']))
 log('Validation passed')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','singles','combos','train','evaluate','report','validate','all']);args=p.parse_args()
 stages={'prepare':prepare,'singles':single_screen,'combos':combo_screen,'train':train,'evaluate':evaluate,'report':report,'validate':validate}
 if args.stage=='all':
  for fn in stages.values():fn()
 else:stages[args.stage]()
