#!/usr/bin/env python3
"""ORACLE diagnostic: regress returns around retrospectively observed releases.

The actual future announcement date selects the entry date. These are
descriptive event-study regressions, never evidence of an implementable
strategy and never inputs to the forecast-entry strategy selection.
Only 1985--2004 observations are processed, with common event-date purging.
"""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mpl-cache'))
import numpy as np
import pandas as pd
import duckdb
import research as base
import horizon_regress

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DATA=HERE/'horizon_oracle_data'
OUT=HERE/'horizon_oracle_results'

def log(s):base.log('ORACLE diagnostic: '+s)

def prepare():
 DATA.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
 cal=base.calendar();train=cal[cal.date.between('1985-01-01','2004-12-31')]
 lo,hi=int(train.idx.min()),int(train.idx.max())
 c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='6GB'")
 c.execute(f"SET temp_directory='{DATA}/tmp'")
 p=DATA/'entries.parquet'
 if not p.exists():
  log('Create actual-date aligned eligible entries, training only')
  c.execute(f"""COPY(WITH events AS (
   SELECT permno,event_idx,row_number() OVER(ORDER BY permno,known_idx) AS event_id
   FROM read_parquet('{ROOT}/results/linked_events.parquet'))
   SELECT p.permno,p.date,p.idx AS entry_idx,e.event_idx AS actual_event_idx,
    CAST(e.event_idx-p.idx+1 AS INTEGER) AS n,e.event_id
   FROM read_parquet('{base.PANEL}/[12]*.parquet') p JOIN events e
    ON p.permno=e.permno AND p.idx BETWEEN e.event_idx-29 AND e.event_idx
   WHERE p.eligible AND e.event_idx BETWEEN {lo+30} AND {hi-40}
   ORDER BY p.idx,p.permno,e.event_id) TO '{p}' (FORMAT PARQUET)""")
 log('Load training return-prefix lookup')
 marks_source=HERE/'horizon_data/marks.parquet'
 # Parent marks include stocks that had any forecast-window eligibility.
 # Audit and independently build missing stocks so oracle entries are not dropped.
 extras=c.execute(f"""SELECT DISTINCT e.permno FROM read_parquet('{p}') e
 ANTI JOIN (SELECT DISTINCT permno FROM read_parquet('{marks_source}')) m USING(permno)""").df()
 if len(extras):
  c.register('extra_stocks',extras)
  extra_path=DATA/'extra_marks.parquet'
  if not extra_path.exists():
   log(f'Construct prefixes for {len(extras)} additional eligible stocks')
   c.execute(f"""COPY(WITH r AS (
    SELECT p.permno,p.idx,coalesce(p.ret,0) AS ret,
     lag(p.idx) OVER(PARTITION BY p.permno ORDER BY p.idx) AS prior_idx
    FROM read_parquet('{base.PANEL}/[12]*.parquet') p SEMI JOIN extra_stocks USING(permno)),
    s AS(SELECT *,sum(CASE WHEN prior_idx IS NULL OR idx-prior_idx>1 THEN 1 ELSE 0 END)
      OVER(PARTITION BY permno ORDER BY idx) AS span FROM r)
    SELECT permno,idx,ret,
     sum(CASE WHEN ret>-1 THEN ln(1+ret) ELSE 0 END) OVER(PARTITION BY permno ORDER BY idx) AS logtotal,
     sum((ret<=-1)::INTEGER) OVER(PARTITION BY permno ORDER BY idx) AS zeros,
     max(idx) OVER(PARTITION BY permno,span) AS span_end
    FROM s ORDER BY permno,idx) TO '{extra_path}' (FORMAT PARQUET)""")
  markquery=f"SELECT permno,idx,ret,logtotal,zeros,span_end FROM read_parquet('{marks_source}') UNION ALL SELECT * FROM read_parquet('{extra_path}')"
 else:markquery=f"SELECT permno,idx,ret,logtotal,zeros,span_end FROM read_parquet('{marks_source}')"
 marks=c.execute(f"SELECT * FROM ({markquery}) WHERE idx BETWEEN {lo} AND {hi} ORDER BY permno,idx").df()
 key=marks.permno.to_numpy(np.int64)*100000+marks.idx.to_numpy(np.int64)
 cum=marks.logtotal.to_numpy(float);zeros=marks.zeros.to_numpy(np.int32)
 span=marks.span_end.to_numpy(np.int32);ret0=marks.ret.to_numpy(float)
 factors=base.metadata()['factors'];blocks={};year_audit=[]
 for year in range(1985,2005):
  log(f'Build frozen entry factors and actual-date labels: {year}')
  f=c.execute(f"SELECT * FROM read_parquet('{p}') WHERE year(date)={year} ORDER BY entry_idx,permno,event_id").df()
  if not len(f):continue
  f['date']=pd.to_datetime(f.date)
  months=f.date.dt.to_period('M')-1;ym=(months.dt.year*100+months.dt.month).to_numpy(np.int64)
  zs=[np.load(base.DATA/f'ranks_finite_{yy}.npz') for yy in sorted(set(ym//100))]
  rk=np.concatenate([z['key'] for z in zs]);rr=np.concatenate([z['ranks'] for z in zs]);[z.close() for z in zs]
  wanted=ym*1000000+f.permno.to_numpy(np.int64);loc=np.searchsorted(rk,wanted)
  valid=(loc<len(rk))&(rk[np.minimum(loc,len(rk)-1)]==wanted)
  rank=np.full((len(f),len(factors)),np.nan,np.float32);rank[valid]=rr[loc[valid]]
  mkey=f.permno.to_numpy(np.int64)*100000+f.entry_idx.to_numpy(np.int64)
  iloc=np.searchsorted(key,mkey);assert np.all(key[iloc]==mkey)
  event=f.actual_event_idx.to_numpy(np.int32)
  desired=event[:,None]+np.arange(1,31,dtype=np.int32)
  exits=np.minimum(desired,span[iloc,None])
  assert np.all(exits>=f.entry_idx.to_numpy()[:,None])
  ekey=f.permno.to_numpy(np.int64)[:,None]*100000+exits
  eloc=np.searchsorted(key,ekey);assert np.all(key[eloc]==ekey)
  before=cum[iloc]-np.where(ret0[iloc]>-1,np.log1p(np.maximum(ret0[iloc],-.999999999999)),0.)
  zbefore=zeros[iloc]-(ret0[iloc]<=-1)
  returns=np.expm1(cum[eloc]-before[:,None]);returns=np.where(zeros[eloc]>zbefore[:,None],-1.,returns)
  duration=exits-f.entry_idx.to_numpy(np.int32)[:,None]+1
  assert (exits<=hi).all() and (event>=lo+30).all() and (event<=hi-40).all()
  values=dict(rank=rank,returns=returns,duration=duration,entry_idx=f.entry_idx.to_numpy(np.int32),forecast_idx=event,exit_idx=exits,event_id=f.event_id.to_numpy(np.int64),permno=f.permno.to_numpy(np.int32),n=f.n.to_numpy(np.int8))
  for k,v in values.items():blocks.setdefault(k,[]).append(v)
  year_audit.append(dict(year=year,opportunities=len(f),missing_rank_rows=int((~valid).sum()),disappearance_shortened_labels=int((exits<desired).sum())))
 combined={k:np.concatenate(v) for k,v in blocks.items()}
 for n in range(1,31):
  log(f'Save actual-date diagnostic cache N={n}')
  keep=combined['n']==n
  np.savez_compressed(DATA/f'opportunities_n{n:02d}.npz',**{k:v[keep] for k,v in combined.items() if k!='n'})
 audit=dict(kind='ORACLE / DESCRIPTIVE ACTUAL-DATE EVENT STUDY',actual_future_dates_used_to_schedule_entries=True,implementable_trading_strategy=False,used_for_tradable_parameter_selection=False,training_start='1985-01-01',training_end='2004-12-31',factor_source_sha256=base.metadata()['sha256'],common_actual_event_min_idx=lo+30,common_actual_event_max_idx=hi-40,additional_stock_prefixes=int(len(extras)),opportunities=int(len(combined['n'])),factor_count=len(factors),n_grid=list(range(1,31)),m_grid=list(range(1,31)),label_rule='Buy actual E-N close, earn C+1..E+M inclusive; stop at first disappearance prior mark. Actual E is retrospective future information. Factor snapshot is previous completed month at entry; frozen.',baseline='Same factor, actual-announcement E-1 close to E+1 close.',year_audit=year_audit)
 (OUT/'oracle_cache_audit.json').write_text(json.dumps(audit,indent=2))
 return audit

def run_screen():
 result=horizon_regress.screen(data_dir=DATA,out_dir=OUT)
 result.insert(0,'study_kind','ORACLE_ACTUAL_DATE_DIAGNOSTIC')
 result.to_csv(OUT/'regression_screen.csv',index=False)
 audit_path=OUT/'regression_screen_audit.json';audit=json.loads(audit_path.read_text())
 audit.update(kind='ORACLE / DESCRIPTIVE ACTUAL-DATE EVENT STUDY',entry_alignment='Observed actual future announcement E, not ex-ante forecast',covariance='One-way ACTUAL-announcement-date clustered CR1, Student t(G-1)',earliest_actual_event_idx=audit.pop('earliest_forecast_idx'),latest_actual_event_idx=audit.pop('latest_forecast_idx'),actual_future_entry_date_used=True,implementable_trading_claim=False,used_for_tradable_selection=False)
 for row in audit['source_audit']:
  row['actual_event_clusters']=row.pop('forecast_clusters')
 audit_path.write_text(json.dumps(audit,indent=2))
 # Descriptive surfaces for each horizon across all factors and older factors.
 cols=['buy_before','sell_after']
 surfaces=[]
 for pool,subset in [('all_factors',result),('published_by2004',result[result.published_by2004])]:
  q=subset[subset.eligible].groupby(cols).agg(median_beta=('beta','median'),median_annual_proxy=('annual_mean_proxy','median'),median_t=('cluster_t','median'),factor_count=('factor','size')).reset_index();q['pool']=pool;surfaces.append(q)
 pd.concat(surfaces).to_csv(OUT/'descriptive_surfaces.csv',index=False)
 best=result[result.eligible].sort_values('sharpe_proxy',ascending=False).groupby('factor').head(1)
 best.to_csv(OUT/'per_factor_descriptive_best.csv',index=False)
 (OUT/'README.txt').write_text('ORACLE ACTUAL-DATE DIAGNOSTIC, DEVELOPMENT 1985-2004 ONLY\n\nThese 188,100 regressions use realized future earnings dates to schedule pre-release entry. They describe historical event concentration; they are not tradable backtest evidence. Do not use these rows to choose the forecast-entry strategy.\n\nregression_screen.csv: all 209 factors x 30 buy leads x 30 sell lags.\ndescriptive_surfaces.csv: horizon surfaces averaged across eligible factors.\nper_factor_descriptive_best.csv: training-only descriptive optima, subject to search bias.\nNo 2005+ outcomes are processed. Baseline is actual E-1 close/E+1 close. Event-return cost proxies are not exact calendar-time neutral portfolios.\n')
 log(f'Complete {len(result):,} ORACLE diagnostic specifications')
 return result

def validate():
 """Independent direct daily compounding and exact close-boundary checks."""
 c=duckdb.connect();checks=0;max_error=0.;counts=[]
 for n in [1,15,30]:
  with np.load(DATA/f'opportunities_n{n:02d}.npz') as z:
   entry=z['entry_idx'];event=z['forecast_idx'];exit_idx=z['exit_idx'];duration=z['duration'];ret=z['returns'];ids=z['permno']
   assert np.all(entry-1==event-n)
   assert np.all(exit_idx<=event[:,None]+np.arange(1,31))
   assert np.all(duration==exit_idx-entry[:,None]+1)
   assert np.all((duration>0)&(duration<=n+np.arange(1,31)))
   chosen=np.linspace(0,len(entry)-1,8,dtype=int)
   for i in chosen:
    raw=c.execute(f"SELECT idx,ret FROM read_parquet('{HERE}/horizon_data/marks.parquet') WHERE permno={int(ids[i])} AND idx BETWEEN {int(entry[i])} AND {int(exit_idx[i,-1])} ORDER BY idx").df()
    if not len(raw):
     raw=c.execute(f"SELECT idx,ret FROM read_parquet('{DATA}/extra_marks.parquet') WHERE permno={int(ids[i])} AND idx BETWEEN {int(entry[i])} AND {int(exit_idx[i,-1])} ORDER BY idx").df()
    for m in [1,7,30]:
     daily=raw[raw.idx<=exit_idx[i,m-1]].ret.to_numpy(float)
     assert len(daily)==duration[i,m-1]
     expected=np.prod(1+daily)-1;max_error=max(max_error,abs(expected-ret[i,m-1]));checks+=1
   counts.append(dict(n=n,opportunities=len(entry)))
 assert max_error<1e-10,max_error
 out=dict(study_kind='ORACLE_ACTUAL_DATE_DIAGNOSTIC',independent_direct_compounding_checks=checks,maximum_return_error=max_error,buy_close_equals_actual_event_minus_N=True,last_return_day_equals_exit_close=True,duration_max_equals_N_plus_M=True,no_future_year_labels=True,counts=counts)
 (OUT/'oracle_validation.json').write_text(json.dumps(out,indent=2));log('Independent return/close-date validation passed');return out

def plots():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 surfaces=pd.read_csv(OUT/'descriptive_surfaces.csv')
 q=surfaces[surfaces.pool=='published_by2004']
 fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
 for ax,field,title,scale in [(axes[0],'median_beta','Median stock-return slope on entry factor rank',100),(axes[1],'median_annual_proxy','Median annualized event-spread proxy after costs',100)]:
  grid=q.pivot(index='sell_after',columns='buy_before',values=field).reindex(index=range(1,31),columns=range(1,31))
  values=grid.to_numpy()*scale;limit=float(np.nanmax(np.abs(values)))
  im=ax.imshow(values,origin='lower',extent=(.5,30.5,.5,30.5),aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
  ax.set(xlabel='Buy N sessions before actual release',ylabel='Sell M sessions after actual release',title=title)
  fig.colorbar(im,ax=ax,label='Percentage points')
 fig.suptitle('ORACLE actual-date diagnostic, 1985–2004; factors published by 2004\nFuture actual dates align entry; these are not tradable calendar-portfolio returns',fontsize=11)
 fig.savefig(OUT/'oracle_descriptive_surfaces.png',dpi=170);plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','screen','validate','plots','all'],default='all',nargs='?');a=p.parse_args()
 if a.stage in ['prepare','all']:prepare()
 if a.stage in ['validate','all']:validate()
 if a.stage in ['screen','all']:run_screen()
 if a.stage in ['plots','all']:plots()
