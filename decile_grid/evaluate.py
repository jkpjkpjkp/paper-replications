#!/usr/bin/env python3
"""Exact post-fee NAV backtest of training-frozen gross-grid selections."""
from pathlib import Path
import sys,os,json,argparse,time,hashlib
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; OUT=HERE/'results'
sys.path.insert(0,str(ROOT/'factor_overlay'))
from portfolio import Ledger
from data import load_year
sys.path.insert(0,str(HERE))
from grid import weights
import numpy as np
import pandas as pd
import duckdb

COSTS=[0.,.0001,.0005,.001,.0025]

def run(start=2005,end=2025,min_names=5,expost=False):
    allspec=json.loads((OUT/'specs.json').read_text())
    frozen=json.loads((OUT/'frozen.json').read_text())
    ids=set(frozen['selected_ids'])
    if expost:
        ex=pd.read_csv(OUT/'grid_expost_best.csv')
        ids=set(ex.loc[ex.period.eq('holdout'),'id'].dropna())
    lookup={s['id']:s for s in allspec}
    ids|={lookup[k]['baseline_id'] for k in list(ids)}
    ids|={s['id'] for s in allspec if s['family']=='timing_only'}
    specs=[s for s in allspec if s['id'] in ids]
    assert len(specs)>0
    name=('expost_' if expost else '')+f'exact_{start}_{end}_min{min_names}'
    (OUT/(name+'_models.json')).write_text(json.dumps(specs,indent=2))
    ledger=[Ledger(len(specs),c) for c in COSTS]
    calendar=pd.read_csv(ROOT/'results/trading_calendar.csv',parse_dates=['date'])
    calendar=calendar[calendar.date.dt.year.between(start,end)]
    out={k:[] for k in ['returns','turnover','fees','borrow','active','n_long','n_short','missing','forced_exit','gross_return','gross_exposure','bankrupt']}
    dates=[]; c=duckdb.connect();c.execute('SET threads=2');c.execute("SET memory_limit='4GB'")
    gross_error=0.
    for year in range(start,end+1):
        z=load_year(year)
        starts=np.r_[0,np.flatnonzero(np.diff(z['idx']))+1];ends=np.r_[starts[1:],len(z['idx'])]
        rows={int(z['idx'][a]):(a,b) for a,b in zip(starts,ends)}
        m=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{ROOT}/coverage5/results_regular65/panel/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        for k in ['ret','retx']:m[k]=np.ma.filled(m[k],np.nan)
        ss=np.r_[0,np.flatnonzero(np.diff(m['idx']))+1];ee=np.r_[ss[1:],len(m['idx'])]
        mr={int(m['idx'][a]):(a,b) for a,b in zip(ss,ee)}
        for day in calendar[calendar.date.dt.year.eq(year)].itertuples():
            a,b=rows.get(int(day.idx),(0,0));zz={k:v[a:b] for k,v in z.items()}
            w=weights(zz,specs,min_names=min_names)
            assert np.max(abs(w.sum(1)))<1e-10
            assert np.max(abs(w))<=1+1e-10  # VW has no per-stock cap
            ma,mb=mr[int(day.idx)]
            rr=[]
            for l in ledger:
                rr.append(l.step(m['permno'][ma:mb],m['ret'][ma:mb],m['retx'][ma:mb],zz['permno'],w,terminal=int(day.idx)==int(calendar.idx.iloc[-1])))
            direct=w@np.nan_to_num(zz['ret'],nan=0.)
            gross_error=max(gross_error,float(np.max(abs(direct-rr[0]['returns']))))
            for k in out:out[k].append(np.stack([r[k] for r in rr]))
            dates.append(day.date)
        print(time.strftime('%H:%M:%S'),name,year,'models',len(specs),flush=True)
    a={k:np.asarray(v) for k,v in out.items()}
    a.update(dates=np.asarray(dates,dtype='datetime64[D]'),costs=np.asarray(COSTS))
    np.savez_compressed(OUT/(name+'.npz'),**a)
    audit=dict(expost_selection=expost,start=start,end=end,min_names=min_names,models=len(specs),days=len(dates),costs=COSTS,
        max_accounting_error=max(l.max_accounting_error for l in ledger),gross_direct_error=gross_error,
        terminal_positions=[l.hold.size for l in ledger],bankrupt_accounts=[int(l.dead.sum()) for l in ledger],
        frozen_sha256=hashlib.sha256((OUT/'frozen.json').read_bytes()).hexdigest())
    assert gross_error<1e-12
    assert audit['max_accounting_error']<1e-10
    assert sum(audit['terminal_positions'])==0
    (OUT/(name+'_audit.json')).write_text(json.dumps(audit,indent=2))
    return a,specs

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=2005);p.add_argument('--end',type=int,default=2025);p.add_argument('--min-names',type=int,default=5)
    p.add_argument('--expost-holdout',action='store_true')
    a=p.parse_args();run(a.start,a.end,a.min_names,a.expost_holdout)
