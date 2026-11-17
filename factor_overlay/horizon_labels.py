#!/usr/bin/env python3
"""Five-session, fixed-share labels for a declared post-result sensitivity.

This does not alter the primary one-day training inputs. Future prices define
labels only; feature rows remain unchanged and labels crossing calendar years
are purged, permitting the existing annual walk-forward training cutoff.
"""
from pathlib import Path
import hashlib
import json
import time
import duckdb
import numpy as np
import pandas as pd
from data import HERE, OUT, PANEL


def log(s):
    print(time.strftime('%H:%M:%S'),s,flush=True)


def _float(x):
    return np.ma.filled(x,np.nan).astype(float)


def make_labels():
    source=OUT/'train.npz'
    z=dict(np.load(source))
    cal=pd.read_csv(OUT/'calendar.csv',parse_dates=['date'])
    year_end=cal.assign(year=cal.date.dt.year).groupby('year').idx.max().to_dict()
    keep=np.array([idx+4<=year_end[int(year)] for idx,year in zip(z['idx'],z['year'])])
    original_n=len(keep)
    z={k:v[keep] for k,v in z.items()}
    z['label_end_idx']=z['idx']+4
    z['horizon']=np.array(5,dtype=np.int8)
    label=np.zeros(len(z['y']),float)
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='4GB'")
    rows_audit=[];verify=[]
    for year in sorted(set(z['year'].tolist())):
        rows=np.flatnonzero(z['year']==year)
        keys=z['idx'][rows].astype(np.int64)*1000000+z['permno'][rows].astype(np.int64)
        unique,inverse=np.unique(keys,return_inverse=True)
        startidx=unique//1000000;permno=unique%1000000
        raw=c.execute(f"SELECT idx,permno,ret,retx FROM read_parquet('{PANEL}/{year}.parquet') ORDER BY idx,permno").fetchnumpy()
        marketkey=raw['idx'].astype(np.int64)*1000000+raw['permno'].astype(np.int64)
        assert (np.diff(marketkey)>0).all()
        ret=_float(raw['ret']);retx=_float(raw['retx'])
        pos=np.ones(len(unique));pnl=np.zeros(len(unique));forced=np.zeros(len(unique),bool)
        missing_ret=np.zeros(len(unique),np.int32)
        missing_retx=np.zeros(len(unique),np.int32)
        observed=np.zeros((len(unique),5),bool)
        returns=np.zeros((len(unique),5));ex_returns=np.zeros((len(unique),5))
        for h in range(5):
            target=(startidx+h)*1000000+permno
            loc=np.searchsorted(marketkey,target)
            found=(loc<len(marketkey))&(marketkey[np.minimum(loc,len(marketkey)-1)]==target)
            observed[:,h]=found
            active=(pos!=0)
            forced|=active&~found
            # An absent listing exits at the previous mark, with no gross P&L.
            pos[~found]=0
            rr=np.zeros(len(unique));rx=np.zeros(len(unique))
            rr[found]=ret[loc[found]];rx[found]=retx[loc[found]]
            missing_ret+=((pos!=0)&found&~np.isfinite(rr))
            missing_retx+=((pos!=0)&found&~np.isfinite(rx))
            rr=np.nan_to_num(rr,nan=0.)
            rx=np.where(np.isfinite(rx),rx,rr)
            returns[:,h]=rr;ex_returns[:,h]=rx
            pnl+=pos*rr
            pos*=1+rx
        signed=z['Xbase'][rows,6].astype(float)
        label[rows]=signed*pnl[inverse]/5.
        # Independent expression: cumulative ex-distribution mark products,
        # stopped at first missing row, times each daily total return. This is
        # fixed shares plus accumulated cash dividends, not daily reinvestment.
        choices=np.linspace(0,len(unique)-1,min(3,len(unique)),dtype=int)
        for u in choices:
            valid_until=np.flatnonzero(~observed[u])
            end=int(valid_until[0]) if len(valid_until) else 5
            rp=returns[u,:end];xp=ex_returns[u,:end]
            scalar=float(np.sum(np.r_[1.,np.cumprod(1+xp)[:-1]]*rp)) if end else 0.
            assert abs(scalar-pnl[u])<1e-13
            verify.append(dict(year=int(year),permno=int(permno[u]),idx=int(startidx[u]),
                               gross_five_day=float(pnl[u]),independent_cash_product=scalar,error=abs(scalar-pnl[u])))
        rows_audit.append(dict(year=int(year),training_records=len(rows),unique_stock_dates=len(unique),
                               forced_exit_unique_labels=int(forced.sum()),
                               forced_exit_training_records=int(forced[inverse].sum()),
                               observed_missing_return_stockdays=int(missing_ret.sum()),
                               observed_missing_retx_stockdays=int(missing_retx.sum())))
        log(f'{year}: {len(rows):,} records; {int(forced.sum())} unique labels forced flat')
    z['y']=label.astype(np.float32)
    assert np.isfinite(z['y']).all()
    assert (z['label_end_idx']<=np.array([year_end[int(y)] for y in z['year']])).all()
    target=OUT/'train5.npz'
    np.savez_compressed(target,**z)
    pd.DataFrame(rows_audit).to_csv(OUT/'train5_year_audit.csv',index=False)
    pd.DataFrame(verify).to_csv(OUT/'train5_independent_labels.csv',index=False)
    report=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                status='post-result targeted five-session horizon sensitivity; not a new untouched primary test',
                source_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                original_training_records=original_n,training_records=len(label),
                records_purged_at_calendar_year_boundary=int((~keep).sum()),
                horizon_sessions=5,
                target='long_or_short_sign * sum_{h=0..4}(fixed_share_mark_before_h * total_return_h) / 5',
                holding='one dollar initial underlying shares; retx updates marks; cash distributions retained as cash; no fees/borrow in gross labels',
                missing_listing='first absent full-panel stock row forces flat at previous mark; remaining horizon returns0; no survivor filtering',
                missing_return='observed missing ret=0; missing retx falls back to effective ret',
                purging='retain only labels with end_idx=t+4 no later than final trading session of source calendar year',
                feature_integrity='all original Xbase,Xcycle,weights,factor,year,idx,permno copied unchanged after boundary mask',
                variance_note='target is average daily payoff over5-session holding; multiply residual-square by5 to estimate five-day residual risk divided by5, a per-day-equivalent five-day risk that includes within-horizon serial covariance; this need not equal true one-day variance',
                independent_checks=len(verify),independent_max_abs_error=max(v['error'] for v in verify),
                forced_exit_unique_labels=sum(r['forced_exit_unique_labels'] for r in rows_audit),
                forced_exit_training_records=sum(r['forced_exit_training_records'] for r in rows_audit),
                observed_missing_return_stockdays=sum(r['observed_missing_return_stockdays'] for r in rows_audit))
    (OUT/'train5_audit.json').write_text(json.dumps(report,indent=2))
    log(f'Saved {target}; {len(label):,} labels, {int((~keep).sum()):,} boundary exclusions')


if __name__=='__main__':
    make_labels()
