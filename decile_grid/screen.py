#!/usr/bin/env python3
"""Exhaustive gross screening; selection reads only 1985--2004 results."""
from __future__ import annotations
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from grid import models, compile_specs, screen_day, weights

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE/'results'
FIELDS = ['returns', 'active', 'n_long', 'n_short']


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def hac_t(x, lag=20):
    """Intercept-only HAC (Bartlett), matching statsmodels default correction=False."""
    x = np.asarray(x, float)
    n = len(x)
    if not n:
        return np.full(x.shape[1:], np.nan)
    centered = x-x.mean(axis=0)
    variance = np.sum(centered*centered, axis=0)
    for k in range(1, min(lag,n-1)+1):
        variance += 2*(1-k/(lag+1))*np.sum(centered[k:]*centered[:-k], axis=0)
    se = np.sqrt(np.maximum(variance, 0))/n
    return np.divide(x.mean(axis=0),se,out=np.full_like(se,np.nan),where=se>1e-15)


def train_summary(specs):
    """This function explicitly refuses any input after December 2004."""
    pieces = []
    for year in range(1985, 2005):
        with np.load(OUT/f'gross_{year}.npz') as z:
            assert (z['dates'] < np.datetime64('2005-01-01')).all()
            pieces.append({k: z[k] for k in FIELDS+['dates']})
    arrays = {k: np.concatenate([p[k] for p in pieces]) for k in FIELDS}
    r = arrays['returns']
    mapping = {s['id']:i for i,s in enumerate(specs)}
    base = np.asarray([mapping[s['baseline_id']] if s['baseline_id'] else j for j,s in enumerate(specs)])
    paired = r-r[:,base]
    mean = r.mean(axis=0)*252
    uplift = paired.mean(axis=0)*252
    activity = arrays['active'].mean(axis=0)
    t = hac_t(r)
    pt = hac_t(paired)
    vol = r.std(axis=0,ddof=1)*np.sqrt(252)
    table = pd.DataFrame(specs)
    table['annual_mean'] = mean
    table['annual_vol'] = vol
    table['sharpe'] = np.divide(mean,vol,out=np.full_like(mean,np.nan),where=vol>0)
    table['t_hac20'] = t
    table['active_fraction'] = activity
    table['mean_long'] = arrays['n_long'].mean(axis=0)
    table['mean_short'] = arrays['n_short'].mean(axis=0)
    table['baseline_annual_mean'] = mean[base]
    table['baseline_active_fraction'] = activity[base]
    table['paired_annual_mean'] = uplift
    table['paired_t_hac20'] = pt
    candidate = (table.id != table.baseline_id)&table.family.ne('timing_only')
    table['selection_eligible'] = candidate & (activity>=.5)&(activity[base]>=.8)&(uplift>0)&(mean>0)
    table.to_csv(OUT/'training_gross_summary.csv',index=False)
    chosen = []
    for (factor,family), group in table.groupby(['factor','family'],sort=False):
        if family=='timing_only':
            continue
        eligible = group[group.selection_eligible]
        for criterion in ['paired_annual_mean','paired_t_hac20']:
            if len(eligible):
                row = eligible.sort_values([criterion,'id'],ascending=[False,True]).iloc[0]
                chosen.append(dict(factor=factor,family=family,criterion=criterion,
                    selected_id=row.id,baseline_id=row.baseline_id,
                    training_annual_mean=float(row.annual_mean),
                    training_paired_annual_mean=float(row.paired_annual_mean),
                    training_paired_t_hac20=float(row.paired_t_hac20),
                    training_active_fraction=float(row.active_fraction),
                    eligible_count=len(eligible)))
            else:
                chosen.append(dict(factor=factor,family=family,criterion=criterion,
                    selected_id=None,baseline_id=None,eligible_count=0))
    selected = sorted({x['selected_id'] for x in chosen if x['selected_id']})
    all_eligible = table[table.selection_eligible]
    primary = None if all_eligible.empty else all_eligible.sort_values(
        ['paired_t_hac20','id'],ascending=[False,True]).iloc[0].id
    frozen = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        training_start='1985-01-01',training_end='2004-12-31',
        training_days=len(r),total_models=len(specs),eligible_candidates=int(table.selection_eligible.sum()),
        minimum_names=5,activity_cutoff=.5,baseline_activity_cutoff=.8,
        choice='per factor/family maximum positive paired mean and maximum paired HAC20 t; own mean positive',
        primary_rule='maximum training paired HAC20 t across all eligible models',
        selected_ids=selected,primary_id=primary,selections=chosen,
        script_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'grid.py',HERE/'screen.py']},
        retrospective=True,later_returns_read=False)
    (OUT/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    log(f'Frozen {len(selected)} unique candidates; primary={primary}; eligible={len(all_eligible)}')


def run(start=1985,end=2004,freeze=True):
    OUT.mkdir(parents=True,exist_ok=True)
    specs = models()
    spec_path = OUT/'specs.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == specs, 'Specification order changed'
    else:
        spec_path.write_text(json.dumps(specs,indent=2)+'\n')
    compiled = compile_specs(specs)
    calendar = pd.read_csv(ROOT/'results/trading_calendar.csv',parse_dates=['date'])
    audits = []
    for year in range(start,end+1):
        target = OUT/f'gross_{year}.npz'
        if target.exists():
            log(f'{year}: reuse completed gross arrays')
            continue
        with np.load(ROOT/f'factor_overlay/data/{year}.npz') as f:
            z = {k:f[k] for k in f.files}
        assert (z['known_idx']<z['idx']).all() and (z['event_idx']<z['idx']).all()
        dates = calendar[calendar.date.dt.year.eq(year)]
        shape = (len(dates),len(specs))
        arrays = dict(returns=np.zeros(shape),active=np.zeros(shape,bool),
            n_long=np.zeros(shape,np.int16),n_short=np.zeros(shape,np.int16))
        dayidx = z['idx']
        starts = np.r_[0,np.flatnonzero(np.diff(dayidx))+1]
        ends = np.r_[starts[1:],len(dayidx)]
        lookup = {int(dayidx[a]):(a,b) for a,b in zip(starts,ends)}
        error = 0.
        for row,day in enumerate(dates.itertuples()):
            a,b = lookup.get(int(day.idx),(0,0))
            d = {k:v[a:b] for k,v in z.items()}
            if a==b:
                continue
            result = screen_day(d,compiled)
            for k in FIELDS:
                arrays[k][row] = result[k]
            # Independently direct weights versus aggregate cell accounting.
            if row in [0,len(dates)//2,len(dates)-1]:
                w = weights(d,specs)
                direct = w @ np.where(np.isfinite(d['ret']),d['ret'],0.)
                error = max(error,float(np.max(abs(direct-result['returns']))))
                assert error<1e-12, (year,row,error)
                assert np.allclose(w.sum(axis=1),0,atol=1e-12)
                assert np.allclose(abs(w).sum(axis=1),2*result['active'],atol=1e-12)
                assert np.array_equal((w>0).sum(axis=1),result['n_long'])
                assert np.array_equal((w<0).sum(axis=1),result['n_short'])
        assert np.isfinite(arrays['returns']).all()
        np.savez_compressed(target,dates=dates.date.to_numpy(dtype='datetime64[D]'),**arrays)
        audits.append(dict(year=year,days=len(dates),models=len(specs),
            stockdays=len(dayidx),max_direct_weight_difference=error,
            chronology_passed=True,direct_check_days=3))
        log(f'{year}: {len(dates)} sessions, {len(specs)} models, direct check error={error:.3g}')
    auditpath = OUT/f'screen_audit_{start}_{end}.json'
    auditpath.write_text(json.dumps(audits,indent=2)+'\n')
    if freeze:
        assert end<=2004, 'A later-period run must use --no-freeze'
        train_summary(specs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start',type=int,default=1985)
    parser.add_argument('--end',type=int,default=2004)
    parser.add_argument('--no-freeze',action='store_true')
    args = parser.parse_args()
    run(args.start,args.end,not args.no_freeze)
