#!/usr/bin/env python3
"""Past-only features for six-factor announcement-cycle overlay research.

Targets are next close-to-close stock returns: panel row t already contains
information available strictly before t. Source characteristics are lagged one
calendar month. All raw source files and earlier analyses are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'data'
PANEL = ROOT / 'coverage5/results_regular65/panel'
SIGNALS = ROOT / 'factor_timing/data/Predictors.parquet'
DOC = Path('/home/jkp/Téléchargements/Chen_Zimmermann_Factor_Zoo_2026_10/SignalDoc.csv')
FACTORS = ['MS', 'CompEquIss', 'InvestPPEInv', 'Cash', 'RIO_Disp', 'RDIPO']
BASE_NAMES = ['factor_center', 'mom_center', 'log_sigma63', 'log_cap', 'log_adv', 'log_price', 'leg']
PHASE_NAMES = ['sin_phase', 'cos_phase', 'sin_2phase', 'cos_2phase', 'clock_scaled', 'pre10', 'post10', 'age_scaled']
CYCLE_NAMES = BASE_NAMES + PHASE_NAMES + [f'{x}_x_leg' for x in PHASE_NAMES] + [f'{x}_x_factor' for x in PHASE_NAMES]


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def load_year(year):
    with np.load(OUT / f'{year}.npz') as z:
        return {k: z[k] for k in z.files}


def features(z, j, rows=None, cycle=False):
    """Return base or augmented features, without imputing from future rows."""
    if rows is None:
        rows = slice(None)
    rank = z['rank'][rows, j]
    leg = z['leg'][rows, j].astype(float)
    center = 2 * rank - 1
    if FACTORS[j] == 'RDIPO':
        center = leg.copy()
    def ln(a):
        return np.log(np.where(a > 0, a, np.nan))
    x = np.column_stack([center, 2*z['mom'][rows]-1, ln(z['sigma63'][rows]),
                         ln(z['lag_cap'][rows]), ln(z['lag_adv'][rows]),
                         ln(z['lag_price'][rows]), leg]).astype(np.float32)
    if cycle:
        p = z['cycle'][rows]
        x = np.column_stack([x, p, p*leg[:, None], p*center[:, None]]).astype(np.float32)
    return x


def connection():
    c = duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='6GB'")
    return c


def source_ranks(c, year):
    # Include prior December, and all month-end snapshots before year end.
    cols = FACTORS + ['Mom12m']
    quoted = ','.join(f'"{x}"' for x in cols)
    f = c.execute(f'''SELECT permno,yyyymm,{quoted} FROM read_parquet('{SIGNALS}')
                     WHERE yyyymm BETWEEN {(year-1)*100+12} AND {year*100+11}
                     ORDER BY yyyymm,permno''').df()
    d = pd.read_csv(DOC).set_index('Acronym')
    raw = f[cols].replace([np.inf, -np.inf], np.nan)
    signed = raw * d.loc[cols, 'Sign'].to_numpy(float)
    ranks = signed.groupby(f.yyyymm).rank(method='average', pct=True).to_numpy(np.float32)
    keys = f.yyyymm.to_numpy(np.int64)*1000000 + f.permno.to_numpy(np.int64)
    assert np.all(np.diff(keys)>0)
    return keys, ranks, raw.RDIPO.to_numpy(np.float32)


def prepare_year(c, year):
    p = OUT / f'{year}.npz'
    if p.exists():
        return
    paths = [str(PANEL / f'{y}.parquet') for y in [year-1, year] if (PANEL / f'{y}.parquet').exists()]
    # RANGE is trading-calendar sessions, not 63 observed rows. The exclusion of
    # CURRENT ROW is critical. Compute risk before filtering eligible stocks.
    f = c.execute(f'''WITH a AS (
       SELECT *, stddev_samp(ret) OVER w AS vol63, count(ret) OVER w AS nvol63
       FROM read_parquet({paths!r})
       WINDOW w AS (PARTITION BY permno ORDER BY idx RANGE BETWEEN 63 PRECEDING AND 1 PRECEDING)
       ) SELECT permno,date,idx,ret,retx,clock,age,period_sessions,
           known_idx,event_idx,lag_price,lag_adv,lag_cap,
           CASE WHEN nvol63>=21 THEN vol63 ELSE NULL END AS sigma63
       FROM a WHERE eligible AND year(date)={year} ORDER BY idx,permno''').df()
    assert (f.known_idx < f.idx).all()
    assert (f.event_idx < f.idx).all()
    assert (f.period_sessions > 0).all()
    month = f.date.dt.to_period('M')-1
    ym = (month.dt.year*100+month.dt.month).to_numpy(np.int64)
    keys = ym*1000000+f.permno.to_numpy(np.int64)
    sourcekey, source_rank, raw_rdipo = source_ranks(c, year)
    loc = np.searchsorted(sourcekey, keys)
    ok = (loc<len(sourcekey)) & (sourcekey[np.minimum(loc,len(sourcekey)-1)] == keys)
    rank = np.full((len(f), len(FACTORS)+1), np.nan, np.float32)
    rawipo = np.full(len(f), np.nan, np.float32)
    rank[ok] = source_rank[loc[ok]]
    rawipo[ok] = raw_rdipo[loc[ok]]
    leg = np.where(rank[:,:6]>2/3, 1, np.where(rank[:,:6]<1/3,-1,0)).astype(np.int8)
    # Preserve binary economic comparison: other firms long, recent no-R&D IPOs short.
    leg[:,5] = np.where(rawipo==0, 1, np.where(rawipo==1,-1,0))
    phase = 2*np.pi*f.age.to_numpy(float)/f.period_sessions.to_numpy(float)
    clock = f.clock.to_numpy(float)
    age = f.age.to_numpy(float)
    cyc = np.column_stack([np.sin(phase), np.cos(phase), np.sin(2*phase), np.cos(2*phase),
                           np.clip(clock,-40,80)/65, (clock>=-10)&(clock<=-1),
                           (age>=2)&(age<=10), np.clip(age,0,80)/65]).astype(np.float32)
    z = {k: f[k].to_numpy(np.float64) for k in ['ret','retx','lag_price','lag_adv','lag_cap','sigma63']}
    z.update({k: f[k].to_numpy(np.int32) for k in ['permno','idx','clock','age','period_sessions','known_idx','event_idx']})
    z.update(date=f.date.to_numpy(dtype='datetime64[D]'), month=ym.astype(np.int32),
             rank=rank[:,:6], mom=rank[:,6], leg=leg, cycle=cyc)
    np.savez_compressed(p, **z)
    log(f'{year}: {len(f):,} eligible stock-days cached')


def training_sample():
    arrays = {k: [] for k in ['Xbase','Xcycle','y','weight','factor','year','idx','permno']}
    coverage = []
    for year in range(1985,2026):
        z = load_year(year)
        dates = np.unique(z['idx'])
        for j, name in enumerate(FACTORS):
            leg = z['leg'][:,j]
            tab = pd.DataFrame({'idx': z['idx'], 'long': leg==1, 'short': leg==-1}).groupby('idx').sum().reindex(dates, fill_value=0)
            coverage.append(dict(year=year,factor=name,stockdays=int((leg!=0).sum()),
                                 mean_long=float(tab.long.mean()),mean_short=float(tab.short.mean()),
                                 min_short=int(tab.short.min()),max_short=int(tab.short.max()),
                                 fraction_both_min5=float(((tab.long>=5)&(tab.short>=5)).mean()),
                                 fraction_both_min20=float(((tab.long>=20)&(tab.short>=20)).mean())))
            rows = np.flatnonzero((z['idx']%5==0)&(leg!=0))
            # Stateless deterministic hash. Sampling does not inspect targets.
            key = (z['permno'][rows].astype(np.uint64)*np.uint64(2654435761) +
                   z['idx'][rows].astype(np.uint64)*np.uint64(2246822519) + np.uint64(j*3266489917))%np.uint64(4294967291)
            sample = pd.DataFrame({'row': rows,'idx':z['idx'][rows], 'leg':leg[rows], 'key':key})
            sample = sample.sort_values(['idx','leg','key']).groupby(['idx','leg'],sort=False).head(30)
            rows = sample.row.to_numpy(int)
            # Drop missing outcomes only after sample choice. No target clipping.
            sample = sample[np.isfinite(z['ret'][rows])]
            rows = sample.row.to_numpy(int)
            count = sample.groupby(['idx','leg']).row.transform('size').to_numpy(float)
            if not len(rows):
                continue
            xb = features(z,j,rows)
            xc = features(z,j,rows,cycle=True)
            n = len(rows)
            values = dict(Xbase=xb,Xcycle=xc,y=(leg[rows]*z['ret'][rows]).astype(np.float32),
                          weight=(30/count).astype(np.float32),factor=np.full(n,j,np.int8),
                          year=np.full(n,year,np.int16),idx=z['idx'][rows],permno=z['permno'][rows])
            for k,v in values.items():
                arrays[k].append(v)
        log(f'{year}: deterministic training sample extracted')
    np.savez_compressed(OUT/'train.npz',**{k:np.concatenate(v) for k,v in arrays.items()})
    pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)


def audit():
    d = pd.read_csv(DOC).set_index('Acronym')
    cal = pd.read_csv(ROOT/'results/trading_calendar.csv',parse_dates=['date'])
    cal[cal.date.dt.year.between(1985,2025)].to_csv(OUT/'calendar.csv',index=False)
    info = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(), factors=FACTORS,
                base_names=BASE_NAMES,cycle_names=CYCLE_NAMES,signal_lag='exact prior calendar month; no forward fill',
                risk='sample stddev of all observed stock returns in idx[t]-63 through idx[t]-1, minimum21; before filtering eligibility',
                chronology='known_idx < idx and event_idx < idx asserted for every eligible row',
                factor_legs='source-universe signed average-tie ranks >2/3 long, <1/3 short; RDIPO raw0 long,raw1 short',
                training_sample='idx%5==0; deterministic hash maximum30 each date/factor/leg; outcomes observed only after sampling',
                training_weight='30/number of observed sampled outcomes for each date/factor/leg; each date-leg equal total weight30',
                missing_features='left missing for train-only imputer; no future information',
                year1985_risk='source panel begins1985; first21 observed sessions lack sigma63 and require train-only imputation',
                sources=dict(panel=str(PANEL),signals=str(SIGNALS),documentation=str(DOC),
                             factor_download_receipt=str(ROOT/'factor_timing/data/download_receipt.json')),
                publications={x:int(d.loc[x,'Year']) for x in FACTORS},
                existing_prior_study='all related holdouts already inspected; retrospective chronological evidence',
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(info,indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start',type=int,default=1985)
    p.add_argument('--end',type=int,default=2025)
    p.add_argument('--sample-only',action='store_true')
    args = p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.sample_only:
        c=connection()
        for year in range(args.start,args.end+1):
            prepare_year(c,year)
    if args.start==1985 and args.end==2025:
        training_sample()
    audit()


if __name__=='__main__':
    main()
