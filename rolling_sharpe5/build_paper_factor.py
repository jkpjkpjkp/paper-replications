#!/usr/bin/env python3
"""Rebuild fixed paper EARC with observed history and exact per-dollar trading fees.

Uses retained linked earnings and conservative analyst-count states, then reads
raw CRSP returns. No existing strategy returns, fitted windows or future event
labels enter construction. Run: .venv/bin/python rolling_sharpe5/build_paper_factor.py
"""
from __future__ import annotations
import hashlib
import argparse
import json
import time
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA = PROJECT.parent
RAW = DATA / 'crsp-daily/data/crsp_a_stock/dsf_v2'
SCENARIOS = [('gross', 0., 0.), ('5bps', .0005, 0.), ('5bps_1pct_borrow', .0005, .01)]
MIN_NAMES_PER_LEG = 1
COMMON = """securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS'
 AND usincflg='Y' AND issuertype='CORP' AND primaryexch IN ('N','A','Q')"""

def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)

def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str, allow_nan=False))

def fingerprint(path):
    with path.open('rb') as h:
        sha = hashlib.file_digest(h, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha}

def build_events(con, event_source='compustat'):
    event_path = PROJECT/'results/linked_events.parquet'
    coverage_path = PROJECT/'coverage5/results/coverage_states.parquet'
    source_details={}
    if event_source=='compustat':
        event_path=DATA/'compustat/ccm/quarterly_announcements_crsp_daily.parquet'
        validation=json.loads((DATA/'compustat/ccm/join_validation.json').read_text())
        assert validation['complete'] is True
        # Explicit whitelist: announcement dates and historical identity only.
        # No announcement-day returns or accounting amounts enter the signal.
        con.execute(f"""CREATE TEMP TABLE source_events AS
          WITH standard AS (
            SELECT permno,datadate,rdq,trim(crsp_cusip) AS cusip,
              crsp_event_session,gvkey
            FROM read_parquet('{event_path}')
            WHERE indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
              AND rdq IS NOT NULL AND permno IS NOT NULL
              AND crsp_event_session IS NOT NULL AND length(trim(crsp_cusip))=8
          ), quarter_first AS (
            SELECT * FROM standard QUALIFY row_number() OVER
              (PARTITION BY permno,datadate ORDER BY rdq,gvkey,cusip)=1
          ), date_unique AS (
            SELECT permno,rdq,cusip,crsp_event_session FROM quarter_first
            QUALIFY row_number() OVER
              (PARTITION BY permno,rdq ORDER BY datadate DESC,gvkey,cusip)=1
          ), sessions AS (
            SELECT s.permno,s.cusip,s.rdq AS anndats,c.idx AS event_idx,c.idx AS known_idx
            FROM date_unique s JOIN calendar c ON s.crsp_event_session=c.date
          )
          SELECT * FROM sessions QUALIFY row_number() OVER
            (PARTITION BY permno,known_idx ORDER BY anndats DESC,cusip)=1
        """)
        source_details=dict(source='Compustat fundq RDQ linked with official historical CCM LC/LU P/C',
            path=str(event_path),deduplication='INDL/STD/D/C; earliest RDQ per permno/datadate; one event per permno/RDQ; if several nontrading dates map same session, latest RDQ; no accounting values or event returns used.',
            availability='No activation or historical revision dates available. RDQ event becomes usable for returns strictly after its mapped event session (one-session conservative lag).',
            inherited_validation=str(DATA/'compustat/ccm/join_validation.json'))
    else:
        # Explicit whitelist: no DIAGNOSTIC_ONLY future values are loaded.
        con.execute(f"""CREATE TEMP TABLE source_events AS
          SELECT permno,trim(cusip) AS cusip,anndats,known_idx,event_idx
          FROM read_parquet('{event_path}')""")
        source_details=dict(source='I/B/E/S act_epsus anndats, retrospective CUSIP linkage',path=str(event_path),
            availability='Retained conservative max(announcement,activation) and previous-close availability states.')
    con.execute(f"""CREATE TEMP TABLE events AS
      WITH history AS (
        SELECT permno,trim(cusip) AS cusip,anndats,known_idx,event_idx,
          lag(anndats) OVER w AS previous_announcement,
          lag(anndats,3) OVER w AS fourth_announcement
        FROM source_events
        WINDOW w AS(PARTITION BY permno ORDER BY known_idx,event_idx)
      ), gates AS (
        SELECT *,coalesce(date_diff('day',fourth_announcement,anndats)<=400
          AND date_diff('day',previous_announcement,anndats) BETWEEN 70 AND 110,
          false) AS regular_paper
        FROM history
      )
      SELECT g.*,coalesce(c.n90,0) AS coverage90_at_announcement,
        c.idx AS coverage_state_idx
      FROM gates g ASOF LEFT JOIN read_parquet('{coverage_path}') c
        ON g.cusip=c.cusip AND g.event_idx>=c.idx
    """)
    stats = con.execute("""SELECT count(*) AS events,
      count(*) FILTER(WHERE regular_paper) AS regular_events,
      count(*) FILTER(WHERE regular_paper AND coverage90_at_announcement>=5) AS qualified_events,
      count(*) FILTER(WHERE coverage_state_idx>event_idx) AS future_coverage,
      count(*) FILTER(WHERE event_idx>known_idx) AS future_announcement,
      min(anndats),max(anndats)
      FROM events""").fetchone()
    assert stats[3] == stats[4] == 0
    assert con.execute('SELECT count(*)-count(DISTINCT(permno,known_idx)) FROM events').fetchone()[0] == 0
    return {**dict(zip(['events','regular_events','qualified_events','future_coverage','future_announcement','first','last'], stats)),**source_details}

def year_data(con, year):
    # Include previous-year history for lagged screens at the first session.
    return con.execute(f"""WITH raw AS (
      SELECT permno,dlycaldt AS date,CAST(dlyret AS DOUBLE) AS ret,
        CAST(dlyretx AS DOUBLE) AS retx,abs(CAST(dlyprc AS DOUBLE)) AS price,
        CAST(dlycap AS DOUBLE) AS cap,conditionaltype,tradingstatusflg,
        ({COMMON}) AS common
      FROM read_parquet(['{RAW}/{year-1}.parquet','{RAW}/{year}.parquet'])
      WHERE dlycaldt>=DATE '{year-1}-11-01'
    ), lagged AS (
      SELECT *,lag(price) OVER w AS lag_price,lag(cap) OVER w AS lag_cap,
        lag(common) OVER w AS lag_common,lag(conditionaltype) OVER w AS lag_cond,
        lag(tradingstatusflg) OVER w AS lag_status,lag(date) OVER w AS lag_date
      FROM raw WINDOW w AS(PARTITION BY permno ORDER BY date)
    ), dated AS (
      SELECT l.*,c.idx FROM lagged l JOIN calendar c USING(date) WHERE year(date)={year}
    ), signals AS (
      SELECT d.permno,d.date,d.idx,d.ret,d.retx,d.lag_date,
        coalesce(d.lag_common AND d.lag_cond='RW' AND d.lag_status='A'
          AND d.lag_price>0 AND d.lag_cap>0,false) AS eligible,
        e.event_idx,e.known_idx,coalesce(e.regular_paper,false) AS regular_paper,
        coalesce(e.coverage90_at_announcement,0) AS coverage90_at_announcement,
        d.idx-e.event_idx AS age
      FROM dated d ASOF LEFT JOIN events e ON d.permno=e.permno AND d.idx>e.known_idx
    )
    SELECT * FROM signals ORDER BY idx,permno
    """).df()

def bisection_cost(old, target, c):
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        value=mid+c*np.abs(mid*target-old).sum()-1
        if value>0:hi=mid
        else:lo=mid
    return (lo+hi)/2

def run(event_source='compustat'):
    started=time.monotonic()
    HERE.mkdir(exist_ok=True)
    con=duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='6GB'")
    calendar=pd.read_csv(PROJECT/'results/trading_calendar.csv',parse_dates=['date'])
    con.register('calendar_frame',calendar)
    con.execute('CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame')
    output_suffix='' if event_source=='compustat' else '_ibes'
    event_audit=build_events(con,event_source)
    log(f'Event gates ready: {event_audit}')
    source_audit=[]
    for year in range(1984,2026):
        path=RAW/f'{year}.parquet'
        receipt=json.loads(path.with_suffix('.done.json').read_text())
        rows=con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        assert receipt['state']=='complete' and rows==receipt['rows']
        assert path.stat().st_size==receipt['bytes']
        source_audit.append(dict(path=str(path),rows=rows,bytes=receipt['bytes'],
                                 receipt_sha256=receipt['sha256'],rehash=False))
    small_inputs=[fingerprint(PROJECT/'results/linked_events.parquet'),
                  fingerprint(PROJECT/'results/trading_calendar.csv'),
                  fingerprint(PROJECT/'coverage5/results/coverage_states.parquet')]
    if event_source=='compustat':
        small_inputs.append(fingerprint(DATA/'compustat/ccm/quarterly_announcements_crsp_daily.parquet'))
    oldids=np.empty(0,dtype=np.int64)
    holdings=np.zeros((3,0))
    costs=np.array([s[1] for s in SCENARIOS])
    borrow_rates=np.array([s[2] for s in SCENARIOS])
    rows=[]
    year_audits=[]
    solver_max_error=0.
    independent_max_error=0.
    accounting_error=0.
    future_selector_count=0
    sampled_ledgers=[]
    for year in range(1985,2026):
        t0=time.monotonic()
        df=year_data(con,year)
        assert not df.duplicated(['permno','date']).any()
        assert not (df.ret.dropna() < -1).any()
        assert not (df.retx.dropna() < -1).any()
        assert not np.isinf(df.ret.to_numpy()).any()
        assert (df.loc[df.lag_date.notna(),'lag_date']<df.loc[df.lag_date.notna(),'date']).all()
        has_event=df.known_idx.notna()
        bad=int((df.loc[has_event,'known_idx']>=df.loc[has_event,'idx']).sum())
        future_selector_count+=bad
        assert bad==0
        year_active=0
        for day_no,(idx,group) in enumerate(df.groupby('idx',sort=False)):
            ids=group.permno.to_numpy(dtype=np.int64)
            assert np.all(np.diff(ids)>0)
            held=oldids[np.any(holdings!=0,axis=0)]
            union=np.union1d(ids,held)
            loc=np.searchsorted(union,ids)
            oldloc=np.searchsorted(oldids,held)
            v=np.zeros((3,len(union)))
            v[:,np.searchsorted(union,held)]=holdings[:,oldloc]
            selected=(group.eligible.to_numpy(dtype=bool)
                      & group.regular_paper.to_numpy(dtype=bool)
                      & (group.coverage90_at_announcement.to_numpy()>=5))
            age=group.age.to_numpy(dtype=float,na_value=np.nan)
            long=selected&(age>=6)&(age<=20)
            short=selected&(age>=36)&(age<=50)
            nlong,nshort=int(long.sum()),int(short.sum())
            active=nlong>=MIN_NAMES_PER_LEG and nshort>=MIN_NAMES_PER_LEG
            year_active+=active
            target=np.zeros(len(union))
            if active:
                target[loc[long]]=1/nlong
                target[loc[short]]=-1/nshort
            rawret=group.ret.to_numpy(dtype=float,na_value=np.nan)
            rawrx=group.retx.to_numpy(dtype=float,na_value=np.nan)
            r=np.zeros(len(union));rx=np.zeros(len(union))
            r[loc]=np.nan_to_num(rawret,nan=0.)
            rx[loc]=np.where(np.isfinite(rawrx),rawrx,r[loc])
            missing=np.zeros(len(union),dtype=bool);missing[loc]=~np.isfinite(rawret)
            absent=~np.isin(union,ids)
            x=np.ones(3)
            for _ in range(12):
                x=1-costs*np.abs(x[:,None]*target-v).sum(axis=1)
            pos=x[:,None]*target
            turn=np.abs(pos-v).sum(axis=1)
            fee=costs*turn
            error=float(np.max(np.abs(x-(1-fee))))
            solver_max_error=max(solver_max_error,error)
            assert error<1e-12
            if day_no==0:
                for ci,(name,c,b) in enumerate(SCENARIOS):
                    independent=bisection_cost(v[ci],target,c)
                    independent_max_error=max(independent_max_error,abs(independent-x[ci]))
                    assert abs(independent-x[ci])<1e-12
                sampled_ledgers.append(dict(date=str(group.date.iloc[0].date()),x=x.tolist(),turnover=turn.tolist()))
            short_exposure=-np.minimum(pos,0).sum(axis=1)
            borrowing=borrow_rates/252*short_exposure
            before=pos@r
            pnl=before-fee-borrowing
            assert np.all(pnl>-1)
            assert np.max(np.abs(pos.sum(axis=1)))<1e-12
            accounting_error=max(accounting_error,float(np.max(np.abs(pnl-before+fee+borrowing))))
            after=pos*(1+rx)/(1+pnl[:,None])
            for ci,(name,_,_) in enumerate(SCENARIOS):
                rows.append(dict(date=group.date.iloc[0],scenario=name,returns=float(pnl[ci]),
                    before_fee=float(before[ci]),turnover=float(turn[ci]),cost=float(fee[ci]),
                    borrow=float(borrowing[ci]),n_long=nlong if active else 0,n_short=nshort if active else 0,
                    active=bool(active),missing_held_returns=int((missing&(pos[ci]!=0)).sum()),
                    absent_exits=int((absent&(v[ci]!=0)).sum())))
            holdings=after[:,loc];oldids=ids
        audit=dict(year=year,raw_stock_days=len(df),sessions=int(df.idx.nunique()),
                   active_sessions=year_active,seconds=round(time.monotonic()-t0,2))
        year_audits.append(audit)
        del df
        log(str(audit))
        # Atomic progress artifact contains completed whole years only.
        save(HERE/'paper_factor_progress.json',dict(last_completed_year=year,elapsed_seconds=time.monotonic()-started,years=year_audits))
    out=pd.DataFrame(rows)
    daily_path=HERE/f'paper_factor_daily{output_suffix}.csv'
    out.to_csv(daily_path,index=False)
    desired=calendar.loc[calendar.date.between('1985-01-01','2025-12-31'),'date'].to_numpy()
    for name,_,_ in SCENARIOS:
        assert np.array_equal(out.loc[out.scenario.eq(name),'date'].to_numpy(),desired)
    assert out.groupby(['date','scenario']).size().eq(1).all()
    assert out.loc[out.scenario.eq('5bps'),'borrow'].eq(0).all()
    assert np.max(np.abs(out.loc[out.scenario.eq('5bps'),'cost']-.0005*out.loc[out.scenario.eq('5bps'),'turnover']))<1e-15
    summary=[]
    for start in ['1985-01-01','2005-01-01','2015-01-01']:
        for name,_,_ in SCENARIOS:
            s=out.loc[out.date.ge(start)&out.scenario.eq(name)]
            summary.append(dict(start=start,scenario=name,days=len(s),annual_mean=float(s.returns.mean()*252),
                annual_cost=float(s.cost.mean()*252),annual_borrow=float(s.borrow.mean()*252),
                sharpe=float(s.returns.mean()/s.returns.std()*np.sqrt(252)),
                active_fraction=float(s.active.mean())))
    audit=dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),seconds=time.monotonic()-started,
        factor='Fixed EARC long age6..20 / short age36..50 trading sessions after observed announcement',
        source_pdf=str(PROJECT/'13699_paper_YrbKK6Zn.pdf'),
        signal='At prior close latest usable strictly advancing event only; current plus3 previous known earnings in400 calendar days; latest gap70..110 calendar days. No future realized earnings.',
        coverage='At event freeze retained n90 distinct analysts with US FY1/FY2 EPS estimates; n90 state<=event_idx. Estimates conservatively available after max(announcement,activation) with previous-close lag. 90 calendar days approximates paper3months.',
        execution='Lagged CRSP US ordinary common shares NYSE/AMEX/Nasdaq, regular-way active, positive previous price/cap. No additional price/size/liquidity screen. Daily equal-weight $1 long/$1 short after-cost NAV, both legs nonempty else cash. All bought/sold dollars charged; separate exact NAV paths. Total returns P&L, ex-dividend value drift; distributions remain cash.',
        scenarios=SCENARIOS,min_names_per_leg=MIN_NAMES_PER_LEG,
        deviations='Primary uses Compustat RDQ via retrospective CCM links; IBES sensitivity uses act_epsus anndats with retrospective CUSIP links. Historical revisions/availability vintages are incomplete. Analyst states are active-update proxy not vendor NUMEST. 90calendar days approximates paper3months. Four observed releases include current, a disclosed reading of prior400days wording. Both legs merely nonempty; archived IBES min20 sensitivity is separate. Windows inclusive6..20/36..50. No financing, cash interest, impact, measured borrow availability, terminal liquidation or collateral rebates. Raw CRSP extends2025, so no2026 realized EARC returns.',
        event_audit=event_audit,input_small_fingerprints=small_inputs,raw_partitions=source_audit,
        inherited_audit_references=[str(PROJECT/'results/input_audit.json'),str(PROJECT/'coverage5/results/input_audit.json'),str(PROJECT/'coverage5/results_regular65/validation_checks.json')],
        years=year_audits,chronology_violations=future_selector_count,
        max_cost_solver_error=solver_max_error,max_independent_bisection_error=independent_max_error,
        max_accounting_error=accounting_error,independent_ledger_samples=sampled_ledgers,
        missing_held_returns=out.groupby('scenario').missing_held_returns.sum().to_dict(),
        absent_exits=out.groupby('scenario').absent_exits.sum().to_dict(),summary=summary,
        concentration=out.loc[out.scenario.eq('gross')].groupby(out.loc[out.scenario.eq('gross'),'date'].dt.year).agg(min_long=('n_long','min'),min_short=('n_short','min'),median_long=('n_long','median'),median_short=('n_short','median')).reset_index().to_dict(orient='records'),
        output=fingerprint(daily_path))
    save(HERE/f'paper_factor_audit{output_suffix}.json',audit)
    log(f'COMPLETE {len(out)} rows in {audit["seconds"]:.1f}s')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-source',choices=['compustat','ibes'],default='compustat')
    run(parser.parse_args().event_source)
