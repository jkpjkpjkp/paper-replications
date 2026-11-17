#!/usr/bin/env python3
"""Five-analyst universe, earnings-date-only strategies. Raw inputs are read-only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
OUT = HERE / "results"
os.environ.setdefault("MPLCONFIGDIR", str(PARENT / ".mpl-cache"))
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("parent_timing", PARENT / "analyze.py")
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
PERIODS = parent.PERIODS
BINS = [(-40,-21),(-20,-11),(-10,-6),(-5,-1),(0,5),(6,10),
        (11,20),(21,35),(36,50),(51,80),(81,130)]
NAMES = [f"{a}:{b}" for a,b in BINS]
STRATEGIES = ["nearest_selected", "nearest_fixed", "known_cycle",
              "selected_liquid", "selected_coverage90", "selected_median_clock",
              "selected_long_only", "covered_equal_weight"]
COSTS = [("gross",0.,0.),("1bp",.0001,.01),("5bps",.0005,.01),
         ("10bps",.001,.01),("25bps",.0025,.02)]
DESIGN = {
 "created_utc":pd.Timestamp.now(tz="UTC").isoformat(),
 "question":"Can distance to the nearest known or predicted quarterly earnings announcement trade profitably among stocks with at least five active analysts?",
 "universe":"All lagged US-incorporated ordinary corporate common shares on NYSE/AMEX/Nasdaq, lagged regular-way/active status and positive lag price/cap. No primary price, size or volume threshold. Coverage >=5 is required at every rebalance.",
 "coverage":"Distinct positive I/B/E/S analyst codes with FY1 or FY2 EPS estimates (fpi 1 or 2, US file) initiated and activated within the preceding 180 calendar days. Multiple forecasts/horizons count once per analyst/CUSIP. Ignore forecast values, reviews and attached future actuals. A 90-day version is a sensitivity. This is an active-coverage proxy, not vendor NUMEST or a withdrawal-aware consensus reconstruction.",
 "information":"Both actuals and estimates become usable for returns only AFTER the close of the first market session strictly after max(announcement date, activation date). Coverage uses previous-session CUSIP. Current returns and prices never select positions.",
 "events":"Quarterly EPS releases from act_epsus, conservatively available, monotonically increasing event states. Exact historical event-date CUSIP linkage; inherited retrospective-link limitations. Quarterly earnings only, not arbitrary corporate news.",
 "prediction":"When parent regular-history criteria hold, prior-year seasonal date = announcement three quarters earlier +364 calendar days, with next gap 70..110 days. Otherwise use a valid median-of-three-gaps forecast, if available. Forecast quality is diagnosed only after signals are constructed. No actual future announcement date enters a signal.",
 "nearest":"Distance in market sessions to latest known actual or a still-future predicted release. Positive after known actual, negative before nearer prediction. Ties choose known actual. Forecast day itself has distance 0; overdue predictions cease being future events. Missing forecasts leave known-actual age. There is no required future realized release.",
 "bins":BINS,
 "selection":"Choose highest and lowest mean calendar-time stock-minus-covered-EW return bins in 1985-2004, requiring >=1000 development days and >=20 names/day. Freeze those two bins before any later return evaluation. Primary = nearest_selected at 10 one-way bps plus 1% annual short borrow.",
 "fixed":"nearest_fixed: long nearest clock -10..-1, short +21..+35. known_cycle comparator: long age +6..+20, short age +36..+50.",
 "sensitivities":"Apply exactly the selected windows to liquid stocks (lag price >=5, cap >=$100m, ADV >=$1m), 90-day coverage, and median-gap nearest clock. Also report selected long leg alone and the whole covered universe EW benchmark. No validation/holdout tuning.",
 "execution":"Daily equal weights, $1 long and $1 short per $1 after-cost NAV; >=20 names in both legs or cash. Long-only is 100% long. Rebalance at preceding close using only known information, then earn supplied daily total return. Exact cost-dependent sizing and NAV; ex-dividend returns drift shares; distributions stay cash. All bought/sold dollars charged. No terminal liquidation, risk-free interest, financing, collateral rebate or measured stock-specific borrow/impact. Missing returns zero; absent security exits at last mark, counted.",
 "costs":COSTS,
 "inference":"Daily calendar-time HAC20; annual mean = daily mean *252, CAGR separately. HAC60 and monthly HAC3 sensitivity; Holm across six L/S strategies within period/cost. Local covered-EW paired excess and local VW-market intercept are diagnostics, not official multifactor alpha.",
 "periods":PERIODS,
 "scope":"Retrospective exploratory chronological validation. Prior related timing results were already present; calendar splits cannot erase researcher or publication hindsight.",
 "source":"https://wrds-www.wharton.upenn.edu/pages/data-announcements/changes-to-crsp-data/"
}

def log(msg): print(time.strftime("%H:%M:%S"),msg,flush=True)
def save(path,obj): path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))
def checksum(p):
    with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    p=OUT/"design.json"
    if p.exists():
        old=json.loads(p.read_text());old.pop("created_utc",None)
        new=DESIGN.copy();new.pop("created_utc",None)
        # JSON represents tuples as lists.
        assert old==json.loads(json.dumps(new)),"Design changed: use a new output directory."
    else: save(p,DESIGN)

def inputs(con):
    log("Verifying raw receipts and derived event/calendar inputs")
    records=[]
    for y in range(1984,2026):
        p=ROOT/f"crsp-daily/data/crsp_a_stock/dsf_v2/{y}.parquet"
        rec=json.loads(p.with_suffix(".done.json").read_text())
        assert rec["state"]=="complete" and p.stat().st_size==rec["bytes"]
        sha=checksum(p);assert sha==rec["sha256"]
        assert con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchall()[0][0]==rec["rows"]
        records.append(dict(path=str(p),rows=rec["rows"],sha256=sha))
    for table in ["det_epsus","act_epsus"]:
        directory=ROOT/f"ibes/all-parquet/ibes/{table}"
        m=json.loads((directory/"manifest.json").read_text())
        assert m["complete"] and m.get("limit") is None
        n=0
        for f in m["files"]:
            p=directory/f["name"];sha=checksum(p)
            assert sha==f["sha256"] and p.stat().st_size==f["bytes"]
            rows=con.execute(f"SELECT count(*) FROM read_parquet('{p}')").fetchall()[0][0]
            assert rows==f["rows"];n+=rows
            records.append(dict(path=str(p),rows=rows,sha256=sha))
        assert n==m["copy_rows"]
    for name in ["linked_events.parquet","trading_calendar.csv"]:
        p=PARENT/"results"/name
        records.append(dict(path=str(p),sha256=checksum(p)))
    cal=pd.read_csv(PARENT/"results/trading_calendar.csv",parse_dates=["date"])
    con.register("cal_frame",cal)
    con.execute("CREATE OR REPLACE TEMP TABLE calendar AS SELECT * FROM cal_frame")
    con.unregister("cal_frame")
    # Only explicitly named information-state columns; diagnostic future dates excluded.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE events AS
      SELECT permno,known_idx,event_idx,regular,predicted_idx,
        seasonal_predicted_idx,seasonal_gap FROM read_parquet('{PARENT}/results/linked_events.parquet')""")
    save(OUT/"input_audit.json",dict(files=records,run_utc=pd.Timestamp.now(tz="UTC").isoformat()))

def coverage(con,reuse=False):
    dest=OUT/"coverage_states.parquet"
    if reuse and dest.exists(): return
    log("Reconstructing distinct-analyst coverage intervals without future actual fields")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE updates AS
      SELECT DISTINCT trim(cusip) AS cusip,CAST(analys AS BIGINT) AS analyst,
        greatest(anndats,actdats) AS available
      FROM read_parquet('{ROOT}/ibes/all-parquet/ibes/det_epsus/*.parquet')
      WHERE usfirm=1 AND measure='EPS' AND fpi IN ('1','2') AND analys>0
        AND length(trim(cusip))=8 AND anndats IS NOT NULL AND actdats IS NOT NULL
        AND greatest(anndats,actdats) BETWEEN DATE '1983-01-01' AND DATE '2025-12-31'""")
    # Count interval unions per analyst, not estimate rows. Start conservatively;
    # expiry is based on calendar age at the preceding close (strictly before
    # first session later than available+lookback, usable from next session).
    for days in [90,180]:
        con.execute(f"""CREATE OR REPLACE TEMP TABLE counts{days} AS
          WITH starts AS (
            SELECT u.*,c.idx+1 AS lo FROM updates u ASOF JOIN calendar c ON u.available<c.date),
          bounds AS (
            SELECT s.*,coalesce(c.idx+1,(SELECT max(idx)+1 FROM calendar)) AS hi
            FROM starts s ASOF LEFT JOIN calendar c
              ON s.available+INTERVAL '{days} days'<c.date),
          prev AS (
            SELECT *,max(hi) OVER(PARTITION BY cusip,analyst ORDER BY lo,hi
              ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_hi
            FROM bounds WHERE lo<hi),
          groups AS (
            SELECT *,sum(CASE WHEN prev_hi IS NULL OR lo>prev_hi THEN 1 ELSE 0 END)
              OVER(PARTITION BY cusip,analyst ORDER BY lo,hi) AS grp FROM prev),
          merged AS (
            SELECT cusip,analyst,grp,min(lo) AS lo,max(hi) AS hi FROM groups GROUP BY ALL),
          deltas AS (
            SELECT cusip,lo AS idx,1 AS d FROM merged UNION ALL
            SELECT cusip,hi AS idx,-1 AS d FROM merged),
          summed AS (SELECT cusip,idx,sum(d) AS d FROM deltas GROUP BY ALL)
          SELECT cusip,idx,CAST(sum(d) OVER(PARTITION BY cusip ORDER BY idx) AS INTEGER) AS n{days}
          FROM summed""")
    con.execute(f"""COPY (
      WITH keys AS (SELECT cusip,idx FROM counts90 UNION SELECT cusip,idx FROM counts180),
      a AS (SELECT k.*,coalesce(n90,0) AS n90 FROM keys k
        ASOF LEFT JOIN counts90 c ON k.cusip=c.cusip AND k.idx>=c.idx)
      SELECT a.*,coalesce(n180,0) AS n180 FROM a
        ASOF LEFT JOIN counts180 c ON a.cusip=c.cusip AND a.idx>=c.idx
      ORDER BY cusip,idx) TO '{dest}' (FORMAT PARQUET)""")
    con.execute("DROP TABLE updates")
    con.execute("DROP TABLE counts90");con.execute("DROP TABLE counts180")

def panel(con,reuse=False):
    folder=OUT/"panel";folder.mkdir(exist_ok=True)
    con.execute(f"CREATE OR REPLACE TEMP VIEW coverage AS SELECT * FROM read_parquet('{OUT}/coverage_states.parquet')")
    for y in range(1985,2026):
        target=folder/f"{y}.parquet"
        if reuse and target.exists():continue
        log(f"Building all-stock coverage and nearest clocks: {y}")
        raw=ROOT/"crsp-daily/data/crsp_a_stock/dsf_v2"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE yr AS
          WITH raw AS (
            SELECT permno,dlycaldt AS date,trim(cusip) AS cusip,
              CAST(dlyret AS DOUBLE) AS ret,CAST(dlyretx AS DOUBLE) AS retx,
              abs(CAST(dlyprc AS DOUBLE)) AS price,CAST(dlycap AS DOUBLE) AS cap,
              abs(CAST(dlyprc AS DOUBLE))*CAST(dlyvol AS DOUBLE) AS dollars,
              conditionaltype,tradingstatusflg,({parent.COMMON}) AS common
            FROM read_parquet(['{raw}/{y-1}.parquet','{raw}/{y}.parquet'])
            WHERE dlycaldt>=DATE '{y-1}-11-01'),
          l AS (
            SELECT *,lag(cusip) OVER w AS lag_cusip,lag(date) OVER w AS lag_date,
              lag(price) OVER w AS lag_price,lag(cap) OVER w AS lag_cap,
              lag(common) OVER w AS lag_common,lag(conditionaltype) OVER w AS lag_cond,
              lag(tradingstatusflg) OVER w AS lag_status,
              avg(dollars) OVER(PARTITION BY permno ORDER BY date
                 ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS lag_adv
            FROM raw WINDOW w AS(PARTITION BY permno ORDER BY date))
          SELECT l.*,c.idx FROM l JOIN calendar c USING(date) WHERE year(date)={y}""")
        con.execute(f"""COPY (
          WITH cov AS (
            SELECT y.*,coalesce(n180,0) AS n180,coalesce(n90,0) AS n90
            FROM yr y ASOF LEFT JOIN coverage c ON y.lag_cusip=c.cusip AND y.idx>=c.idx),
          e AS (
            SELECT y.*,s.known_idx,s.event_idx,s.regular,s.predicted_idx,
              s.seasonal_predicted_idx,s.seasonal_gap
            FROM cov y ASOF LEFT JOIN events s ON y.permno=s.permno AND y.idx>s.known_idx),
          clocks AS (
            SELECT *,idx-event_idx AS age,
              CASE WHEN regular AND seasonal_gap BETWEEN 70 AND 110
                  THEN seasonal_predicted_idx
                WHEN regular THEN predicted_idx END AS forecast_idx,
              coalesce(lag_common AND lag_cond='RW' AND lag_status='A'
                AND lag_price>0 AND lag_cap>0 AND n180>=5,false) AS eligible
            FROM e)
          SELECT permno,date,idx,ret,retx,lag_cusip,lag_date,lag_price,lag_cap,lag_adv,
            n180,n90,eligible,known_idx,event_idx,regular,age,forecast_idx,predicted_idx,
            CASE WHEN forecast_idx-idx>=0 AND forecast_idx-idx<age
                THEN idx-forecast_idx ELSE age END AS clock,
            CASE WHEN regular AND predicted_idx-idx>=0 AND predicted_idx-idx<age
                THEN idx-predicted_idx ELSE age END AS median_clock
          FROM clocks ORDER BY idx,permno
          ) TO '{target}' (FORMAT PARQUET)""")
    return folder

def bins(con,folder):
    cases=" ".join(f"WHEN clock BETWEEN {a} AND {b} THEN '{n}'" for (a,b),n in zip(BINS,NAMES))
    d=con.execute(f"""WITH p AS (
       SELECT * FROM read_parquet('{folder}/[12]*.parquet') WHERE eligible),
      market AS (SELECT date,avg(coalesce(ret,0)) AS covered_ew FROM p GROUP BY date),
      b AS (SELECT *,CASE {cases} END AS bin FROM p)
      SELECT b.date,bin,count(*) AS n,avg(coalesce(ret,0)) AS returns,
        avg(coalesce(ret,0))-first(covered_ew) AS excess
      FROM b JOIN market USING(date) WHERE bin IS NOT NULL
      GROUP BY b.date,bin ORDER BY b.date,bin""").df()
    d.to_csv(OUT/"daily_bins.csv",index=False)
    rows=[]
    for period,(start,end) in PERIODS.items():
        for name,g in d.loc[d.date.between(start,end)&(d.n>=20)].groupby("bin"):
            f=mean_fit(g.excess);ci=f.conf_int()[0]*1e4
            rows.append(dict(period=period,bin=name,days=len(g),mean_stocks=float(g.n.mean()),
                excess_bps=float(g.excess.mean()*1e4),t=float(f.tvalues[0]),p=float(f.pvalues[0]),
                low_bps=float(ci[0]),high_bps=float(ci[1])))
    stats=pd.DataFrame(rows)
    for period,z in stats.groupby("period"):
        stats.loc[z.index,"p_holm"]=multipletests(z.p,method="holm")[1]
    stats.to_csv(OUT/"bin_statistics.csv",index=False)
    train=stats.loc[(stats.period=="development")&(stats.days>=1000)].set_index("bin")
    choice=dict(long=train.excess_bps.idxmax(),short=train.excess_bps.idxmin(),
                selection_period="1985-2004",selection_metric="calendar-time mean excess versus covered EW")
    dest=OUT/"selected_windows.json"
    if dest.exists(): assert json.loads(dest.read_text())==choice
    else: save(dest,choice)
    log(f"Frozen development windows: {choice}")
    return choice,stats

def bin_mask(clock,name):
    a,b=map(int,name.split(":"));return (clock>=a)&(clock<=b)

def weights(f,choice):
    elig=f.eligible.to_numpy(dtype=bool)
    clock=f.clock.to_numpy(dtype=float,na_value=np.nan)
    age=f.age.to_numpy(dtype=float,na_value=np.nan)
    median=f[DESIGN.get("alternative_clock_column","median_clock")].to_numpy(dtype=float,na_value=np.nan)
    l=bin_mask(clock,choice["long"]);s=bin_mask(clock,choice["short"])
    liquid=(f.lag_price.to_numpy()>=5)&(f.lag_cap.to_numpy()>=100000)&(f.lag_adv.to_numpy()>=1e6)
    reg=f.regular.fillna(False).to_numpy(dtype=bool)
    masks=[(l,s),((clock>=-10)&(clock<=-1),(clock>=21)&(clock<=35)),
           (reg&(age>=6)&(age<=20),reg&(age>=36)&(age<=50)),
           (l&liquid,s&liquid),(l&(f.n90.to_numpy()>=5),s&(f.n90.to_numpy()>=5)),
           (bin_mask(median,choice["long"]),bin_mask(median,choice["short"])),
           (l,np.zeros(len(f),bool)),(elig,np.zeros(len(f),bool))]
    w=np.zeros((len(STRATEGIES),len(f)));nl=[];ns=[]
    for i,(lm,ss) in enumerate(masks):
        lm=lm&elig;ss=ss&elig;nlong=int(lm.sum());nshort=int(ss.sum())
        active=nlong>=20 and (i>=6 or nshort>=20)
        nl.append(nlong if active else 0);ns.append(nshort if active else 0)
        if active:
            w[i,lm]=1/nlong
            if i<6:w[i,ss]=-1/nshort
    return w,np.array(nl),np.array(ns)

def execute(v,w,r,rx,c,b):
    # Cost-specific old marked values are in units of today's pretrade NAV.
    x=np.ones(len(w))
    for _ in range(10):x=1-c*np.abs(x[:,None]*w-v).sum(axis=1)
    positions=x[:,None]*w
    turn=np.abs(positions-v).sum(axis=1);fee=c*turn
    assert np.max(np.abs(x-(1-fee)))<1e-11
    short=-np.minimum(positions,0).sum(axis=1);borrow=b/252*short
    before=positions@r;pnl=before-fee-borrow
    assert np.all(1+pnl>0),"Insolvency"
    after=positions*(1+rx)/(1+pnl[:,None])
    return pnl,after,turn,fee,borrow,before

def simulate(con,folder,choice):
    rows=[];oldids=np.array([],dtype=np.int64)
    holdings=np.zeros((len(COSTS),len(STRATEGIES),0))
    for y in range(1985,2026):
        log(f"Simulating exact cost-specific portfolios: {y}")
        d=con.execute(f"SELECT * FROM read_parquet('{folder}/{y}.parquet') ORDER BY idx,permno").df()
        market=con.execute(f"SELECT date,vw_market FROM read_parquet('{PARENT}/results/panel/market_{y}.parquet')").df().set_index("date").vw_market
        for _,f in d.groupby("idx",sort=True):
            date=pd.Timestamp(f.date.iloc[0])
            # Positions and candidate stocks only. Today's raw records for every
            # previously held name are retained, including names failing screens.
            held=oldids[np.any(holdings!=0,axis=(0,1))]
            keep=f.eligible.to_numpy()|np.isin(f.permno.to_numpy(),held)
            f=f.loc[keep].copy()
            ids=f.permno.to_numpy(dtype=np.int64);union=np.union1d(held,ids)
            loc=np.searchsorted(union,ids)
            prevloc=np.searchsorted(oldids,held)
            v=np.zeros((len(COSTS),len(STRATEGIES),len(union)))
            v[:,:,np.searchsorted(union,held)]=holdings[:,:,prevloc]
            target,nl,ns=weights(f,choice)
            w=np.zeros((len(STRATEGIES),len(union)));w[:,loc]=target
            rawret=f.ret.to_numpy(dtype=float,na_value=np.nan)
            rawrx=f.retx.to_numpy(dtype=float,na_value=np.nan)
            r=np.zeros(len(union));rx=r.copy()
            r[loc]=np.nan_to_num(rawret,nan=0.)
            rx[loc]=np.where(np.isfinite(rawrx),rawrx,np.nan_to_num(rawret,nan=0.))
            gone=~np.isin(union,ids)
            missing=np.zeros(len(union),bool);missing[loc]=~np.isfinite(rawret)
            outputs=[]
            for ci,(label,c,b) in enumerate(COSTS):
                pnl,after,turn,fee,borrow,before=execute(v[ci],w,r,rx,c,b)
                outputs.append(after[:,loc])
                for si,name in enumerate(STRATEGIES):
                    rows.append(dict(date=date,strategy=name,scenario=label,returns=pnl[si],
                       before_fee=before[si],turnover=turn[si],cost=fee[si],borrow=borrow[si],
                       n_long=int(nl[si]),n_short=int(ns[si]),active=bool(nl[si]>0),
                       missing_held_returns=int((missing&(w[si]!=0)).sum()),
                       absent_exits=int((gone&(v[ci,si]!=0)).sum()),
                       vw_market=float(market.loc[date])))
            holdings=np.stack(outputs);oldids=ids
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"daily_returns.csv",index=False)
    return df

def mean_fit(r,lags=20):
    return sm.OLS(np.asarray(r),np.ones((len(r),1)),missing="raise").fit(
        cov_type="HAC",cov_kwds={"maxlags":lags})

def statistics(df):
    rows=[];robust=[]
    for period,(start,end) in PERIODS.items():
        z=df.loc[df.date.between(start,end)]
        for (name,scenario),g in z.groupby(["strategy","scenario"]):
            g=g.sort_values("date");r=g.returns.to_numpy();fit=mean_fit(r)
            nav=np.cumprod(1+r);draw=nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1
            controlled=sm.OLS(r,sm.add_constant(g.vw_market.to_numpy())).fit(
                cov_type="HAC",cov_kwds={"maxlags":20})
            sd=r.std(ddof=1);ci=fit.conf_int()[0]*252
            rows.append(dict(period=period,strategy=name,scenario=scenario,days=len(r),
                annual_mean=float(r.mean()*252),cagr=float(nav[-1]**(252/len(r))-1),
                volatility=float(sd*np.sqrt(252)),sharpe=float(r.mean()/sd*np.sqrt(252)),
                max_drawdown=float(draw.min()),t=float(fit.tvalues[0]),p=float(fit.pvalues[0]),
                ci_low=float(ci[0]),ci_high=float(ci[1]),annual_turnover=float(g.turnover.mean()*252),
                annual_cost=float(g.cost.mean()*252),annual_borrow=float(g.borrow.mean()*252),
                active_fraction=float(g.active.mean()),mean_longs=float(g.n_long.mean()),mean_shorts=float(g.n_short.mean()),
                vw_intercept=float(controlled.params[0]*252),vw_intercept_t=float(controlled.tvalues[0]),
                vw_beta=float(controlled.params[1])))
            if period in ["holdout","recent"]:
                for lag in [20,60]:
                    f=mean_fit(r,lag);lo,hi=f.conf_int()[0]*252
                    robust.append(dict(period=period,strategy=name,scenario=scenario,
                        frequency="daily",lags=lag,annual_mean=float(r.mean()*252),
                        t=float(f.tvalues[0]),p=float(f.pvalues[0]),ci_low=float(lo),ci_high=float(hi)))
                m=g.set_index("date").returns.resample("ME").apply(lambda x:np.prod(1+x)-1)
                f=mean_fit(m,3);lo,hi=f.conf_int()[0]*12
                robust.append(dict(period=period,strategy=name,scenario=scenario,
                    frequency="monthly",lags=3,annual_mean=float(m.mean()*12),
                    t=float(f.tvalues[0]),p=float(f.pvalues[0]),ci_low=float(lo),ci_high=float(hi)))
    stats=pd.DataFrame(rows)
    stats["p_holm_6_ls"]=np.nan
    for _,z in stats.loc[stats.strategy.isin(STRATEGIES[:6])].groupby(["period","scenario"]):
        stats.loc[z.index,"p_holm_6_ls"]=multipletests(z.p,method="holm")[1]
    stats.to_csv(OUT/"performance.csv",index=False)
    pd.DataFrame(robust).to_csv(OUT/"inference_robustness.csv",index=False)
    annual=df.assign(year=df.date.dt.year).groupby(["strategy","scenario","year"]).agg(
        returns=("returns",lambda x:np.prod(1+x)-1),turnover=("turnover","sum"),cost=("cost","sum"),borrow=("borrow","sum"))
    annual.to_csv(OUT/"annual_returns.csv")
    gross=stats.loc[stats.scenario=="gross",["period","strategy","annual_mean","annual_turnover","active_fraction"]].copy()
    ls=gross.strategy.isin(STRATEGIES[:6])
    gross["approx_breakeven_one_way_bps"]=(gross.annual_mean-.01*gross.active_fraction*ls)/gross.annual_turnover*1e4
    gross.to_csv(OUT/"breakeven_costs.csv",index=False)
    return stats

def diagnostics(con,folder):
    p=f"{folder}/[12]*.parquet"
    yearly=con.execute(f"""SELECT year(date) AS year,count(*) FILTER(WHERE eligible) AS covered_stockdays,
      count(DISTINCT permno) FILTER(WHERE eligible) AS stocks,
      count(*) FILTER(WHERE eligible AND clock IS NOT NULL) AS timed_stockdays,
      avg(n180) FILTER(WHERE eligible) AS analysts,
      count(*) FILTER(WHERE eligible AND ret IS NULL) AS missing_returns
      FROM read_parquet('{p}') GROUP BY 1 ORDER BY 1""").df()
    yearly.to_csv(OUT/"universe_by_year.csv",index=False)
    universe=con.execute(f"""SELECT count(DISTINCT permno),count(*) FROM read_parquet('{p}') WHERE eligible""").fetchall()[0]
    # Prediction diagnostics use excluded future labels only in this function.
    ev=con.execute(f"SELECT * FROM read_parquet('{PARENT}/results/linked_events.parquet')").df()
    ev=ev.loc[ev.regular & ev.seasonal_gap.between(70,110)].copy()
    nxt=ev.next_actual_date_DIAGNOSTIC_ONLY
    ev=ev.loc[nxt.notna()]
    error=(ev.seasonal_predicted_date-ev.next_actual_date_DIAGNOSTIC_ONLY).dt.days
    info=dict(unique_covered_stocks=int(universe[0]),covered_stockdays=int(universe[1]),
        seasonal_forecast_diagnostics_all_linked_regular_events=dict(n=int(len(error)),
            median_absolute_calendar_days=float(error.abs().median()),within_7_days=float(error.abs().le(7).mean())),
        coverage_states=int(con.execute(f"SELECT count(*) FROM read_parquet('{OUT}/coverage_states.parquet')").fetchall()[0][0]))
    save(OUT/"universe_audit.json",info)
    return info

def report(df,stats,bs,choice,audit):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(1,2,figsize=(14,5),layout="constrained")
    for period,color in [("development","#73899b"),("holdout","#12866c")]:
        x=bs.loc[bs.period==period].set_index("bin").reindex(NAMES)
        ax[0].plot(np.arange(len(x)),x.excess_bps,marker="o",label=period,color=color)
        if period=="holdout":
            ax[0].fill_between(np.arange(len(x)),x.low_bps,x.high_bps,alpha=.15,color=color)
    ax[0].axhline(0,color="#777",lw=.7);ax[0].set_xticks(np.arange(len(NAMES)),NAMES,rotation=60)
    ax[0].set_ylabel("Daily return minus covered EW (basis points)")
    ax[0].set_xlabel("Nearest release clock: negative = forecast ahead; positive = known past")
    ax[0].set_title("Timing pattern and holdout HAC20 intervals");ax[0].legend()
    for name,scenario,label in [("nearest_selected","gross","Selected, gross"),
       ("nearest_selected","10bps","Selected, 10 bps + 1% borrow"),
       ("nearest_fixed","10bps","Fixed pre/after, 10 bps + 1% borrow"),
       ("known_cycle","10bps","Known cycle, 10 bps + 1% borrow")]:
        g=df.loc[(df.date>="2015-01-01")&(df.strategy==name)&(df.scenario==scenario)]
        ax[1].plot(g.date,np.cumprod(1+g.returns),label=label)
    ax[1].set_ylabel("Growth of $1, 200% target gross exposure")
    ax[1].set_title("2015–2025 chronological evaluation");ax[1].legend(fontsize=8)
    fig.savefig(OUT/"timing_results.png",dpi=180);plt.close(fig)
    def table(z):
        cols=["period","strategy","scenario","annual_mean","cagr","sharpe","max_drawdown","t","p_holm_6_ls","annual_turnover"]
        z=z[cols].copy()
        for c in ["annual_mean","cagr","max_drawdown"]:z[c]=z[c].map(lambda x:f"{x:.2%}")
        return z.round(3).to_html(index=False,border=0)
    h=stats.loc[(stats.period=="holdout")&(stats.strategy=="nearest_selected")&(stats.scenario=="10bps")].iloc[0]
    g=stats.loc[(stats.period=="holdout")&(stats.strategy=="nearest_selected")&(stats.scenario=="gross")].iloc[0]
    summary=f"The frozen timing-only rule earns {h.annual_mean:.2%} annual arithmetic return ({h.cagr:.2%} CAGR) in 2015–2025 after 10 bps per dollar traded and 1% annual short borrow, versus {g.annual_mean:.2%} gross. Its net Sharpe is {h.sharpe:.2f}; HAC20 95% annual-mean interval {h.ci_low:.2%} to {h.ci_high:.2%}; six-strategy Holm p={h.p_holm_6_ls:.3g}."
    body=f"""<!doctype html><html><head><meta charset="utf-8"><title>Five analysts: earnings date trading</title>
    <style>body{{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.55 system-ui;color:#243849}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border-bottom:1px solid #dde3e8;padding:7px;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}pre{{white-space:pre-wrap;font-size:13px}}.note{{padding:18px;background:#eef5f4;border-left:4px solid #12866c}}</style></head><body>
    <h1>Can the nearest earnings announcement date support a trading rule?</h1>
    <p class="note">{summary}</p>
    <p>This study specifically requires at least five historically active EPS analysts. It includes {audit['unique_covered_stocks']:,} distinct eligible securities and {audit['covered_stockdays']:,} covered stock-days over 1985–2025. It uses all qualifying US ordinary corporate common stocks, with no primary price, capitalization or liquidity threshold beyond positive lagged price/capitalization and tradable lagged status. The supplied data support quarterly earnings releases; these results do not cover every kind of corporate announcement.</p>
    <h2>The concrete rule</h2><p>At each previous close, retain stocks with at least five distinct analysts who issued an available FY1 or FY2 EPS forecast in the prior 180 calendar days. For each stock, find the nearest latest-known earnings release or predicted future release. Count market sessions: negative before a forecast, positive after a known release, ties to the known release. Buy equal weights at clock <b>{choice['long']}</b>; short equal weights at clock <b>{choice['short']}</b>. Rebalance daily. Hold $1 long and $1 short per $1 after-cost capital, and hold cash when either leg has fewer than 20 names. The two windows were selected only on 1985–2004 mean returns relative to the same covered universe, and frozen before later evaluations. Analyst count determines eligibility; EPS values, surprises, prices and momentum do not determine trade direction.</p>
    <p>{DESIGN['prediction']} {DESIGN.get('regularity','')} Overdue forecasts are dropped as future candidates. No eventual actual release date is used as a trading signal. Release-date forecasting is approximate: across all linked regular seasonal events, median absolute error is {audit['seasonal_forecast_diagnostics_all_linked_regular_events']['median_absolute_calendar_days']:.0f} calendar days. This diagnostic is broader than the five-analyst sample.</p>
    <img src="timing_results.png" alt="Nearest release timing pattern and chronological strategy performance">
    <h2>Holdout: all reported methods at 10 bps + 1% borrow</h2>
    {table(stats.loc[(stats.period=='holdout')&(stats.scenario=='10bps')])}
    <p>The first six strategies in the frozen design are long/short methods. Long-only and covered equal-weight benchmark are 100% long and have market exposure; their positive raw returns alone are not timing alpha. The covered benchmark is independently rebalanced with its own fees. The liquid, 90-day coverage and {DESIGN.get('alternative_clock_label','median-clock')} variants reuse the primary windows without retuning. Known-cycle is the prior study's +6..20 versus +36..50 comparator; it does not select on the nearest clock.</p>
    <h2>The frozen primary rule across periods and costs</h2>
    {table(stats.loc[stats.strategy=='nearest_selected'])}
    <h2>Interpretation and practical limits</h2><p>Annual mean = daily arithmetic mean ×252. CAGR compounds actual daily portfolio returns; Sharpe assumes zero cash return. HAC intervals and the six-test Holm adjustment concern portfolio mean return, not proof of causal or fully risk-adjusted alpha. The recent period overlaps the holdout. The observed cost scenarios are assumptions, not historical measured spreads, market impact or stock-specific borrow fees; short availability, financing and collateral rebates are absent.</p>
    <p>This is retrospective exploratory research. Related full-period results already existed in the directory, and date splitting cannot erase that prior knowledge. Analyst coverage is a trailing-update proxy, not a vendor historical NUMEST observation: withdrawals and review-only updates are unavailable to this reconstruction. Detail rows and event links come from a later database vintage, so not all historical revisions or identifier changes can be reconstructed. The conservative availability lag reduces timestamp leakage without eliminating vintage/linkage bias. Exact CUSIP linkage can exclude names and is not the official IBES/CRSP link.</p>
    <p>Raw CIZ total returns are used as supplied, including their supplied delisting treatment; no legacy delisting return is added. Per-scenario holdings and NAV include trading-cost feedback. Missing held returns are carried at zero and counted; disappearing securities are closed at their last mark and counted. No terminal liquidation fee is charged. The raw panel retains all securities' returns so a name dropping the coverage or trading screen does not silently remove a return needed to mark an existing holding. Daily full rebalancing closes ineligible positions at the prior mark.</p>
    <h2>Reproduce and audit</h2><p>Run <code>../.venv/bin/python analyze.py</code> then <code>../.venv/bin/python validate.py</code> from this folder. <code>--reuse-panel</code> reuses coverage and stock-day intermediates. Inputs are checksum verified; raw files and the earlier analyses are unchanged. Full daily/annual returns, bin results, cost sensitivity, frozen design, source hashes, coverage states, and validation outputs are included.</p>
    <p>Data-format reference checked during this analysis: <a href="{DESIGN['source']}">WRDS: Changes to CRSP Data</a>. Variable meanings are taken from the supplied WRDS schema files. The legacy CRSP format ends with the December 2024 release; this study uses the local CIZ data through December 2025.</p>
    <h2>Frozen research design</h2><pre>{json.dumps(DESIGN,indent=2)}</pre></body></html>"""
    (OUT/"report.html").write_text(body)
    (OUT/"summary.txt").write_text(summary+"\n"+json.dumps(choice,indent=2)+"\n")
    log(summary)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-panel",action="store_true")
    parser.add_argument("--reuse-returns",action="store_true")
    args=parser.parse_args();freeze()
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1500MB'")
    con.execute(f"SET temp_directory='{OUT}/duckdb_tmp'")
    inputs(con);coverage(con,args.reuse_panel);folder=panel(con,args.reuse_panel)
    choice,bs=bins(con,folder)
    df=pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"]) if args.reuse_returns else simulate(con,folder,choice)
    stats=statistics(df);audit=diagnostics(con,folder);report(df,stats,bs,choice,audit)

if __name__=="__main__":main()
