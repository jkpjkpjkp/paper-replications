#!/usr/bin/env python3
"""Local-only earnings-clock research. Run with .venv/bin/python analyze.py.

Raw inputs are never changed. The chronological test is retrospective and is
not a record of live trading. No realized future earnings date enters a signal.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path

import duckdb
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
BINS = [(2, 5), (6, 20), (21, 35), (36, 50), (51, 65), (66, 80)]
BIN_NAMES = [f"since_{a}_{b}" for a, b in BINS]
PERIODS = {"development": ("1985-01-01", "2004-12-31"),
           "validation": ("2005-01-01", "2014-12-31"),
           "holdout": ("2015-01-01", "2025-12-31"),
           "recent": ("2020-01-01", "2025-12-31"),
           "all": ("1985-01-01", "2025-12-31")}
COMMON = """securitytype='EQTY' AND securitysubtype='COM' AND sharetype='NS'
 AND usincflg='Y' AND issuertype='CORP' AND primaryexch IN ('N','A','Q')"""


def log(s):
    print(time.strftime("%H:%M:%S"), s, flush=True)


def json_save(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False))


def path_sql(paths):
    return "[" + ",".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def audit_inputs(con, verify_hashes=True):
    raw = ROOT / "crsp-daily/data/crsp_a_stock/dsf_v2"
    paths = [raw / f"{y}.parquet" for y in range(1982, 2026)]
    audits = []
    for p in paths:
        receipt = json.loads(p.with_suffix(".done.json").read_text())
        n, lo, hi = con.execute(f"SELECT count(*),min(dlycaldt),max(dlycaldt) FROM read_parquet('{p}')").fetchall()[0]
        if receipt["state"] != "complete" or n != receipt["rows"] or p.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Incomplete or inconsistent CRSP partition: {p}")
        checksum = hashlib.file_digest(p.open("rb"), "sha256").hexdigest() if verify_hashes else None
        if checksum and checksum != receipt["sha256"]:
            raise ValueError(f"Checksum mismatch: {p}")
        audits.append(dict(year=int(p.stem), n_rows=n, first=lo, last=hi, sha256=checksum))
    actual_dir = ROOT / "ibes/all-parquet/ibes/act_epsus"
    manifest = json.loads((actual_dir / "manifest.json").read_text())
    if not manifest["complete"] or manifest.get("limit") is not None:
        raise ValueError("EPS actuals table is incomplete or a connectivity sample")
    actual_paths = [actual_dir / f["name"] for f in manifest["files"]]
    for p, info in zip(actual_paths, manifest["files"]):
        if p.stat().st_size != info["bytes"]:
            raise ValueError(f"Actuals size mismatch: {p}")
        if verify_hashes and hashlib.file_digest(p.open("rb"), "sha256").hexdigest() != info["sha256"]:
            raise ValueError(f"Actuals checksum mismatch: {p}")
    actual = path_sql(actual_paths)
    assert con.execute(f"SELECT count(*) FROM read_parquet({actual})").fetchall()[0][0] == manifest["copy_rows"]
    canonical = con.execute(f"SELECT count(*),min(dlycaldt),max(dlycaldt) FROM read_parquet('{ROOT}/crsp-daily/parquet/*.parquet')").fetchall()[0]
    audit = dict(crsp_raw_partitions=audits, actuals_manifest=manifest,
                 canonical_directory=dict(n_rows=canonical[0], first=canonical[1], last=canonical[2]),
                 raw_source_used=str(raw), run_utc=pd.Timestamp.now(tz="UTC").isoformat())
    json_save(OUT / "input_audit.json", audit)
    return paths, actual, audit


def prepare_events(con, paths, actual):
    source = path_sql(paths)
    con.execute(f"""CREATE TEMP TABLE calendar AS
        SELECT dlycaldt AS date, row_number() OVER (ORDER BY dlycaldt)-1 AS idx
        FROM (SELECT DISTINCT dlycaldt FROM read_parquet({source}))""")
    calendar = con.execute("SELECT * FROM calendar ORDER BY idx").df()
    dates = calendar.date.to_numpy(dtype="datetime64[D]")
    con.execute(f"""CREATE TEMP TABLE actual_candidates AS
        SELECT ticker,trim(cusip) AS cusip,pends,anndats,actdats,anntims,
               greatest(anndats,actdats) AS availability_date
        FROM read_parquet({actual})
        WHERE pdicity='QTR' AND measure='EPS' AND usfirm=1
          AND anndats BETWEEN DATE '1982-01-01' AND DATE '2025-12-31'
          AND datediff('day',pends,anndats) BETWEEN 0 AND 180
          AND length(trim(cusip))=8 AND actdats IS NOT NULL
        QUALIFY row_number() OVER(PARTITION BY trim(cusip),pends
                                  ORDER BY actdats,anndats,ticker)=1""")
    con.execute("""CREATE TEMP TABLE events AS
        SELECT a.*,c.date AS event_session,c.idx AS event_idx
        FROM actual_candidates a ASOF LEFT JOIN calendar c ON a.anndats<=c.date""")
    log("Linking earnings to contemporaneous CRSP CUSIPs; ambiguous matches will be dropped")
    con.execute(f"""CREATE TEMP TABLE linked_events AS
        WITH matches AS (
          SELECT e.*,r.permno,
            count(*) OVER(PARTITION BY e.cusip,e.pends) AS n_matches
          FROM events e JOIN read_parquet({source}) r
            ON e.cusip=trim(r.cusip) AND e.event_session=r.dlycaldt
          WHERE {COMMON})
        SELECT * EXCLUDE(n_matches) FROM matches WHERE n_matches=1""")
    ev = con.execute("SELECT * FROM linked_events ORDER BY permno,availability_date,anndats").df()
    counts = dict(actual_candidates=con.execute("SELECT count(*) FROM actual_candidates").fetchall()[0][0],
                  linked_unambiguously=len(ev), linked_permnos=int(ev.permno.nunique()))
    # Treat unknown timezone and missing intraday timestamps conservatively:
    # observe at the close of the FIRST session strictly AFTER availability_date.
    # Credit returns only on subsequent sessions (asof with strict inequality).
    ev["known_idx"] = np.searchsorted(dates, ev.availability_date.to_numpy(dtype="datetime64[D]"), side="right")
    ev = ev.sort_values(["permno", "known_idx", "event_idx"])
    ev = ev.drop_duplicates(["permno", "known_idx"], keep="last")
    prev_max = ev.groupby("permno").event_idx.cummax().groupby(ev.permno).shift()
    ev = ev.loc[prev_max.isna() | (ev.event_idx > prev_max)].copy()
    ev["gap_days"] = ev.groupby("permno").anndats.diff().dt.days
    ev["forecast_gap"] = ev.groupby("permno").gap_days.transform(lambda s: s.rolling(3, min_periods=3).median())
    # Four observed announcements, including current, over the preceding 400 days.
    fourth = ev.groupby("permno").anndats.shift(3)
    ev["regular"] = ((ev.anndats - fourth).dt.days.le(400)
                       & ev.gap_days.between(70, 110)
                       & ev.forecast_gap.between(70, 110))
    forecast = ev.anndats + pd.to_timedelta(ev.forecast_gap.fillna(90).round(), unit="D")
    ev["predicted_idx"] = np.searchsorted(dates, forecast.to_numpy(dtype="datetime64[D]"), side="left")
    # searchsorted clips every forecast beyond the observed calendar to its end.
    # Disable these right-censored pre-announcement windows, rather than wrongly
    # treating every 2026 forecast as the first session of 2026.
    ev.loc[ev.predicted_idx==len(dates),"predicted_idx"]=len(dates)+365
    ev["predicted_date"] = forecast
    seasonal_forecast=fourth+pd.Timedelta(days=364)
    ev["seasonal_predicted_idx"] = np.searchsorted(dates,seasonal_forecast.to_numpy(dtype="datetime64[D]"),side="left")
    ev.loc[ev.seasonal_predicted_idx==len(dates),"seasonal_predicted_idx"]=len(dates)+365
    ev["seasonal_gap"]=(seasonal_forecast-ev.anndats).dt.days
    ev["seasonal_predicted_date"]=seasonal_forecast
    # Future actual dates are exported ONLY to measure forecast error; not in states.
    ev["next_actual_date_DIAGNOSTIC_ONLY"] = ev.groupby("permno").anndats.shift(-1)
    ev["forecast_error_days_DIAGNOSTIC_ONLY"] = (forecast - ev.next_actual_date_DIAGNOSTIC_ONLY).dt.days
    counts["monotone_information_states"] = len(ev)
    counts["regular_information_states"] = int(ev.regular.sum())
    good_forecast = ev.loc[ev.regular & ev.forecast_error_days_DIAGNOSTIC_ONLY.notna(), "forecast_error_days_DIAGNOSTIC_ONLY"]
    counts["forecast_median_absolute_error_calendar_days"] = float(good_forecast.abs().median())
    counts["forecast_within_7_calendar_days"] = float(good_forecast.abs().le(7).mean())
    seasonal_error=(seasonal_forecast-ev.next_actual_date_DIAGNOSTIC_ONLY).dt.days
    good_seasonal=seasonal_error[ev.regular & seasonal_error.notna()]
    counts["seasonal_forecast_median_absolute_error_calendar_days"]=float(good_seasonal.abs().median())
    counts["seasonal_forecast_within_7_calendar_days"]=float(good_seasonal.abs().le(7).mean())
    # DuckDB writes Parquet without requiring an additional PyArrow dependency.
    con.register("event_frame", ev)
    con.execute(f"COPY event_frame TO '{OUT}/linked_events.parquet' (FORMAT PARQUET)")
    con.execute("""CREATE TEMP TABLE states AS SELECT permno,known_idx,event_idx,
        predicted_idx,seasonal_predicted_idx,seasonal_gap,regular FROM event_frame""")
    con.unregister("event_frame")
    calendar.to_csv(OUT / "trading_calendar.csv", index=False)
    json_save(OUT / "event_audit.json", counts)
    log(str(counts))
    return dates, counts


def build_panel(con, paths, skip=False):
    folder = OUT / "panel"
    folder.mkdir(exist_ok=True)
    for y in range(1985, 2026):
        target = folder / f"{y}.parquet"
        if skip and target.exists():
            continue
        log(f"Constructing {y}: lagged liquidity screens, market benchmarks and earnings clocks")
        inputs = path_sql([p for p in paths if int(p.stem) in (y-1, y)])
        con.execute(f"""CREATE OR REPLACE TEMP TABLE yr AS
          WITH base AS (
            SELECT permno,dlycaldt AS date,cusip,
              CAST(dlyret AS DOUBLE) AS ret,CAST(dlyretx AS DOUBLE) AS retx,
              abs(CAST(dlyprc AS DOUBLE)) AS price,CAST(dlycap AS DOUBLE) AS cap,
              CAST(dlyvol AS DOUBLE)*abs(CAST(dlyprc AS DOUBLE)) AS dollars,
              conditionaltype,tradingstatusflg,({COMMON}) AS is_common
            FROM read_parquet({inputs})
            WHERE dlycaldt>=DATE '{y-1}-11-01'),
          lagged AS (
            SELECT *,lag(price) OVER w AS lag_price,lag(cap) OVER w AS lag_cap,
              lag(date) OVER w AS lag_date,
              lag(conditionaltype) OVER w AS lag_condition,
              lag(tradingstatusflg) OVER w AS lag_status,
              lag(is_common) OVER w AS lag_common,
              avg(dollars) OVER(PARTITION BY permno ORDER BY date
                ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) AS lag_adv
            FROM base WINDOW w AS(PARTITION BY permno ORDER BY date))
          SELECT l.*,c.idx FROM lagged l JOIN calendar c USING(date)
          WHERE year(date)={y} AND lag_common AND lag_cap>0
            AND lag_condition='RW' AND lag_status='A'""")
        ndup = con.execute("SELECT count(*)-count(DISTINCT (permno,date)) FROM yr").fetchall()[0][0]
        assert ndup == 0, f"Duplicate CRSP keys: {y}"
        assert con.execute("SELECT count(*) FROM yr WHERE ret < -1").fetchall()[0][0] == 0, f"Invalid return: {y}"
        con.execute("""CREATE OR REPLACE TEMP TABLE markets AS
          SELECT date,idx,avg(ret) AS ew_market,
            sum(ret*lag_cap) FILTER(WHERE ret IS NOT NULL)
               /sum(lag_cap) FILTER(WHERE ret IS NOT NULL) AS vw_market,
            count(*) AS universe_n FROM yr GROUP BY date,idx""")
        con.execute(f"COPY markets TO '{OUT}/panel/market_{y}.parquet' (FORMAT PARQUET)")
        con.execute(f"""COPY (
          SELECT r.permno,r.date,r.idx,r.ret,r.retx,r.lag_price,r.lag_cap,r.lag_adv,
            s.known_idx,s.event_idx,s.predicted_idx,s.seasonal_predicted_idx,s.seasonal_gap,s.regular,
            r.idx-s.event_idx AS age,s.predicted_idx-r.idx AS to_predicted,
            s.seasonal_predicted_idx-r.idx AS to_seasonal_predicted,
            m.ew_market,m.vw_market,
            r.ret-m.ew_market AS abnormal_ew,r.ret-m.vw_market AS abnormal_vw,
            CASE WHEN s.predicted_idx-r.idx>=0
                       AND s.predicted_idx-r.idx<r.idx-s.event_idx
                 THEN r.idx-s.predicted_idx ELSE r.idx-s.event_idx END
              AS nearest_predicted_signed_clock
          FROM yr r ASOF LEFT JOIN states s
            ON r.permno=s.permno AND r.idx>s.known_idx
          JOIN markets m ON r.date=m.date
          WHERE r.lag_price>=5 AND r.lag_cap>=100000 AND r.lag_adv>=1000000
          ) TO '{target}' (FORMAT PARQUET)""")
    return folder


def bin_return_tables(con, folder):
    cases = " ".join(f"WHEN age BETWEEN {a} AND {b} THEN '{n}'" for (a,b),n in zip(BINS,BIN_NAMES))
    sql = f"""WITH b AS (SELECT *,CASE {cases} ELSE NULL END AS bin
        FROM read_parquet('{folder}/[12][0-9][0-9][0-9].parquet')
        WHERE regular)
        SELECT date,bin,count(*) AS n,count(ret) AS n_returns,
          avg(coalesce(ret,0)) AS raw_return,
          avg(coalesce(ret,0)-ew_market) AS ew_abnormal,
          avg(coalesce(ret,0)-vw_market) AS vw_abnormal
        FROM b WHERE bin IS NOT NULL GROUP BY date,bin ORDER BY date,bin"""
    bins = con.execute(sql).df()
    bins.to_csv(OUT / "daily_timing_bins.csv", index=False)
    stats = []
    for pname,(start,end) in PERIODS.items():
        for b in BIN_NAMES:
            z = bins.loc[(bins.date>=start)&(bins.date<=end)&(bins.bin==b)&(bins.n>=20)]
            if len(z)<100:
                continue
            f=sm.OLS(z.ew_abnormal.to_numpy(),np.ones((len(z),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":20})
            ci=f.conf_int()[0]
            stats.append(dict(period=pname,bin=b,days=len(z),mean_stocks=z.n.mean(),
                abnormal_bps_daily=float(f.params[0]*10000),t_hac20=float(f.tvalues[0]),p_hac20=float(f.pvalues[0]),
                ci_low_bps=float(ci[0]*10000),ci_high_bps=float(ci[1]*10000)))
    stats=pd.DataFrame(stats)
    for p in stats.period.unique():
        mask=stats.period==p
        stats.loc[mask,"p_holm_6bins"]=multipletests(stats.loc[mask,"p_hac20"],method="holm")[1]
    stats.to_csv(OUT / "timing_bin_statistics.csv",index=False)
    train=stats.loc[stats.period=="development"].set_index("bin")
    # Select direction/windows using development ONLY; freeze before later periods.
    long_bin=train.abnormal_bps_daily.idxmax()
    short_bin=train.abnormal_bps_daily.idxmin()
    log(f"Development-selected windows: long {long_bin}, short {short_bin}")
    return bins,stats,long_bin,short_bin


def masks_for_frame(d, long_bin, short_bin):
    reg=d.regular.fillna(False).to_numpy(dtype=bool)
    age=d.age.to_numpy(dtype=float,na_value=np.nan)
    pred=d.to_predicted.to_numpy(dtype=float,na_value=np.nan)
    b={name:reg&(age>=a)&(age<=z) for name,(a,z) in zip(BIN_NAMES,BINS)}
    pre=reg&(pred>=1)&(pred<=10)&(age>=20)&(age<=80)
    seasonal=d.to_seasonal_predicted.to_numpy(dtype=float,na_value=np.nan)
    seasonal_gap=d.seasonal_gap.to_numpy(dtype=float,na_value=np.nan)
    pre_seasonal=reg&(seasonal>=1)&(seasonal<=10)&(age>=20)&(age<=80)&(seasonal_gap>=70)&(seasonal_gap<=110)
    return {
        "cycle":(b["since_6_20"],b["since_36_50"]),
        "post_earnings":(b["since_2_5"],b["since_21_35"]),
        "predicted_pre":(pre,b["since_21_35"]&~pre),
        "exploratory_seasonal_pre":(pre_seasonal,b["since_21_35"]&~pre_seasonal),
        "development_selected":(b[long_bin],b[short_bin]),
        "cycle_large_liquid":(b["since_6_20"]&(d.lag_cap.to_numpy()>=1e6)&(d.lag_adv.to_numpy()>=5e6),
                              b["since_36_50"]&(d.lag_cap.to_numpy()>=1e6)&(d.lag_adv.to_numpy()>=5e6))}


def simulate(con, folder, long_bin, short_bin):
    previous={}
    previous_pnl={}
    rows=[]
    for y in range(1985,2026):
        log(f"Backtesting {y} with drift-adjusted turnover")
        d=con.execute(f"SELECT * FROM read_parquet('{folder}/{y}.parquet') ORDER BY idx,permno").df()
        for idx,day in d.groupby("idx",sort=True):
            date=pd.Timestamp(day.date.iloc[0])
            ew=float(day.ew_market.iloc[0]);vw=float(day.vw_market.iloc[0])
            ids=day.permno.to_numpy(dtype=np.int64)
            returns=day.ret.to_numpy(dtype=float,na_value=np.nan)
            price_returns=day.retx.to_numpy(dtype=float,na_value=np.nan)
            masks=masks_for_frame(day,long_bin,short_bin)
            for name,(lm,ss) in masks.items():
                nl=int(lm.sum());ns=int(ss.sum())
                active=nl>=20 and ns>=20
                w=np.zeros(len(day))
                if active:
                    w[lm]=1/nl;w[ss]=-1/ns
                target={int(p):float(v) for p,v in zip(ids,w) if v}
                old=previous.get(name,{})
                denom=1+previous_pnl.get(name,0.)
                drift={p:v/denom for p,v in old.items()}
                turnover=sum(abs(target.get(p,0)-drift.get(p,0)) for p in target.keys()|drift.keys())
                r0=np.nan_to_num(returns,nan=0.)
                pnl=float(np.dot(w,r0))
                # Ex-dividend price changes determine shares to trade. Distributions
                # remain cash; total-return P&L includes both ordinary/special payouts.
                rx=np.where(np.isfinite(price_returns),price_returns,r0)
                previous[name]={int(p):float(v*(1+r)) for p,v,r in zip(ids,w,rx) if v}
                previous_pnl[name]=pnl
                rows.append(dict(date=date,strategy=name,n_long=nl,n_short=ns,active=active,
                    long_return=float(r0[lm].mean()) if active else 0.,
                    short_underlying_return=float(r0[ss].mean()) if active else 0.,
                    gross=pnl,turnover=turnover,
                    net_1bp_1pct_borrow=pnl-.0001*turnover-(.01/252 if active else 0),
                    net_10bps_1pct_borrow=pnl-.001*turnover-(.01/252 if active else 0),
                    net_25bps_2pct_borrow=pnl-.0025*turnover-(.02/252 if active else 0),
                    missing_held_returns=int((~np.isfinite(returns)&(w!=0)).sum()),
                    ew_market=ew,vw_market=vw))
    df=pd.DataFrame(rows)
    df.to_csv(OUT / "daily_strategies.csv",index=False)
    return df


def performance_stats(df):
    result=[]
    for period,(start,end) in PERIODS.items():
        for strategy,g in df.loc[df.date.between(start,end)].groupby("strategy"):
            g=g.sort_values("date")
            for return_type in ["gross","net_1bp_1pct_borrow","net_10bps_1pct_borrow","net_25bps_2pct_borrow"]:
                r=g[return_type].to_numpy()
                fit=sm.OLS(r,np.ones((len(g),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":20})
                X=sm.add_constant(g[["vw_market"]].to_numpy(),has_constant="add")
                controlled=sm.OLS(r,X,missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":20})
                nav=np.cumprod(1+r);draw=nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1
                ci=fit.conf_int()[0]*252
                result.append(dict(period=period,strategy=strategy,return_type=return_type,n_days=len(g),
                    active_fraction=float(g.active.mean()),annual_mean=float(r.mean()*252),
                    annual_geometric=float(nav[-1]**(252/len(g))-1),annual_vol=float(r.std(ddof=1)*np.sqrt(252)),
                    sharpe_zero_cash=float(r.mean()/r.std(ddof=1)*np.sqrt(252)),
                    t_hac20=float(fit.tvalues[0]),p_hac20=float(fit.pvalues[0]),
                    annual_ci_low=float(ci[0]),annual_ci_high=float(ci[1]),
                    vw_controlled_annual_intercept=float(controlled.params[0]*252),
                    vw_controlled_intercept_t=float(controlled.tvalues[0]),vw_beta=float(controlled.params[1]),
                    max_drawdown=float(draw.min()),annual_turnover=float(g.turnover.mean()*252),
                    mean_long_stocks=float(g.loc[g.active,"n_long"].mean()),
                    mean_short_stocks=float(g.loc[g.active,"n_short"].mean()),
                    missing_held_return_observations=int(g.missing_held_returns.sum())))
    s=pd.DataFrame(result)
    for (period,rt),z in s.groupby(["period","return_type"]):
        s.loc[z.index,"p_holm_all_strategies"]=multipletests(z.p_hac20,method="holm")[1]
    s.to_csv(OUT / "strategy_statistics.csv",index=False)
    breakeven=s.loc[s.return_type=="gross",["period","strategy","annual_mean","annual_turnover","active_fraction"]].copy()
    breakeven["breakeven_one_way_bps_with_1pct_borrow"]=(breakeven.annual_mean-.01*breakeven.active_fraction)/breakeven.annual_turnover*10000
    breakeven.to_csv(OUT / "breakeven_costs.csv",index=False)
    annual=df.assign(year=df.date.dt.year).groupby(["strategy","year"]).agg(
        gross=("gross",lambda x:np.prod(1+x)-1),
        net_1bp_1pct_borrow=("net_1bp_1pct_borrow",lambda x:np.prod(1+x)-1),
        net_10bps_1pct_borrow=("net_10bps_1pct_borrow",lambda x:np.prod(1+x)-1),
        turnover=("turnover","sum"),active_days=("active","sum"))
    annual.to_csv(OUT / "annual_strategy_returns.csv")
    return s


def figures_and_report(df,stats,binstats,events,audit,long_bin,short_bin):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(13,4.8),layout="constrained")
    h=binstats.loc[binstats.period=="holdout"].set_index("bin").reindex(BIN_NAMES)
    x=np.arange(len(h))
    axs[0].bar(x,h.abnormal_bps_daily,color=["#318c78" if v>=0 else "#d17059" for v in h.abnormal_bps_daily])
    axs[0].errorbar(x,h.abnormal_bps_daily,yerr=[h.abnormal_bps_daily-h.ci_low_bps,h.ci_high_bps-h.abnormal_bps_daily],fmt="none",color="#243849",capsize=3)
    axs[0].set_xticks(x,[f"{a}–{b}" for a,b in BINS]);axs[0].axhline(0,color="#555",lw=.7)
    axs[0].set_xlabel("Trading days since last earnings announcement")
    axs[0].set_ylabel("Mean daily market-adjusted return (bps)")
    axs[0].set_title("2015–2025 timing pattern; HAC 95% intervals")
    cycle=df.loc[(df.date>="2015-01-01")&(df.strategy=="cycle")]
    axs[1].plot(cycle.date,np.cumprod(1+cycle.gross),label="Cycle, before costs",color="#777",ls="--",lw=1.3)
    for name,label in [("cycle","Cycle, after costs"),("predicted_pre","Median-gap predicted pre-earnings, net"),("exploratory_seasonal_pre","Exploratory seasonal pre-earnings, net")]:
        z=df.loc[(df.date>="2015-01-01")&(df.strategy==name)]
        axs[1].plot(z.date,np.cumprod(1+z.net_1bp_1pct_borrow),label=label,lw=1.5)
    axs[1].set_ylabel("Growth of $1 (200% gross exposure)")
    axs[1].set_title("Chronological holdout; net = 1 bp trading + 1% borrow")
    axs[1].legend(fontsize=7,loc="best")
    fig.savefig(OUT / "earnings_timing.png",dpi=180)
    plt.close(fig)
    hold=stats.loc[(stats.period=="holdout")&(stats.return_type!="net_25bps_2pct_borrow")].copy()
    hold["annual mean"]=hold.annual_mean.map(lambda v:f"{v:.2%}")
    hold["95% interval"]=hold.apply(lambda r:f"{r.annual_ci_low:.2%} to {r.annual_ci_high:.2%}",axis=1)
    hold["market-controlled intercept"]=hold.vw_controlled_annual_intercept.map(lambda v:f"{v:.2%}")
    table=hold[["strategy","return_type","annual mean","95% interval","sharpe_zero_cash","t_hac20","p_holm_all_strategies","market-controlled intercept","max_drawdown"]].round(3).to_html(index=False,border=0)
    subperiod=stats.loc[(stats.strategy=="cycle")&(stats.return_type=="net_1bp_1pct_borrow"),["period","annual_mean","annual_geometric","sharpe_zero_cash","t_hac20","annual_turnover","active_fraction"]].round(4).to_html(index=False,border=0)
    cycle_gross=stats.loc[(stats.period=="holdout")&(stats.strategy=="cycle")&(stats.return_type=="gross")].iloc[0]
    cycle_net=stats.loc[(stats.period=="holdout")&(stats.strategy=="cycle")&(stats.return_type=="net_1bp_1pct_borrow")].iloc[0]
    significant=stats.loc[(stats.period=="holdout")&(stats.return_type=="gross")&(stats.p_holm_all_strategies<.05)]
    report=f"""<!doctype html><html><head><meta charset='utf-8'><title>Earnings timing research</title>
    <style>body{{max-width:1200px;margin:45px auto;padding:0 25px;font:16px/1.55 system-ui;color:#243849}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dde3e8;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.note{{background:#eff5f6;padding:18px;border-left:4px solid #318c78}}code{{font-size:13px}}</style></head><body>
    <h1>Can earnings timing alone earn abnormal returns?</h1>
    <p><b>The fixed earnings-cycle rule does not establish robust recent alpha.</b> It earns {cycle_gross.annual_mean:.2%} a year before costs in 2015–2025 (HAC t={cycle_gross.t_hac20:.2f}; 95% annual-mean interval {cycle_gross.annual_ci_low:.2%} to {cycle_gross.annual_ci_high:.2%}). Even 1 bp per dollar traded plus 1% annual short borrow reduces its annual mean to {cycle_net.annual_mean:.2%}. Of six gross holdout variants, the number with Holm-adjusted p&lt;0.05 is {len(significant)}.</p>
    <p>Retrospective research using only the supplied local CRSP and I/B/E/S observations. Development: 1985–2004; validation: 2005–2014; chronological holdout: 2015–2025. No trading-window choices are fit on the holdout. This split does not undo prior knowledge of the published earnings-cycle hypothesis.</p>
    <div class='note'>The processed CRSP directory ends in 1976, but the raw download contains the later years. This run uses checksum-verified raw annual Parquet files from 1982–2025, and the completed, untruncated EPS actuals table. The uncompleted I/B/E/S download as a whole is not assumed complete.</div>
    <h2>Holdout results</h2>{table}
    <p>Annual mean is daily arithmetic mean ×252. Sharpe uses zero cash return, not an available risk-free series. Strategy returns are on $1 capital supporting $1 long and $1 short (200% gross exposure). Net scenarios deduct one-way costs per dollar traded and annual short borrow. The reported market-controlled intercept uses a locally constructed value-weighted common-stock market return; it is not a Fama–French alpha.</p>
    <img src='earnings_timing.png' alt='Timing bins and net holdout strategy performance'>
    <h2>The fixed cycle rule across time: net of 1 bp + 1% borrow</h2>{subperiod}
    <h2>Trading rules</h2><ul>
    <li><b>Cycle:</b> equal-weight long +6 to +20 trading days after the latest known announcement; equal-weight short +36 to +50. Calendar sessions, not security-specific trading days, define age.</li>
    <li><b>Post earnings:</b> long +2 to +5; short +21 to +35.</li>
    <li><b>Predicted pre:</b> long 1–10 trading sessions before the predicted next announcement; short +21 to +35, excluding overlap with the long leg. Predicted next date = last announcement + median of the three preceding calendar gaps, rounded to a day. Actual future announcement dates never enter a signal. Predicted dates beyond the last observed market session are excluded from pre-announcement windows.</li>
    <li><b>Exploratory seasonal pre:</b> same pre-announcement/mid-cycle trade, predicting the next date from the announcement three quarters ago +364 calendar days (same weekday, 52 weeks later). Require the implied next gap to be 70–110 days. This predictor was added after the first holdout evaluation and is explicitly exploratory. It needs a new independent test before any trading claim.</li>
    <li><b>Development selected:</b> long {html.escape(long_bin)}, short {html.escape(short_bin)}. Select the highest/lowest development-period market-adjusted bin, then freeze those bins.</li>
    <li><b>Cycle large/liquid:</b> same fixed cycle with previous-close capitalization ≥$1 billion and past-20-session average dollar volume ≥$5 million.</li></ul>
    <h2>Construction and safeguards</h2><ul>
    <li>Prior-session US-incorporated ordinary corporate common shares on NYSE/AMEX/Nasdaq; no survivor-only selection. Baseline previous-close price ≥$5, capitalization ≥$100 million, previous 20 observations’ average dollar volume ≥$1 million; prior conditional type RW and prior trading status A. Missing returns are carried at zero for P&amp;L and explicitly counted; returns are not winsorized. CRSP total returns are used as supplied; no separate legacy delisting return is added.</li>
    <li>Quarterly EPS only. Fiscal period must end no later than the announcement and no more than 180 days earlier. Use first-activated observation per CUSIP/fiscal period; exclude missing activation dates and CUSIPs. The I/B/E/S ticker is never joined directly to a CRSP ticker.</li>
    <li>Match eight-character I/B/E/S CUSIP to historical CRSP CUSIP on the first market session on/after the announcement. Only unique security matches survive. This is a local exact-CUSIP linkage, not an official historical I/B/E/S–PERMNO link. Retrospective CUSIP fields may lose older names or firms; linkage selection remains a limitation.</li>
    <li>Economic event day zero = first session on/after the announcement date. Unknown intraday timezone is handled conservatively: assume availability at the close of the first session strictly after max(announcement date, activation date), and begin crediting returns only the following session. Late activation of old data cannot replace a newer known announcement.</li>
    <li>Require at least four known announcements within 400 calendar days and last gap 70–110 days. The median of the three preceding known gaps must also be 70–110. This information screen uses past available observations, without requiring a future announcement.</li>
    <li>Both legs need ≥20 stocks; otherwise the strategy holds cash. Daily equal-weight rebalancing. Turnover includes membership changes and drift from prior ex-dividend price moves. Trades occur at the preceding close; current-day returns and liquidity never determine that day’s holdings.</li>
    <li>Inference aggregates stocks into calendar-time portfolios, using Newey–West/HAC standard errors with 20 daily lags. Holm correction covers six strategy variants within each period/cost scenario; six timing-bin tests are corrected separately. The recent period overlaps the holdout and is a diagnostic, not an independent replication.</li></ul>
    <h2>Coverage and prediction quality</h2><pre>{html.escape(json.dumps(events,indent=2))}</pre>
    <p>The prediction diagnostics use future announcements only after signals have been fixed. All such fields in <code>linked_events.parquet</code> carry a DIAGNOSTIC_ONLY suffix and are absent from the signal-state table. Stock-day panel outputs contain <code>abnormal_ew = ret − ew_market</code>, <code>abnormal_vw = ret − vw_market</code>, the past/forecast clocks, and a signed nearest predicted clock (positive after a past announcement, negative before a nearer predicted one).</p>
    <h2>What these results can establish</h2>
    <p>Positive stock-minus-market returns describe a benchmark-relative timing pattern. A positive net long–short result with stable holdout behavior is stronger evidence of a tradable candidate. Neither establishes a fully risk-adjusted alpha: official risk-free, size, value, profitability, investment and momentum factors, timestamped schedule revisions, short availability, stock-specific borrow fees, market impact, and financing are absent. Analyst-coverage restrictions in the supplied paper are not imposed, so this is a timing-only extension rather than an exact replication.</p>
    <h2>Reproduce and inspect</h2><p>From <code>/home/jkp/data/earnings-timing</code>, run <code>.venv/bin/python analyze.py</code>. Outputs include full daily returns, turnover, yearly returns, confidence intervals, input checksums, event coverage, and exact code. <code>--reuse-panel</code> reuses the existing derived panel; remove it to recompute from source.</p>
    <p>Methods reference: Linnainmaa &amp; Zhang, <i>The Earnings Announcement Return Cycle</i> (November 2018), supplied redacted methods copy. Data-format reference: <a href='https://wrds-www.wharton.upenn.edu/pages/data-announcements/changes-to-crsp-data/'>WRDS CRSP CIZ transition documentation</a>. Statistical workflow: Kassis, Agarwal, He, Patel &amp; Brueckner (2026), <a href='https://doi.org/10.48550/arXiv.2609.00065'>Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>.</p>
    </body></html>"""
    (OUT / "report.html").write_text(report)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reuse-panel",action="store_true")
    parser.add_argument("--skip-checksums",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    con=duckdb.connect()
    con.execute("SET threads=2")
    con.execute("SET memory_limit='1800MB'")
    con.execute(f"SET temp_directory='{OUT}/duckdb_tmp'")
    log("Auditing actual local inputs, including checksums")
    paths,actual,audit=audit_inputs(con,not args.skip_checksums)
    dates,events=prepare_events(con,paths,actual)
    folder=build_panel(con,paths,skip=args.reuse_panel)
    bins,binstats,long_bin,short_bin=bin_return_tables(con,folder)
    df=simulate(con,folder,long_bin,short_bin)
    stats=performance_stats(df)
    figures_and_report(df,stats,binstats,events,audit,long_bin,short_bin)
    log("Complete. Holdout results after 1 bp per dollar traded and 1% annual borrow:")
    print(stats.loc[(stats.period=="holdout")&(stats.return_type=="net_1bp_1pct_borrow"),
          ["strategy","annual_mean","sharpe_zero_cash","t_hac20","p_holm_all_strategies","annual_turnover"]].to_string(index=False),flush=True)


if __name__=="__main__":
    main()
