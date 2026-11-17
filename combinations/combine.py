#!/usr/bin/env python3
"""Paired tests of earnings-clock overlays on reversal and momentum baselines.

Only local price/earnings observations are used. All rules are frozen in
design.json before strategy returns are computed. No additional agent work.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
ROOT=PARENT.parent
OUT=HERE/"results"
os.environ.setdefault("MPLCONFIGDIR",str(PARENT/".mpl-cache"))
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OVERLAYS=["baseline","cycle_tilt","cycle_veto","earnings_blackout"]
STRATEGIES=[f"{b}__{o}" for b in ["reversal","momentum"] for o in OVERLAYS]
COSTS=[("gross",0.,0.),("1bp",.0001,.01),("5bps",.0005,.01),
       ("10bps",.001,.01),("25bps",.0025,.01)]
PERIODS={"development":("1985-01-01","2004-12-31"),
         "validation":("2005-01-01","2014-12-31"),
         "holdout":("2015-01-01","2025-12-31"),
         "recent":("2020-01-01","2025-12-31"),
         "all":("1985-01-01","2025-12-31")}
DESIGN={
 "created_utc":pd.Timestamp.now(tz="UTC").isoformat(),
 "primary_cost_scenario":"10bps", "borrow_annual":.01,
 "baseline_reversal":"Prior five complete market sessions; long bottom quintile of cumulative return, short top quintile; rebalance daily.",
 "baseline_momentum":"Prior sessions t-252 through t-22; at least 200 of 231 observations; long top quintile, short bottom quintile; rebalance first session of each month using previous-close information.",
 "universe":"Same lagged price/cap/liquidity universe as parent analysis; no earnings-coverage restriction on either baseline.",
 "cycle_tilt":"At baseline rebalance, multiply existing long weights by 1+0.5*clock and short magnitudes by 1-0.5*clock; renormalize each leg. clock=+1 at +6..20, -1 at +36..50, else 0; require regular known earnings history.",
 "cycle_veto":"At baseline rebalance, remove longs at +36..50 and shorts at +6..20; redistribute within the existing baseline leg.",
 "earnings_blackout":"At baseline rebalance, remove both sides at age 0..5 after a known announcement or 1..5 sessions before the seasonal predicted next announcement; redistribute within baseline leg.",
 "prediction":"Announcement three quarters ago plus 364 calendar days; same weekday 52 weeks later. Require regular history and predicted quarterly gap 70..110 days. No realized future dates.",
 "trade_dates":"Overlays use only baseline scheduled rebalances. A held stock leaving the lagged eligible universe is liquidated at its last mark, with a charged exit trade, on any date.",
 "weights":"Equal weights within each baseline leg; $1 long and $1 short per $1 after-cost equity at rebalance. Drift between rebalances; no volatility targeting.",
 "selection":"Report all six paired overlays. Independently choose best overlay per baseline on development-period mean incremental net return at 10 bps, then freeze for later periods.",
 "inference":"Paired daily return differences; HAC 20 and 60 daily lags, monthly HAC 3; Holm across six baseline-overlay pairs for each period/cost. Primary = holdout, 10 bps, daily HAC20.",
 "scope":"Retrospective exploratory extension. Prior standalone earnings-clock holdout results were already seen. This is not live or untouched prospective evidence.",
 "sources":["https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html",
            "https://www.nber.org/papers/w2533",
            "https://doi.org/10.48550/arXiv.2609.00065"]}


def log(s):print(time.strftime("%H:%M:%S"),s,flush=True)


def save_json(path,obj):path.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))


def freeze_design():
    p=OUT/"design.json"
    if p.exists():
        old=json.loads(p.read_text())
        a={k:v for k,v in old.items() if k!="created_utc"}
        b={k:v for k,v in DESIGN.items() if k!="created_utc"}
        if a!=b:raise ValueError("An existing frozen design differs; use a new results directory.")
    else:save_json(p,DESIGN)


def audit_sources():
    parent=json.loads((PARENT/"results/input_audit.json").read_text())
    audited=[]
    for item in parent["crsp_raw_partitions"]:
        if item["year"]<1984:continue
        p=ROOT/f"crsp-daily/data/crsp_a_stock/dsf_v2/{item['year']}.parquet"
        with p.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        assert sha==item["sha256"],p
        audited.append({"path":str(p),"sha256":sha})
    for y in range(1985,2026):
        p=PARENT/f"results/panel/{y}.parquet"
        with p.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        audited.append({"path":str(p),"sha256":sha})
    save_json(OUT/"input_audit.json",{"parent_event_audit":json.loads((PARENT/"results/event_audit.json").read_text()),"files":audited})


def features(con,reuse=False):
    folder=OUT/"features";folder.mkdir(exist_ok=True)
    cal=pd.read_csv(PARENT/"results/trading_calendar.csv",parse_dates=["date"])
    con.register("calendar_frame",cal)
    con.execute("CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame")
    con.unregister("calendar_frame")
    for y in range(1985,2026):
        out=folder/f"{y}.parquet"
        if reuse and out.exists():continue
        log(f"Building strictly lagged baseline signals: {y}")
        p=PARENT/f"results/panel/{y}.parquet"
        raw=ROOT/"crsp-daily/data/crsp_a_stock/dsf_v2"
        files=f"['{raw}/{y-1}.parquet','{raw}/{y}.parquet']"
        con.execute(f"CREATE OR REPLACE TEMP TABLE eligible AS SELECT * FROM read_parquet('{p}')")
        con.execute(f"""COPY(
          WITH r AS (
            SELECT x.permno,c.idx,CAST(x.dlyret AS DOUBLE) AS r
            FROM read_parquet({files}) x JOIN calendar c ON x.dlycaldt=c.date
            SEMI JOIN (SELECT DISTINCT permno FROM eligible) e ON x.permno=e.permno),
          windows AS (
            SELECT permno,idx,
              count(r) OVER w5 AS n5,
              count(*) FILTER(WHERE r=-1) OVER w5 AS zero5,
              sum(CASE WHEN r>-1 THEN ln(1+r) END) OVER w5 AS log5,
              count(r) OVER wm AS nm,
              count(*) FILTER(WHERE r=-1) OVER wm AS zerom,
              sum(CASE WHEN r>-1 THEN ln(1+r) END) OVER wm AS logm
            FROM r WINDOW
              w5 AS(PARTITION BY permno ORDER BY idx RANGE BETWEEN 5 PRECEDING AND 1 PRECEDING),
              wm AS(PARTITION BY permno ORDER BY idx RANGE BETWEEN 252 PRECEDING AND 22 PRECEDING))
          SELECT e.*,
            CASE WHEN w.n5=5 THEN CASE WHEN zero5>0 THEN -1 ELSE exp(log5)-1 END END AS reversal5,
            CASE WHEN w.nm>=200 THEN CASE WHEN zerom>0 THEN -1 ELSE exp(logm)-1 END END AS momentum12_2,
            n5,nm FROM eligible e JOIN windows w USING(permno,idx)
          ) TO '{out}' (FORMAT PARQUET)""")
    return folder


def target_weights(frame):
    """Every overlay is confined to the baseline's own candidate positions."""
    n=len(frame);w=np.zeros((8,n));elig=np.zeros((8,n),dtype=bool)
    reg=frame.regular.fillna(False).to_numpy(dtype=bool)
    age=frame.age.to_numpy(dtype=float,na_value=np.nan)
    soon=frame.to_seasonal_predicted.to_numpy(dtype=float,na_value=np.nan)
    gap=frame.seasonal_gap.to_numpy(dtype=float,na_value=np.nan)
    early=reg&(age>=6)&(age<=20);late=reg&(age>=36)&(age<=50)
    clock=early.astype(float)-late.astype(float)
    blackout=((age>=0)&(age<=5))|(reg&(soon>=1)&(soon<=5)&(gap>=70)&(gap<=110))
    nlong=np.zeros(8,dtype=int);nshort=np.zeros(8,dtype=int)
    for start,col,direction in [(0,"reversal5",-1),(4,"momentum12_2",1)]:
        sig=frame[col].to_numpy(dtype=float,na_value=np.nan)*direction
        good=np.flatnonzero(np.isfinite(sig));elig[start:start+4,np.isfinite(sig)]=True
        if len(good)<100:continue
        # Deterministic tie break on permno, which is frame's sort order.
        order=good[np.argsort(sig[good],kind="stable")];k=int(len(good)*.2)
        short=np.zeros(n,dtype=bool);long=np.zeros(n,dtype=bool)
        short[order[:k]]=True;long[order[-k:]]=True
        for j,overlay in enumerate(OVERLAYS):
            l=long.copy();s=short.copy();a=np.ones(n);b=np.ones(n)
            if overlay=="cycle_tilt":a=1+.5*clock;b=1-.5*clock
            elif overlay=="cycle_veto":l&=~late;s&=~early
            elif overlay=="earnings_blackout":l&=~blackout;s&=~blackout
            row=start+j;nlong[row]=l.sum();nshort[row]=s.sum()
            if min(l.sum(),s.sum())<20:continue
            w[row,l]=a[l]/a[l].sum();w[row,s]=-b[s]/b[s].sum()
    return w,elig,nlong,nshort


def execute_step(v,w,eligible,rebal,r,rx,cost,borrow):
    """Exact after-cost sizing at rebalances; otherwise carry existing shares.

    v is signed marked position value / pre-trade NAV. Solving x=1-cost*turn
    sizes new weights to after-cost NAV rather than assuming baseline costs
    were paid by somebody else. Borrow is charged on current short value.
    """
    new=np.where(eligible,v,0.)
    trade=np.abs(new-v).sum(axis=1)
    fee=cost*trade
    if rebal.any():
        wr=w[rebal];vr=v[rebal]
        x=np.ones(rebal.sum())
        for _ in range(8):x=1-cost*np.abs(wr*x[:,None]-vr).sum(axis=1)
        new[rebal]=wr*x[:,None]
        trade[rebal]=np.abs(new[rebal]-vr).sum(axis=1)
        fee[rebal]=cost*trade[rebal]
        assert np.max(np.abs(x-(1-fee[rebal])))<1e-11
    short=-np.minimum(new,0).sum(axis=1)
    borrowing=borrow/252*short
    pnl=(new*r).sum(axis=1)-fee-borrowing
    assert np.all(1+pnl>0),"Portfolio insolvency needs explicit handling"
    after=new*(1+rx)/(1+pnl[:,None])
    return pnl,trade,borrowing,after,np.abs(new).sum(axis=1),fee


def simulate(con,folder):
    rows=[];old_ids=np.array([],dtype=np.int64)
    # Cost-specific positions permit drift/fee feedback to differ across scenarios.
    holdings=np.zeros((len(COSTS),8,0));previous_month=None
    for y in range(1985,2026):
        log(f"Simulating paired baseline/overlay portfolios: {y}")
        d=con.execute(f"SELECT * FROM read_parquet('{folder}/{y}.parquet') ORDER BY idx,permno").df()
        for idx,f in d.groupby("idx",sort=True):
            ids=f.permno.to_numpy(dtype=np.int64);union=np.union1d(old_ids,ids)
            oldloc=np.searchsorted(union,old_ids);newloc=np.searchsorted(union,ids)
            v=np.zeros((len(COSTS),8,len(union)));v[:,:,oldloc]=holdings
            tw,el,nl,ns=target_weights(f)
            w=np.zeros((8,len(union)));w[:,newloc]=tw
            eligible=np.zeros((8,len(union)),dtype=bool);eligible[:,newloc]=el
            r=np.zeros(len(union));rx=r.copy()
            ret=f.ret.to_numpy(dtype=float,na_value=np.nan)
            retx=f.retx.to_numpy(dtype=float,na_value=np.nan)
            r[newloc]=np.nan_to_num(ret,nan=0.)
            rx[newloc]=np.where(np.isfinite(retx),retx,np.nan_to_num(ret,nan=0.))
            date=pd.Timestamp(f.date.iloc[0]);month=(date.year,date.month)
            reb=np.array([True]*4+[month!=previous_month]*4)
            ew=float(f.ew_market.iloc[0]);vw=float(f.vw_market.iloc[0])
            next_positions=[]
            for ci,(label,c,b) in enumerate(COSTS):
                pnl,turn,borrowing,after,gross_exposure,fee=execute_step(v[ci],w,eligible,reb,r,rx,c,b)
                next_positions.append(after[:,newloc])
                for si,name in enumerate(STRATEGIES):
                    rows.append(dict(date=date,strategy=name,baseline=name.split("__")[0],overlay=name.split("__")[1],
                        scenario=label,returns=float(pnl[si]),turnover=float(turn[si]),
                        cost=float(fee[si]),borrow=float(borrowing[si]),gross_exposure=float(gross_exposure[si]),
                        scheduled_rebalance=bool(reb[si]),
                        n_long=int(nl[si]) if reb[si] else int((after[si]>0).sum()),
                        n_short=int(ns[si]) if reb[si] else int((after[si]<0).sum()),
                        ew_market=ew,vw_market=vw))
            holdings=np.stack(next_positions);old_ids=ids;previous_month=month
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"daily_returns.csv",index=False)
    return df


def mean_fit(r,lags=20):
    return sm.OLS(np.asarray(r),np.ones((len(r),1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":lags})


def statistics(df):
    perf=[];paired=[];robust=[]
    for period,(start,end) in PERIODS.items():
        sub=df.loc[df.date.between(start,end)]
        for (name,scenario),g in sub.groupby(["strategy","scenario"]):
            g=g.sort_values("date");r=g.returns.to_numpy();fit=mean_fit(r)
            nav=np.cumprod(1+r);draw=nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1
            perf.append(dict(period=period,strategy=name,baseline=g.baseline.iloc[0],overlay=g.overlay.iloc[0],scenario=scenario,
                n_days=len(g),annual_mean=float(r.mean()*252),annual_geometric=float(nav[-1]**(252/len(r))-1),
                annual_vol=float(r.std(ddof=1)*np.sqrt(252)),sharpe=float(r.mean()/r.std(ddof=1)*np.sqrt(252)),
                max_drawdown=float(draw.min()),annual_turnover=float(g.turnover.mean()*252),
                annual_trading_cost=float(g.cost.mean()*252),annual_borrow=float(g.borrow.mean()*252),
                t_hac20=float(fit.tvalues[0]),p_hac20=float(fit.pvalues[0]),
                mean_gross_exposure=float(g.gross_exposure.mean()),
                scheduled_rebalances=int(g.scheduled_rebalance.sum())))
        for baseline in ["reversal","momentum"]:
            for scenario,_,_ in COSTS:
                z=sub.loc[(sub.baseline==baseline)&(sub.scenario==scenario)]
                b=z.loc[z.overlay=="baseline"].set_index("date").sort_index()
                for overlay in OVERLAYS[1:]:
                    g=z.loc[z.overlay==overlay].set_index("date").sort_index()
                    assert g.index.equals(b.index)
                    diff=g.returns-b.returns;fit=mean_fit(diff)
                    ci=fit.conf_int()[0]*252
                    # Incremental return controlled for the original strategy and
                    # market exposure; this is not a full factor alpha.
                    X=sm.add_constant(np.column_stack([b.returns,b.vw_market,b.ew_market-b.vw_market]),has_constant="add")
                    control=sm.OLS(diff.to_numpy(),X,missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":20})
                    paired.append(dict(period=period,baseline=baseline,overlay=overlay,scenario=scenario,
                        annual_baseline=float(b.returns.mean()*252),annual_combined=float(g.returns.mean()*252),
                        annual_increment=float(diff.mean()*252),increment_ci_low=float(ci[0]),increment_ci_high=float(ci[1]),
                        incremental_t_hac20=float(fit.tvalues[0]),incremental_p_hac20=float(fit.pvalues[0]),
                        baseline_annual_turnover=float(b.turnover.mean()*252),combined_annual_turnover=float(g.turnover.mean()*252),
                        annual_turnover_increment=float((g.turnover-b.turnover).mean()*252),
                        annual_before_fee_increment=float(((g.returns+g.cost+g.borrow)-(b.returns+b.cost+b.borrow)).mean()*252),
                        annual_trading_cost_increment=float((g.cost-b.cost).mean()*252),
                        annual_borrow_increment=float((g.borrow-b.borrow).mean()*252),
                        controlled_increment_annual=float(control.params[0]*252),controlled_increment_t=float(control.tvalues[0]),
                        baseline_loading=float(control.params[1]),market_loading=float(control.params[2]),
                        tracking_error_annual=float(diff.std(ddof=1)*np.sqrt(252)),information_ratio=float(diff.mean()/diff.std(ddof=1)*np.sqrt(252))))
                    if period=="holdout":
                        for lag in [5,20,60]:
                            f=mean_fit(diff,lag)
                            robust.append(dict(baseline=baseline,overlay=overlay,scenario=scenario,frequency="daily",hac_lags=lag,
                                annual_increment=float(diff.mean()*252),t=float(f.tvalues[0]),p=float(f.pvalues[0])))
                        # Monthly paired returns of separately compounded strategies.
                        m=g.returns.resample("ME").apply(lambda a:np.prod(1+a)-1)-b.returns.resample("ME").apply(lambda a:np.prod(1+a)-1)
                        f=mean_fit(m,3)
                        robust.append(dict(baseline=baseline,overlay=overlay,scenario=scenario,frequency="monthly",hac_lags=3,
                            annual_increment=float(m.mean()*12),t=float(f.tvalues[0]),p=float(f.pvalues[0])))
    perf=pd.DataFrame(perf);paired=pd.DataFrame(paired)
    for (_,scenario),z in paired.groupby(["period","scenario"]):
        paired.loc[z.index,"p_holm_six_pairs"]=multipletests(z.incremental_p_hac20,method="holm")[1]
    perf.to_csv(OUT/"performance.csv",index=False)
    paired.to_csv(OUT/"paired_incremental_results.csv",index=False)
    pd.DataFrame(robust).to_csv(OUT/"inference_robustness.csv",index=False)
    # Winner selection is isolated to development, primary 10bps scenario.
    train=paired.loc[(paired.period=="development")&(paired.scenario=="10bps")]
    chosen={b:z.loc[z.annual_increment.idxmax(),"overlay"] for b,z in train.groupby("baseline")}
    save_json(OUT/"development_choices.json",chosen)
    annual=df.assign(year=df.date.dt.year).groupby(["baseline","overlay","scenario","year"]).agg(
        return_compounded=("returns",lambda x:np.prod(1+x)-1),turnover=("turnover","sum"),trading_cost=("cost","sum"))
    annual.to_csv(OUT/"annual_returns.csv")
    return perf,paired,chosen


def report(df,perf,paired,chosen):
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),layout="constrained")
    for col,b in enumerate(["reversal","momentum"]):
        z=df.loc[(df.date>="2015-01-01")&(df.baseline==b)&(df.scenario=="10bps")]
        for overlay in OVERLAYS:
            g=z.loc[z.overlay==overlay]
            axes[0,col].plot(g.date,np.cumprod(1+g.returns),label=overlay.replace("_"," "),lw=1.3)
        axes[0,col].set_yscale("log");axes[0,col].set_ylabel("Growth of $1, log scale")
        axes[0,col].set_title(f"{b.title()}: 2015–2025, net of 10 bps + 1% borrow")
        axes[0,col].legend(fontsize=8)
        p=paired.loc[(paired.period=="holdout")&(paired.baseline==b)&(paired.scenario=="10bps")].set_index("overlay").reindex(OVERLAYS[1:])
        x=np.arange(3)
        axes[1,col].bar(x,p.annual_increment*100,color=["#318c78" if a>=0 else "#d17059" for a in p.annual_increment])
        axes[1,col].errorbar(x,p.annual_increment*100,yerr=[(p.annual_increment-p.increment_ci_low)*100,(p.increment_ci_high-p.annual_increment)*100],fmt="none",capsize=4,color="#243849")
        axes[1,col].set_xticks(x,[s.replace("_"," ") for s in OVERLAYS[1:]])
        axes[1,col].axhline(0,color="#555",lw=.8)
        axes[1,col].set_ylabel("Annual incremental return (percentage points)")
        axes[1,col].set_title("Combined strategy minus its own baseline; HAC 95% intervals")
    fig.savefig(OUT/"incremental_edge.png",dpi=170);plt.close(fig)
    p=paired.loc[(paired.period=="holdout")&(paired.scenario=="10bps")].copy()
    display=p[["baseline","overlay","annual_baseline","annual_combined","annual_increment","increment_ci_low","increment_ci_high","annual_before_fee_increment","annual_trading_cost_increment","p_holm_six_pairs"]].copy()
    for c in display.columns:
        if c.startswith("annual_") or c.startswith("increment_ci"):display[c]=display[c].map(lambda a:f"{a:.2%}")
    display.p_holm_six_pairs=display.p_holm_six_pairs.map(lambda a:f"{a:.3g}")
    table=display.to_html(index=False,border=0)
    selected=pd.concat([paired.loc[(paired.baseline==b)&(paired.overlay==o)&(paired.scenario=="10bps")] for b,o in chosen.items()])
    selected=selected[["period","baseline","overlay","annual_increment","incremental_t_hac20","annual_turnover_increment"]].round(4).to_html(index=False,border=0)
    # Monetary attribution at a lower cost must be visible; high baseline fees
    # never make incremental orders free.
    costs=paired.loc[(paired.period=="holdout"),["baseline","overlay","scenario","annual_baseline","annual_combined","annual_increment","annual_trading_cost_increment","annual_turnover_increment"]].copy()
    costs=costs.round(4).to_html(index=False,border=0)
    sig=p.loc[(p.annual_increment>0)&(p.p_holm_six_pairs<.05)]
    title="Do earnings clocks improve an existing trading strategy?"
    text=f"""<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>
    <style>body{{max-width:1250px;margin:45px auto;padding:0 24px;font:16px/1.55 system-ui;color:#243849}}h1,h2{{line-height:1.2}}table{{font-size:12px;border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.scroll{{overflow:auto}}.note{{padding:18px;background:#edf4f4;border-left:4px solid #318c78}}</style></head><body>
    <h1>{title}</h1><p>We test an added edge by comparing each combined strategy directly with its own baseline, on identical dates and at the same transaction-cost assumptions. High baseline trading costs do not make new trades free.</p>
    <div class='note'>Primary comparison: 2015–2025; 10 bps one-way transaction cost per dollar bought or sold, plus 1% annual short borrow. Positive incremental net-return tests surviving Holm correction across six pairs: <b>{len(sig)}</b>. These are retrospective tests; the standalone earnings-clock holdout had already been inspected.</div>
    <h2>Holdout net returns and paired increments</h2><div class='scroll'>{table}</div>
    <p>Returns are annualized daily arithmetic means. The increment is measured in percentage points, not a relative percentage increase. Initial target exposure is $1 long/$1 short per $1 after-cost equity. Between monthly momentum rebalances, exposure drifts. A positive increment can improve an unprofitable baseline without making it profitable. Missing official risk factors and trade-level costs prevent a fully risk-adjusted alpha claim.</p>
    <img src='incremental_edge.png' alt='Net strategy performance and paired incremental returns'>
    <h2>Baseline implementations</h2><ul>
    <li><b>Daily short-term reversal:</b> rank cumulative returns over the preceding five market sessions; equal-weight long the bottom quintile, short the top quintile; rebalance every session. Five valid previous-session returns are required. This is a transparent high-turnover implementation of the short-term reversal family.</li>
    <li><b>Monthly momentum:</b> rank cumulative returns over t−252 through t−22, omitting the most recent month; at least 200 valid returns out of 231 sessions. Equal-weight long the top quintile, short the bottom quintile. Rebalance at the start of each month using previous-close information, and otherwise carry shares. This is a 12–2 momentum implementation, not a replication of the French value-weighted size-balanced factor.</li></ul>
    <h2>Three fixed overlays, applied only to baseline positions</h2><ul>
    <li><b>Cycle tilt:</b> earnings clock is +1 at 6–20 sessions after the latest known earnings date, −1 at 36–50, and zero otherwise. Multiply existing long weights by 1+0.5×clock and short magnitudes by 1−0.5×clock; normalize each leg back to its target exposure.</li>
    <li><b>Cycle veto:</b> remove late-cycle longs and early-cycle shorts; redistribute within each baseline leg.</li>
    <li><b>Earnings blackout:</b> remove both sides in sessions 0–5 after a known announcement and 1–5 before the seasonal predicted next announcement. Future dates are predicted using the corresponding prior-year quarter +364 days, with a 70–110-day expected gap and regular known announcement history.</li></ul>
    <p>Overlay trades occur only at the baseline scheduled rebalances. No overlay adds a stock outside that baseline’s existing candidate leg. A leg with fewer than 20 stocks holds cash. If a held name leaves the lagged tradable universe, it is liquidated at its last mark with charged exit turnover, even between scheduled rebalances. This convention is shared by the baseline and all its variants.</p>
    <h2>Development selection, frozen for later periods</h2><div class='scroll'>{selected}</div>
    <p>Best overlay per baseline is selected using 1985–2004 mean incremental return after 10 bps costs only. Validation is 2005–2014; chronological holdout is 2015–2025; 2020–2025 is an overlapping recent diagnostic. All three overlays are reported, not just the development winner. None of the strengths, thresholds or rebalance schedules is tuned on combination holdout returns.</p>
    <h2>Trading-cost sensitivity and attribution</h2><div class='scroll'>{costs}</div>
    <p>Cost-specific holdings are simulated separately. At each rebalance, x=1−cost×sum|x×target−marked holdings| sizes positions to equity remaining after trading costs. Costs are charged on the final net position changes of each combined portfolio, so baseline and overlay orders can offset. Carrying shares between rebalances avoids assuming daily equal-weight trading for monthly momentum. Borrow is charged against actual short exposure. Total returns determine P&amp;L; ex-dividend returns determine share-value drift; distributions remain cash. Turnover counts every dollar bought and sold, not half the sum. Financing, collateral rebates, market impact, stock-specific borrow and terminal liquidation are not modeled.</p>
    <h2>Data and inference</h2><p>The local universe, earnings linkage and availability controls are inherited from the parent analysis. Prior-close price ≥$5, capitalization ≥$100 million and past-20-observation average dollar volume ≥$1 million; US ordinary corporate common shares on NYSE/AMEX/Nasdaq. Baselines include firms without earnings links; their cycle clock is zero. Exact historical-CUSIP matching is retrospective and can miss old identities. Unknown event times are treated conservatively: wait until the close of the first session after max(announcement, activation), and credit returns only from later sessions.</p>
    <p>Paired daily combined-minus-baseline return series receive HAC/Newey–West standard errors with 20 lags and six-pair Holm correction. Alternative daily lags 5/60 and monthly HAC3 are provided. Incremental regressions control for the baseline return, the local value-weighted market and equal-weight-minus-value-weight market return; these controls are proxies, not official factor alpha. Prices are not intraday execution quotes. Signal availability and execution are modeled at the previous close.</p>
    <h2>Reproduce</h2><p>Run <code>../.venv/bin/python combine.py</code> from <code>earnings-timing/combinations</code>. Then run <code>../.venv/bin/python validate.py</code>. Derived feature files can be reused with <code>--reuse-features</code>. See design.json, input_audit.json, daily_returns.csv, performance.csv, paired_incremental_results.csv and inference_robustness.csv for the full record.</p>
    <h2>References</h2><p><a href='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html'>Kenneth French: daily momentum factor construction</a>; <a href='https://www.nber.org/papers/w2533'>Bruce N. Lehmann: Fads, Martingales, and Market Efficiency</a>. Statistical workflow reference: Kassis, Agarwal, He, Patel &amp; Brueckner (2026), <a href='https://doi.org/10.48550/arXiv.2609.00065'>Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>.</p>
    </body></html>"""
    (OUT/"report.html").write_text(text)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-features",action="store_true")
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    freeze_design();log("Verifying parent panels and raw price inputs")
    audit_sources()
    con=duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1800MB'")
    con.execute(f"SET temp_directory='{OUT}/duckdb_tmp'")
    folder=features(con,args.reuse_features)
    df=simulate(con,folder)
    perf,paired,chosen=statistics(df)
    report(df,perf,paired,chosen)
    log("Primary holdout comparison, net of 10bps + 1% borrow:")
    print(paired.loc[(paired.period=="holdout")&(paired.scenario=="10bps"),
       ["baseline","overlay","annual_baseline","annual_combined","annual_increment","incremental_t_hac20","p_holm_six_pairs","annual_trading_cost_increment"]].to_string(index=False),flush=True)


if __name__=="__main__":main()
