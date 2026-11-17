#!/usr/bin/env python3
"""Quick equal-dollar SPY short against the frozen pre-forecast long basket.

Uses raw CRSP total/ex-dividend returns, exact per-scenario dollar ledgers,
and bisection for after-fee sizing. Both legs rebalance at the previous close.
The strategy is exploratory: the long window and some later results were seen.
"""
from pathlib import Path
import hashlib
import json
import os
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parent.parent/".mpl-cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE/"results_regular65"
ROOT=HERE.parent.parent
SPY=84398
COSTS={"gross":(0.,0.),"1bp":(.0001,.01),"5bps":(.0005,.01),
       "10bps":(.001,.01),"25bps":(.0025,.02)}
PERIODS={"development_available":("1994-01-01","2004-12-31"),
 "validation":("2005-01-01","2014-12-31"),"holdout":("2015-01-01","2025-12-31"),
 "recent":("2020-01-01","2025-12-31"),"all_available":("1994-01-01","2025-12-31")}
DESIGN={"frozen_utc":pd.Timestamp.now(tz="UTC").isoformat(),
 "requested_check":"Short a market index against the prior announcement-date long basket to check market-relative performance.",
 "long":"Same frozen regular-65/5-analyst universe and nearest signed forecast clock -5..-1. Equal-weight long basket, >=20 names, otherwise cash on BOTH sides. No return, size, analyst forecast value, surprise or beta affects target direction or relative long weights.",
 "short":"SPY (CRSP PERMNO 84398), equal dollar magnitude to total long basket at every prior-close rebalance. 100% long +100% short per after-fee NAV; net marked target dollars exactly zero. End-of-day drift is corrected at next rebalance.",
 "execution":"Exact scenario-specific absolute dollar holdings/NAV, total-return P&L, ex-dividend drift, bisection of invest+c*sum(abs(invest*w-old))=pretradeNAV. All orders incl SPY and cash entry/exit charged. Short SPY dividend effects are included through total returns. Zero cash return, no collateral rebate/financing/margin cost, no terminal liquidation.",
 "costs":COSTS,"primary_scenario":"10bps","periods":PERIODS,
 "scope":"Quick exploratory market hedge, not a newly selected optimal rule or untouched out-of-sample evidence. SPY returns available from 1993; use full years 1994 onward. Timing window stays the original 1985-2004-selected window; no later-period retuning.",
 "inference":"Calendar-time portfolio returns, daily HAC20 primary, HAC60 and monthly HAC3 sensitivities. Raw annual mean and CAGR separately; dollar neutrality does not ensure beta neutrality or official multifactor alpha."}

def save(p,obj):p.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False))
def fit(r,lag=20):return sm.OLS(np.asarray(r),np.ones((len(r),1))).fit(cov_type="HAC",cov_kwds={"maxlags":lag})

def report(d,stats):
    h=stats.loc[stats.period.eq("holdout")&stats.scenario.eq("1bp")].iloc[0]
    h10=stats.loc[stats.period.eq("holdout")&stats.scenario.eq("10bps")].iloc[0]
    v=stats.loc[stats.period.eq("validation")&stats.scenario.eq("1bp")].iloc[0]
    plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout="constrained")
    for scenario,label in [("gross","No fees/borrow"),("1bp","1 bp + 1% borrow"),
          ("5bps","5 bps + 1% borrow"),("10bps","10 bps + 1% borrow")]:
        z=d.loc[d.date.ge("2015-01-01")&d.scenario.eq(scenario)]
        axes[0].plot(z.date,np.cumprod(1+z.returns),label=label)
    axes[0].set_title("Long announcement basket / short SPY, 2015–2025")
    axes[0].set_ylabel("Growth of $1, 200% gross target exposure");axes[0].legend(fontsize=8)
    annual=d.loc[d.date.ge("2015-01-01")&d.scenario.eq("1bp")].copy()
    annual=annual.groupby(annual.date.dt.year).returns.apply(lambda x:np.prod(1+x)-1)
    axes[1].bar(annual.index,annual.values*100,color=["#16836c" if x>=0 else "#cb6655" for x in annual])
    axes[1].axhline(0,color="#555",lw=.7);axes[1].set_ylabel("Calendar-year return (%)")
    axes[1].set_title("Dollar-neutral, net 1 bp + 1% borrow")
    fig.savefig(OUT/"spy_hedge_results.png",dpi=180);plt.close(fig)
    cols=["period","scenario","annual_mean","cagr","sharpe","max_drawdown","t_hac20","p_hac20","annual_turnover"]
    table=stats.loc[stats.period.isin(["validation","holdout"]),cols].copy()
    for col in ["annual_mean","cagr","max_drawdown"]:table[col]=table[col].map(lambda x:f"{x:.2%}")
    audit_path=OUT/"audit_long_only_summary.json"
    audit=json.loads(audit_path.read_text()) if audit_path.exists() else None
    audit_text=""
    if audit:
        total=sum(audit["actual_release_stockdays"].values())
        actual=audit["actual_release_stockdays"]["on_actual"]/total
        after=audit["actual_release_stockdays"]["after_actual"]/total
        audit_text=(f"<h2>Why five-day stock holdings could produce a much larger annual portfolio return</h2>"
          f"<p>An independent absolute-dollar ledger rebuilt from raw CRSP matches the prior long-only result. "
          f"The portfolio rotated its full capital among different companies: it was invested on {audit['invested_sessions']:,} of "
          f"{audit['market_sessions']:,} sessions, or {audit['mean_invested_sessions_per_year']:.1f} sessions per year, "
          f"averaging {audit['mean_stocks_when_invested']:.0f} stocks while active. A stock holding spell averaged "
          f"{audit['mean_spell_sessions']:.2f} sessions (median {audit['median_spell_sessions']:.0f}, maximum {audit['max_spell_sessions']}); "
          f"the unweighted mean raw return per completed stock spell was {audit['raw_equal_spell_weight_mean_stock_return']:.2%}. "
          f"Thus this was not the same portfolio invested for just 20 sessions a year.</p>"
          f"<p>Forecast day zero was excluded. Forecast errors still caused {actual:.1%} of held stock-days to fall on an actual "
          f"announcement-date market session and {after:.1%} after the nearest actual release. These are ex-post labels from linked "
          f"release dates; they do not adjust for announcement time of day. They show that the strategy does not reliably avoid "
          f"actual earnings-event exposure. The raw dates used in this diagnostic never select positions.</p>")
    summary=(f"The equal-dollar SPY hedge reduces the original 2015-2025 long-only 14.86% CAGR to {h.cagr:.2%} "
       f"at 1 bp per dollar traded plus 1% annual SPY borrow. Annual arithmetic mean {h.annual_mean:.2%}; "
       f"HAC20 95% interval {h.ci_low:.2%} to {h.ci_high:.2%}; t={h.t_hac20:.2f}, p={h.p_hac20:.3f}. "
       f"At 10 bps, CAGR {h10.cagr:.2%}. Earlier 2005-2014 CAGR at 1 bp is {v.cagr:.2%}. "
       "This quick check does not establish reliable market-relative timing advantage.")
    html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Announcement timing: SPY hedge check</title>
      <style>body{{max-width:1200px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#233847}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px;width:100%}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{padding:18px;background:#eef4f5;border-left:4px solid #16836c}}</style>
      </head><body><h1>Announcement timing with an equal-dollar SPY short</h1><p class="note">{summary}</p>
      <h2>Exact position rule</h2><p>Use the same >=5-active-analyst regular-quarterly stock universe: four known releases,
      three past gaps each 50–80 market sessions and median 60–70, latest known release no more than 80 sessions old.
      Forecast the next release from that median session gap. Equal-weight the stocks whose nearest release clock is -5..-1.
      When at least 20 qualify, buy $1 of the basket and short $1 of SPY per $1 after-cost NAV. Otherwise hold cash on both legs.
      Rebalance at every preceding close; net target holding is exactly zero. Price moves cause end-of-day drift before the next rebalance.</p>
      <p>SPY is the actual ETF in the raw CRSP files (PERMNO 84398), not a fitted market-regression hedge. Its total return includes
      the dividend expense of the short. Each scenario has independent dollar holdings and NAV, and both legs pay trading costs.
      Borrow costs apply only to actual short exposure while active. Cash earns zero; financing, collateral rebates and terminal
      liquidation costs are absent. Dollar neutrality does not guarantee beta neutrality; the observed holdout SPY beta at 1 bp
      is {h.spy_beta:.3f}.</p><img src="spy_hedge_results.png" alt="Dollar-neutral SPY hedge results">
      <h2>Chronological comparisons</h2>{table.round(3).to_html(index=False,border=0)}
      <p>Annual mean is daily arithmetic mean ×252; CAGR compounds all calendar-time portfolio returns, including cash sessions.
      Gross has no trading/borrow charges; 1/5/10 bps use 1% annual short borrow, and 25 bps uses 2%.
      The reported confidence interval concerns annual arithmetic mean, not CAGR. HAC20 p-values are unadjusted exploratory diagnostics.
      This checks the existing window without retuning; it is not a claim to an optimal strategy or an untouched holdout.</p>
      {audit_text}<h2>Reproduce</h2><p>Run <code>../.venv/bin/python audit_long_only.py</code> and
      <code>../.venv/bin/python hedge_spy.py</code>. Inspect <code>spy_hedge_design.json</code>,
      <code>spy_hedge_daily.csv</code>, <code>spy_hedge_performance.csv</code>, <code>spy_hedge_robustness.csv</code>,
      <code>spy_hedge_input_audit.json</code>, and <code>spy_hedge_validation.json</code>. Raw input hashes, daily target dollar neutrality,
      the gross long-minus-SPY identity, and fee/borrow arithmetic are checked.
      The independent audit also supplies complete holding spells and its raw-return dollar ledger.</p></body></html>"""
    (OUT/"spy_hedge_report.html").write_text(html)
    (OUT/"spy_hedge_summary.txt").write_text(summary+"\n")
    main_report=OUT/"report.html"
    if main_report.exists():
        body=main_report.read_text()
        start="<!-- SPY_HEDGE_UPDATE_START -->";end="<!-- SPY_HEDGE_UPDATE_END -->"
        if start in body:
            a=body.index(start);b=body.index(end,a)+len(end);body=body[:a]+body[b:]
        note=(start+f"<p class='note'><b>Updated market-neutral check:</b> {summary} "
          "<a href='spy_hedge_report.html'>Read the equal-dollar SPY hedge audit and results.</a></p>"+end)
        body=body.replace("<h2>The concrete rule</h2>",note+"<h2>The concrete rule</h2>")
        main_report.write_text(body)
    print(summary)

def main():
    dest=OUT/"spy_hedge_design.json"
    if dest.exists():
        old=json.loads(dest.read_text());old.pop("frozen_utc")
        new=DESIGN.copy();new.pop("frozen_utc")
        assert old==json.loads(json.dumps(new)),"Existing hedge definitions differ."
    else:save(dest,DESIGN)
    c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='1200MB'")
    cal=pd.read_csv(HERE.parent/"results/trading_calendar.csv",parse_dates=["date"])
    portfolios={k:dict(nav=1.,positions={}) for k in COSTS}
    raw=ROOT/"crsp-daily/data/crsp_a_stock/dsf_v2"
    rows=[];audits=[]
    for year in range(1994,2026):
        print("SPY dollar-neutral hedge",year,flush=True)
        source=raw/f"{year}.parquet";receipt=json.loads(source.with_suffix(".done.json").read_text())
        with source.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
        assert sha==receipt["sha256"] and source.stat().st_size==receipt["bytes"]
        panel=OUT/f"panel/{year}.parquet"
        with panel.open("rb") as f:panel_sha=hashlib.file_digest(f,"sha256").hexdigest()
        audits.append(dict(raw=str(source),raw_sha256=sha,panel=str(panel),panel_sha256=panel_sha))
        candidates=c.execute(f"""SELECT p.permno,p.date,p.idx,CAST(r.dlyret AS DOUBLE) AS ret,
          CAST(r.dlyretx AS DOUBLE) AS retx FROM read_parquet('{panel}') p
          JOIN read_parquet('{source}') r ON p.permno=r.permno AND p.date=r.dlycaldt
          WHERE p.eligible AND p.clock BETWEEN -5 AND -1 ORDER BY idx,permno""").df()
        market=c.execute(f"""SELECT dlycaldt AS date,CAST(dlyret AS DOUBLE) AS ret,
          CAST(dlyretx AS DOUBLE) AS retx,ticker,cusip FROM read_parquet('{source}')
          WHERE permno={SPY} ORDER BY date""").df().set_index("date")
        assert (market.ticker=="SPY").all() and market.ret.notna().all() and not market.index.duplicated().any()
        days=cal.loc[cal.date.dt.year.eq(year)]
        assert days.date.isin(market.index).all()
        grouped={int(k):g for k,g in candidates.groupby("idx",sort=False)}
        for entry in days.itertuples():
            dt=pd.Timestamp(entry.date);idx=int(entry.idx);f=grouped.get(idx)
            active=f is not None and len(f)>=20
            if not active:f=candidates.iloc[:0]
            n=len(f)
            assert SPY not in f.permno.to_numpy()
            ids=f.permno.to_numpy(dtype=int)
            if active:
                ids=np.r_[ids,SPY]
                w=np.r_[np.full(n,1/n),-1.]
                r=np.r_[f.ret.fillna(0).to_numpy(),float(market.loc[dt,"ret"])]
                rx=np.r_[f.retx.fillna(f.ret).fillna(0).to_numpy(),float(market.loc[dt,"retx"])]
            else:w=np.array([]);r=np.array([]);rx=np.array([])
            idset=set(map(int,ids))
            for scenario,(cost,borrow_rate) in COSTS.items():
                pf=portfolios[scenario];nav=pf["nav"];old=pf["positions"]
                old_in=np.array([old.get(int(p),0.) for p in ids])
                exits=sum(abs(v) for p,v in old.items() if p not in idset)
                lo=0.;hi=nav
                if active:
                    for _ in range(48):
                        x=(lo+hi)/2;trade=float(np.abs(x*w-old_in).sum())+exits
                        if x+cost*trade>nav:hi=x
                        else:lo=x
                    x=(lo+hi)/2;positions=x*w
                    trade=float(np.abs(positions-old_in).sum())+exits
                else:
                    positions=np.array([]);trade=sum(abs(v) for v in old.values())
                fee=cost*trade
                long=float(np.maximum(positions,0).sum());short=float(-np.minimum(positions,0).sum())
                assert abs(long-short)/nav<1e-12
                borrowing=borrow_rate/252*short
                before=float(positions@r)
                new_nav=nav-fee-borrowing+before
                assert new_nav>0
                signed={int(p):float(v*(1+vret)) for p,v,vret in zip(ids,positions,rx)}
                returns=new_nav/nav-1
                rows.append(dict(date=dt,scenario=scenario,returns=returns,nav=new_nav,
                    before_fee=before/nav,cost=fee/nav,borrow=borrowing/nav,turnover=trade/nav,
                    active=active,n_longs=n,long_exposure=long/nav,short_exposure=short/nav,
                    target_net_exposure=(long-short)/nav,end_net_exposure=sum(signed.values())/new_nav,
                    spy_return=float(market.loc[dt,"ret"])))
                pf.update(nav=new_nav,positions=signed)
    d=pd.DataFrame(rows);d.to_csv(OUT/"spy_hedge_daily.csv",index=False)
    save(OUT/"spy_hedge_input_audit.json",audits)
    assert np.allclose(d.returns,d.before_fee-d.cost-d.borrow,atol=1e-12)
    assert np.max(np.abs(d.target_net_exposure))<1e-12
    results=[];robust=[]
    for period,(start,end) in PERIODS.items():
        for scenario,g in d.loc[d.date.between(start,end)].groupby("scenario"):
            g=g.sort_values("date");r=g.returns.to_numpy();f=fit(r);ci=f.conf_int()[0]*252
            nav=np.cumprod(1+r);draw=nav/np.maximum.accumulate(np.r_[1,nav])[1:]-1
            sd=r.std(ddof=1)
            beta=sm.OLS(r,sm.add_constant(g.spy_return.to_numpy())).fit(cov_type="HAC",cov_kwds={"maxlags":20})
            results.append(dict(period=period,scenario=scenario,n_days=len(r),annual_mean=float(r.mean()*252),
              cagr=float(nav[-1]**(252/len(r))-1),sharpe=float(r.mean()/sd*np.sqrt(252)),
              annual_vol=float(sd*np.sqrt(252)),max_drawdown=float(draw.min()),
              t_hac20=float(f.tvalues[0]),p_hac20=float(f.pvalues[0]),ci_low=float(ci[0]),ci_high=float(ci[1]),
              annual_turnover=float(g.turnover.mean()*252),annual_cost=float(g.cost.mean()*252),
              annual_borrow=float(g.borrow.mean()*252),active_fraction=float(g.active.mean()),
              spy_beta=float(beta.params[1]),spy_controlled_intercept=float(beta.params[0]*252),
              spy_controlled_t=float(beta.tvalues[0])))
            if period in ["validation","holdout"]:
                h=fit(r,60);low,high=h.conf_int()[0]*252
                robust.append(dict(period=period,scenario=scenario,frequency="daily",lags=60,
                    annual_mean=float(r.mean()*252),ci_low=float(low),ci_high=float(high),t=float(h.tvalues[0]),p=float(h.pvalues[0])))
                m=g.set_index("date").returns.resample("ME").apply(lambda x:np.prod(1+x)-1)
                h=fit(m,3);low,high=h.conf_int()[0]*12
                robust.append(dict(period=period,scenario=scenario,frequency="monthly",lags=3,
                    annual_mean=float(m.mean()*12),ci_low=float(low),ci_high=float(high),t=float(h.tvalues[0]),p=float(h.pvalues[0])))
    stats=pd.DataFrame(results);stats.to_csv(OUT/"spy_hedge_performance.csv",index=False)
    pd.DataFrame(robust).to_csv(OUT/"spy_hedge_robustness.csv",index=False)
    d.assign(year=d.date.dt.year).groupby(["scenario","year"]).agg(
       returns=("returns",lambda x:np.prod(1+x)-1),turnover=("turnover","sum"),
       cost=("cost","sum"),borrow=("borrow","sum")).to_csv(OUT/"spy_hedge_annual.csv")
    # Explicit raw identity in the gross scenario: long basket return - SPY.
    gross=d.loc[d.scenario.eq("gross")].set_index("date")
    original=pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"])
    long=original.loc[original.strategy.eq("selected_long_only")&original.scenario.eq("gross")].set_index("date")
    shared=gross.index.intersection(long.index)
    expected=long.loc[shared,"returns"]-np.where(gross.loc[shared,"active"],gross.loc[shared,"spy_return"],0)
    assert np.allclose(gross.loc[shared,"returns"],expected,atol=1e-11)
    save(OUT/"spy_hedge_validation.json",dict(
       raw_spy_identity="PERMNO 84398, ticker SPY; complete daily returns 1994-2025",
       dollar_neutral_at_every_rebalance=True,max_abs_target_net_exposure=float(np.max(np.abs(d.target_net_exposure))),
       gross_return_equals_existing_long_basket_minus_spy_on_identical_active_days=True,
       cost_and_borrow_pnl_identities=True,scenario_specific_dollar_nav_and_holdings=True,
       raw_input_checksum_validation=True))
    report(d,stats)
    print(stats.loc[stats.period.isin(["validation","holdout"]),
       ["period","scenario","annual_mean","cagr","sharpe","t_hac20","ci_low","ci_high","max_drawdown","spy_beta"]].round(5).to_string(index=False))

if __name__=="__main__":main()
