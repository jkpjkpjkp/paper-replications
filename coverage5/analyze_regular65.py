#!/usr/bin/env python3
"""User-requested regular-quarter focus, with a separate frozen result set."""
import argparse
import json
import shutil
from pathlib import Path
import analyze as base

OUT=base.HERE/"results_regular65"
SOURCE=base.HERE/"results"
base.OUT=OUT
base.BINS=[(-40,-21),(-20,-11),(-10,-6),(-5,-1),(0,5),(6,10),(11,20),(21,35)]
base.NAMES=[f"{a}:{b}" for a,b in base.BINS]
base.STRATEGIES[5]="selected_seasonal_clock"
base.DESIGN.update({
 "created_utc":base.pd.Timestamp.now(tz="UTC").isoformat(),
 "question":"Can nearest-announcement timing trade profitably among five-analyst stocks with regular approximately 65-trading-session earnings announcements?",
 "regularity":"At the previous close, require four known releases: each of the last three inter-release gaps must be 50–80 market sessions and their median 60–70. Require latest-known release age no more than 80 sessions. This gate uses only historical announcements; it does not require a future realized release.",
 "universe":base.DESIGN["universe"]+" Additionally require the explicitly defined regular 65-session historical release pattern and no overdue clock beyond 80 sessions.",
 "prediction":"Primary next release forecast = latest-known release session plus the median of its last three observed inter-release session gaps (60–70). This forecasts in market-session units. Alternative seasonal forecast uses the prior-year release plus 364 calendar days when its gap is 70–110 calendar days, otherwise the primary forecast. No realized future release date enters either signal.",
 "nearest":"Primary signed clock chooses the latest known release or the still-future median-session forecast, ties to the known release. Positive after known release, negative before forecast, zero on forecast day. Expired forecasts are not treated as actual releases. Only clock windows -40 through +35 are candidates; stale +81..130 windows are excluded by design.",
 "bins":base.BINS,
 "selection":"Choose highest and lowest mean calendar-time stock-minus-regular-covered-EW bins on 1985-2004, with >=1000 days and >=20 names/day. Freeze before later evaluation. Primary nearest_selected, 10 bps one-way +1% annual short borrow. All variants and benchmark use the same regular-history gate.",
 "sensitivities":"Apply selected bins without retuning to liquid names, 90-day analyst coverage, and a seasonal nearest clock. Fixed nearest-pre/midcycle and known-past cycle comparators, selected long leg alone, and regular covered EW benchmark are all reported.",
 "alternative_clock_column":"seasonal_clock",
 "alternative_clock_label":"seasonal-clock",
 "scope":"Retrospective exploratory revision requested by the user after the development sample selected an implausibly stale +81..130-session short window. The broad-universe preliminary run is retained separately. Related timing results were already present; this is not a prospective or untouched holdout. No regular-universe holdout return is used to set these definitions.",
 "strategies":base.STRATEGIES,
})

def panel(con,reuse=False):
    folder=OUT/"panel";folder.mkdir(exist_ok=True)
    # The original panel retained ALL raw securities, including ineligible ones.
    # Reusing its lagged fields does not inherit a liquidity or regularity filter.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE history65 AS
      WITH lags AS (
        SELECT permno,known_idx,event_idx,seasonal_predicted_idx,seasonal_gap,
          event_idx-lag(event_idx,1) OVER w AS g1,
          lag(event_idx,1) OVER w-lag(event_idx,2) OVER w AS g2,
          lag(event_idx,2) OVER w-lag(event_idx,3) OVER w AS g3
        FROM read_parquet('{base.PARENT}/results/linked_events.parquet')
        WINDOW w AS(PARTITION BY permno ORDER BY known_idx)),
      gaps AS (SELECT *,list_median([g1,g2,g3]) AS median_gap FROM lags)
      SELECT *,coalesce(g1 BETWEEN 50 AND 80 AND g2 BETWEEN 50 AND 80
        AND g3 BETWEEN 50 AND 80 AND median_gap BETWEEN 60 AND 70,false) AS regular65
      FROM gaps""")
    refs=[]
    for y in range(1985,2026):
        dest=folder/f"{y}.parquet";src=SOURCE/f"panel/{y}.parquet"
        refs.append(dict(path=str(src),sha256=base.checksum(src)))
        if reuse and dest.exists():continue
        base.log(f"Applying historical regular-65-session gate: {y}")
        con.execute(f"""COPY (
          WITH s AS (
            SELECT p.* EXCLUDE(eligible,regular,forecast_idx,predicted_idx,clock,median_clock),
              coalesce(h.regular65,false) AS regular,
              coalesce(p.eligible AND h.regular65 AND p.age BETWEEN 0 AND 80,false) AS eligible,
              CAST(h.median_gap AS BIGINT) AS period_sessions,h.g1,h.g2,h.g3,
              p.event_idx+CAST(h.median_gap AS BIGINT) AS predicted_idx,
              CASE WHEN h.seasonal_gap BETWEEN 70 AND 110 AND
                   h.seasonal_predicted_idx<=(SELECT max(idx) FROM calendar)
                THEN h.seasonal_predicted_idx
                ELSE p.event_idx+CAST(h.median_gap AS BIGINT) END AS seasonal_forecast_idx
            FROM read_parquet('{src}') p LEFT JOIN history65 h USING(permno,known_idx,event_idx)),
          clocks AS (
            SELECT *,predicted_idx AS forecast_idx,
              CASE WHEN predicted_idx-idx>=0 AND predicted_idx-idx<age
                THEN idx-predicted_idx ELSE age END AS clock,
              CASE WHEN seasonal_forecast_idx-idx>=0 AND seasonal_forecast_idx-idx<age
                THEN idx-seasonal_forecast_idx ELSE age END AS seasonal_clock
            FROM s)
          SELECT *,CASE WHEN regular AND predicted_idx-idx>=0 AND predicted_idx-idx<age
            THEN idx-predicted_idx ELSE age END AS median_clock
          FROM clocks ORDER BY idx,permno
        ) TO '{dest}' (FORMAT PARQUET)""")
    base.save(OUT/"panel_input_audit.json",refs)
    return folder

def diagnostics(con,folder):
    info=base.diagnostics(con,folder)
    q=f"""SELECT count(*),avg(period_sessions),min(g1),max(g1),
       count(DISTINCT permno),avg(CAST(clock BETWEEN -40 AND 35 AS INTEGER))
       FROM read_parquet('{folder}/[12]*.parquet') WHERE eligible"""
    z=con.execute(q).fetchall()[0]
    info["regular65"]=dict(stockdays=int(z[0]),mean_predicted_period_sessions=float(z[1]),
        min_last_gap=int(z[2]),max_last_gap=int(z[3]),unique_stocks=int(z[4]),
        fraction_with_candidate_clock=float(z[5]))
    # Date forecast accuracy, using future labels only in this diagnostic table.
    ev=con.execute(f"""SELECT e.permno,e.known_idx,e.event_idx,
      h.median_gap,h.regular65 FROM read_parquet('{base.PARENT}/results/linked_events.parquet') e
      JOIN history65 h USING(permno,known_idx,event_idx) ORDER BY e.permno,e.known_idx""").df()
    ev["next_idx_DIAGNOSTIC_ONLY"]=ev.groupby("permno").event_idx.shift(-1)
    v=ev.loc[ev.regular65 & ev.next_idx_DIAGNOSTIC_ONLY.notna()]
    error=v.event_idx+v.median_gap-v.next_idx_DIAGNOSTIC_ONLY
    info["median_session_forecast_diagnostics_all_regular65_events"]=dict(n=len(error),
        median_absolute_sessions=float(error.abs().median()),within_5_sessions=float(error.abs().le(5).mean()))
    base.save(OUT/"universe_audit.json",info)
    return info

def long_only_review(df,stats,audit):
    """Report the already specified long-only candidate without selecting anew."""
    h=stats.loc[(stats.period=="holdout")&(stats.strategy=="selected_long_only")&(stats.scenario=="1bp")].iloc[0]
    v=stats.loc[(stats.period=="validation")&(stats.strategy=="selected_long_only")&(stats.scenario=="1bp")].iloc[0]
    h10=stats.loc[(stats.period=="holdout")&(stats.strategy=="selected_long_only")&(stats.scenario=="10bps")].iloc[0]
    summary=(f"The prespecified long-only candidate buys equal weights in stocks 1-5 sessions before the median-gap forecast, "
       f"holding cash when fewer than 20 qualify. In 2015-2025 it earns {h.annual_mean:.2%} annual arithmetic return "
       f"({h.cagr:.2%} CAGR; Sharpe {h.sharpe:.2f}; max drawdown {h.max_drawdown:.2%}) at 1 bp trading cost, with no shorts or borrow. "
       f"It is invested on {h.active_fraction:.1%} of sessions. At 10 bps its annual arithmetic return is {h10.annual_mean:.2%}. "
       f"Earlier 2005-2014 validation earns {v.annual_mean:.2%} at 1 bp, with a VW-market-controlled intercept of {v.vw_intercept:.2%} "
       f"(t={v.vw_intercept_t:.2f}), versus holdout intercept {h.vw_intercept:.2%} (t={h.vw_intercept_t:.2f}). "
       "These raw returns and exploratory market controls do not establish stable timing alpha.")
    fig,ax=base.plt.subplots(figsize=(10.5,4.5),layout="constrained")
    for name,scenario,label in [("selected_long_only","1bp","Pre-forecast long-only, 1 bp"),
      ("selected_long_only","5bps","Pre-forecast long-only, 5 bps"),
      ("selected_long_only","10bps","Pre-forecast long-only, 10 bps"),
      ("covered_equal_weight","1bp","Regular covered universe EW, 1 bp")]:
        g=df.loc[(df.date>="2015-01-01")&(df.strategy==name)&(df.scenario==scenario)]
        ax.plot(g.date,base.np.cumprod(1+g.returns),label=label)
    ax.set_title("Long-only candidate and always-invested covered benchmark, 2015–2025")
    ax.set_ylabel("Growth of $1; zero return on cash");ax.legend(fontsize=8)
    fig.savefig(OUT/"long_only_candidate.png",dpi=180);base.plt.close(fig)
    cols=["period","scenario","annual_mean","cagr","sharpe","max_drawdown","vw_intercept","vw_intercept_t"]
    table=stats.loc[stats.strategy.eq("selected_long_only")&stats.period.isin(["development","validation","holdout"])&
      stats.scenario.isin(["1bp","5bps","10bps"]),cols].copy()
    for k in ["annual_mean","cagr","max_drawdown","vw_intercept"]:table[k]=table[k].map(lambda x:f"{x:.2%}")
    section=f"""<h2>Long-only candidate: better recent returns, weaker earlier validation</h2><p>{summary}</p>
      <p>The long-only rule was included in the frozen design before regular-universe evaluation; it reuses the development-selected long window. It has 100% long exposure when active and otherwise holds cash; the benchmark remains invested every session. The curves therefore have different market exposure and are not an exposure-matched alpha test. At 1 bp, the holdout raw annual-mean 95% HAC20 interval is {h.ci_low:.2%} to {h.ci_high:.2%}; the market-controlled t-statistic is unadjusted for examining alternative strategies. No official risk-free or multifactor series is available.</p>
      <img src="long_only_candidate.png" alt="Long-only cost sensitivity versus covered equal-weight benchmark">
      {table.round(3).to_html(index=False,border=0)}"""
    html=OUT/"report.html";body=html.read_text()
    body=body.replace("<h2>Interpretation and practical limits</h2>",section+"<h2>Interpretation and practical limits</h2>")
    error=audit["median_session_forecast_diagnostics_all_regular65_events"]
    old=(f"Release-date forecasting is approximate: across all linked regular seasonal events, median absolute error is "
       f"{audit['seasonal_forecast_diagnostics_all_linked_regular_events']['median_absolute_calendar_days']:.0f} calendar days. "
       "This diagnostic is broader than the five-analyst sample.")
    body=body.replace(old,f"Across all linked regular-65 events, the primary median-session forecast has median absolute error "
       f"{error['median_absolute_sessions']:.0f} trading sessions; {error['within_5_sessions']:.1%} fall within five sessions. "
       "Future realized dates are used only for this diagnostic, which is broader than the five-analyst sample.")
    html.write_text(body)
    with (OUT/"summary.txt").open("a") as f:f.write("\n"+summary+"\n")
    base.save(OUT/"trading_method.json",dict(
       regularity=base.DESIGN["regularity"],coverage=base.DESIGN["coverage"],
       forecast=base.DESIGN["prediction"],
       long_only="Equal-weight long nearest signed clock -5 through -1; at least 20 names; daily rebalance; otherwise cash; 100% long when invested.",
       long_short="Equal-weight long nearest clock -5 through -1; short nearest clock -40 through -21; at least 20 per leg; daily rebalance; otherwise cash; 200% gross target.",
       evidence="Low-cost long-only candidate, without stable market-controlled performance in 2005-2014. The long/short timing rule lacks reliable gross holdout evidence and loses after moderate assumed costs.",
       selection="Both windows selected on 1985-2004 only. Retrospective exploratory evaluation, not prospective evidence."))

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-panel",action="store_true")
    parser.add_argument("--reuse-returns",action="store_true");args=parser.parse_args()
    base.freeze()
    con=base.duckdb.connect();con.execute("SET threads=2");con.execute("SET memory_limit='1500MB'")
    con.execute(f"SET temp_directory='{OUT}/duckdb_tmp'")
    base.inputs(con)
    source=SOURCE/"coverage_states.parquet";target=OUT/"coverage_states.parquet"
    if not target.exists():shutil.copyfile(source,target)
    assert base.checksum(source)==base.checksum(target)
    base.save(OUT/"coverage_input_audit.json",dict(source=str(source),sha256=base.checksum(source)))
    folder=panel(con,args.reuse_panel)
    choice,bs=base.bins(con,folder)
    df=base.pd.read_csv(OUT/"daily_returns.csv",parse_dates=["date"]) if args.reuse_returns else base.simulate(con,folder,choice)
    stats=base.statistics(df);audit=diagnostics(con,folder);base.report(df,stats,bs,choice,audit)
    html=OUT/"report.html"
    body=html.read_text().replace("<h1>Can the nearest earnings announcement date support a trading rule?</h1>",
       "<h1>Nearest earnings timing among regular 65-session, five-analyst stocks</h1>")
    body=body.replace("Run <code>../.venv/bin/python analyze.py</code> then <code>../.venv/bin/python validate.py</code>",
       "Run <code>../.venv/bin/python analyze_regular65.py</code> then <code>../.venv/bin/python validate_regular65.py</code>")
    body=body.replace("stocks with at least five distinct analysts who issued an available FY1 or FY2 EPS forecast in the prior 180 calendar days.",
       "stocks with at least five distinct analysts who issued an available FY1 or FY2 EPS forecast in the prior 180 calendar days, and the regular quarterly history defined below.")
    html.write_text(body)
    long_only_review(df,stats,audit)

if __name__=="__main__":main()
