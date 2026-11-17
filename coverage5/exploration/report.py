#!/usr/bin/env python3
"""Readable comparison of all frozen date-only long/short families."""
import json
import numpy as np
import pandas as pd
import explore as e

LABELS={"nearest_median":"Median forecast windows","nearest_seasonal":"Seasonal forecast windows",
 "since_known":"Days since known release","learned_median":"Learned median cycle",
 "learned_seasonal":"Learned seasonal cycle","fixed_periodic":"Fixed quarterly cosine"}

def rule(m):
    tail="Daily target weights."
    if m["adjustment"]<1:tail=f"Move {m['adjustment']:.0%} toward the target daily, retaining part of the old book and rebalancing both legs to equal dollars."
    if m["type"]=="window":
        return f"Clock {m['clock']}: long [{m['long_lo']}, {m['long_hi']}], short [{m['short_lo']}, {m['short_hi']}], equal weight. {tail}"
    if m["type"]=="learned":
        description=f"{m['harmonics']}-harmonic curve fitted on 1985–2004 date-only cross-sectional returns, using {m['phase']} expected periods."
    else:description=f"Cosine with peak {m['offset_sessions']} sessions into a 65-session cycle."
    weighting="Center scores within today's universe; allocate long/short dollars in proportion to positive/negative scores." if m["weighting"]=="score" else "Equal weight the top and bottom score terciles, including ties."
    return description+" "+weighting+" "+tail

def pct(x):return f"{x:.2%}"

def main():
    s=pd.read_csv(e.OUT/"performance.csv");a=pd.read_csv(e.OUT/"adaptive_performance.csv")
    choices=json.loads((e.OUT/"frozen_choices.json").read_text());models=json.loads((e.OUT/"final_models.json").read_text())
    screen=json.loads((e.OUT/"screen_audit.json").read_text());by={m["id"]:m for m in models}
    primary_ids=[x["model"] for x in choices["family_winners_10bps"]]
    low_ids=[x["model"] for x in choices["family_winners_1bp"]]
    primary=s.loc[s.model.isin(primary_ids)]
    rows=[]
    for ident in primary_ids:
        z=s.loc[s.model.eq(ident)]
        def value(period,scenario,col):return float(z.loc[z.period.eq(period)&z.scenario.eq(scenario),col].iloc[0])
        rows.append({"family":LABELS[by[ident]["family"]],"model":ident,
            "development 10bps":pct(value("development","10bps","cagr")),
            "validation 10bps":pct(value("validation","10bps","cagr")),
            "holdout gross":pct(value("holdout","gross","cagr")),
            "holdout 1bp":pct(value("holdout","1bp","cagr")),
            "holdout 10bps":pct(value("holdout","10bps","cagr")),
            "holdout Sharpe 10bps":round(value("holdout","10bps","sharpe"),2)})
    table=pd.DataFrame(rows);table.to_csv(e.OUT/"family_comparison.csv",index=False)
    low=[]
    for ident in low_ids:
        z=s.loc[s.model.eq(ident)&s.scenario.eq("1bp")]
        low.append(dict(family=LABELS[by[ident]["family"]],model=ident,
            development=pct(float(z.loc[z.period.eq("development"),"cagr"].iloc[0])),
            validation=pct(float(z.loc[z.period.eq("validation"),"cagr"].iloc[0])),
            holdout=pct(float(z.loc[z.period.eq("holdout"),"cagr"].iloc[0])),
            holdout_t=round(float(z.loc[z.period.eq("holdout"),"t"].iloc[0]),2)))
    low=pd.DataFrame(low);low.to_csv(e.OUT/"low_cost_comparison.csv",index=False)
    adapt_rows=[]
    for ident,z in a.groupby("model"):
        get=lambda p,c:float(z.loc[z.period.eq(p)&z.scenario.eq(c),"cagr"].iloc[0])
        adapt_rows.append(dict(model=ident,validation_1bp=pct(get("validation","1bp")),
            validation_10bps=pct(get("validation","10bps")),holdout_1bp=pct(get("holdout","1bp")),
            holdout_10bps=pct(get("holdout","10bps"))))
    adapt_table=pd.DataFrame(adapt_rows);adapt_table.to_csv(e.OUT/"adaptive_comparison.csv",index=False)
    champion=choices["overall_10bps"];cheap=choices["overall_1bp"]
    h=s.loc[s.model.eq(champion)&s.period.eq("holdout")&s.scenario.eq("10bps")].iloc[0]
    hcheap=s.loc[s.model.eq(cheap)&s.period.eq("holdout")&s.scenario.eq("1bp")].iloc[0]
    summary=(f"Screened {screen['ordered_window_pairs']:,} ordered window pairs, then exactly simulated "
       f"{screen['shortlisted_window_pairs']} window pairs and {screen['curve_models']} smooth-curve models across five cost scenarios. "
       "Also tested 25%/10% slower adjustment variants of development winners and eight annual-refit models. "
       "All portfolios are dollar-neutral at each daily rebalance. "
       f"The 10-bp development winner ({champion}) earns {h.cagr:.2%} CAGR in 2015–2025; "
       f"the independent 1-bp development winner ({cheap}) earns {hcheap.cagr:.2%} CAGR at 1 bp in that period. "
       "No frozen family winner or annual-refit variant establishes a reliable positive 2015–2025 return after costs.")
    (e.OUT/"summary.txt").write_text(summary+"\n")
    # Main figures use only models chosen on development, avoiding a later-period maximum.
    e.plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"font.size":9})
    fig,axes=e.plt.subplots(1,2,figsize=(14,4.8),layout="constrained")
    hrows=primary.loc[primary.period.eq("holdout")&primary.scenario.eq("10bps")].set_index("model").loc[primary_ids]
    x=np.arange(len(hrows));y=hrows.annual_mean.to_numpy()*100
    axes[0].errorbar(x,y,yerr=np.vstack([y-hrows.ci_low.to_numpy()*100,hrows.ci_high.to_numpy()*100-y]),
        fmt="o",capsize=4,color="#bb5849")
    shortlabels={"learned_seasonal":"Learned\nseasonal","learned_median":"Learned\nmedian",
        "nearest_seasonal":"Seasonal\nwindows","nearest_median":"Median\nwindows",
        "since_known":"Known-age\nwindows","fixed_periodic":"Fixed\ncycle"}
    axes[0].set_xticks(x,[shortlabels[by[i]["family"]] for i in primary_ids])
    axes[0].axhline(0,color="#666",lw=.8);axes[0].set_ylabel("Annual arithmetic mean return (%)")
    axes[0].set_title("2015–2025: 10 bps + 1% borrow; HAC20 95% intervals")
    with np.load(e.OUT/"final.npz") as z:cache={k:z[k] for k in z.files}
    dates=pd.to_datetime(cache["dates"]);mask=dates>="2015-01-01";j=next(k for k,m in enumerate(models) if m["id"]==champion)
    for ci in [0,1,2,3]:
        axes[1].plot(dates[mask],np.cumprod(1+cache["returns"][mask,ci,j]),label=e.COSTS[ci][0])
    axes[1].set_title("Development-selected smooth seasonal rule: cost sensitivity")
    axes[1].set_ylabel("Growth of $1; 200% gross target exposure");axes[1].legend()
    fig.savefig(e.OUT/"exploration_results.png",dpi=180);e.plt.close(fig)
    fig,ax=e.plt.subplots(figsize=(10,4),layout="constrained")
    phase=np.linspace(0,65,261);theta=phase/65*2*np.pi
    for family in ["learned_median","learned_seasonal","fixed_periodic"]:
        ident=next(i for i in primary_ids if by[i]["family"]==family);m=by[ident]
        if m["type"]=="fixed":
            values=np.cos(theta-2*np.pi*m["offset_sessions"]/65)
            values*=np.hypot(*by[champion]["coefficients"])
        else:
            values=sum(m["coefficients"][2*k]*np.sin((k+1)*theta)+m["coefficients"][2*k+1]*np.cos((k+1)*theta) for k in range(m["harmonics"]))
        ax.plot(phase,values*1e4,label=LABELS[family])
    ax.axhline(0,color="#666",lw=.8);ax.set_xlabel("Sessions since known release, scaled to a 65-session quarter")
    ax.set_ylabel("Fitted training score (bps/day)")
    ax.set_title("Date-only scoring curves frozen on 1985–2004\nFixed cosine scaled to the learned seasonal amplitude",fontsize=10);ax.legend(fontsize=8)
    fig.savefig(e.OUT/"timing_scores.png",dpi=180);e.plt.close(fig)
    rules="".join(f"<li><b>{LABELS[by[i]['family']]}</b> (<code>{i}</code>): {rule(by[i])}</li>" for i in primary_ids)
    signnote="Negative nearest clocks are before a forecast; positive clocks are after a known actual release. Forecast day zero is allowed in this search. Known-age clocks are always sessions since an actual release already available to the strategy. Forecasts and all regularity checks use past available releases only."
    phase_peak=np.arctan2(by[champion]["coefficients"][0],by[champion]["coefficients"][1])%(2*np.pi)*65/(2*np.pi) if by[champion]["type"]=="learned" and by[champion]["harmonics"]==1 else None
    peaktext=(f"Its first-harmonic score peaks about {phase_peak:.1f} sessions after a known release in a 65-session cycle, with a trough half a quarter later. Each day the score is centered across the eligible universe; positive scores receive long dollars and negative scores short dollars. " if phase_peak is not None else "")
    html=f"""<!doctype html><html><head><meta charset="utf-8"><title>Date-only long/short exploration</title>
    <style>body{{max-width:1350px;margin:40px auto;padding:0 25px;font:16px/1.55 system-ui;color:#25394a}}.note{{padding:18px;background:#edf4f4;border-left:4px solid #18816b}}img{{width:100%}}table{{border-collapse:collapse;font-size:12px;width:100%}}td,th{{border-bottom:1px solid #dde3e8;padding:8px;text-align:right}}td:first-child,th:first-child{{text-align:left}}code{{font-size:12px}}li{{margin:10px 0}}.table{{overflow:auto}}</style></head><body>
    <h1>Long/short rules using only announcement timing</h1><p class="note">{summary}</p>
    <p>Universe: at least five active annual-EPS analysts under the prior trailing-180-day proxy, and four known quarterly releases with the three past gaps each 50–80 market sessions and median 60–70. Latest-known release age must be no more than 80 sessions. Eligibility and timing are known before the return session under the existing conservative activation lag. No return momentum, earnings surprise, EPS level, valuation, analyst opinion or liquidity feature determines trade direction.</p>
    <p>Every portfolio targets $1 long and $1 short per $1 after-cost NAV, net holding zero and 200% gross exposure at each daily rebalance. Each leg must contain at least 20 names; otherwise both legs hold cash. Price moves cause drift before the next rebalance. Dollar neutrality does not guarantee beta or factor neutrality.</p>
    <h2>Frozen family winners: CAGR comparison</h2><p>Each row was chosen using development-period net Sharpe at 10 bps, before this family's later-period evaluation. Development = 1985–2004; validation = 2005–2014; holdout = 2015–2025. All entries are CAGR; 1 bp is 0.01% per dollar bought or sold, and 10 bps is 0.10%.</p>
    <div class="table">{table.drop(columns='model').to_html(index=False,border=0)}</div>
    <img src="exploration_results.png" alt="Frozen strategy means, uncertainty and cost sensitivity">
    <h2>What the development winner actually does</h2><p>{peaktext}{rule(by[champion])}</p>
    <p>At 10 bps, its holdout annual arithmetic mean is {h.annual_mean:.2%}, HAC20 95% interval {h.ci_low:.2%} to {h.ci_high:.2%}, Sharpe {h.sharpe:.2f}, annual dollar turnover {h.annual_turnover:.1f} times NAV. Its positive training performance did not persist. These are portfolio returns on capital, not returns per individual stock trade.</p>
    <img src="timing_scores.png" alt="Training-only timing score shapes">
    <h2>Exact rules for the six families</h2><p>{signnote}</p><ul>{rules}</ul>
    <p>Slow adjustment changes holding behavior: some names remain held after leaving their original signal window, while they remain in the eligible regular-release universe. It reduces trading but does not promise a fixed stock holding period. The smoothing is applied to the actual existing portfolio and both legs are then rescaled to equal dollars, with final net orders determining costs.</p>
    <h2>Separate low-cost selection objective</h2><p>These six rows are independently chosen by 1985–2004 Sharpe at 1 bp, rather than cherry-picked on later returns. They may be different models from the primary table. Costs include 1% annual borrow on the actual short leg.</p>
    <div class="table">{low.to_html(index=False,border=0)}</div>
    <p>The low-cost development champion uses {rule(by[cheap])} Its holdout annual-mean interval at 1 bp is {hcheap.ci_low:.2%} to {hcheap.ci_high:.2%}; t={hcheap.t:.2f}. It also fails to establish a reliable positive edge.</p>
    <h2>Adaptive annual refits</h2><p>After inspecting the frozen-rule results, eight additional variants were specified and all are reported here: median/seasonal cycle, expanding/trailing-10-year fit, and full/10% daily adjustment. Their first-harmonic coefficients are refitted at the start of each year using only prior complete calendar years. No evaluation year trains its own model. This is an exploratory extension; no variant is selected on these outcomes.</p>
    <div class="table">{adapt_table.to_html(index=False,border=0)}</div>
    <h2>How the search and costs were handled</h2><p>The coarse screen ranked all {screen['ordered_window_pairs']:,} ordered disjoint window pairs using separate-leg returns, joint active-day cash gates, turnover with price drift, and approximate fees. It deliberately omitted pair-specific NAV normalization. Only the shortlisted window pairs and smooth curves then received the exact cost-specific simulator. The report therefore identifies best models within this screened search, not a global mathematical optimum.</p>
    <p>Exact sizing solves x = 1 − c Σ|xw − v|. Each cost scenario has its own holdings and NAV. CRSP total returns determine P&amp;L; ex-dividend returns determine share-value drift and distributions remain cash. Buy/sell dollars on both legs are charged. Borrow is 1% annually for 1/5/10 bps and 2% for 25 bps; gross has no fees or borrow. No cash interest, financing, collateral rebates, measured market impact, stock-specific borrow availability, or terminal liquidation fee is included. Missing returns are carried at zero and counted.</p>
    <h2>Evidence limits and audit</h2><p>This is retrospective exploration after previous related later-period results were seen. Chronological selection limits direct fitting to later returns, but cannot erase researcher hindsight. The 7,650-pair search creates substantial selection risk. HAC20/60 and monthly HAC3 checks and Holm adjustments across the six frozen family winners are supplied; adaptive variants also have combined 14-rule adjustments. No small in-sample p-value is treated as trading evidence. Exact CUSIP linkage, database-vintage limitations and analyst-coverage-proxy limitations are inherited.</p>
    <p>Independent checks verify zero net targets, exact first-day after-cost sizing, cash exits, reproduction of every selected model's training prefix, training-only winner selection, raw-return two-day ledgers, and the fact that changing current returns cannot change signal targets. Annual refit logs verify strict training/evaluation separation.</p>
    <h2>Reproduce and inspect</h2><p>From this directory, use the existing Python environment to run <code>explore.py prepare</code>, <code>explore.py screen</code>, <code>explore.py train</code>, <code>explore.py evaluate</code>, <code>validate.py</code>, <code>adaptive.py</code>, and <code>report.py</code>. Exact frozen rules, coefficients, choices, screening scores, all daily/annual returns, uncertainty, checksums and validation records are in this results folder. See the directory README for the full command.</p>
    </body></html>"""
    (e.OUT/"report.html").write_text(html)
    e.save(e.OUT/"selected_rules.json",dict(primary=by[champion],low_cost=by[cheap],
        family_winners=[dict(by[i],readable_rule=rule(by[i])) for i in primary_ids],
        user_constraint="Equal long/short dollars at each daily rebalance, signals from release dates only."))
    print(summary)

if __name__=="__main__":main()
