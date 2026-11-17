#!/usr/bin/env python3
"""Independent zero-cost reproduction of existing announcement-only rules.

Run from the study root: .venv/bin/python gross_reproduction/reproduce.py
Reads historical event states and eligibility from the existing validated study,
but rebuilds median clocks and daily portfolios, using raw CRSP total returns.
No optimization, transaction fees, short borrow, or new signal fitting.
"""
from pathlib import Path
import hashlib
import html
import json
import os

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl-cache"))
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL = ROOT / "coverage5/results_regular65/panel"
RAW = ROOT.parent / "crsp-daily/data/crsp_a_stock/dsf_v2"
EXPL = ROOT / "coverage5/exploration/results"
REGULAR = ROOT / "coverage5/results_regular65"
PERIODS = {
    "development": ("1985-01-01", "2004-12-31"),
    "validation": ("2005-01-01", "2014-12-31"),
    "holdout": ("2015-01-01", "2025-12-31"),
    "post2004": ("2005-01-01", "2025-12-31"),
    "recent": ("2020-01-01", "2025-12-31"),
    "all": ("1985-01-01", "2025-12-31"),
}
# These are existing rules, inspected before this reproduction. All five are
# reported, including the original overall training winner, W_103_74.
RULES = {
    "W_37_3": ("Median forecast windows", "median_clock_new", -5, 14, -35, -16),
    "W_103_74": ("Seasonal forecast windows", "seasonal_clock_new", -5, 14, -30, -11),
    "nearest_selected": ("Original narrow windows", "median_clock_new", -5, -1, -40, -21),
    "known_cycle": ("Known-announcement cycle", "age_new", 6, 20, 36, 50),
    "fixed_cos_10_score": ("Fixed quarterly cosine", "cosine", None, None, None, None),
}


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stats(d):
    rows = []
    for model, g in d.groupby("model", sort=False):
        for period, (start, end) in PERIODS.items():
            z = g.loc[g.date.between(start, end)].sort_values("date")
            r = z.returns.to_numpy()
            assert len(r) and np.isfinite(r).all() and (r > -1).all()
            row = dict(model=model, period=period, start=str(z.date.min().date()),
                       end=str(z.date.max().date()), days=len(r), daily_bp=r.mean()*1e4,
                       daily_bp_per_gross_dollar=r.mean()*5e3,
                       annual_arithmetic=r.mean()*252,
                       cagr=np.expm1(np.log1p(r).sum()*252/len(r)),
                       annual_vol=r.std(ddof=1)*np.sqrt(252),
                       active_fraction=float(z.active.mean()),
                       active_day_bp=float(z.loc[z.active, "returns"].mean()*1e4))
            row["sharpe_zero_cash"] = row["annual_arithmetic"]/row["annual_vol"]
            for lag in (20, 60):
                fit = sm.OLS(r, np.ones((len(r), 1)), missing="raise").fit(
                    cov_type="HAC", cov_kwds={"maxlags": lag})
                low, high = fit.conf_int()[0]*1e4
                row.update({f"ci_low_bp_hac{lag}": low, f"ci_high_bp_hac{lag}": high,
                            f"t_hac{lag}": fit.tvalues[0], f"p_hac{lag}": fit.pvalues[0]})
            rows.append(row)
    result = pd.DataFrame(rows)
    for _, idx in result.groupby("period").groups.items():
        for lag in (20, 60):
            result.loc[idx, f"p_holm_reported_hac{lag}"] = multipletests(
                result.loc[idx, f"p_hac{lag}"], method="holm")[1]
    return result


def rebuild():
    OUT.mkdir(parents=True, exist_ok=True)
    design = dict(
        purpose="Reproduce the 2-4 bp/day magnitude at zero costs, using existing timing-only rules.",
        rules=RULES, periods=PERIODS, transaction_cost=0, short_borrow=0,
        exposure="$1 long and $1 short per $1 NAV; 200% gross; rebalance daily; >=20 names per side.",
        mean="Arithmetic daily return across ALL market sessions, including cash days; 1 bp=0.0001.",
        universe="Existing regular-65-session, >=5 active analyst proxy, lagged eligible US common stocks.",
        timing="Past available announcements only; median of last three session gaps predicts next release.",
        selection="No new optimization. W_37_3 was an earlier training-selected median-family winner at 1 bp. "
                  "It is highlighted here after inspecting existing gross returns. W_103_74 was the overall "
                  "1-bp winner and has the highest training gross Sharpe among the prior exact shortlist. "
                  "Report both and three fixed/previous comparators. This is retrospective reproduction.",
        missing="Missing raw returns count as zero; no cash interest, financing or rebates.",
        inherited_limits="Exact-CUSIP retrospective links, incomplete historical data vintages, daily close execution.",
    )
    save(OUT / "design.json", design)
    c = duckdb.connect()
    c.execute("SET threads=2")
    c.execute("SET memory_limit='1400MB'")
    c.execute(f"SET temp_directory='{OUT}/duckdb_tmp'")
    # Reconstruct the historical median directly from event states, independently
    # of the saved stock-day clock. Lagging is by information-availability order.
    c.execute(f"""CREATE TEMP TABLE hist AS
      WITH gaps AS (
        SELECT permno, known_idx, event_idx,
          event_idx-lag(event_idx) OVER w AS a,
          lag(event_idx) OVER w-lag(event_idx,2) OVER w AS b,
          lag(event_idx,2) OVER w-lag(event_idx,3) OVER w AS c
        FROM read_parquet('{ROOT}/results/linked_events.parquet')
        WINDOW w AS (PARTITION BY permno ORDER BY known_idx))
      SELECT *,list_median([a,b,c]) AS period_new FROM gaps""")
    calendar = pd.read_csv(ROOT / "results/trading_calendar.csv", parse_dates=["date"])
    calendar = calendar.loc[calendar.date.between("1985-01-01", "2025-12-31"), ["date"]]
    expected_sha = {Path(x["path"]).name: x["sha256"]
                    for x in json.loads((EXPL / "input_audit.json").read_text())["files"]}
    annual_audits, pieces = [], []
    for year in range(1985, 2026):
        panel = PANEL / f"{year}.parquet"
        checksum = sha(panel)
        assert checksum == expected_sha[panel.name], (year, "source panel changed")
        c.execute(f"""CREATE OR REPLACE TEMP TABLE stock AS
          WITH states AS (
            SELECT p.*,h.known_idx AS known_new,h.event_idx AS event_new,
              h.a,h.b,h.c,h.period_new, p.idx-h.event_idx AS age_new,
              h.event_idx+h.period_new-p.idx AS to_forecast
            FROM (SELECT * FROM read_parquet('{panel}') WHERE eligible) p
            ASOF LEFT JOIN hist h ON p.permno=h.permno AND p.idx>h.known_idx),
          clocks AS (
            SELECT *,CASE WHEN to_forecast>=0 AND to_forecast<age_new
                      THEN -to_forecast ELSE age_new END AS median_clock_new,
              CASE WHEN seasonal_forecast_idx-idx>=0 AND seasonal_forecast_idx-idx<age_new
                THEN idx-seasonal_forecast_idx ELSE age_new END AS seasonal_clock_new,
              cos(2*pi()*age_new/period_new-2*pi()*10/65) AS cosine_raw
            FROM states),
          returns AS (
            SELECT p.*,r.permno AS raw_id,cast(r.dlyret AS DOUBLE) AS raw_ret,
              coalesce(cast(r.dlyret AS DOUBLE),0) AS r
            FROM clocks p LEFT JOIN read_parquet('{RAW}/{year}.parquet') r
              ON p.permno=r.permno AND p.date=r.dlycaldt)
          SELECT *,cosine_raw-avg(cosine_raw) OVER (PARTITION BY date) AS cosine
          FROM returns""")
        a = c.execute("""SELECT count(*) AS stockdays,
          count(*)-count(DISTINCT (permno,date)) AS duplicates,
          count(*) FILTER(WHERE raw_id IS NULL) AS unmatched_raw,
          count(*) FILTER(WHERE ret IS DISTINCT FROM raw_ret) AS raw_return_mismatch,
          count(*) FILTER(WHERE known_new IS NULL OR known_new>=idx OR known_new<>known_idx
            OR event_new<>event_idx OR a<>g1 OR b<>g2 OR c<>g3
            OR period_new<>period_sessions OR period_new NOT BETWEEN 60 AND 70
            OR a NOT BETWEEN 50 AND 80 OR b NOT BETWEEN 50 AND 80 OR c NOT BETWEEN 50 AND 80
            OR age_new<>age OR age_new NOT BETWEEN 0 AND 80 OR n180<5
            OR lag_date>=date OR clock<>median_clock_new OR seasonal_clock<>seasonal_clock_new
            OR NOT isfinite(r) OR r< -1) AS chronology_or_data_error,
          count(*) FILTER(WHERE raw_ret IS NULL) AS missing_returns
          FROM stock""").df().iloc[0].to_dict()
        a = {k:int(v) for k,v in a.items()}
        for key in ["duplicates", "unmatched_raw", "raw_return_mismatch", "chronology_or_data_error"]:
            assert a[key] == 0, (year, key, a[key])
        annual_audits.append(dict(year=year, panel_sha256=checksum, **a))
        for model, (_, clock, lo, hi, slo, shi) in RULES.items():
            if clock == "cosine":
                lm, ss = "cosine>1e-16", "cosine< -1e-16"
                lr = f"sum(cosine*r) FILTER(WHERE {lm})/nullif(sum(cosine) FILTER(WHERE {lm}),0)"
                sr = f"sum(cosine*r) FILTER(WHERE {ss})/nullif(sum(cosine) FILTER(WHERE {ss}),0)"
            else:
                lm, ss = f"{clock} BETWEEN {lo} AND {hi}", f"{clock} BETWEEN {slo} AND {shi}"
                lr, sr = f"avg(r) FILTER(WHERE {lm})", f"avg(r) FILTER(WHERE {ss})"
            g = c.execute(f"""SELECT date,count(*) FILTER(WHERE {lm}) AS long_candidates,
              count(*) FILTER(WHERE {ss}) AS short_candidates,{lr} AS long_return,
              {sr} AS short_return,
              count(*) FILTER(WHERE raw_ret IS NULL AND (({lm}) OR ({ss}))) AS missing_candidates
              FROM stock GROUP BY date ORDER BY date""").df()
            g = calendar.loc[calendar.date.dt.year.eq(year)].merge(g, how="left", on="date", validate="one_to_one")
            g[["long_candidates", "short_candidates", "missing_candidates"]] = g[
                ["long_candidates", "short_candidates", "missing_candidates"]].fillna(0).astype(int)
            g["active"] = (g.long_candidates>=20)&(g.short_candidates>=20)
            g["returns"] = np.where(g.active, g.long_return-g.short_return, 0.)
            assert g.returns.notna().all()
            g.loc[~g.active, ["long_return", "short_return"]] = 0.
            g["model"] = model
            g["transaction_cost"] = 0.
            g["short_borrow"] = 0.
            pieces.append(g)
        print(f"Rebuilt {year}: {a['stockdays']:,} eligible stock-days; raw-return and chronology checks passed", flush=True)
    d = pd.concat(pieces, ignore_index=True).sort_values(["model", "date"])
    assert not d.duplicated(["model", "date"]).any()
    assert d.groupby("model").size().eq(len(calendar)).all()
    d.to_csv(OUT / "daily_returns.csv", index=False)
    # Compare every reconstructed day with the prior simulator, not just its means.
    old_expl = pd.read_csv(EXPL / "daily_returns.csv", parse_dates=["date"])
    old_reg = pd.read_csv(REGULAR / "daily_returns.csv", parse_dates=["date"]).rename(columns={"strategy":"model"})
    old = pd.concat([old_expl[old_expl.model.isin(["W_37_3", "W_103_74", "fixed_cos_10_score"])],
                     old_reg[old_reg.model.isin(["nearest_selected", "known_cycle"])]], ignore_index=True)
    old = old.loc[old.scenario.eq("gross")]
    assert old.cost.eq(0).all() and old.borrow.eq(0).all()
    z = d.merge(old[["date", "model", "returns", "active"]], on=["date", "model"],
                validate="one_to_one", suffixes=("_new", "_old"))
    assert len(z)==len(d)
    error = float((z.returns_new-z.returns_old).abs().max())
    assert error<1e-12 and (z.active_new==z.active_old).all()
    checks = dict(years=annual_audits, compared_daily_portfolios=len(z), max_abs_daily_return_difference=error,
                  all_sessions_per_model=len(calendar), no_new_fit_or_window_search=True,
                  raw_total_returns_used=True, median_clocks_rebuilt_from_past_events=True,
                  seasonal_forecast_inherited_from_validated_panel=True,
                  inherited_analyst_coverage_and_security_linkage_audit=str(REGULAR / "validation_checks.json"))
    save(OUT / "validation.json", checks)
    perf = stats(d)
    perf.to_csv(OUT / "performance.csv", index=False)
    # Show multiplicity over the entire previously reported frozen final set.
    all_prior = old_expl.loc[old_expl.scenario.eq("gross")].copy()
    all_prior["active"] = all_prior.active.astype(bool)
    full = stats(all_prior)
    full.to_csv(OUT / "all_11_prior_models_gross.csv", index=False)
    return d, perf, checks


def report(d, perf, checks):
    labels = {k:v[0] for k,v in RULES.items()}
    main = perf[perf.model.eq("W_37_3")].set_index("period")
    h, v, p, a = [main.loc[x] for x in ["holdout", "validation", "post2004", "all"]]
    summary = (
        f"The 2-4 bp/day gross magnitude is reproducible with an existing median-forecast timing rule (W_37_3): "
        f"{v.daily_bp:.3f} bp/day in 2005-2014, {h.daily_bp:.3f} in 2015-2025, "
        f"{p.daily_bp:.3f} in 2005-2025, and {a.daily_bp:.3f} in 1985-2025. "
        f"2015-2025 arithmetic annual mean {h.annual_arithmetic:.2%}; CAGR {h.cagr:.2%}. "
        f"HAC20 95% daily-mean interval [{h.ci_low_bp_hac20:.3f}, {h.ci_high_bp_hac20:.3f}] bp, "
        f"p={h.p_hac20:.4f}. This reproduces a historical point estimate, not a statistically established "
        "2-4 bp/day recent edge. Returns use $1 long and $1 short per $1 NAV (200% gross).\n")
    (OUT / "summary.txt").write_text(summary)
    order = list(RULES)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True, layout="constrained")
    for ax, period, title in zip(axes, ["validation", "holdout"], ["2005–2014", "2015–2025"]):
        z = perf.loc[perf.period.eq(period)].set_index("model").loc[order]
        yy = np.arange(len(order))
        ax.axvspan(2, 4, color="#d9e9e5", label="Target: 2–4 bp/day")
        ax.axvline(0, color="#939b9f", lw=1)
        ax.errorbar(z.daily_bp, yy, xerr=np.array([z.daily_bp-z.ci_low_bp_hac20,
                    z.ci_high_bp_hac20-z.daily_bp]), fmt="o", color="#087b72", capsize=4)
        ax.set_title(title)
        ax.set_xlabel("Gross daily mean (bp per $1 NAV)")
        ax.set_xlim(-2, 5.5)
        ax.grid(axis="x", alpha=.15)
    axes[0].set_yticks(np.arange(len(order)), [labels[m] for m in order])
    axes[0].invert_yaxis()
    fig.suptitle("Announcement timing alone, zero costs | points and 95% HAC20 intervals")
    fig.savefig(OUT / "gross_daily_returns.png", dpi=180)
    plt.close(fig)
    table = perf.pivot(index="model", columns="period", values="daily_bp").loc[order,
              ["development", "validation", "holdout", "post2004", "recent", "all"]]
    table.index = [labels[x] for x in table.index]
    detail = main[["daily_bp", "annual_arithmetic", "cagr", "ci_low_bp_hac20", "ci_high_bp_hac20",
                   "p_hac20", "p_hac60", "active_fraction", "daily_bp_per_gross_dollar"]]
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
      <title>Zero-cost announcement-timing reproduction</title>
      <style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#203844}}
      table{{border-collapse:collapse;font-size:13px;width:100%}}td,th{{padding:8px;border-bottom:1px solid #dbe4e6;text-align:right}}
      td:first-child,th:first-child{{text-align:left}}img{{width:100%}}.note{{background:#ecf4f1;padding:18px}}code{{font-size:14px}}</style>
      <h1>Can announcement timing alone produce 2–4 bp/day before costs?</h1>
      <p class="note">{html.escape(summary)}</p>
      <h2>The reproducible rule</h2>
      <p>Keep the existing universe of US common stocks with at least five active annual-EPS analysts and regular
      earnings history. Require the last three known announcement gaps to be 50–80 trading sessions each,
      with a median of 60–70, and the latest known release no more than 80 sessions old.
      Forecast the next release as the latest release plus the median of those three gaps.</p>
      <p>At each previous close, choose the nearer of the latest known release and a nonexpired forecast (ties
      choose the past release). The clock is positive after a known release and negative before a forecast;
      a forecast date is zero. Buy clock −5 through +14; short clock −35 through −16. Equal weight each side,
      $1 long and $1 short per $1 NAV, rebalance daily. Hold cash if either side has fewer than 20 names.
      The positive-clock part uses a known actual release, not the days following an expired forecast.</p>
      <p>Trade direction uses announcement timing alone. Analyst counts define eligibility. Earnings values,
      surprises, return momentum and external factor scores do not enter the signal. Window endpoints came
      from earlier training on 1985–2004; no endpoints were optimized in this reproduction.</p>
      <h2>Gross arithmetic daily means, including cash days</h2>{table.round(3).to_html(border=0)}
      <p>1 bp = 0.01%; 2–4 bp/day corresponds to 5.04–10.08% arithmetic annual return at 252 sessions.
      Every figure above sets both transaction fees and short borrow to zero. Cash earns zero.
      At 100% total gross exposure ($0.50 long/$0.50 short per $1 NAV), these daily means halve:
      the median-window holdout becomes {h.daily_bp/2:.3f} bp/day.</p>
      <img src="gross_daily_returns.png" alt="Five existing timing rules with 95 percent HAC intervals; the 2 to 4 basis point target is shaded">
      <h2>Uncertainty and selection</h2>
      <p>The median-window rule was a training-selected winner within its family at 1 bp cost; this report
      highlights it after seeing prior gross results. It was not the overall training winner. That winner,
      the seasonal-window rule, has a weaker later-period gross mean and is shown alongside it.
      The original narrow-window rule also remains below 2 bp/day in both later decades. Therefore the
      evidence supports reproduction for one existing specification, not every announcement-only rule.</p>
      <p>The 2015–2025 median-window confidence interval includes zero even before adjustment for examining
      alternatives. HAC60 sensitivity and Holm adjustment across these five rules are in performance.csv;
      all_11_prior_models_gross.csv covers the entire prior frozen final set. Neither adjustment removes
      the historical strategy-search and researcher hindsight. The full sample includes training years;
      the chronological later periods are retrospective, not untouched prospective evidence.</p>
      <div style="overflow-x:auto">{detail.round(5).to_html(border=0)}</div>
      <h2>Independent reproduction checks</h2>
      <p>Rebuilt {sum(x['stockdays'] for x in checks['years']):,} eligible stock-days from historical event states
      and raw CRSP total returns. Checked each year for duplicate securities/dates, raw-return mismatches,
      event availability, historical gap calculations, median clocks, regularity and coverage.
      All checks passed. All {checks['compared_daily_portfolios']:,} daily portfolio returns across five rules
      match the saved simulations within {checks['max_abs_daily_return_difference']:.2g} in decimal return.
      Source-panel SHA-256 checksums match the prior input audit. Missing returns are counted and carried
      as zero, consistent with the original execution convention.</p>
      <p>This rerun reuses previously validated analyst coverage and historical security linkage;
      it does not rebuild raw analyst feeds. Seasonal forecast dates are inherited from the validated panel.
      Historical-vintage and retrospective CUSIP-linkage limitations remain. Returns are historical
      self-financing portfolio returns, not a standard multifactor alpha estimate.</p>
      <h2>Files and reproduction</h2>
      <p>Run <code>.venv/bin/python gross_reproduction/reproduce.py</code> from the study root.
      Outputs: <a href="daily_returns.csv">daily returns</a>, <a href="performance.csv">performance and uncertainty</a>,
      <a href="all_11_prior_models_gross.csv">all prior models</a>, <a href="validation.json">audit</a>,
      <a href="design.json">definitions</a>, and <a href="summary.txt">summary</a>.</p>
      <p>Software/workflow: local pinned DuckDB, pandas, NumPy, statsmodels and Matplotlib; statsmodels skill
      from Scientific Agent Skills. Reference: Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026).
      <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>.
      Metadata checked against the latest arXiv record on 2026-11-02. This is a workflow reference, not evidence
      for the earnings strategy.</p></html>"""
    (OUT / "report.html").write_text(page)
    print(summary, flush=True)


if __name__ == "__main__":
    report(*rebuild())
