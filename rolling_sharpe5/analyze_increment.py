"""Chronological factor-pool selection and EARC incremental-return analysis.

Run audit/extend_selection_audit.py and build_paper_factor.py first.
All returns in this script are decimal, monthly unless explicitly daily.
The incumbent library has no turnover: its returns remain gross throughout.
"""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LIBRARY = ROOT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
PERIODS = {"1990-2025": (1990, 2025), "2005-2025": (2005, 2025),
           "2015-2025": (2015, 2025), "2020-2025": (2020, 2025)}


def hac_mean(values, lag=3, annualization=12):
    x = np.asarray(values, dtype=float)
    assert np.isfinite(x).all(), "HAC must not compress missing calendar periods"
    if len(x) < 24:
        return {"n": len(x)}
    if np.std(x, ddof=1) == 0:
        return {"n": len(x), "annual_mean": x.mean() * annualization,
                "annual_vol": 0., "sharpe": np.nan, "t": np.nan, "p": np.nan,
                "ci_low": np.nan, "ci_high": np.nan,
                "cagr": (1 + x[0]) ** annualization - 1}
    m = sm.OLS(x, np.ones((len(x), 1)), missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
    ci = m.conf_int()[0] * annualization
    return {"n": len(x), "annual_mean": x.mean() * annualization,
            "annual_vol": np.std(x, ddof=1) * np.sqrt(annualization),
            "sharpe": x.mean() / np.std(x, ddof=1) * np.sqrt(annualization),
            "t": float(m.tvalues[0]), "p": float(m.pvalues[0]),
            "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "cagr": float(np.expm1(np.log1p(x).sum() * annualization / len(x)))
                    if np.min(x) > -1 else np.nan}


def monthly_compound(daily):
    assert np.isfinite(daily.to_numpy()).all(), "Missing daily returns must not become zero"
    return daily.add(1).groupby(daily.index.to_period("M")).prod().sub(1)


def validate_inputs(factors, counts, members, daily):
    assert factors.index.is_unique and factors.index.is_monotonic_increasing
    assert factors.index.max() == pd.Period("2025-12", freq="M")
    assert daily.groupby(["date", "scenario"]).size().eq(1).all()
    assert np.isfinite(daily[["returns", "before_fee", "turnover", "cost", "borrow"]].to_numpy()).all()
    assert (daily[["turnover", "cost", "borrow"]].to_numpy() >= -1e-12).all()
    assert np.allclose(daily.returns, daily.before_fee - daily.cost - daily.borrow,
                       atol=1e-11, rtol=1e-10)
    for scenario, fee in [("gross", 0.), ("5bps", .0005), ("5bps_1pct_borrow", .0005)]:
        x = daily[daily.scenario.eq(scenario)]
        assert len(x) > 10000, (scenario, len(x))
        assert np.allclose(x.cost, x.turnover * fee, atol=1e-11)
        if scenario != "5bps_1pct_borrow":
            assert np.allclose(x.borrow, 0)
    keys = ["target_year", "method", "test", "availability"]
    assert not counts.duplicated(keys).any()
    assert not members.duplicated(keys + ["factor"]).any()
    for row in counts.itertuples():
        assert row.window_end == f"{row.target_year-1}-12"
        assert row.window_start == f"{row.target_year-5}-01"
    pub = members[members.availability.eq("published_before_year")]
    assert pub.publication_year.lt(pub.target_year).all()


def make_pools(factors, counts, members, paper):
    rows, beta_rows = [], []
    # One-sided selection is retained in the audit, not promoted after viewing returns.
    use_counts = counts[counts.test.eq("two_sided")]
    member_groups = {k: sorted(g.factor.tolist()) for k, g in members.groupby(
        ["target_year", "method", "test", "availability"])}
    for cfg, group in use_counts.groupby(["method", "availability"], sort=True):
        method, availability = cfg
        for row in group.itertuples():
            year = row.target_year
            names = member_groups.get((year, method, "two_sided", availability), [])
            assert len(names) == row.factor_count
            idx = factors.index[factors.index.year == year]
            if year > 2025:
                continue
            f = factors.reindex(index=idx, columns=names)
            k = len(names)
            # Do not silently renormalize or fill an unavailable selected factor.
            base = f.sum(axis=1, min_count=k) / k if k else pd.Series(0., index=idx)
            train_index = pd.period_range(f"{year-5}-01", f"{year-1}-12", freq="M")
            train_f = factors.reindex(index=train_index, columns=names)
            assert not train_f.isna().any().any()
            for scenario in paper:
                y = paper[scenario].reindex(idx)
                combined = (base * k + y) / (k + 1)
                # Annual OLS hedge for strict small pools only; estimate beta with
                # an intercept on PREVIOUS 60 months. Do not subtract that intercept.
                residual = pd.Series(np.nan, index=idx)
                condition = np.nan
                train_y = paper[scenario].reindex(train_index)
                if k == 0:
                    residual = y.copy()
                elif method.startswith("each_year") and train_y.notna().all():
                    design = np.column_stack([np.ones(60), train_f.to_numpy()])
                    rank = np.linalg.matrix_rank(design)
                    condition = np.linalg.cond(design)
                    assert k <= 12 and rank == k + 1 and condition < 1e8
                    coef = np.linalg.lstsq(design, train_y.to_numpy(), rcond=None)[0]
                    residual = y - f.dot(coef[1:])
                    for name, b in zip(names, coef[1:]):
                        beta_rows.append({"target_year": year, "method": method,
                            "availability": availability, "scenario": scenario,
                            "factor": name, "beta": b, "train_intercept": coef[0],
                            "train_end": str(train_index.max()), "condition": condition})
                for month in idx:
                    rows.append({"month": str(month), "year": year, "method": method,
                        "availability": availability, "scenario": scenario,
                        "factor_count": k, "pool_gross": base.loc[month],
                        "paper_return": y.loc[month], "augmented_mixed_cost": combined.loc[month],
                        "increment": combined.loc[month] - base.loc[month],
                        "oos_spanning_residual": residual.loc[month],
                        "missing_selected_returns": int(f.loc[month].isna().sum())})
    pd.DataFrame(beta_rows).to_csv(OUT / "oos_betas.csv", index=False)
    return pd.DataFrame(rows)


def main():
    factor_path = LIBRARY / "Portfolios/Full Sets OP/PredictorLSretWide.parquet"
    factors = duckdb.read_parquet(str(factor_path)).df().set_index("date") / 100.
    factors.index = factors.index.to_period("M")
    counts = pd.read_csv(HERE / "audit/extended_selection_counts.csv")
    members = pd.read_csv(HERE / "audit/extended_selection_members.csv")
    daily = pd.read_csv(HERE / "paper_factor_daily.csv", parse_dates=["date"])
    daily = daily.sort_values(["date", "scenario"])
    validate_inputs(factors, counts, members, daily)
    paper = monthly_compound(daily.pivot(index="date", columns="scenario", values="returns"))
    paper.to_csv(OUT / "paper_monthly.csv", index_label="month")
    pnl = make_pools(factors, counts, members, paper)
    pnl.to_csv(OUT / "monthly_comparison.csv", index=False)
    counts.to_csv(OUT / "yearly_pool_counts.csv", index=False)
    members.to_csv(OUT / "yearly_pool_members.csv", index=False)
    members[members.target_year.eq(2026)].to_csv(OUT / "pool_2026.csv", index=False)

    statistics = []
    for cfg, g in pnl.groupby(["method", "availability", "scenario"], sort=True):
        for period, (start, end) in PERIODS.items():
            z = g[g.year.between(start, end)].sort_values("month")
            for series in ["pool_gross", "paper_return", "augmented_mixed_cost", "increment", "oos_spanning_residual"]:
                # Keep consecutive months: an interior missing value is not compressed
                # into its neighbor for a HAC calculation.
                if z[series].isna().any():
                    continue
                for lag in [3, 6, 12]:
                    statistics.append(dict(zip(["method", "availability", "scenario"], cfg)) |
                        {"period": period, "series": series, "hac_lags": lag,
                         "empty_pool_months": int(z.factor_count.eq(0).sum()),
                         "missing_months": int(z[series].isna().sum())} | hac_mean(z[series], lag))
    stats = pd.DataFrame(statistics)
    # Four declared, overlapping reporting periods. Holm within a fixed design
    # and outcome; this does not correct retrospective research across projects.
    stats["p_holm_4periods"] = np.nan
    family = ["method", "availability", "scenario", "series", "hac_lags"]
    for _, idx in stats.groupby(family).groups.items():
        good = stats.loc[idx, "p"].dropna()
        if len(good):
            stats.loc[good.index, "p_holm_4periods"] = multipletests(good, method="holm")[1]
    stats.to_csv(OUT / "inference.csv", index=False)

    # The pooled interpretation has missing option-factor returns during 2023.
    # Report the continuous 1990-2022 comparison separately, not a spliced full
    # sample, and do not retroactively remove the unavailable factor names.
    alt_rows = []
    for cfg, g in pnl[pnl.method.str.startswith("pooled60")].groupby(
            ["method", "availability", "scenario"], sort=True):
        z = g[g.year.between(1990, 2022)].sort_values("month")
        for series in ["pool_gross", "paper_return", "augmented_mixed_cost", "increment"]:
            if z[series].isna().any():
                continue
            for lag in [3, 6, 12]:
                alt_rows.append(dict(zip(["method", "availability", "scenario"], cfg)) |
                    {"period": "1990-2022 (complete contiguous alternative sample)",
                     "series": series, "hac_lags": lag} | hac_mean(z[series], lag))
    pd.DataFrame(alt_rows).to_csv(OUT / "pooled_interpretation_sensitivity.csv", index=False)

    # Official daily FF5+momentum controls are a separate diagnostic when a
    # strict factor pool is empty. The zero-net EARC spread is not reduced by RF.
    ff = pd.read_csv(ROOT.parent / "factors-daily/ff5_momentum_daily.csv", parse_dates=["date"]).set_index("date")
    risk_rows = []
    controls = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    for scenario, g in daily.groupby("scenario"):
        merged = g.set_index("date")[["returns"]].join(ff[controls], how="left")
        for period, (start, end) in PERIODS.items():
            x = merged[merged.index.year.to_series().between(start, end).to_numpy()]
            assert not x.isna().any().any(), "Incomplete FF control dates"
            for lag in [20, 60]:
                model = sm.OLS(x.returns, sm.add_constant(x[controls], has_constant="add"), missing="raise").fit(
                    cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
                ci = model.conf_int().loc["const"] * 252
                risk_rows.append({"scenario": scenario, "period": period, "hac_lags": lag,
                    "n_days": len(x), "annual_alpha": model.params["const"] * 252,
                    "t": model.tvalues["const"], "p": model.pvalues["const"],
                    "ci_low": ci.iloc[0], "ci_high": ci.iloc[1]})
    risks = pd.DataFrame(risk_rows)
    risks.to_csv(OUT / "ff6_alpha_daily.csv", index=False)

    # Ex-post alpha against the complete selected pool (not merely its EW
    # average). Every year's selected factors get separate slopes, with one
    # common intercept. Pool membership is predetermined; slopes are estimated
    # on evaluation returns for inference, NOT described as tradable OOS weights.
    dynamic_rows = []
    strict_members = members[members.method.str.startswith("each_year") & members.test.eq("two_sided")]
    all_months = paper.index[paper.index.year >= 1990]
    for (method, availability), chosen in strict_members.groupby(["method", "availability"]):
        controls_by_year = pd.DataFrame(index=all_months)
        for member in chosen.itertuples():
            if member.target_year > 2025:
                continue
            column = f"{member.target_year}:{member.factor}"
            controls_by_year[column] = 0.
            use = all_months[all_months.year == member.target_year]
            controls_by_year.loc[use, column] = factors.loc[use, member.factor]
        for scenario in paper:
            for period, (start, end) in PERIODS.items():
                idx = all_months[(all_months.year >= start) & (all_months.year <= end)]
                x = controls_by_year.loc[idx]
                x = x.loc[:, x.abs().sum().gt(0)]
                x = sm.add_constant(x, has_constant="add")
                assert np.linalg.matrix_rank(x.to_numpy()) == x.shape[1]
                for lag in [3, 6, 12]:
                    fit = sm.OLS(paper.loc[idx, scenario], x, missing="raise").fit(
                        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
                    ci = fit.conf_int().loc["const"] * 12
                    dynamic_rows.append({"method": method, "availability": availability,
                        "scenario": scenario, "period": period, "hac_lags": lag,
                        "n_months": len(idx), "slope_count": x.shape[1]-1,
                        "annual_alpha": fit.params["const"]*12,
                        "t": fit.tvalues["const"], "p": fit.pvalues["const"],
                        "ci_low": ci.iloc[0], "ci_high": ci.iloc[1]})
    dynamic = pd.DataFrame(dynamic_rows)
    dynamic.to_csv(OUT / "dynamic_pool_alpha.csv", index=False)

    annual = []
    for (scenario, year), z in daily.groupby(["scenario", daily.date.dt.year]):
        annual.append({"scenario": scenario, "year": year,
            "return": (1 + z.returns).prod() - 1,
            "annual_mean_daily": z.returns.mean() * 252,
            "turnover": z.turnover.sum(), "cost_sum": z.cost.sum(),
            "borrow_sum": z.borrow.sum(), "active_fraction": z.active.mean()})
    pd.DataFrame(annual).to_csv(OUT / "paper_annual.csv", index=False)

    primary = stats[(stats.method.eq("each_year_iid")) &
                    (stats.availability.eq("published_before_year")) &
                    (stats.scenario.eq("5bps")) & stats.hac_lags.eq(3)]
    primary.to_csv(OUT / "primary_results.csv", index=False)
    missing = pnl.groupby(["method", "availability", "scenario"]).agg(
        missing_months=("pool_gross", lambda x: int(x.isna().sum())),
        missing_factor_observations=("missing_selected_returns", "sum"))
    missing.to_csv(OUT / "missing_returns_audit.csv")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
    for method, label in [("each_year_iid", "Each of 5 years: IID"),
                          ("each_year_hac1", "Each of 5 years: HAC(1)"),
                          ("pooled60_hac3", "Pooled 60 months: HAC(3), alternative")]:
        c = counts[counts.method.eq(method) & counts.availability.eq("published_before_year") & counts.test.eq("two_sided")]
        axes[0].plot(c.target_year, c.factor_count, label=label)
    axes[0].set(ylabel="Selected factors", title="Annual pool membership; only information through prior December")
    axes[0].legend(fontsize=8)
    for scenario, label in [("gross", "EARC gross"), ("5bps", "EARC after 5bp"), ("5bps_1pct_borrow", "EARC after 5bp + 1% borrow")]:
        p = paper[scenario].loc["1990":]
        axes[1].plot(p.index.to_timestamp(), (1 + p).cumprod(), label=label)
    axes[1].set(ylabel="Wealth, initial = 1", title="Fixed earnings-cycle strategy; 200% target gross exposure")
    axes[1].legend(fontsize=8)
    a = primary[primary.series.eq("increment")].set_index("period").reindex(PERIODS)
    axes[2].errorbar(np.arange(len(a)), a.annual_mean * 100,
        yerr=np.vstack([(a.annual_mean-a.ci_low)*100, (a.ci_high-a.annual_mean)*100]), fmt="o", capsize=5)
    axes[2].set(xticks=np.arange(len(a)), xticklabels=a.index,
                ylabel="Annual increment (%)", title="Add EARC as one equal-weight factor: mean and HAC(3) 95% CI")
    axes[2].axhline(0, color="black", linewidth=.8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)

    sources = [factor_path, LIBRARY / "SignalDoc.csv", HERE / "paper_factor_daily.csv",
               HERE / "build_paper_factor.py", Path(__file__), HERE / "audit/extend_selection_audit.py"]
    manifest = {str(p.relative_to(ROOT.parent)): {"bytes": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources}
    summary = {
        "primary_selection": "positive mean and two-sided IID t p<0.05 in EACH of prior 5 complete calendar years; 12 months each; publication year < target year",
        "robustness_selection": "each-year HAC1; one-sided tests in audit; pooled60 IID/HAC3 explicitly alternative interpretation",
        "annual_selection_multiplicity": "per-factor p<0.05, not a 212-factor family-wise/FDR claim",
        "main_test": "equal weight existing factors + one EARC; paired monthly incremental mean; HAC3 with6/12 sensitivity; Holm across4 periods",
        "factor_costs": "EARC exact one-way5bp; incumbent factor library GROSS because holdings/turnover unavailable",
        "spanning": "strict rules only: betas fit prior60months including intercept; test mean of y_current-F_current*beta_prior; no intercept subtraction; hedge implementation costs unavailable",
        "dynamic_pool_alpha": "separate ex-post regression: common alpha and each selected factor times target-year indicator; annual slopes fitted on evaluation sample, not implementable OOS weights",
        "empty_pool": "zero-return cash; retained, never deleted; incremental return then equals EARC net return",
        "current_year": "2026 pool available; no2026 returns in sources, cannot test2026 performance",
        "paper_net_return_convention": "zero-net spread on1unit capital; no RF subtracted; financing/collateral excluded",
        "inference_scope": "retrospective chronological evaluation; generated-beta HAC is a diagnostic conditional on fitted rule, not full discovery-adjusted Sharpe-spanning inference",
        "sources": manifest,
        "software": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "statsmodels": sm.__version__},
        "verified": {"source_months_unique": True, "prior_year_only_windows": True,
                     "publication_dates_prior": True, "cost_identity": True,
                     "accounting_identity": True, "no_missing_factor_imputation": True},
    }
    (OUT / "design_and_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(primary[primary.series.isin(["paper_return", "increment", "oos_spanning_residual"])].to_string(index=False))
    print("2026 counts:\n", counts[counts.target_year.eq(2026) & counts.test.eq("two_sided")].to_string(index=False))
    print("FF6 alpha:\n", risks[risks.scenario.eq("5bps") & risks.hac_lags.eq(20)].to_string(index=False))
    print("Missing selected factor observations:\n", missing.to_string())


if __name__ == "__main__":
    main()
