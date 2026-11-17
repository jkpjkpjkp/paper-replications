"""Minimum-ten annual pools: paired return, Sharpe, and out-of-sample tests."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(PARENT))
from analyze_increment import hac_mean, monthly_compound
from optimization import optimize_weights

PERIODS = {"1990-2025": (1990, 2025), "2005-2025": (2005, 2025),
           "2015-2025": (2015, 2025), "2020-2025": (2020, 2025)}


def sharpe(r):
    r = np.asarray(r, dtype=float)
    return float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(12))


def paired_sharpe(a, b, lag=3, bootstrap_n=0):
    """HAC delta-method and paired circular block bootstrap, conditional on paths.

    IF(SR)=sqrt(12)*[(r-mu)/sigma-mu*((r-mu)^2-sigma^2)/(2*sigma^3)].
    Use population moments for this asymptotic linearization, then multiply
    by sqrt((n-1)/n) so both point estimate and influence use sample SD.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and len(a) >= 24
    assert np.isfinite(a).all() and np.isfinite(b).all()
    n = len(a)

    def influence(x):
        mu = x.mean()
        u = x - mu
        sigma = x.std(ddof=0)
        assert sigma > 0
        return np.sqrt(12) * (u/sigma - mu*(u*u-sigma*sigma)/(2*sigma**3)) * np.sqrt((n-1)/n)

    d = sharpe(a) - sharpe(b)
    v = influence(a) - influence(b)
    model = sm.OLS(v, np.ones((n, 1)), missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
    se = float(model.bse[0])
    if abs(d) < 1e-12 and se < 1e-12:
        # Identical or positively scaled paths have exactly equal Sharpe;
        # floating-point residuals must not produce a spurious small p-value.
        d, se = 0., 0.
    crit = student_t.ppf(.975, n-1)
    result = {"base_sharpe": sharpe(b), "augmented_sharpe": sharpe(a),
        "sharpe_difference": d, "sharpe_delta_se": se,
        "sharpe_delta_p": float(2*student_t.sf(abs(d/se), n-1)) if se > 0 else 1.,
        "sharpe_delta_ci_low": d-crit*se, "sharpe_delta_ci_high": d+crit*se}
    if bootstrap_n:
        rng = np.random.default_rng(491725)
        block_length = 12
        starts = rng.integers(n, size=(bootstrap_n, int(np.ceil(n/block_length))))
        idx = ((starts[:, :, None] + np.arange(block_length)) % n).reshape(bootstrap_n, -1)[:, :n]
        aa, bb = a[idx], b[idx]
        ds = np.sqrt(12)*(aa.mean(axis=1)/aa.std(axis=1, ddof=1) - bb.mean(axis=1)/bb.std(axis=1, ddof=1))
        result.update({"bootstrap_draws": bootstrap_n, "bootstrap_block_months": block_length,
            "sharpe_boot_ci_low": float(np.quantile(ds, .025)),
            "sharpe_boot_ci_high": float(np.quantile(ds, .975)),
            "bootstrap_scope": "paired fixed realized strategy paths; selection/weights not refitted"})
    return result


def max_drawdown(x):
    wealth = np.r_[1., np.cumprod(1+np.asarray(x))]
    return float(np.min(wealth/np.maximum.accumulate(wealth)-1))


def make_paths(factors, paper, counts, members):
    paths, weights, betas, fitting = [], [], [], []
    groups = {k: g.sort_values("selection_rank") for k, g in members.groupby(["target_year", "method", "availability"])}
    for row in counts.itertuples():
        if row.target_year > 2025:
            continue
        key = (row.target_year, row.method, row.availability)
        names = groups[key].factor.tolist()
        k = len(names)
        assert k >= 10 and k == row.factor_count
        train_idx = pd.period_range(f"{row.target_year-5}-01", f"{row.target_year-1}-12", freq="M")
        test_idx = pd.period_range(f"{row.target_year}-01", f"{row.target_year}-12", freq="M")
        train_f = factors.loc[train_idx, names]
        test_f = factors.loc[test_idx, names]
        assert np.isfinite(train_f.to_numpy()).all()
        base_ew = test_f.sum(axis=1, min_count=k)/k
        w_base, diag_base = optimize_weights(train_f.to_numpy())
        base_opt = test_f.dot(w_base)
        common = {"target_year": row.target_year, "method": row.method, "availability": row.availability}
        fitting.append(common | {"scenario": "incumbent_gross", "model": "baseline"} | diag_base)
        for f, w in zip(names, w_base):
            weights.append(common | {"scenario": "incumbent_gross", "model": "baseline", "factor": f, "weight": w})
        for scenario in paper:
            train_y = paper.loc[train_idx, scenario]
            test_y = paper.loc[test_idx, scenario]
            aug_ew = (k*base_ew+test_y)/(k+1)
            train_all = np.column_stack([train_f.to_numpy(), train_y.to_numpy()])
            w_aug, diag_aug = optimize_weights(train_all)
            if w_aug[-1] <= 1e-8:
                # With zero EARC weight the incumbent covariance block/objective
                # is unchanged. Reuse its solution to avoid solver noise being
                # mislabeled as a gain from adding an unused asset.
                w_aug = np.r_[w_base, 0.]
                diag_aug["canonicalized_zero_earc"] = True
            else:
                diag_aug["canonicalized_zero_earc"] = False
            aug_opt = test_f.dot(w_aug[:-1])+test_y*w_aug[-1]
            fitting.append(common | {"scenario": scenario, "model": "augmented"} | diag_aug)
            for f, w in zip(names+["EARC"], w_aug):
                weights.append(common | {"scenario": scenario, "model": "augmented", "factor": f, "weight": w})
            x = np.column_stack([np.ones(len(train_idx)), train_f.to_numpy()])
            assert np.linalg.matrix_rank(x) == x.shape[1]
            coef = np.linalg.lstsq(x, train_y.to_numpy(), rcond=None)[0]
            residual = test_y-test_f.dot(coef[1:])
            for f, beta in zip(names, coef[1:]):
                betas.append(common | {"scenario": scenario, "factor": f, "beta": beta,
                    "train_intercept": coef[0], "train_end": str(train_idx.max()), "condition_number": np.linalg.cond(x)})
            for i, month in enumerate(test_idx):
                paths.append({"month": str(month), "year": row.target_year,
                    "method": row.method, "availability": row.availability, "scenario": scenario,
                    "factor_count": k, "base_ew": base_ew.iloc[i], "augmented_ew": aug_ew.iloc[i],
                    "delta_ew": aug_ew.iloc[i]-base_ew.iloc[i],
                    "base_optimized": base_opt.iloc[i], "augmented_optimized": aug_opt.iloc[i],
                    "delta_optimized": aug_opt.iloc[i]-base_opt.iloc[i],
                    "paper": test_y.iloc[i], "oos_residual": residual.iloc[i],
                    "earc_weight_optimized": w_aug[-1],
                    "missing_factor_returns": int(test_f.iloc[i].isna().sum())})
    return pd.DataFrame(paths), pd.DataFrame(weights), pd.DataFrame(betas), pd.DataFrame(fitting)


def main():
    factor_path = ROOT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10/Portfolios/Full Sets OP/PredictorLSretWide.parquet"
    factors = duckdb.read_parquet(str(factor_path)).df().set_index("date")/100
    factors.index = factors.index.to_period("M")
    daily = pd.read_csv(PARENT / "paper_factor_daily.csv", parse_dates=["date"])
    assert np.allclose(daily.returns, daily.before_fee-daily.cost-daily.borrow)
    paper = monthly_compound(daily.pivot(index="date", columns="scenario", values="returns"))
    counts = pd.read_csv(OUT / "selection_counts.csv")
    members = pd.read_csv(OUT / "selection_members.csv")
    paths, weights, betas, fitting = make_paths(factors, paper, counts, members)
    paths.to_csv(OUT / "monthly_paths.csv", index=False)
    weights.to_csv(OUT / "annual_weights.csv", index=False)
    betas.to_csv(OUT / "annual_oos_betas.csv", index=False)
    fitting.to_csv(OUT / "optimization_diagnostics.csv", index=False)

    summaries, alpha_rows, missing = [], [], []
    for cfg, group in paths.groupby(["method", "availability", "scenario"]):
        info = dict(zip(["method", "availability", "scenario"], cfg))
        period_set = PERIODS.copy()
        if group.missing_factor_returns.gt(0).any():
            first_gap_year = int(group.loc[group.missing_factor_returns.gt(0), "year"].min())
            for start in [1990, 2005, 2015, 2020]:
                if first_gap_year-start >= 2:
                    period_set[f"{start}-{first_gap_year-1}_complete_prefix"] = (start, first_gap_year-1)
        for period, (start, end) in period_set.items():
            z = group[group.year.between(start, end)].sort_values("month")
            assert len(z) == (end-start+1)*12
            if z.missing_factor_returns.gt(0).any():
                missing.append(info | {"period": period, "missing_months": int(z.missing_factor_returns.gt(0).sum()),
                    "status": "not computed; unavailable selected factor returns never filled/replaced ex post"})
                continue
            for allocation in ["ew", "optimized"]:
                b, a = z[f"base_{allocation}"], z[f"augmented_{allocation}"]
                for lag in [3, 6, 12]:
                    gain = hac_mean(a-b, lag)
                    bstats, astats = hac_mean(b, lag), hac_mean(a, lag)
                    sr = paired_sharpe(a, b, lag, bootstrap_n=5000 if lag == 3 else 0)
                    summaries.append(info | {"period": period, "allocation": allocation, "hac_lags": lag,
                        "n_months": len(z), "base_annual_mean": bstats["annual_mean"],
                        "augmented_annual_mean": astats["annual_mean"],
                        "delta_annual_mean": gain["annual_mean"], "delta_mean_p": gain["p"],
                        "delta_mean_ci_low": gain["ci_low"], "delta_mean_ci_high": gain["ci_high"],
                        "base_annual_vol": bstats["annual_vol"], "augmented_annual_vol": astats["annual_vol"],
                        "base_cagr": bstats["cagr"], "augmented_cagr": astats["cagr"],
                        "base_max_drawdown": max_drawdown(b), "augmented_max_drawdown": max_drawdown(a),
                        "base_paper_correlation": b.corr(z.paper),
                        "mean_earc_weight": float(z.earc_weight_optimized.mean()) if allocation=="optimized" else float((1/(z.factor_count+1)).mean())} | sr)
            for lag in [3, 6, 12]:
                # Evaluation regression on the aggregated EW pool: one control,
                # distinguished from full-pool training-only beta residual.
                model = sm.OLS(z.paper.to_numpy(), sm.add_constant(z.base_ew.to_numpy(), has_constant="add"), missing="raise").fit(
                    cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True}, use_t=True)
                ci = model.conf_int()[0]*12
                resid_stats = hac_mean(z.oos_residual, lag)
                alpha_rows.append(info | {"period": period, "hac_lags": lag,
                    "aggregate_pool_alpha_annual": model.params[0]*12,
                    "aggregate_pool_alpha_p": model.pvalues[0],
                    "aggregate_pool_alpha_ci_low": ci[0], "aggregate_pool_alpha_ci_high": ci[1],
                    "oos_full_pool_residual_annual": resid_stats["annual_mean"],
                    "oos_full_pool_residual_p": resid_stats["p"],
                    "oos_full_pool_residual_ci_low": resid_stats["ci_low"],
                    "oos_full_pool_residual_ci_high": resid_stats["ci_high"]})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "performance_and_advantage.csv", index=False)
    pd.DataFrame(alpha_rows).to_csv(OUT / "alpha_diagnostics.csv", index=False)
    pd.DataFrame(missing).to_csv(OUT / "unavailable_comparisons.csv", index=False)
    primary = summary[(summary.method=="min10_annual_iid") & (summary.availability=="published_before_year") & (summary.scenario=="5bps") & (summary.hac_lags==3)]
    primary.to_csv(OUT / "primary_results.csv", index=False)
    records = {"selection": "Retain prior5 every-year significant factors; fill to10 by largest worst-year signed t, pooled t tie-break. Method variants explicitly labeled.",
        "portfolio": "Same total capital/gross budget: k factor baseline EW; add EARC as1/(k+1). No factor direction flipped.",
        "optimization": "Sensitivity only: annual prior60monthly data; gamma5 long-only simplex mean-variance,50% covariance shrink to diagonal. Both baseline and augmented use identical rule.",
        "costs": "EARC exact5bp per bought/sold dollar; incumbent factors gross; allocation changes and hedge costs unavailable and omitted.",
        "sharpe_inference": "Paired HAC influence-function delta method; paired12month circular block bootstrap5000 draws. Conditional on constructed realized paths; no full selection/optimizer rerun inside bootstrap.",
        "alpha": "One-control aggregate-pool regression alpha is distinct from OOS full-pool regression residual with prior60 beta and intercept retained. No saturated yearly10slope/12month alpha.",
        "sample": "1990-2025, previously inspected retrospective periods. No current2026returns. Missing selected OOS returns make full period unavailable; report continuous prefix instead of imputation.",
        "multiplicity": "Predeclared primary min10_annual_iid,published_before_year,5bps,EW. Othermethods/weights/periods sensitivity; pvalues not discovery-adjusted across all variants.",
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), HERE/"optimization.py", HERE/"select_pools.py", PARENT/"paper_factor_daily.csv", OUT/"selection_members.csv"]}}
    (OUT / "design.json").write_text(json.dumps(records, indent=2), encoding="utf8")
    print(primary[["period", "allocation", "base_annual_mean", "augmented_annual_mean", "delta_annual_mean", "delta_mean_p", "base_sharpe", "augmented_sharpe", "sharpe_boot_ci_low", "sharpe_boot_ci_high", "mean_earc_weight"]].to_string(index=False))
    print("Unavailable full-period comparisons:", len(missing))


if __name__ == "__main__":
    main()
