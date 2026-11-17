#!/usr/bin/env python3
"""Descriptive training-only factor/entry/exit response-surface summaries.

The response is annual_mean_proxy from the event-weighted screen, NOT an exact
calendar portfolio return or alpha. Factor fixed effects plus a common quadratic
N/M timing surface summarize the grid. Cluster-by-factor intervals describe
variation across factor response surfaces; factors share stocks and signals,
so those intervals are not independent-cluster trading-profit evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster


HERE = Path(__file__).resolve().parent
NAMES = ["N_centered", "M_centered", "N_centered_squared",
         "M_centered_squared", "N_centered_times_M_centered"]


def features(n, m, center=15.5):
    n, m = np.asarray(n, float)-center, np.asarray(m, float)-center
    return np.column_stack([n, m, n*n, m*m, n*m])


def _scalar(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, dict):
        return {str(k): _scalar(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_scalar(v) for v in value]
    return value


def multiple_testing_summary(all_rows):
    p = all_rows.cluster_p.to_numpy(float)
    finite = np.isfinite(p) & (p >= 0) & (p <= 1)
    valid = all_rows.loc[finite]
    result = dict(total_specifications=len(all_rows), finite_p_values=int(finite.sum()),
                  family="All finite individual event-regression p-values, before eligibility filtering",
                  interpretation="Training-only gross slope associations, not post-cost portfolio profits")
    for method, label in [("fdr_bh", "BH"), ("fdr_by", "BY_dependence_sensitivity"), ("holm", "Holm")]:
        rejected, adjusted, _, _ = multipletests(p[finite], alpha=.05, method=method)
        result[label] = dict(
            rejected_5pct=int(rejected.sum()),
            positive_slope_rejected=int((rejected & (valid.beta.to_numpy() > 0)).sum()),
            negative_slope_rejected=int((rejected & (valid.beta.to_numpy() < 0)).sum()),
            eligible_positive_slope_rejected=int((rejected & valid.eligible.to_numpy(bool)
                                                 & (valid.beta.to_numpy() > 0)).sum()),
            minimum_adjusted_p=float(adjusted.min()) if len(adjusted) else None,
        )
    return result


def describe(directory):
    directory = Path(directory)
    source = directory/"regression_screen.csv"
    all_rows = pd.read_csv(source)
    keep = (all_rows.eligible.astype(bool)
            & np.isfinite(all_rows.annual_mean_proxy)
            & np.isfinite(all_rows.buy_before) & np.isfinite(all_rows.sell_after))
    rows = all_rows.loc[keep].reset_index(drop=True)
    assert len(rows), "No finite eligible training grid cells"
    factors, group = np.unique(rows.factor.to_numpy(), return_inverse=True)
    ng = len(factors)
    assert ng > 5
    counts = np.bincount(group, minlength=ng).astype(float)
    center = 15.5
    x = features(rows.buy_before, rows.sell_after, center)
    y = rows.annual_mean_proxy.to_numpy(float)
    group_y = np.bincount(group, weights=y, minlength=ng)/counts
    group_x = np.vstack([np.bincount(group, weights=x[:, j], minlength=ng)/counts
                         for j in range(x.shape[1])]).T
    xd, yd = x-group_x[group], y-group_y[group]
    fit = sm.OLS(yd, xd, missing="raise").fit()
    beta = fit.params
    # FWL absorbs ng factor intercepts. Count them in the finite-sample CR1
    # correction instead of treating the demeaned fit as only five parameters.
    covariance = cov_cluster(fit, group, use_correction=False)
    covariance *= ng/(ng-1) * (len(rows)-1)/(len(rows)-ng-len(NAMES))
    se = np.sqrt(np.diag(covariance))
    t_values = beta/se
    p = 2*student_t.sf(np.abs(t_values), ng-1)
    critical = student_t.ppf(.975, ng-1)
    coefficients = pd.DataFrame(dict(
        parameter=NAMES, coefficient=beta, cluster_se=se,
        ci95_low=beta-critical*se, ci95_high=beta+critical*se,
        t=t_values, p=p,
        unit=["annual-return decimal per extra day at N=M=15.5",
              "annual-return decimal per extra day at N=M=15.5",
              "annual-return decimal per squared centered N day",
              "annual-return decimal per squared centered M day",
              "annual-return decimal per centered N*M day squared"],
        status="DESCRIPTIVE_EVENT_PROXY_NOT_PORTFOLIO_ALPHA",
    ))
    coefficients.to_csv(directory/"surface_coefficients.csv", index=False)

    intercept = group_y-group_x@beta
    effects = pd.DataFrame(dict(
        factor=factors, eligible_grid_cells=counts.astype(int),
        raw_mean_annual_proxy=group_y, fitted_factor_intercept_at_N15_5_M15_5=intercept,
    ))
    effects["raw_mean_rank"] = effects.raw_mean_annual_proxy.rank(ascending=False, method="min").astype(int)
    effects["factor_intercept_rank"] = effects.fitted_factor_intercept_at_N15_5_M15_5.rank(ascending=False, method="min").astype(int)
    raw_best = rows.sort_values(["annual_mean_proxy", "buy_before", "sell_after"],
                               ascending=[False, True, True]).groupby("factor", sort=False).head(1)
    best_columns = ["factor", "buy_before", "sell_after", "annual_mean_proxy", "sharpe_proxy", "delta_annual_proxy"]
    raw_best = raw_best[best_columns].rename(columns={
        "buy_before":"raw_best_N", "sell_after":"raw_best_M",
        "annual_mean_proxy":"raw_best_annual_proxy", "sharpe_proxy":"raw_best_sharpe_proxy",
        "delta_annual_proxy":"raw_best_delta_annual_proxy_vs_N1_M1"})
    effects = effects.merge(raw_best, on="factor", validate="one_to_one")
    low_n, high_n = int(all_rows.buy_before.min()), int(all_rows.buy_before.max())
    low_m, high_m = int(all_rows.sell_after.min()), int(all_rows.sell_after.max())
    effects["raw_best_N_at_boundary"] = effects.raw_best_N.isin([low_n,high_n])
    effects["raw_best_M_at_boundary"] = effects.raw_best_M.isin([low_m,high_m])
    effects["raw_best_any_boundary"] = effects.raw_best_N_at_boundary | effects.raw_best_M_at_boundary

    all_n, all_m = np.meshgrid(np.arange(low_n,high_n+1), np.arange(low_m,high_m+1), indexing="ij")
    grid_x = features(all_n.ravel(), all_m.ravel(), center)
    timing_fit = grid_x@beta
    global_fit = intercept.mean()+timing_fit
    grid = pd.DataFrame(dict(buy_before=all_n.ravel(), sell_after=all_m.ravel(),
                             common_timing_component=timing_fit,
                             equal_factor_mean_fitted_annual_proxy=global_fit))
    grid["status"] = "DESCRIPTIVE_EVENT_PROXY_NOT_PORTFOLIO_ALPHA"
    grid.to_csv(directory/"surface_fitted_grid.csv", index=False)
    g_best = int(np.argmax(global_fit))
    # Timing coefficients are shared across factors. Restrict each factor's
    # reported fitted optimum to its observed eligible grid cells.
    rows["fitted_annual_proxy"] = intercept[group]+x@beta
    fitted_best = rows.sort_values(["fitted_annual_proxy", "buy_before", "sell_after"],
                                  ascending=[False, True, True]).groupby("factor", sort=False).head(1)
    effects = effects.merge(fitted_best[["factor","buy_before","sell_after","fitted_annual_proxy"]]
                            .rename(columns={"buy_before":"fitted_best_N", "sell_after":"fitted_best_M",
                                             "fitted_annual_proxy":"fitted_best_annual_proxy"}),
                            on="factor", validate="one_to_one")
    effects["status"] = "DESCRIPTIVE_TRAINING_GRID_NOT_SELECTED_PORTFOLIO"
    effects = effects.sort_values(["factor_intercept_rank", "factor"])
    effects.to_csv(directory/"surface_factor_effects.csv", index=False)

    predicted = intercept[group]+x@beta
    sse = float(np.sum((y-predicted)**2))
    total = float(np.sum((y-y.mean())**2))
    within = float(np.sum(yd**2))
    best = rows.sort_values(["annual_mean_proxy", "factor", "buy_before", "sell_after"],
                           ascending=[False, True, True, True]).iloc[0]
    old = rows[rows.published_by2004.astype(bool)]
    old_best = old.sort_values(["annual_mean_proxy", "factor", "buy_before", "sell_after"],
                              ascending=[False,True,True,True]).iloc[0] if len(old) else None
    def best_receipt(row):
        if row is None:
            return None
        return {k:row[k] for k in ["factor","buy_before","sell_after","annual_mean_proxy",
                                   "sharpe_proxy","delta_annual_proxy"]}
    summary = dict(
        status="DESCRIPTIVE_TRAINING_EVENT_PROXY_NOT_PORTFOLIO_ALPHA",
        source=str(source), training_period="1985-2004 only; common purged event sample",
        response="annual_mean_proxy from event-weighted screening approximation",
        model="y_f,N,M = factor_intercept_f + b1*(N-15.5) + b2*(M-15.5) + b3*(N-15.5)^2 + b4*(M-15.5)^2 + b5*(N-15.5)*(M-15.5) + residual",
        sample="Uniform weight for every finite eligible factor×N×M grid cell",
        absorbed_effects="Factor-specific intercepts; exact within-factor demeaning of both response and features",
        covariance="CR1 by factor, absorbed-intercept degrees of freedom included; Student t(G-1)",
        covariance_caveat="Factor clusters share stocks, event returns and correlated signals. Standard errors are descriptive across-factor diagnostics, not calibrated independent-cluster profit evidence.",
        total_grid_cells=len(all_rows), fitted_cells=len(rows), factor_clusters=ng,
        centered_at_days=center, coefficients=coefficients[["parameter","coefficient","cluster_se","ci95_low","ci95_high","p"]].to_dict("records"),
        additive_R2=1-sse/total, within_factor_timing_R2=1-sse/within,
        factor_intercepts_only_R2=1-within/total,
        residual_rmse=float(np.sqrt(sse/len(rows))),
        fitted_common_surface_optimum=dict(buy_before=int(all_n.ravel()[g_best]),
                                          sell_after=int(all_m.ravel()[g_best]),
                                          equal_factor_mean_fitted_annual_proxy=float(global_fit[g_best])),
        raw_grid_best=best_receipt(best), raw_grid_best_published_by2004=best_receipt(old_best),
        raw_factor_optimum_boundaries=dict(
            factor_count=len(effects),
            N_at_lower=int((effects.raw_best_N==low_n).sum()),
            N_at_upper=int((effects.raw_best_N==high_n).sum()),
            M_at_lower=int((effects.raw_best_M==low_m).sum()),
            M_at_upper=int((effects.raw_best_M==high_m).sum()),
            any_boundary=int(effects.raw_best_any_boundary.sum()),
            both_N_and_M_boundary=int((effects.raw_best_N_at_boundary & effects.raw_best_M_at_boundary).sum()),
            interpretation="Boundary maxima indicate unresolved horizon shape; no extension of search or later-period selection is performed here."),
        individual_regression_multiple_testing=multiple_testing_summary(all_rows),
    )
    (directory/"surface_summary.json").write_text(json.dumps(_scalar(summary), indent=2, allow_nan=False))
    print(json.dumps(_scalar(dict(directory=str(directory), fitted_cells=len(rows), factor_clusters=ng,
                                  additive_R2=summary["additive_R2"],
                                  within_factor_timing_R2=summary["within_factor_timing_R2"],
                                  fitted_optimum=summary["fitted_common_surface_optimum"],
                                  raw_grid_best=summary["raw_grid_best"])), indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path,
                        default=[HERE/"horizon_results", HERE/"horizon_oracle_results"])
    args=parser.parse_args()
    for directory in args.directories:
        describe(directory)
