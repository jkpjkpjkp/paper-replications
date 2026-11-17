from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ROOT, month_id, month_label
from .statistics import bootstrap_mean, factor_alpha, hac_mean, holm, performance, stationary_indices

FF6 = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def main():
    config = json.loads((ROOT / "config.json").read_text())
    returns = pd.read_parquet(ROOT / "results/returns.parquet")
    factors = pd.read_parquet(ROOT / "data/factors.parquet")
    data = returns.merge(factors, on="month", how="left", validate="many_to_one")
    assert np.isfinite(data[FF6].to_numpy()).all()
    rows = []
    lag_rows = []
    for (method, fraction, scenario), sample in data.groupby(["method", "fraction", "scenario"], sort=True):
        sample = sample.sort_values("month")
        y = sample["return"].to_numpy()
        assert len(y) == 252
        result = {"method": method, "fraction": fraction, "scenario": scenario,
                  **performance(y, config["primary_hac_lag"])}
        capm = factor_alpha(y, sample[["mkt_rf"]], config["primary_hac_lag"])
        ff = factor_alpha(y, sample[FF6], config["primary_hac_lag"])
        result.update({"capm_alpha_annual": capm["alpha_annual"], "capm_alpha_t": capm["alpha_t"],
                       "capm_beta": capm["market_beta"], "capm_beta_ci_low": capm["market_beta_ci_low"],
                       "capm_beta_ci_high": capm["market_beta_ci_high"],
                       "ff6_alpha_annual": ff["alpha_annual"], "ff6_alpha_t": ff["alpha_t"],
                       "ff6_alpha_p": ff["alpha_p"], "ff6_alpha_ci_low": ff["alpha_ci_low"],
                       "ff6_alpha_ci_high": ff["alpha_ci_high"], "ff6_r_squared": ff["r_squared"],
                       "turnover_monthly": sample.turnover.mean(),
                       "transaction_cost_annual": sample.transaction_cost.mean() * 12,
                       "borrow_cost_annual": sample.borrow_cost.mean() * 12,
                       "missing_positions": int(sample.missing_count.sum()),
                       "missing_gross_weight_sum": sample.missing_gross_weight.sum(),
                       "unresolved_payoff_positions": int(sample.unresolved_payoff_count.sum()),
                       "unresolved_payoff_gross_weight_sum": sample.unresolved_payoff_gross_weight.sum(),
                       "mean_long_names": sample.n_long.mean(), "mean_short_names": sample.n_short.mean()})
        rows.append(result)
        if fraction == config["primary_quantile"] and scenario == "primary":
            for lag in config["hac_lags"]:
                lag_rows.append({"method": method, **hac_mean(y, lag),
                                 **{"ff6_" + k: v for k, v in factor_alpha(y, sample[FF6], lag).items()
                                    if k != "factor_loadings"}})
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results/summary.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(ROOT / "results/hac_sensitivity.csv", index=False)
    primary_rows = data[(data.fraction == config["primary_quantile"]) & (data.scenario == "primary")]
    a = primary_rows[primary_rows.method.eq("arbnn")].sort_values("month")
    b = primary_rows[primary_rows.method.eq("xgboost")].sort_values("month")
    np.testing.assert_array_equal(a.month, b.month)
    increment = a["return"].to_numpy() - b["return"].to_numpy()
    primary = performance(a["return"], config["primary_hac_lag"])
    primary_alpha = factor_alpha(a["return"], a[FF6], config["primary_hac_lag"])
    difference = hac_mean(increment, config["primary_hac_lag"])
    adjusted = holm([primary["mean_p"], primary_alpha["alpha_p"], difference["mean_p"]])
    family = []
    for name, estimate, p, corrected, low, high in zip(
        config["primary_hypothesis_family"],
        [primary["mean_annual"], primary_alpha["alpha_annual"], difference["mean_annual"]],
        [primary["mean_p"], primary_alpha["alpha_p"], difference["mean_p"]], adjusted,
        [primary["mean_ci_low"], primary_alpha["alpha_ci_low"], difference["mean_ci_low"]],
        [primary["mean_ci_high"], primary_alpha["alpha_ci_high"], difference["mean_ci_high"]]):
        family.append({"hypothesis": name, "estimate_annual": float(estimate), "two_sided_hac_p": float(p),
                       "holm_p": float(corrected), "individual_95ci_low": float(low),
                       "individual_95ci_high": float(high),
                       "significant_positive_gain": bool(estimate > 0 and corrected < 0.05)})
    indices = stationary_indices(len(a), config["bootstrap_repetitions"],
                                 config["bootstrap_block_months"], config["bootstrap_seed"])
    bootstrap = {"method": "circular stationary bootstrap; percentile mean-return interval",
                 "mean_block_months": config["bootstrap_block_months"],
                 "arbnn_net": bootstrap_mean(a["return"], indices),
                 "arbnn_minus_xgboost_net": bootstrap_mean(increment, indices)}
    primary_payload = {"sample": "2005-01 through 2025-12", "n_months": 252,
                       "long_exposure": 1.0, "short_exposure": -1.0, "gross_exposure": 2.0,
                       "one_way_bps": 10.0, "annual_short_borrow": 0.01,
                       "family": family, "primary_performance": primary,
                       "primary_ff6": primary_alpha, "paired_increment": difference,
                       "bootstrap": bootstrap}
    (ROOT / "results/primary_inference.json").write_text(json.dumps(primary_payload, indent=2) + "\n")
    # Fixed blocks and two additional predeclared broad/recent summaries.
    periods = config["test_blocks"] + [[2005, 2014], [2020, 2025]]
    period_rows = []
    for method in ("arbnn", "xgboost", "ridge", "momentum", "reversal"):
        table = primary_rows[primary_rows.method.eq(method)]
        for start, end in periods:
            subset = table[table.month.between(month_id(f"{start}-01-01"), month_id(f"{end}-12-01"))]
            period_rows.append({"method": method, "period": f"{start}-{end}",
                                **performance(subset["return"], config["primary_hac_lag"]),
                                "ff6_alpha_annual": factor_alpha(subset["return"], subset[FF6],
                                                                 config["primary_hac_lag"])["alpha_annual"]})
    pd.DataFrame(period_rows).to_csv(ROOT / "results/subperiods.csv", index=False)
    # Influence diagnostics are reported; no observations are removed.
    influence = a[["month", "return", "gross_return", "turnover", "transaction_cost", "borrow_cost"]].copy()
    influence["date"] = [month_label(m) for m in influence.month]
    influence["paired_increment"] = increment
    influence["absolute_return_rank"] = influence["return"].abs().rank(ascending=False, method="min")
    influence.to_csv(ROOT / "results/monthly_influence.csv", index=False)
    sensitivity_rows = []
    for lag in config["hac_lags"]:
        sensitivity_rows.append({"series": "arbnn_minus_xgboost_net", **hac_mean(increment, lag)})
    pd.DataFrame(sensitivity_rows).to_csv(ROOT / "results/increment_hac_sensitivity.csv", index=False)
    primary_turnover = a.turnover.mean()
    breakeven = (a.gross_return.mean() - a.borrow_cost.mean()) / primary_turnover * 10000
    (ROOT / "results/cost_breakeven.json").write_text(json.dumps({
        "one_way_bps_approx": float(breakeven),
        "meaning": "Mean-return breakeven using primary-path turnover and 1% annual borrow; not alpha breakeven",
        "zero_bps_means": "no transaction cost"}, indent=2) + "\n")
    print(json.dumps(primary_payload, indent=2))


if __name__ == "__main__":
    main()
