from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import ROOT


def percent(value):
    return f"{100 * value:.2f}%"


def main():
    config = json.loads((ROOT / "config.json").read_text())
    primary = json.loads((ROOT / "results/primary_inference.json").read_text())
    summary = pd.read_csv(ROOT / "results/summary.csv")
    periods = pd.read_csv(ROOT / "results/subperiods.csv")
    verification = json.loads((ROOT / "results/verification.json").read_text())
    metadata = json.loads((ROOT / "data/metadata.json").read_text())
    sample = summary[summary.fraction.eq(.1) & summary.scenario.eq("primary")].set_index("method")
    a = sample.loc["arbnn"]
    gross = summary[summary.method.eq("arbnn") & summary.fraction.eq(.1) & summary.scenario.eq("gross")].iloc[0]
    tests = primary["family"]
    verdict = ("The primary test finds significant positive net returns."
               if tests[0]["significant_positive_gain"] else
               "The primary test does not find a significant positive net gain.")
    text = ["# ArbNN on real U.S. stocks: equal-dollar long–short results", "",
            f"**{verdict}** ArbNN's annualized arithmetic mean is "
            f"**{percent(a.mean_annual)} net**, versus {percent(gross.mean_annual)} before costs. "
            f"Its net Sharpe is **{a.sharpe:.2f}**. This result is specific to the declared monthly "
            "market-feature experiment; it is not evidence that every possible finance application must fail.", "",
            "## Design and sample", "",
            f"- CRSP CIZ U.S. corporate common stocks, **2005-01 through 2025-12**, "
            f"**252 months** and **{verification['oos_stock_months']:,} stock-month forecasts**.",
            f"- Dynamic historical universe, no current-constituent screen: roughly "
            f"{metadata['minimum_monthly_universe']:,}–{metadata['maximum_monthly_universe']:,} stocks per month; "
            "price >= $5, >= $1m average daily dollar volume, above the NYSE 20th size percentile, one share class per issuer.",
            "- 35 trailing market-derived predictors. Predictors stop at t-2; eligibility is assessed at t-1. "
            "Five refits, each using 120 training months, 24 validation months and a one-month label embargo. "
            "All 15 models (five blocks × three seeds) completed. No test-based direction flip or seed selection.",
            "- Primary strategy averages the three model scores, buys the top 10% and shorts the bottom 10%. "
            "Each leg has equal stock weights and equal count; long +100%, short -100%, net 0%, gross 200% of NAV.",
            "- Net case: **10 bp per dollar bought/sold plus 1% annual short borrow**. "
            "Turnover uses prior positions after return and NAV drift; initial entry and both legs are charged. "
            "Monthly total returns already account for dividends, including short dividend payments.", "",
            "The [locked protocol](PROTOCOL.md), [initial hashes](protocol_initial.sha256), "
            "[configuration](config.json), and [measurement clarification](MEASUREMENT_NOTE.md) "
            "document all choices and the pre-result extension of the missing-payoff stress.", "",
            "## Primary comparison", "",
            "Annual return below is 12 × mean monthly return. It differs from CAGR. "
            "All rows use the same held-out months, universe rule and 10 bp + 1% borrow assumptions. "
            "Newey–West uses six monthly lags and small-sample correction.", "",
            "| Model/signal | Net annual mean | CAGR | Sharpe | NW t(mean) | Net FF6 alpha/year |",
            "|---|---:|---:|---:|---:|---:|"]
    for method, label in [("arbnn", "ArbNN ensemble"), ("xgboost", "XGBoost ensemble"),
                          ("decompiled", "Decompiled ArbNN"), ("ridge", "Ridge"),
                          ("momentum", "12-to-2 momentum"), ("reversal", "Lagged reversal")]:
        row = sample.loc[method]
        text.append(f"| {label} | {percent(row.mean_annual)} | {percent(row.cagr)} | {row.sharpe:.2f} | "
                    f"{row.mean_t:.2f} | {percent(row.ff6_alpha_annual)} |")
    text += ["", "## Significance and incremental value", "",
             "The three primary hypotheses form one predeclared family. P-values are two-sided, "
             "Holm-adjusted across that family. A negative estimate never counts as a positive gain. "
             "The 95% intervals below are individual HAC intervals, not simultaneous family intervals.", "",
             "| Primary quantity | Annual estimate | 95% HAC CI | Raw p | Holm p | Significant positive? |",
             "|---|---:|---|---:|---:|---|"]
    labels = ["ArbNN net mean", "ArbNN net FF5 + momentum alpha", "ArbNN − XGBoost net mean"]
    for label, test in zip(labels, tests):
        text.append(f"| {label} | {percent(test['estimate_annual'])} | "
                    f"[{percent(test['individual_95ci_low'])}, {percent(test['individual_95ci_high'])}] | "
                    f"{test['two_sided_hac_p']:.4f} | {test['holm_p']:.4f} | "
                    f"{'yes' if test['significant_positive_gain'] else 'no'} |")
    bootstrap = primary["bootstrap"]
    boot, delta = bootstrap["arbnn_net"], bootstrap["arbnn_minus_xgboost_net"]
    text += ["", f"The 5,000-replication stationary bootstrap (mean block 12 months) gives a net annual-mean "
             f"95% interval of **[{percent(boot['ci_low'])}, {percent(boot['ci_high'])}]**, "
             f"and a paired increment interval of **[{percent(delta['ci_low'])}, {percent(delta['ci_high'])}]**. "
             "No stock-row or seed pseudo-replication is used. HAC(3) and HAC(12) results are saved separately.", "",
             f"The CAPM market beta is **{a.capm_beta:.3f}**, 95% CI "
             f"[{a.capm_beta_ci_low:.3f}, {a.capm_beta_ci_high:.3f}]. Equal-dollar neutrality therefore "
             "does **not** establish beta neutrality. The FF6 regression is an attribution diagnostic, "
             "not an executable hedge constructed with future information.", "",
             f"ArbNN's maximum net drawdown is **{percent(a.max_drawdown)}**. "
             f"Average total absolute monthly traded notional is **{a.turnover_monthly:.2f} × NAV**. "
             f"Transaction costs average {percent(a.transaction_cost_annual)} of NAV per year, "
             f"with {percent(a.borrow_cost_annual)} annual borrow expense.", "",
             "## Fixed cost and basket sensitivities", "",
             "| Basket per leg | Scenario | Annual mean | Sharpe |",
             "|---|---|---:|---:|"]
    cost_names = {"gross": "Gross: 0 bp, no borrow", "low": "5 bp + 1% borrow",
                  "primary": "10 bp + 1% borrow", "stress": "25 bp + 3% borrow",
                  "primary_adverse_missing": "10 bp + adverse missing/partial payoff stress"}
    for fraction in config["quantiles"]:
        for scenario in ("gross", "low", "primary", "stress", "primary_adverse_missing"):
            row = summary[summary.method.eq("arbnn") & summary.fraction.eq(fraction) & summary.scenario.eq(scenario)].iloc[0]
            text.append(f"| {int(fraction * 100)}% | {cost_names[scenario]} | {percent(row.mean_annual)} | {row.sharpe:.2f} |")
    text += ["", "## Stability by held-out period", "",
             "| Period | Months | Net annual mean | Sharpe |",
             "|---|---:|---:|---:|"]
    for row in periods[periods.method.eq("arbnn")].itertuples():
        text.append(f"| {row.period} | {row.n_months} | {percent(row.mean_annual)} | {row.sharpe:.2f} |")
    completions = [json.loads(p.read_text()) for p in sorted((ROOT / "results").glob("fold_*/seed*/complete.json"))]
    kept_initial = sum(row["best_epoch"] == 0 for row in completions)
    pd.DataFrame(completions).to_json(ROOT / "results/training_summary.json", orient="records", indent=2)
    text += ["", f"Validation selected the initial compiled soft model (epoch 0) for **{kept_initial}/15** fits. "
             "Training checkpoints were retained by validation IC, even when this did not favor fine-tuning. "
             "Individual seed results and all baseline/quantile/cost combinations are retained in summary.csv; "
             "they were not searched for a winning combination.", "",
             "## Data and execution limitations", "",
             f"The whole prepared universe contains two missing realized returns. The primary ArbNN baskets "
             f"selected **{int(a.missing_positions)}** of those records. They selected "
             f"**{int(a.unresolved_payoff_positions)}** G/M/V-flagged partial or unresolved terminal-payoff records. "
             "Those finite returns are not automatically complete cash outcomes; the adverse scenario explicitly "
             "stresses them, without dropping affected stocks or merging a second legacy delisting return.", "",
             "Actual stock-loan locates, borrow fees, financing spreads, bid/ask fills and market impact were not "
             "observed. Costs are declared scenarios, not transaction records. Monthly close-to-close return "
             "accounting and terminal cash conversion are approximations. Current-vintage historical data "
             "may contain revisions; the test is retrospective and not a prospective trading trial. "
             "These results cover this monthly U.S. market-feature design, not fundamental signals, other "
             "markets, frequencies or an exhaustive hyperparameter search.", "",
             "## Validation and files", "",
             f"Artifact verification: **{verification['status']}**. "
             f"It checked all **{verification['unique_held_return_rows_checked']:,}** distinct held stock/month "
             "returns against unfiltered CRSP, replayed 128 rows from each model checkpoint, checked every "
             "month's net/gross exposures and issuer uniqueness, and independently recomputed primary P&L "
             "and HAC mean/FF6 covariance. The original protocol hashes passed.", "",
             "- [Primary test details](results/primary_inference.json)",
             "- [All model/cost/basket statistics](results/summary.csv)",
             "- [Monthly returns and turnover](results/returns.csv)",
             "- [Verification](results/verification.json)",
             "- [HAC lag sensitivities](results/hac_sensitivity.csv)",
             "- [Paired increment sensitivities](results/increment_hac_sensitivity.csv)",
             "- [Subperiods](results/subperiods.csv)",
             "- [Reproduction instructions](../README.md)", "",
             "![Held-out performance and cost sensitivity](results/performance.png)", "",
             "## Sources and software", "",
             "- CRSP CIZ monthly source and local audit: the shared `crsp-monthly/README.txt` and `validation.json`.",
             "- [CRSP official current guide](https://indexes.morningstar.com/docs/guide/crsp-us-stock-databases-guide-for-flat-file-format-2-0?isRdp=true): monthly return definitions and Aggregate Delisting Flag table.",
             "- [Official French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html): monthly FF5 and momentum archives retained with checksums.",
             "- [Statsmodels HAC documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html).",
             "- Statistical workflow reference: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
             "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. "
             "[arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065), current record v2.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(text))
    plot(summary)
    print(ROOT / "REPORT.md")


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    returns = pd.read_parquet(ROOT / "results/returns.parquet")
    primary = returns[returns.fraction.eq(.1) & returns.scenario.eq("primary")].copy()
    primary["date"] = pd.to_datetime(primary.date)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    colors = {"arbnn": "#d35f27", "xgboost": "#266c99", "ridge": "#6d7c88", "momentum": "#459b72"}
    for method, label in [("arbnn", "ArbNN"), ("xgboost", "XGBoost"), ("ridge", "Ridge"), ("momentum", "Momentum")]:
        p = primary[primary.method.eq(method)].sort_values("month")
        nav = np.cumprod(1 + p["return"].to_numpy())
        axes[0, 0].plot(p.date, nav, label=label, color=colors[method], linewidth=1.4)
        if method in {"arbnn", "xgboost"}:
            drawdown = nav / np.maximum.accumulate(np.r_[1, nav])[1:] - 1
            axes[1, 1].plot(p.date, drawdown * 100, label=label, color=colors[method], linewidth=1.2)
    axes[0, 0].set(title="Net NAV (initial NAV = 1)", ylabel="NAV, log scale", yscale="log")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set(title="Net drawdown", ylabel="Drawdown (%)")
    axes[1, 1].legend(fontsize=8)
    calendar = primary[primary.method.isin(["arbnn", "xgboost"])].copy()
    calendar["year"] = calendar.month // 12
    annual = calendar.groupby(["year", "method"])["return"].apply(lambda r: np.prod(1 + r) - 1).unstack()
    x = np.arange(len(annual))
    axes[0, 1].bar(x - .19, annual.arbnn * 100, .38, label="ArbNN", color=colors["arbnn"])
    axes[0, 1].bar(x + .19, annual.xgboost * 100, .38, label="XGBoost", color=colors["xgboost"])
    axes[0, 1].set_xticks(x[::2], annual.index[::2], rotation=45)
    axes[0, 1].axhline(0, color="black", lw=.7)
    axes[0, 1].set(title="Calendar-year net returns", ylabel="Return (%)")
    axes[0, 1].legend(fontsize=8)
    cases = ["gross", "low", "primary", "stress", "primary_adverse_missing"]
    labels = ["Gross", "5 bp / 1%", "10 bp / 1%", "25 bp / 3%", "Adverse payoff"]
    bars = summary[summary.method.eq("arbnn") & summary.fraction.eq(.1)].set_index("scenario").loc[cases]
    values = bars.mean_annual.to_numpy() * 100
    error = np.stack([bars.mean_annual - bars.mean_ci_low, bars.mean_ci_high - bars.mean_annual]) * 100
    axes[1, 0].errorbar(np.arange(5), values, yerr=error, fmt="o", color=colors["arbnn"], capsize=4)
    axes[1, 0].axhline(0, color="black", lw=.7)
    axes[1, 0].set_xticks(np.arange(5), labels, rotation=12)
    axes[1, 0].set(title="ArbNN costs: mean and individual 95% HAC CI", ylabel="Annualized arithmetic mean (%)")
    for ax in axes.flat:
        ax.grid(alpha=.18)
    fig.suptitle("CRSP U.S. stocks · 2005–2025 · +100% long / −100% short\n"
                 "Net case: 10 bp per traded dollar + 1% annual short borrow", fontsize=12)
    fig.savefig(ROOT / "results/performance.png", dpi=170)
    fig.savefig(ROOT / "results/performance.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
