"""Generate the measured-result report; never substitute reference values."""
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transalpha.data import ROOT, save_json
from transalpha.evaluation import block_bootstrap_mean


def main():
    config = json.loads((ROOT / "config.json").read_text())
    paths = sorted((ROOT / "results").glob("*/*/metrics.json"))
    rows, by_group = [], {}
    for path in paths:
        m = json.loads(path.read_text())
        row = {k: m[k] for k in ["universe", "variant", "seed", "ic", "rank_ic", "icir", "rank_icir",
                                "best_epoch", "validation_ic", "parameters", "seconds"]}
        row.update({"gross_sharpe": m["gross"]["sharpe"], "net_sharpe": m["net"]["sharpe"],
                    "gross_ar": m["gross"]["annualized_return"], "net_ar": m["net"]["annualized_return"],
                    "net_mdd": m["net"]["maximum_drawdown"],
                    "turnover": m["mean_daily_traded_notional_over_nav"],
                    "frozen_position_days": m["frozen_position_days"],
                    "maximum_absolute_net_exposure": m["maximum_absolute_net_exposure"],
                    "maximum_gross_exposure": m["maximum_gross_exposure"]})
        rows.append(row)
        by_group.setdefault((m["universe"], m["variant"]), []).append(pd.read_csv(path.parent / "daily.csv"))
    expected = sum(len(config["seeds"]) * (len(config["variants"]) if u in config["ablation_universes"] else 2)
                   for u in config["universes"])
    status = {"scope": config["scope"], "expected_runs": expected, "completed_runs": len(paths),
              "daily_suite_complete": len(paths) == expected,
              "exact_intraday_replication": "Pending author factor data and missing protocol details",
              "report": str(ROOT / "results" / "REPORT.txt")}
    if (ROOT / "results" / "audit.json").exists():
        audit = json.loads((ROOT / "results" / "audit.json").read_text())
        status["audited_runs"] = len(audit["runs"])
        status["all_completed_runs_audited"] = bool(audit["passed"] and len(audit["runs"]) == len(paths))
    save_json(ROOT / "STATUS.json", status)
    if not rows:
        print(json.dumps(status))
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / "results" / "per_seed.csv", index=False)
    metrics = ["ic", "rank_ic", "icir", "rank_icir", "gross_sharpe", "net_sharpe", "gross_ar", "net_ar", "net_mdd", "turnover"]
    summary = frame.groupby(["universe", "variant"])[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    summary["seeds_completed"] = [len(by_group[(r.universe, r.variant)]) for r in summary.itertuples()]
    summary.to_csv(ROOT / "results" / "summary.csv", index=False)
    statistics = {"scope": "Circular 10-session block bootstrap of daily metrics after averaging fixed seeds; descriptive single-year intervals",
                  "by_model": {}, "paired_transalpha_minus_linear": {}, "paired_transalpha_minus_ablations": {}}
    for (universe, variant), daily in by_group.items():
        dates = daily[0].signal_date.tolist()
        if not all(d.signal_date.tolist() == dates for d in daily):
            raise ValueError("Seed dates differ")
        statistics["by_model"][universe + "/" + variant] = {
            column: block_bootstrap_mean(np.mean([d[column].to_numpy() for d in daily], axis=0))
            for column in ["ic", "rank_ic", "net_return"]}
    for universe in config["universes"]:
        a = by_group.get((universe, "transalpha"), [])
        b = by_group.get((universe, "linear"), [])
        if len(a) == len(config["seeds"]) and len(b) == len(config["seeds"]):
            statistics["paired_transalpha_minus_linear"][universe] = {
                column: block_bootstrap_mean(np.mean([d[column].to_numpy() for d in a], axis=0) -
                                            np.mean([d[column].to_numpy() for d in b], axis=0))
                for column in ["ic", "rank_ic", "net_return"]}
    for universe in config["ablation_universes"]:
        a = by_group.get((universe, "transalpha"), [])
        for variant in ["no_csd", "no_tag", "no_sp"]:
            b = by_group.get((universe, variant), [])
            if len(a) == len(config["seeds"]) and len(b) == len(config["seeds"]):
                statistics["paired_transalpha_minus_ablations"][universe + "/" + variant] = {
                    column: block_bootstrap_mean(np.mean([d[column].to_numpy() for d in a], axis=0) -
                                                np.mean([d[column].to_numpy() for d in b], axis=0))
                    for column in ["ic", "rank_ic", "net_return"]}
    save_json(ROOT / "results" / "statistics.json", statistics)
    lines = ["TransAlpha: measured daily CSI adaptation", "=" * 41, "",
             "Completion: %d / %d planned fits." % (len(paths), expected),
             "This is an independent architecture/loss reconstruction and daily-frequency experiment.",
             "The paper's private 400-factor 15-minute performance has NOT been replicated.", "",
             "Fixed protocol: 3 seeds; train 2021-01-04 to 2023-09-28, validation 2023-10-09",
             "to 2023-12-29, test 2024-01-02 to 2024-12-31. Daily Alpha158, 16-session context.",
             "665 training decisions, 58 validation decisions, 242 test decisions per universe.",
             "Reconstructed parameter counts: 19,006 with daily D=158; 34,978 with paper D=400.",
             "Target/execution: signal at t close, entry t+1 close, exit t+2 close; split labels purged.",
             "Checkpoints maximize validation IC. Settings and seeds fixed before test evaluation.", "",
             "Measured results: mean +/- sample SD across seeds (SD is not a confidence interval).", "",
             "Universe  Model        n     IC mean(sd)       RankIC mean(sd)    Net AR       Net Sharpe     Net MDD"]
    for r in summary.itertuples():
        def sd(x):
            return 0. if pd.isna(x) else x
        lines.append(f"{r.universe:9} {r.variant:12} {r.seeds_completed:1d}  {r.ic_mean: .4f}({sd(r.ic_std):.4f})   "
                     f"{r.rank_ic_mean: .4f}({sd(r.rank_ic_std):.4f})   {r.net_ar_mean: .2%}   "
                     f"{r.net_sharpe_mean: .3f}({sd(r.net_sharpe_std):.3f})  {r.net_mdd_mean:.2%}")
    lines += ["", "Paired TransAlpha-minus-linear daily IC, 95% 10-session block intervals:"]
    for universe, stats in statistics["paired_transalpha_minus_linear"].items():
        ic = stats["ic"]
        lines.append(f"  {universe}: {ic['mean']:.5f}, [{ic['ci95'][0]:.5f}, {ic['ci95'][1]:.5f}]")
    lines += ["", "Paired full-model-minus-ablation daily IC, same unadjusted block intervals:"]
    for name, stats in statistics["paired_transalpha_minus_ablations"].items():
        ic = stats["ic"]
        lines.append(f"  {name}: {ic['mean']:.5f}, [{ic['ci95'][0]:.5f}, {ic['ci95'][1]:.5f}]")
    full = summary[(summary.universe == "csi300") & (summary.variant == "transalpha")]
    no_sp = summary[(summary.universe == "csi300") & (summary.variant == "no_sp")]
    if len(full) and len(no_sp) and no_sp.iloc[0].ic_mean > full.iloc[0].ic_mean:
        lines += ["  Mean pooling has higher mean test IC than Smart Pooling in this daily experiment.",
                  "  The adaptation does not establish a consistent benefit from every proposed component."]
    lines += ["", "Published Table 3 reference (different inputs/frequency; not a matched comparison):",
              "  CSI300:  IC .0664, RankIC .0713, AR 27.30%, Sharpe 3.1769",
              "  CSI500:  IC .0697, RankIC .0696, AR 28.40%, Sharpe 3.1336",
              "  CSI1000: IC .0966, RankIC .0732, AR 28.23%, Sharpe 2.8341", "",
              "Portfolio assumptions: target +50% NAV long / -50% short in top/bottom 50.",
              "Daily rebalancing; 10bp on long sales and short covers, drift-aware; terminal",
              "marked liquidation fee included. Block untradable new entries, retain halted",
              "positions, use last observed marks. Gross/net differences include the specified",
              "fee only. No borrow, financing, spread, slippage, price-limit, or short-eligibility model.",
              "This idealized A-share portfolio is not established as directly executable.",
              f"Maximum observed gross exposure: {frame.maximum_gross_exposure.max():.9f}.",
              f"Maximum absolute net exposure at rebalance: {frame.maximum_absolute_net_exposure.max():.9f}.",
              "ICIR uses daily mean/population SD; Sharpe uses sqrt(252), sample SD and zero RF.",
              "IR is omitted because the paper does not specify its active-return benchmark.", "",
              "Interpretation: daily measurements evaluate this disclosed reconstruction only.",
              "A positive IC or Sharpe does not reproduce the intraday Table 3. A failure to beat",
              "the linear control here also does not falsify performance on the unavailable alphas.",
              "Date-block intervals account for serial dependence within 2024; they do not measure",
              "uncertainty from vendor history, feature substitution, protocol choices, or market regimes.", "",
              "Open requirements for exact replication:",
              "  Author 400-factor 15-minute HDF5 data, stock/return alignment and preprocessing.",
              "  Chosen optimizer/grid/head/dropout/stopping settings and seeds.",
              "  Precise leverage, backtest rules, IR benchmark, and fee interpretation.",
              "  Original CSI2000/full-market fits and the full published baseline suite.", "",
              "Artifacts: per_seed.csv, summary.csv, statistics.json, overview.png; per-run model.pt,",
              "validation/test_predictions.npz, daily.csv, history.json, metrics.json; audit.json.",
              "Paper: https://ijcai.org/proceedings/2026/580",
              "Data: https://huggingface.co/datasets/joshuaxql/qlib_data",
              "Source/data/code details: README.txt and sources/reconstruction_notes.txt."]
    report = "\n".join(lines) + "\n"
    (ROOT / "results" / "REPORT.txt").write_text(report)
    (ROOT / "results" / "REPORT.html").write_text("<!doctype html><meta charset='utf-8'><title>TransAlpha reproduction</title>"
        "<style>body{font:15px system-ui;margin:3rem;max-width:1200px;color:#172538}pre{white-space:pre-wrap;line-height:1.5}img{max-width:100%}</style>"
        "<pre>" + html.escape(report) + "</pre><img src='overview.png' alt='Daily IC and net Sharpe by universe'>")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, column, title in zip(axes, ["ic", "net_sharpe"], ["Daily cross-sectional IC", "Net annualized Sharpe"]):
        for j, variant in enumerate(["transalpha", "linear"]):
            sub = summary[summary.variant == variant].set_index("universe").reindex(config["universes"])
            ax.bar(np.arange(3) + (j - .5) * .34, sub[column + "_mean"], width=.34,
                   yerr=sub[column + "_std"].fillna(0), capsize=3, label=variant,
                   color=["#235d9f", "#b87541"][j])
        ax.set_xticks(np.arange(3), ["CSI300", "CSI500", "CSI1000"])
        ax.axhline(0, color="#667", linewidth=.6)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle("TransAlpha daily adaptation — mean ± seed SD; not the private intraday benchmark", fontsize=11)
    fig.savefig(ROOT / "results" / "overview.png", dpi=170)
    plt.close(fig)
    print(report)


if __name__ == "__main__":
    main()
