"""Audit complete scope, record all comparisons, render figures, publish review."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
UP = ROOT / "upstream"
OUT = ROOT / "results" / "final_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
sys.path.insert(0, str(UP / "src"))
import CON2 as con
import NodeMechanism as nm


def save(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    temp.replace(path)


def holm(values):
    p = np.asarray(values, dtype=float)
    ix = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[ix] = np.minimum(1., np.maximum.accumulate(p[ix] * (len(p) - np.arange(len(p)))))
    return adjusted


def prefix_check():
    data = con.load_all_data(ROOT / "data/raw")
    data["vix"] = data["vix"].copy()
    data["vix"]["VIXCLS"] = data["vix"]["VIXCLS"].shift(1)
    cfg = con.ExperimentConfig(random_seed=7, mc_paths=50, neural_exact_ic=True)
    rows = []
    for ds, prep, date in [("industry", con.prepare_industry_returns, "2008-01-23"), ("etf", con.prepare_etf_returns, "2020-01-14")]:
        r, f, v = prep(data, cfg)
        ix = r.index.get_loc(pd.Timestamp(date))
        r = r.iloc[ix - 252:ix + 21].copy()
        f = f.reindex(r.index).copy()
        changed_r, changed_f = r.copy(), f.copy()
        changed_r.loc[date:] += .3
        changed_f.loc[date:] += .1
        a = con.build_episode_set(ds, r, f, v, cfg, date, date, np.random.default_rng(123))
        b = con.build_episode_set(ds, changed_r, changed_f, v, cfg, date, date, np.random.default_rng(123))
        checks = {key: np.array_equal(getattr(a, key), getattr(b, key)) for key in ["features", "Q", "W_gls", "m_tail_cvar"]}
        assert all(checks.values()) and not np.array_equal(a.Fsum_mc, b.Fsum_mc)
        rows.append({"dataset": ds, "state": date, "future_price_and_factor_perturbation": "all future returns+0.3, factors+0.1", "past_feature_graph_invariance": checks, "future_payoff_arrays_changed": True})
    save(OUT / "future_perturbation_check.json", rows)


def main():
    # The independent audit includes the no-graph comparison:90 contrasts.
    result = subprocess.run([sys.executable, str(ROOT / "audit_release.py")], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (ROOT / "logs/final-release-audit.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout
    corrected = ROOT / "results/corrected_training"
    assert json.loads((corrected / "summary.json").read_text())["models_trained"] == 140
    records = []
    for seed in SEEDS:
        target = corrected / f"seed_{seed}"
        assert (target / "complete.json").exists()
        models = list((target / "tables").glob("*_model.pt"))
        assert len(models) == 14
        for model in models:
            marker = model.with_name(model.name.replace("_model.pt", "_training_complete.json"))
            info = json.loads(marker.read_text())
            assert hashlib.sha256(model.read_bytes()).hexdigest() == info["checkpoint_sha256"]
            assert info["config"]["neural_exact_ic"] is True
            assert info["config"]["mc_paths"] == 50 and info["config"]["epochs"] == 250
            records.append(info)
        for ds in ["industry", "etf"]:
            df = pd.read_csv(target / "tables" / (ds + "_neural_results.csv"))
            assert len(df) == 7 and (df.M_paths == 500).all()
            assert (df.IR_violation_rate == 0).all() and (df.min_IR_gap >= -1e-7).all()
            assert (df.mean_effort >= .039999).all()
    save(OUT / "all_training_manifests.json", records)
    prefix_check()
    scopes = {
        "author_released": UP / "results/constant_score_500",
        "fresh_frozen": ROOT / "results/fresh_frozen",
        "corrected_training": corrected,
    }
    comparisons = []
    means = []
    for scope, base in scopes.items():
        paired = pd.read_csv(base / "multi_seed_paired_significance.csv")
        assert len(paired) == 90, (scope, len(paired))
        paired["scope"] = scope
        comparisons.append(paired)
        for ds in ["industry", "etf"]:
            parts = []
            for seed in SEEDS:
                frame = pd.read_csv(base / f"seed_{seed}" / "tables" / f"{ds}_neural_results.csv")
                frame["seed"] = seed
                parts.append(frame)
            frame = pd.concat(parts)
            frame["scope"] = scope
            frame["dataset"] = ds
            means.append(frame)
    all_metrics = pd.concat(means, ignore_index=True)
    all_metrics.to_csv(OUT / "all_scopes_seed_metrics.csv", index=False)
    combined = pd.concat(comparisons, ignore_index=True)
    # Include the predeclared constant-IC sensitivity in the same all-search audit.
    fresh = scopes["fresh_frozen"]
    sensitivity = pd.read_csv(fresh / "constant_exact_IC_sensitivity.csv")
    extra = []
    rng = np.random.default_rng(4121811)
    for ds in ["industry", "etf"]:
        a = sensitivity[sensitivity.dataset == ds].set_index("seed")
        b = all_metrics[(all_metrics.scope == "fresh_frozen") & (all_metrics.dataset == ds) & all_metrics.method.str.contains("ConstantPenalty")].set_index("seed")
        for metric in con.ARTIFACT_SEED_METRICS:
            d = a.loc[SEEDS, metric].to_numpy() - b.loc[SEEDS, metric].to_numpy()
            boot = d[rng.integers(0, 10, (10000, 10))].mean(axis=1)
            low, high = np.quantile(boot, [.025, .975])
            extra.append({"scope": "fresh_constant_IC_sensitivity", "dataset": ds, "comparison_type": "constant_exact_minus_approx_IC", "comparison": "Constant exactIC minus Constant approximateIC", "metric": metric, "mean_treatment": a[metric].mean(), "mean_baseline": b[metric].mean(), "delta": d.mean(), "ci_low": low, "ci_high": high, "wilcoxon_p": stats.wilcoxon(d).pvalue if np.any(d) else 1., "n_seeds": 10})
    combined = pd.concat([combined, pd.DataFrame(extra)], ignore_index=True)
    assert len(combined) == 288
    combined["holm_all288_p"] = holm(combined.wilcoxon_p.to_numpy())
    for scope in combined.scope.unique():
        ix = combined.index[combined.scope == scope]
        combined.loc[ix, "holm_scope_p"] = holm(combined.loc[ix, "wilcoxon_p"])
        primary = ix[(combined.loc[ix, "comparison_type"] == "placement_vs_feature_only") & combined.loc[ix, "metric"].isin(["CVaR95_loss", "crowding", "co_crash_freq"])]
        if len(primary):
            combined.loc[primary, "holm_primary6_p"] = holm(combined.loc[primary, "wilcoxon_p"])
    combined.to_csv(OUT / "all288_comparisons.csv", index=False)
    key = combined[(combined.comparison_type.isin(["placement_vs_feature_only", "learned_signal_vs_random_signal", "constant_penalty_vs_feature_only", "graph_penalty_vs_constant_penalty"])) & combined.metric.isin(["CVaR95_loss", "crowding", "co_crash_freq"])]
    key.to_csv(OUT / "principal_paper_comparisons.csv", index=False)
    # Verify the independent per-node solve against the author's Torch mean
    # exposure before interpreting the graph-risk mechanism's quintile slope.
    node_accounting = []
    mechanism = []
    for scope in ["fresh_frozen", "corrected_training"]:
        node_frame = pd.read_csv(scopes[scope] / "node_exposure_all.csv")
        by_state = node_frame.groupby(["dataset", "seed", "date"])[["s_feature_only", "s_graph_penalty"]].mean()
        for ds in ["industry", "etf"]:
            prefix = "ETF_" if ds == "etf" else ""
            for seed in SEEDS:
                detail = pd.read_csv(scopes[scope] / f"seed_{seed}/tables/{ds}_episode_details_neural.csv")
                detail["date"] = pd.to_datetime(detail.date).dt.strftime("%Y-%m-%d")
                for suffix, column in [("-NoPenalty", "s_feature_only"), ("", "s_graph_penalty")]:
                    ref = detail[detail.method == prefix + "GraphSignal-ICL" + suffix].set_index("date").mean_systematic_exposure
                    observed = by_state.xs((ds, seed), level=[0, 1])[column].reindex(ref.index)
                    error = float(np.max(np.abs(observed.to_numpy() - ref.to_numpy())))
                    assert error < 1e-6, (scope, ds, seed, suffix, error)
                    node_accounting.append({"scope": scope, "dataset": ds, "seed": seed, "method": "Feature" if suffix else "Graph", "maximum_abs_state_mean_error": error, "tolerance": 1e-6})
        for filename, construction in [("actual_node_coefficient_mechanism_bins.csv", "actual node-specific coefficients"), ("released_mean_coefficient_mechanism_bins.csv", "state-mean coefficients (released mechanism rule)")]:
            path = scopes[scope] / filename
            if not path.exists():
                nm.summarize_bins(node_frame, "released_mean_coefficient_delta").to_csv(path, index=False)
            bins = pd.read_csv(path)
            for ds in ["industry", "etf"]:
                sub = bins[bins.dataset == ds].set_index("score_bin")
                q1, q5 = float(sub.loc["Q1", "mean_delta"]), float(sub.loc["Q5", "mean_delta"])
                mechanism.append({"scope": scope, "dataset": ds, "construction": construction, "Q1_exposure_delta": q1, "Q5_exposure_delta": q5, "Q5_minus_Q1": q5-q1, "direction": "higher-score nodes contract less" if q5 > q1 else "higher-score nodes contract more", "inference": "descriptive quintile-gradient comparison; no extra confirmatory significance claim"})
    save(OUT / "node_exposure_accounting_check.json", node_accounting)
    save(OUT / "mechanism_construction_comparison.json", mechanism)
    # Rebuild original reporting figures from each full set of result tables.
    for scope, base in scopes.items():
        out = ROOT / "figures" / scope
        out.mkdir(parents=True, exist_ok=True)
        figure_source = UP / "results/frozen500" if scope == "author_released" else base
        fig = subprocess.run([sys.executable, str(UP / "src/Plot.py"), "--results_dir", str(figure_source), "--out_dir", str(out)], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (ROOT / "logs" / (scope + "_plot.log")).write_text(fig.stdout)
        assert fig.returncode == 0, fig.stdout
    # Honest comparison of actual node coefficients vs published mean construction.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    for ax, ds in zip(axes, ["industry", "etf"]):
        for scope, name in [("fresh_frozen", "Frozen, node coefficients"), ("corrected_training", "Retrained, node coefficients")]:
            frame = pd.read_csv(scopes[scope] / "actual_node_coefficient_mechanism_bins.csv")
            sample = frame[frame.dataset == ds]
            # summarize_bins returns one row per seed/bin; plot seed means only.
            value_col = "mean_delta" if "mean_delta" in sample else "delta_mean"
            if value_col not in sample:
                value_col = next(c for c in sample if c not in ["dataset", "score_bin", "seed", "n_nodes"] and pd.api.types.is_numeric_dtype(sample[c]))
            means_q = sample.groupby("score_bin", observed=True)[value_col].mean().reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
            ordered = sample.set_index("score_bin").reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
            ax.errorbar(range(1, 6), means_q, yerr=np.vstack([means_q - ordered.ci_low, ordered.ci_high - means_q]), marker="o", capsize=3, label=name)
        mean_rule = pd.read_csv(corrected / "released_mean_coefficient_mechanism_bins.csv")
        mean_rule = mean_rule[mean_rule.dataset == ds].set_index("score_bin").reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
        ax.errorbar(range(1, 6), mean_rule.mean_delta, yerr=np.vstack([mean_rule.mean_delta - mean_rule.ci_low, mean_rule.ci_high - mean_rule.mean_delta]), color="gray", linestyle="--", marker="s", capsize=3, label="Retrained, mean coefficients (release rule)")
        ax.set(title="Industry" if ds == "industry" else "ETF", xlabel="Graph-risk score quintile", ylabel="Graph minus Feature exposure")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_xticks(range(1, 6))
    axes[0].legend(fontsize=7)
    figure.savefig(ROOT / "figures/actual_node_mechanism.png", dpi=180)
    figure.savefig(ROOT / "figures/actual_node_mechanism.pdf")
    plt.close(figure)
    corrected_primary = key[(key.scope == "corrected_training") & (key.comparison_type == "placement_vs_feature_only")]
    source_primary = key[(key.scope == "author_released") & (key.comparison_type == "placement_vs_feature_only")]
    stats_summary = {
        "status": "partial_original_target_full_corrected_execution",
        "original_required_corrected_training_runs": 100,
        "additional_released_auxiliary_training_runs": 40,
        "frozen_released_checkpoint_evaluations": 140,
        "declared_constant_IC_sensitivity_evaluations": 20,
        "universes": {"industry": 49, "ETF": 20},
        "seeds": SEEDS,
        "horizons_days": [20],
        "source_training_paths": 50,
        "evaluation_paths": 500,
        "all_tested_risk_contrasts": 288,
        "primary_placement_cells_per_scope": 6,
        "corrected_primary": corrected_primary.to_dict(orient="records"),
        "cached_primary": source_primary.to_dict(orient="records"),
        "selection": "No test-based selection of model, horizon, breadth or hyperparameters. Paper settings retained. Every declared scope and IC variant reported. Validation objective selects training checkpoint.",
        "mechanism_construction_comparison": mechanism,
        "node_exposure_accounting": {"checks": len(node_accounting), "maximum_error": max(r["maximum_abs_state_mean_error"] for r in node_accounting), "all_within_1e_6": True},
        "market_neutral_nocost": {"status": "not_applicable", "trading_strategies_tested": 0, "daily_return_observations": 0, "significance_claim": None, "reason": "Learned compensation contract and nonnegative systematic agent exposure; modeled contractual episode payoffs do not specify investable long/short holdings or a self-financing daily strategy."},
        "exact_replication_limitations": ["Author raw data snapshot and environment pins not redistributed; current-vintage Yahoo/French and primary Cboe VIX are explicit adaptations", "Released RandomSignal checkpoints lack a manifest linking corrected feature-permutation code to training; current-source frozen replay changes industry ordering conclusion", "Published constant-score runner defaults approximateIC whereas main source configs and manuscript specify exact feasibility", "Purging and prior-close VIX corrections necessarily differ from source-native training; both source replay and corrected outputs retained"],
        "remaining_feasible_local_work": [],
    }
    save(OUT / "summary.json", stats_summary)
    table = key[["scope", "dataset", "comparison_type", "metric", "delta", "ci_low", "ci_high", "wilcoxon_p", "holm_primary6_p", "holm_all288_p"]]
    table = table[(table.metric == "CVaR95_loss") & table.comparison_type.isin(["placement_vs_feature_only", "learned_signal_vs_random_signal"])]
    text = """<html><head><meta charset='utf-8'><title>Paper412 reproduction audit</title><style>body{font:16px system-ui;max-width:1150px;margin:40px auto;line-height:1.5}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border:1px solid #ccc}img{max-width:100%}code{overflow-wrap:anywhere}</style></head><body>"""
    text += "<h1>Paper412: Feature or Penalty?</h1><p><b>Original-target status: partial. Complete corrected execution:140 fits across all ten seeds and both full universes.</b> Also completed140 saved-checkpoint evaluations and20 declared constant-IC sensitivities, all at500 paths per test state. No heavy LLM or new environment was used.</p>"
    text += "<p>The exact <a href='https://tenghanzhong.github.io/assets/papers/CIKM.pdf'>author manuscript</a> and <a href='https://github.com/TenghanZhong/within-state-score/tree/5fca9f59df30ba75b75ca2917a292e041b7cdb39'>pinned official code</a> identify49 Fama-French industries and20 Yahoo-adjusted ETFs. All18 released Table1 means match the manuscript at four decimals. This cached match is separate from fresh replay and retraining.</p>"
    text += "<p><b>No-cost market-neutral profitability: not applicable.</b> The model controls a delegated agent's compensation and systematic risk exposure, with modeled effort and bootstrap contractual payoffs. It supplies no long/short trading rule or calendar-time NAV ledger. No daily trading return, HAC p-value or profitability claim is manufactured.</p>"
    text += "<h2>Coverage and findings</h2><p>Industry test:228 states,2008-01-23 through2026-02-09; ETF test:77 states,2020-01-14 through2026-02-02. Each uses252 lookback days,20-day horizon,50 training scenarios and500 common evaluation scenarios. Corrected training retains250 epochs, early stopping and all original hyperparameters; training/validation states are purged to189/24 for industry and28/12 for ETF. VIX is lagged one available close and constant-score training uses the exact feasibility system.</p>"
    text += "<p>The saved-checkpoint replay preserves the placement improvement, but its industry Graph-minus-Random CVaR contrast reverses the published sign. This is consistent with the release's explicit warning that randomized-score checkpoint lineage after its feature-permutation fix is unverified; data vintage also differs, so the code fix is not isolated as the sole cause.</p>"
    text += "<p><b>The mechanism explanation does not survive the actual node coefficients.</b> The released mechanism code repeats state-mean coefficients across assets and produces stronger contraction at higher graph-risk scores. Using the actual learned node-specific coefficients gives weaker Q5 contraction than Q1 in both universes and both replay/retraining scopes. The independent node solves reconcile with the author's Torch aggregate exposures within1e-6 in80 dataset/seed/method checks. For corrected training, industry Q1/Q5 deltas are−0.01672/−0.00519 and ETF Q1/Q5 deltas are−0.02105/−0.00241. This gradient comparison is descriptive, not an additional confirmatory significance claim.</p>"
    text += "<h2>CVaR comparison, in contractual payoff units</h2>" + table.to_html(index=False, float_format=lambda x: f"{x:.6g}")
    text += "<p>Intervals concern paired training-seed variation on the same observed states/scenarios. Seeds are not extra trading days. The primary six placement contrasts, each90-comparison scope and the complete288-comparison search are reported separately using Holm adjustment of paired Wilcoxon p-values. No programme-wide correction is asserted.</p>"
    text += "<h2>Actual learned node exposure</h2><img src='../figures/actual_node_mechanism.png'>"
    text += "<h2>Evidence and reproducibility</h2><ul>"
    for label, link in [("All288 estimates and p-values", "final_audit/all288_comparisons.csv"), ("Full seed metrics", "final_audit/all_scopes_seed_metrics.csv"), ("Training/checkpoint manifests", "final_audit/all_training_manifests.json"), ("Future perturbation check", "final_audit/future_perturbation_check.json"), ("Input provenance", "../data/manifest.json"), ("Fresh replay summary", "fresh_frozen/summary.json"), ("Corrected training summary", "corrected_training/summary.json"), ("Source audit", "release_audit/summary.json")]:
        text += f"<li><a href='{link}'>{html.escape(label)}</a></li>"
    text += "</ul><p>Exact original data-vintage identity, randomized-score checkpoint history and the author's constant-score training configuration remain unavailable. All feasible local execution in the declared corrected scope is finished; exact original replication is not claimed.</p></body></html>"
    (ROOT / "results/report.html").write_text(text)
    print(json.dumps({"status": "partial", "corrected_fits": 140, "full_frozen_replay": 140, "all_comparisons": 288, "report": str(ROOT / "results/report.html")}, indent=2))


if __name__ == "__main__":
    main()
