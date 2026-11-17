from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from arboreal.data import DATASETS


def report(root):
    root = Path(root)
    results = [json.loads(p.read_text()) for p in sorted(root.glob("*/seed*/result.json"))]
    if not results:
        raise ValueError(f"No completed results in {root}")
    settings = json.loads((root / "suite_config.json").read_text())
    rows = []
    for result in results:
        for method, values in result["test"].items():
            row = {"dataset": result["dataset"], "seed": result["seed"], "method": method,
                   "best_epoch": result["best_epoch"], "epochs_run": result["epochs_run"],
                   "training_seconds": result["training_seconds"],
                   "compile_hard_max_abs_error": result["compile_hard_max_abs_error"],
                   "soft_hard_mean_abs_margin_gap": result["soft_hard_mean_abs_margin_gap"],
                   "saturated_train_node_fraction": result["saturated_train_node_fraction"], **values}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(root / "metrics.csv", index=False)
    summary = []
    for name in settings["datasets"]:
        for method in ("xgboost", "compiled_soft", "arbnn_soft", "decompiled_hard"):
            subset = df[(df.dataset == name) & (df.method == method)]
            if subset.empty:
                continue
            metric = "auc" if DATASETS[name][0].startswith("clf") else "rmse"
            vals = subset[metric].to_numpy()
            summary.append({"dataset": name, "method": method, "metric": metric,
                            "n_seeds": len(vals), "mean": vals.mean(),
                            "std": vals.std(ddof=1) if len(vals) > 1 else float("nan"),
                            "paper": DATASETS[name][1 if method in {"arbnn_soft", "decompiled_hard"} else 2]})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(root / "summary.csv", index=False)
    expected = len(settings["datasets"]) * len(settings["seeds"])
    complete = len(results) == expected
    text = ["# Independent ArbNN reproduction results", "",
            f"Completed **{len(results)}/{expected}** configured dataset/seed runs. "
            f"Status: {'complete' if complete else 'partial'}. CPU/device: `{settings['device']}`.", "",
            "These results implement Appendix Algorithm 1. They do not establish a numerical "
            "reproduction of the paper: author code, exact splits/search settings and TabCredit "
            "were not available. See [PROTOCOL.md](../../PROTOCOL.md).", "",
            "Each value below is mean ± sample standard deviation across independent model seeds "
            "on one fixed train/validation/test split. Classification: AUC, higher is better. "
            "Regression: RMSE on targets standardized using training statistics, lower is better. "
            "Native-unit RMSE is in metrics.csv. Paper columns are reference values, not directly "
            "comparable estimates under an identical protocol.", "",
            "| Dataset | Metric | XGBoost | Initial soft | Trained soft | Decompiled hard | Paper ArbNN |", 
            "|---|---|---:|---:|---:|---:|---:|"]
    wins = 0
    for name in settings["datasets"]:
        subset = summary_df[summary_df.dataset == name].set_index("method")
        if subset.empty:
            continue
        metric = subset.iloc[0]["metric"]
        def fmt(method):
            row = subset.loc[method]
            if row["n_seeds"] == 1:
                return f"{row['mean']:.4f} (1 seed)"
            return f"{row['mean']:.4f} ± {row['std']:.4f}"
        text.append(f"| {name} | {metric} | {fmt('xgboost')} | {fmt('compiled_soft')} | "
                    f"{fmt('arbnn_soft')} | {fmt('decompiled_hard')} | {DATASETS[name][1]:.3f} |")
        delta = subset.loc["arbnn_soft", "mean"] - subset.loc["xgboost", "mean"]
        wins += delta > 0 if metric == "auc" else delta < 0
    n_datasets = len({r["dataset"] for r in results})
    text.extend(["", f"Trained soft ArbNN improves the mean held-out primary metric over XGBoost on "
                 f"**{wins}/{n_datasets}** datasets. This is a descriptive count, not a significance test.", "",
                 f"Maximum absolute margin error for XGBoost → compiled hard tree: "
                 f"**{max(r['compile_hard_max_abs_error'] for r in results):.3g}**.", "",
                 "Parameter/topology round trips preserve rules; finite-temperature soft and hard "
                 "predictions can differ. Per-run mean absolute prediction differences are saved in metrics.csv. "
                 "The explicit stump counterexample and routing-sign audit are in "
                 "[method_audit.json](../method_audit.json).", "",
                 f"The mean fraction of saturated training-node activations (|tanh| > 0.9999) is "
                 f"**{np.mean([r['saturated_train_node_fraction'] for r in results]):.2%}**. "
                 "On native feature scales, τ1=100 often leaves little threshold gradient away from split boundaries.", "",
                 f"Sum of per-run elapsed fine-tuning times: "
                 f"**{sum(r['training_seconds'] for r in results) / 60:.2f} minutes** "
                 "(runs may overlap), excluding baseline tuning, data preparation and compilation.", "",
                 "## Artifacts and validation", "",
                 "Every run includes its config, XGBoost model, trained ArbNN checkpoint, exported if/else "
                 "rules, validation learning curve, and held-out predictions with source row IDs. "
                 "`scripts/verify.py` independently re-scores all saved metrics and replays the first 128 "
                 "held-out rows per run against checkpoints and symbolic rules. "
                 "No test scores are accessed during baseline tuning, epoch selection or early stopping.", "",
                 "## Sources", "",
                 "- [Paper and review discussion](https://openreview.net/forum?id=NDM1ltvBOf)",
                 "- [PyTorch Frame TabularBenchmark loader](https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/datasets/tabular_benchmark.py)",
                 "- [PyTorch Frame Yandex loader](https://github.com/pyg-team/pytorch-frame/blob/master/torch_frame/datasets/yandex.py)", ""])
    fidelity_lines = []
    for task, metric, label in (("classification", "auc", "AUC"), ("regression", "rmse", "standardized RMSE")):
        relevant = [r for r in results if r["task"] == task]
        if relevant:
            differences = [abs(r["test"]["arbnn_soft"][metric] - r["test"]["decompiled_hard"][metric])
                           for r in relevant]
            fidelity_lines.append(f"Maximum absolute soft-versus-decompiled-hard {label} difference: "
                                  f"**{max(differences):.6f}** across {len(relevant)} runs.")
    location = text.index("## Artifacts and validation")
    text[location:location] = fidelity_lines + [""]
    alignment_path = root / "routing_alignment.csv"
    if alignment_path.exists():
        alignment = pd.read_csv(alignment_path)
        location = text.index("## Artifacts and validation")
        text[location:location] = [
            f"Initial soft-routing argmax agrees with XGBoost's leaf on "
            f"**{alignment.leaf_argmax_agreement.min():.2%}–{alignment.leaf_argmax_agreement.max():.2%}** "
            "of tree/sample pairs across datasets (first 1024 training rows, first model seed). "
            "This audit does not tune temperatures or use held-out labels. See routing_alignment.csv. "
            "Exact threshold equality can leave tanh at zero while XGBoost takes its >= branch.", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    make_plot(summary_df, root)
    print(root / "REPORT.md")


def make_plot(summary, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, metric, title in zip(axes, ("auc", "rmse"),
                                 ("Classification (AUC; higher is better)",
                                  "Regression (standardized RMSE; lower is better)")):
        subset = summary[summary.metric == metric]
        names = list(dict.fromkeys(subset.dataset))
        for i, (method, label) in enumerate((("xgboost", "XGBoost"),
                                            ("arbnn_soft", "ArbNN soft"),
                                            ("decompiled_hard", "Decompiled hard"))):
            vals = subset[subset.method == method].set_index("dataset").loc[names]
            ax.errorbar(vals["mean"], np.arange(len(names)) + (i - 1) * 0.17,
                        xerr=vals["std"].fillna(0), fmt="o", capsize=3, ms=4, label=label)
        ax.set_yticks(np.arange(len(names)), names)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11)
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Held-out score (mean ± sample SD)")
    axes[0].legend(loc="lower left", fontsize=8)
    counts = summary.n_seeds.unique()
    seed_label = str(int(counts[0])) if len(counts) == 1 else f"{int(min(counts))}–{int(max(counts))}"
    fig.suptitle(f"Independent ArbNN reproduction · fixed splits · {seed_label} model seeds", fontsize=13)
    fig.savefig(root / "benchmark.png", dpi=180)
    fig.savefig(root / "benchmark.svg")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="artifacts/benchmark")
    report(p.parse_args().root)
