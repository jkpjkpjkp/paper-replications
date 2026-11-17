#!/usr/bin/env python3
"""Generate standalone, publication-readable figures from saved runs."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

OUT = ROOT/"figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .18, "savefig.dpi": 180})
COLORS = {"STCO1": "#1769aa", "STCO2": "#c8422b", "UBAH": "#555555",
          "BEST": "#b08900", "BCRP": "#398256"}
NAMES = ["NYSE(O)", "NYSE(N)", "TSE", "MSCI"]
SLUGS = ["nyse-o", "nyse-n", "tse", "msci"]


def save(fig, name):
    fig.savefig(OUT/f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT/f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    metrics = pd.read_csv(ROOT/"results/main/metrics.csv")
    targets = pd.read_csv(ROOT/"sources/paper_targets.csv")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    for ax, name in zip(axes.flat, NAMES):
        d = metrics[metrics.dataset == name]
        for strategy in ["UBAH", "BEST", "BCRP", "STCO1", "STCO2"]:
            sub = d[(d.strategy == strategy) & ((d.window == 0) | (d.window == 4))]
            ax.plot(sub.gamma, sub.wealth, label=strategy+(" (window 4)" if strategy=="STCO2" else ""),
                    color=COLORS[strategy], lw=1.8,
                    linestyle={"UBAH": "--", "BEST": ":", "BCRP": "--",
                               "STCO1": "-", "STCO2": "-."}[strategy])
        sub = d[(d.strategy == "STCO2") & (d.window == 5)]
        ax.plot(sub.gamma, sub.wealth, label="STCO2 (window 5)", color="#82499a", lw=1.8, linestyle=":")
        for s, marker in [("STCO1", "o"), ("STCO2", "s")]:
            paper = targets[(targets.dataset == name) & (targets.strategy == s)]
            ax.scatter(paper.gamma, paper.paper_wealth, marker=marker, facecolors="none",
                       edgecolors=COLORS[s], s=70, zorder=5, label=f"Paper {s}")
        ax.set(yscale="log", title=name, xlabel="Transaction rate", ylabel="Final wealth (log scale)")
        ax.xaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle("STCO: independent reproduction on the four author-supplied datasets", fontsize=16)
    save(fig, "cost_sensitivity")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), layout="constrained")
    for ax, name, slug in zip(axes.flat, NAMES, SLUGS):
        for strategy in ["UBAH", "BCRP", "STCO1", "STCO2"]:
            win = 4 if strategy == "STCO2" else 0
            run = np.load(ROOT/f"results/main/runs/{slug}_{strategy}_g0.0025_w{win}.npz")
            ax.plot(np.arange(len(run["wealth"]))+1, run["wealth"], color=COLORS[strategy],
                    label=strategy, lw=1.5, linestyle={"UBAH": ":", "BCRP": "--",
                                                      "STCO1": "-", "STCO2": "-."}[strategy])
        ax.set(title=name, yscale="log", xlabel="Trading day", ylabel="Cumulative wealth")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4)
    fig.suptitle("Self-financing wealth paths at a 0.25% transaction rate", fontsize=16)
    save(fig, "wealth_paths")

    comparison = pd.read_csv(ROOT/"results/main/paper_comparison.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, name in zip(axes.flat, NAMES):
        labels, values, colors = [], [], []
        for strategy in ["STCO1", "STCO2"]:
            for rate in [0., .0025, .005]:
                r = comparison[(comparison.dataset == name) & (comparison.strategy == strategy)
                               & np.isclose(comparison.gamma, rate)].iloc[0]
                labels.append(f"{strategy} | {rate:.2%}")
                values.append(np.log10(r.wealth_ratio_to_paper))
                colors.append(COLORS[strategy])
        ax.barh(np.arange(len(values)), values, color=colors, height=.65)
        ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(np.arange(len(values)), labels, fontsize=10)
        ax.invert_yaxis()
        ax.set(title=name, xlabel="log10(reproduced wealth / reported wealth)")
    fig.suptitle("Agreement with Table 5 (STCO2 window = 4; zero means exact agreement)", fontsize=14)
    save(fig, "paper_agreement")

    trace = pd.read_csv(ROOT/"results/validation/convergence_trace.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].plot(trace.iteration, trace.objective, color=COLORS["STCO1"])
    axes[0].set(title="DC objective decreases", xlabel="PSGA iteration", ylabel="Equation (8) objective")
    axes[1].semilogy(trace.iteration, np.maximum(trace.distance_to_limit, 1e-15), color=COLORS["STCO2"])
    axes[1].set(title="Distance to the computed limit", xlabel="PSGA iteration", ylabel="Euclidean distance")
    fig.suptitle("Direct PSGA validation case (numerical check, not a rate proof)", fontsize=14)
    save(fig, "convergence")


if __name__ == "__main__":
    main()
