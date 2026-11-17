"""
Generate the filter-response plot for the rebuttal: shows the learned
spectral polynomial filter bank vs the ideal high-pass branch response,
illustrating that the polynomial bank does NOT achieve a sharp boundary
at lambda -> 2 on its own (motivating R4-C4).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = Path("results")
FIG_DIR = Path("rebuttal_figures")
FIG_DIR.mkdir(exist_ok=True)


def plot_dataset(ds):
    p = RESULT_DIR / f"filter_response_{ds}.json"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return
    with open(p) as f:
        d = json.load(f)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    cmap = plt.get_cmap("viridis")

    # Polynomial bands
    K = d["num_bands"]
    for j, b in enumerate(d["spectral_bands"]):
        lam = np.array(b["lambda_grid"])
        r = np.array(b["response"])
        # Normalize for visualization (each band is independently scaled in the model)
        ax.plot(lam, r / (np.max(np.abs(r)) + 1e-12),
                color=cmap(j / max(K - 1, 1)),
                linewidth=2.0,
                label=f"Polynomial band $B_{{{j}}}$")

    # High-pass branches
    for k, h in enumerate(d["highpass_branches"]):
        lam = np.array(h["lambda_grid"])
        r = np.array(h["response"]) / (2 ** h["iters"])  # normalise to [0,1]
        ax.plot(lam, r,
                color="crimson",
                linestyle="--",
                linewidth=2.5,
                label=f"High-pass branch $H_{{{k+1}}}$ (iters={h['iters']})")

    ax.axvspan(1.5, 2.0, color="orange", alpha=0.10,
               label="Heterophilic frequency band ($\\lambda > 1.5$)")
    ax.set_xlabel("Eigenvalue $\\lambda$")
    ax.set_ylabel("Filter response (normalised)")
    ax.set_title(f"Learned filter responses on {ds} (h={d['homophily']:.3f})")
    ax.set_xlim(0, 2)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"filter_response_{ds}.pdf"
    fig.savefig(out)
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=160)
    plt.close(fig)
    print(f"  [saved] {out}")


if __name__ == "__main__":
    for ds in ["yelp", "tolokers", "weibo"]:
        plot_dataset(ds)
