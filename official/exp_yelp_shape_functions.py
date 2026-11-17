"""A15: Yelp per-feature shape functions visualization."""
import argparse
import json
import os
import random
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fraudgnam_detector import FraudGNAMDetector
from extract_interpretability import (
    extract_shape_functions_for_all_features,
    set_seed,
)
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def main(out_pdf):
    device = "cuda:0"
    data = Dataset("yelp")
    data.split(0, 0)
    set_seed(SEED_LIST[0])
    train_cfg = {"device": device, "epochs": 200, "patience": 50,
                 "metric": "AUPRC", "inductive": False, "seed": SEED_LIST[0]}
    model_cfg = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}
    det = FraudGNAMDetector(train_cfg, model_cfg, data)
    score = det.train()
    print(f"Yelp trained AUROC = {score['AUROC']:.4f}")

    feats = data.graph.ndata["feature"].to(device)
    shapes = extract_shape_functions_for_all_features(det.model, feats, num_points=100)
    if shapes is None:
        print("Yelp uses grouped MLP — skipping per-feature shapes")
        return

    # Select top-12 most-influential features by output range
    ranges = [(idx, max(s["y"]) - min(s["y"])) for idx, s in shapes.items()]
    ranges.sort(key=lambda x: -x[1])
    top12 = [idx for idx, _ in ranges[:12]]
    print("Top-12 features by shape-function range:", top12)

    fig, axes = plt.subplots(3, 4, figsize=(13, 8))
    for ax, idx in zip(axes.flat, top12):
        s = shapes[idx]
        ax.plot(s["x"], s["y"], color="#2c3e50", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.fill_between(s["x"], 0, s["y"], where=np.array(s["y"]) > 0,
                         alpha=0.2, color="#e74c3c", label="↑ fraud")
        ax.fill_between(s["x"], 0, s["y"], where=np.array(s["y"]) < 0,
                         alpha=0.2, color="#3498db", label="↓ fraud")
        ax.set_title(f"Feature {idx}", fontsize=10)
        ax.set_xlabel("$x_i$", fontsize=8)
        ax.set_ylabel("$f_i(x_i)$", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Yelp per-feature NAM shape functions (top-12 by range; AUROC = {score['AUROC']:.3f})",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    print("Saved", out_pdf)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_pdf", required=True)
    args = p.parse_args()
    main(args.out_pdf)
