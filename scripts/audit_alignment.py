"""Measure compilation leaf selection consistency on training rows only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from arboreal.data import load
from arboreal.model import ArborealNetwork

p = argparse.ArgumentParser()
p.add_argument("--root", default="artifacts/benchmark")
p.add_argument("--data-root", default="data")
args = p.parse_args()
root = Path(args.root)
settings = json.loads((root / "suite_config.json").read_text())
torch.set_num_threads(2)
rows = []
for name in settings["datasets"]:
    seed = settings["seeds"][0]
    dest = root / name / f"seed{seed}"
    if not (dest / "xgboost.json").exists():
        raise ValueError(f"Missing baseline checkpoint: {dest}")
    data = load(Path(args.data_root) / "processed" / f"{name}-split{settings['split_seed']}.npz")
    X = data["X_train"][:1024]
    booster = xgb.Booster(model_file=dest / "xgboost.json")
    booster.set_param({"nthread": 2})
    model = ArborealNetwork.from_xgboost(booster, X[:512], tau1=settings["tau1"], tau2=settings["tau2"])
    with torch.no_grad():
        tensor = torch.from_numpy(X)
        selected = model.leaf_probabilities(tensor).argmax(-1).numpy()
        ids = np.full(model.leaf_mask.shape, -1, dtype=np.int64)
        for t, leaf_ids in enumerate(model.leaf_ids):
            ids[t, :len(leaf_ids)] = leaf_ids
        selected_ids = ids[np.arange(len(model.trees))[None, :], selected]
        original = booster.predict(xgb.DMatrix(X), pred_leaf=True).astype(np.int64)
        original = original.reshape(len(X), len(model.trees))
        equal = selected_ids == original
        # Quantify exact equality on active numeric projections. At H=0,
        # tanh(H)=0 for every finite tau, while XGBoost follows the right branch.
        projections = tensor[:, model.features] - model.f
        boundary_count = ((projections == 0) & model.node_mask).sum().item()
    rows.append({"dataset": name, "seed": seed, "sample_split": "train",
                 "sample_n": len(X), "n_trees": len(model.trees),
                 "leaf_argmax_agreement": float(equal.mean()),
                 "all_trees_agree_fraction": float(equal.all(axis=1).mean()),
                 "exact_projection_boundary_fraction": boundary_count / (len(X) * model.node_mask.sum().item()),
                 "tau1": model.tau1, "tau2": model.tau2})
pd.DataFrame(rows).to_csv(root / "routing_alignment.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
