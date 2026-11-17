"""
EXP-F: "Single-branch collapsed MLP" baseline on Reddit and questions.

R2-W2: "On Reddit/tolokers, performance drops are blamed on sparse labels &
heterophily, but there's no systematic look at how the scalar-sum
constraint itself limits capacity under low supervision."

We test this directly: replace the entire FraudGNAM scalar-sum architecture
with a single 2-layer MLP that ingests *concatenated* features +
spectrally-filtered features (same input information, no decomposition
constraint).  If the collapsed MLP outperforms FraudGNAM on Reddit, that
would support the reviewer's hypothesis.  If not, the bottleneck is data,
not architecture.
"""

import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def laplacian_powers(graph, x, K):
    """Compute (L_hat/2)^k * x for k = 0..K."""
    with graph.local_scope():
        deg_inv_sqrt = torch.pow(graph.in_degrees().float().clamp(min=1), -0.5).unsqueeze(-1).to(x.device)
        outs = [x]
        h = x
        for _ in range(K):
            graph.ndata["h"] = h * deg_inv_sqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            h = (h - graph.ndata.pop("h") * deg_inv_sqrt) / 2.0
            outs.append(h)
        return outs


class CollapsedMLP(nn.Module):
    """Concatenate raw feats + (L/2)^k feats and feed into a single MLP head."""

    def __init__(self, in_feats, K=2, h_feats=64, dropout=0.1):
        super().__init__()
        self.K = K
        self.in_feats = in_feats
        d_in = in_feats * (K + 1)
        self.net = nn.Sequential(
            nn.Linear(d_in, h_feats), nn.LayerNorm(h_feats),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h_feats, h_feats), nn.LayerNorm(h_feats),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h_feats, 2),
        )

    def forward(self, graph):
        x = graph.ndata["feature"]
        powers = laplacian_powers(graph, x, self.K)
        cat = torch.cat(powers, dim=-1)
        return self.net(cat)


def reck(y_true, y_score, k):
    idx = np.argsort(-y_score)[:k]
    return float(y_true[idx].sum()) / max(1, int(y_true.sum()))


def train_one(model, graph, mask_split, weights, epochs=200, lr=0.01, patience=50):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_val = -1
    best_test = None
    bad = 0

    train_mask = graph.ndata["train_mask"].bool()
    val_mask = graph.ndata["val_mask"].bool()
    test_mask = graph.ndata["test_mask"].bool()
    labels = graph.ndata["label"].long()

    for e in range(epochs):
        model.train()
        logits = model(graph)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=weights)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_val = labels[val_mask].cpu().numpy()
        s_val = probs[val_mask.cpu().numpy()]
        try:
            v_auprc = average_precision_score(y_val, s_val)
        except Exception:
            v_auprc = 0.0

        if v_auprc > best_val:
            best_val = v_auprc; bad = 0
            y_test = labels[test_mask].cpu().numpy()
            s_test = probs[test_mask.cpu().numpy()]
            try:
                auc = roc_auc_score(y_test, s_test)
                ap = average_precision_score(y_test, s_test)
                rk = reck(y_test, s_test, int(y_test.sum()))
            except Exception:
                auc, ap, rk = 0.5, 0.0, 0.0
            best_test = {"AUROC": auc, "AUPRC": ap, "RecK": rk}
        else:
            bad += 1
            if bad > patience:
                break
    return best_test


def run(dataset_name, gpu, trials, K, h_feats, out_json):
    device = f"cuda:{gpu}"
    auc_list, pre_list, rec_list = [], [], []
    for t in range(trials):
        torch.cuda.empty_cache()
        data = Dataset(dataset_name)
        data.split(0, t)
        seed = SEED_LIST[t]
        set_seed(seed)

        graph = data.graph.to(device)
        in_feats = graph.ndata["feature"].shape[1]
        model = CollapsedMLP(in_feats, K=K, h_feats=h_feats).to(device)

        labels = graph.ndata["label"].long()
        train_mask = graph.ndata["train_mask"].bool()
        n_pos = (labels[train_mask] == 1).sum().item()
        n_neg = (labels[train_mask] == 0).sum().item()
        # Class weight inversely proportional to class frequency
        w_pos = max(n_neg / max(1, n_pos), 1.0)
        weights = torch.tensor([1.0, w_pos], device=device, dtype=torch.float)

        score = train_one(model, graph, None, weights)
        if score is None:
            score = {"AUROC": float("nan"), "AUPRC": float("nan"), "RecK": float("nan")}
        auc_list.append(score["AUROC"]); pre_list.append(score["AUPRC"]); rec_list.append(score["RecK"])
        del model, data, graph

    summary = {
        "method": f"CollapsedMLP_K{K}_h{h_feats}",
        "dataset": dataset_name,
        "trials": trials,
        "auc_per_trial": auc_list,
        "pre_per_trial": pre_list,
        "rec_per_trial": rec_list,
        "AUROC_mean": float(np.nanmean(auc_list)),
        "AUROC_std":  float(np.nanstd(auc_list)),
        "AUPRC_mean": float(np.nanmean(pre_list)),
        "AUPRC_std":  float(np.nanstd(pre_list)),
        "RecK_mean":  float(np.nanmean(rec_list)),
        "RecK_std":   float(np.nanstd(rec_list)),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> {dataset_name} K={K} CollapsedMLP "
          f"AUROC {summary['AUROC_mean']:.4f}±{summary['AUROC_std']:.3f}  "
          f"AUPRC {summary['AUPRC_mean']:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--h_feats", type=int, default=64)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.dataset, args.gpu, args.trials, args.K, args.h_feats, args.out_json)
