"""A17: Label-noise robustness on Yelp.
Flip a fraction of training labels and measure test AUROC degradation.
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

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from refined_detector import RefinedFraudGNAMDetector
from models.secgfd_detector import SECGFDDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def corrupt_labels(graph, train_mask, flip_rate, seed):
    """Flip flip_rate fraction of train labels (preserves test/val labels)."""
    rng = np.random.default_rng(seed)
    labels = graph.ndata["label"].clone()
    train_idx = torch.nonzero(train_mask, as_tuple=True)[0].cpu().numpy()
    n_flip = int(len(train_idx) * flip_rate)
    flip_idx = rng.choice(train_idx, size=n_flip, replace=False)
    for v in flip_idx:
        labels[v] = 1 - labels[v]
    graph.ndata["label"] = labels
    return graph, n_flip


def run(method, dataset_name, gpu, trials, flip_rates, out_json):
    device = f"cuda:{gpu}"
    results = {}
    for fr in flip_rates:
        auc_list = []
        for t in range(trials):
            torch.cuda.empty_cache()
            data = Dataset(dataset_name)
            data.split(0, t)
            seed = SEED_LIST[t]
            set_seed(seed)

            train_mask = data.graph.ndata["train_mask"]
            data.graph, n_flip = corrupt_labels(data.graph, train_mask, fr, seed)

            train_cfg = {"device": device, "epochs": 200, "patience": 50,
                         "metric": "AUPRC", "inductive": False, "seed": seed}
            model_cfg = {"model": method, "lr": 0.01, "drop_rate": 0}

            try:
                if method == "FraudGNAM":
                    det = RefinedFraudGNAMDetector(train_cfg, model_cfg, data)
                else:
                    det = SECGFDDetector(train_cfg, model_cfg, data)
                score = det.train()
                auc_list.append(float(score["AUROC"]))
                print(f"  {method} | flip={fr*100:.0f}% | trial {t}: "
                      f"flipped {n_flip} labels, AUROC={score['AUROC']:.4f}",
                      flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                auc_list.append(float("nan"))
            finally:
                try: del det
                except: pass
                del data
        results[str(fr)] = {
            "auc_list": auc_list,
            "AUROC_mean": float(np.nanmean(auc_list)),
            "AUROC_std": float(np.nanstd(auc_list)),
        }

    summary = {"method": method, "dataset": dataset_name,
               "flip_rates": flip_rates, "results": results}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> {method} ({dataset_name}) saved to {out_json}", flush=True)
    for fr, r in results.items():
        print(f"  flip={float(fr)*100:.0f}%: AUROC {r['AUROC_mean']:.4f}±{r['AUROC_std']:.3f}",
              flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=["FraudGNAM", "SECGFD"])
    p.add_argument("--dataset", default="yelp")
    p.add_argument("--gpu", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)  # 5 trials × 4 noise rates
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.method, args.dataset, args.gpu, args.trials,
        flip_rates=[0.0, 0.05, 0.10, 0.20], out_json=args.out_json)
