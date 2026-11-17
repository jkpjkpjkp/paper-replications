"""
Threshold sensitivity grid (R2-W10 / R2-comment).

Re-runs the FULL refined-rule FraudGNAM under perturbed cutoff values:

  * tau_order in {0.80, 0.85, 0.90}      (homophily->order rule, original)
  * tau_nce   in {0.90, 0.95, 1.00}      (homophily->NCE/HP rule, original)
  * n_f_R1    in {100, 200, 300}         (NEW R1 sparse-fraud rule)
  * h_R3      in {0.65, 0.70, 0.75}      (NEW R3 extreme-heterophily rule)

Only datasets that *change configuration* under a given perturbation are
re-run; otherwise the original refined-rule numbers are reused.
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

from models.fraudgnam_detector import FraudGNAMDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class TunableFraudGNAMDetector(FraudGNAMDetector):
    """FraudGNAM detector with explicitly overridable thresholds."""
    _thresholds = {
        "tau_order": 0.85,
        "tau_nce": 0.95,
        "n_f_R1": 200,
        "h_R3": 0.70,
    }

    def __init__(self, train_config, model_config, data):
        # Call parent to compute homophily and base config
        super().__init__(train_config, model_config, data)

        graph = self.source_graph
        labels_np = self.labels.cpu().numpy()
        avg_degree = graph.num_edges() / max(graph.num_nodes(), 1)
        in_feats = graph.ndata["feature"].shape[1]
        homophily = self.homophily
        train_fraud = int(labels_np[self.train_mask.cpu().numpy()].sum())

        T = TunableFraudGNAMDetector._thresholds
        # Re-derive order with custom thresholds
        if homophily < T["tau_order"]:
            new_order = 4
        elif avg_degree < 30:
            new_order = 3
        else:
            new_order = 2
        # NEW R1
        if train_fraud < T["n_f_R1"]:
            new_order = 2
        # R2 (Weibo) is only triggered by static h>0.97 ∧ d≥300; not perturbed here
        if homophily > 0.97 and in_feats >= 300:
            new_order = 4

        # Re-derive HP with custom tau_nce
        if homophily < T["tau_order"]:
            new_hp = 2
        elif homophily < T["tau_nce"]:
            new_hp = 1
        else:
            new_hp = 0
        # NEW R3 with tunable h_R3
        if homophily < T["h_R3"]:
            new_hp = 0

        # NCE rule: shares tau_nce
        new_nce = (train_fraud >= 500) and (homophily < T["tau_nce"])

        old_order = model_config.get("order", 2)
        old_hp = model_config.get("num_highpass", 0)
        old_nce = model_config.get("use_nce", False)
        if (new_order != old_order) or (new_hp != old_hp) or (new_nce != old_nce):
            model_config["order"] = new_order
            model_config["num_highpass"] = new_hp
            model_config["use_nce"] = new_nce
            self.model = type(self.model)(**model_config).to(
                train_config["device"]
            )
            if model_config.get("use_pairwise", False):
                tf = graph.ndata["feature"][self.train_mask]
                tl = self.labels[self.train_mask]
                self.model.set_important_pairs(tf, tl)


def run(dataset_name, gpu, trials, thresholds, out_json):
    device = f"cuda:{gpu}"
    TunableFraudGNAMDetector._thresholds = thresholds

    auc_list, pre_list, rec_list, time_list = [], [], [], []
    for t in range(trials):
        torch.cuda.empty_cache()
        data = Dataset(dataset_name)
        data.split(0, t)
        seed = SEED_LIST[t]
        set_seed(seed)

        train_cfg = {"device": device, "epochs": 200, "patience": 50,
                     "metric": "AUPRC", "inductive": False, "seed": seed}
        model_cfg = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

        det = TunableFraudGNAMDetector(train_cfg, model_cfg, data)
        t0 = time.time()
        try:
            score = det.train()
            auc_list.append(float(score["AUROC"]))
            pre_list.append(float(score["AUPRC"]))
            rec_list.append(float(score["RecK"]))
        except Exception as e:
            print(f"  ERROR trial {t}: {type(e).__name__}: {e}", flush=True)
            auc_list.append(float("nan"))
            pre_list.append(float("nan"))
            rec_list.append(float("nan"))
        time_list.append(time.time() - t0)
        del det, data

    summary = {
        "thresholds": thresholds,
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
    print(f">>> [{dataset_name} | {thresholds}] AUROC "
          f"{summary['AUROC_mean']:.4f}±{summary['AUROC_std']:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--tau_order", type=float, default=0.85)
    p.add_argument("--tau_nce", type=float, default=0.95)
    p.add_argument("--n_f_R1", type=int, default=200)
    p.add_argument("--h_R3", type=float, default=0.70)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    th = {"tau_order": args.tau_order, "tau_nce": args.tau_nce,
          "n_f_R1": args.n_f_R1, "h_R3": args.h_R3}
    run(args.dataset, args.gpu, args.trials, th, args.out_json)
