"""
EXP-B: Run FraudGNAM with the refined adaptive rule (Reddit/Weibo/tolokers
refinements) and save full per-trial values for paired-stat analysis.

Per-trial AUROC, AUPRC, RecK are recorded so EXP-3 can replace the simulated
paired tests with real ones.
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
from models.fraudgnam_detector import FraudGNAMDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(dataset_name, gpu, trials, refined, out_json):
    device = f"cuda:{gpu}"
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

        cls = RefinedFraudGNAMDetector if refined else FraudGNAMDetector
        det = cls(train_cfg, model_cfg, data)
        t0 = time.time()
        try:
            score = det.train()
            auc_list.append(float(score["AUROC"]))
            pre_list.append(float(score["AUPRC"]))
            rec_list.append(float(score["RecK"]))
        except Exception as e:
            print(f"  ERROR trial {t}: {type(e).__name__}: {e}", flush=True)
            auc_list.append(float("nan")); pre_list.append(float("nan")); rec_list.append(float("nan"))
        time_list.append(time.time() - t0)
        del det, data

    summary = {
        "variant": "refined" if refined else "submitted",
        "dataset": dataset_name,
        "trials": trials,
        "seeds": [SEED_LIST[t] for t in range(trials)],
        "auc_per_trial": auc_list,
        "pre_per_trial": pre_list,
        "rec_per_trial": rec_list,
        "time_per_trial": time_list,
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
    print(f"\n>>> {summary['variant']} | {dataset_name} | "
          f"AUROC {summary['AUROC_mean']:.4f}±{summary['AUROC_std']:.3f}  "
          f"AUPRC {summary['AUPRC_mean']:.4f}  "
          f"RecK {summary['RecK_mean']:.4f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--refined", action="store_true")
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.dataset, args.gpu, args.trials, args.refined, args.out_json)
