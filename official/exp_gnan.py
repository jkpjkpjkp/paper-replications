"""
GNAN baseline runner. Saves per-trial AUROC/AUPRC/RecK on matched seeds
so it can be paired-tested against FraudGNAM.
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

from models.gnan_detector import GNANDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run(dataset_name, gpu, trials, K, h_feats, out_json):
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
        model_cfg = {"model": "GNAN", "lr": 0.01, "drop_rate": 0.1,
                     "K": K, "h_feats": h_feats}

        det = GNANDetector(train_cfg, model_cfg, data)
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
        "method": f"GNAN_K{K}_h{h_feats}",
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
    print(f">>> GNAN {dataset_name} K={K} "
          f"AUROC {summary['AUROC_mean']:.4f}±{summary['AUROC_std']:.3f}  "
          f"AUPRC {summary['AUPRC_mean']:.4f}±{summary['AUPRC_std']:.3f}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--K", type=int, default=2)
    p.add_argument("--h_feats", type=int, default=32)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.dataset, args.gpu, args.trials, args.K, args.h_feats, args.out_json)
