"""
Rebuttal ablation wrapper: runs one (variant, dataset) pair and saves to a
unique JSON/xlsx path, avoiding save-ID collisions when 4 GPUs run in parallel.

Usage:
    python rebuttal_run.py --variant no_highpass --dataset tolokers --gpu 0 --trials 10 \
        --out_json results/tolokers_no_highpass.json
"""
import argparse
import json
import os
import random
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from run_ablation import (
    ABLATION_CONFIGS,
    AblationFraudGNAMDetector,
    DATASET_MAP,
    seed_list,
    set_seed,
)
from utils import Dataset


def run_one(variant, dataset_name, gpu, trials, out_json):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    overrides = ABLATION_CONFIGS[variant]
    AblationFraudGNAMDetector._ablation_overrides = overrides

    auc_list, pre_list, rec_list, times = [], [], [], []
    errors = 0

    for t in range(trials):
        torch.cuda.empty_cache()
        print(f"=== {dataset_name} | {variant} | trial {t} (gpu={gpu}) ===", flush=True)

        data = Dataset(dataset_name)
        data.split(0, t)
        seed = seed_list[t]
        set_seed(seed)

        train_config = {
            "device": device,
            "epochs": 200,
            "patience": 50,
            "metric": "AUPRC",
            "inductive": False,
            "seed": seed,
        }
        model_config = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

        try:
            detector = AblationFraudGNAMDetector(train_config, model_config, data)
            t0 = time.time()
            score = detector.train()
            dt = time.time() - t0
            auc_list.append(float(score["AUROC"]))
            pre_list.append(float(score["AUPRC"]))
            rec_list.append(float(score["RecK"]))
            times.append(dt)
        except Exception as e:
            print(f"  ERROR trial {t}: {type(e).__name__}: {e}", flush=True)
            errors += 1
            # record sentinel to keep alignment
            auc_list.append(float("nan"))
            pre_list.append(float("nan"))
            rec_list.append(float("nan"))
            times.append(float("nan"))
        finally:
            try:
                del detector
            except Exception:
                pass
            del data

    summary = {
        "variant": variant,
        "dataset": dataset_name,
        "trials": trials,
        "errors": errors,
        "auc_list": auc_list,
        "pre_list": pre_list,
        "rec_list": rec_list,
        "time_list": times,
        "AUROC_mean": float(np.nanmean(auc_list)),
        "AUROC_std": float(np.nanstd(auc_list)),
        "AUPRC_mean": float(np.nanmean(pre_list)),
        "AUPRC_std": float(np.nanstd(pre_list)),
        "RecK_mean": float(np.nanmean(rec_list)),
        "RecK_std": float(np.nanstd(rec_list)),
        "time_mean": float(np.nanmean(times)),
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"\n>>> SAVED {out_json}\n"
        f"AUROC {summary['AUROC_mean']:.4f} ± {summary['AUROC_std']:.4f} | "
        f"AUPRC {summary['AUPRC_mean']:.4f} ± {summary['AUPRC_std']:.4f} | "
        f"RecK {summary['RecK_mean']:.4f} ± {summary['RecK_std']:.4f} | "
        f"t_avg {summary['time_mean']:.1f}s | errors={errors}",
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True,
                   choices=list(ABLATION_CONFIGS.keys()))
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    run_one(args.variant, args.dataset, args.gpu, args.trials, args.out_json)
