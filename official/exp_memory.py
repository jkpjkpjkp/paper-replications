"""
EXP-E: Per-variant peak GPU memory profiling on Yelp and tolokers.

Addresses R1-W1 (scalability/memory overhead).
"""

import argparse
import json
import os
import random
import sys
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_ablation import AblationFraudGNAMDetector, ABLATION_CONFIGS
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def profile_variant(dataset_name, variant, gpu):
    device = f"cuda:{gpu}"
    torch.cuda.set_device(gpu)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    data = Dataset(dataset_name)
    data.split(0, 0)
    set_seed(SEED_LIST[0])

    AblationFraudGNAMDetector._ablation_overrides = ABLATION_CONFIGS[variant]
    train_cfg = {"device": device, "epochs": 30, "patience": 50,  # short
                 "metric": "AUPRC", "inductive": False, "seed": SEED_LIST[0]}
    model_cfg = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    det = AblationFraudGNAMDetector(train_cfg, model_cfg, data)
    n_params = sum(p.numel() for p in det.model.parameters() if p.requires_grad)
    det.train()

    peak_mb = torch.cuda.max_memory_allocated(gpu) / 1024 / 1024
    del det, data
    torch.cuda.empty_cache()
    return {"peak_MB": float(peak_mb), "n_params": int(n_params)}


def run(dataset_name, gpu, out_json):
    variants = ["full", "no_highpass", "no_nce", "no_efi", "order2", "order4"]
    results = {}
    for v in variants:
        try:
            r = profile_variant(dataset_name, v, gpu)
            print(f"  [{dataset_name} | {v}] params={r['n_params']/1000:.1f}k  "
                  f"peak={r['peak_MB']:.0f} MB", flush=True)
            results[v] = r
        except Exception as e:
            print(f"  [{dataset_name} | {v}] ERROR: {e}", flush=True)
            results[v] = {"peak_MB": None, "n_params": None, "error": str(e)}

    summary = {"dataset": dataset_name, "variants": results}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.dataset, args.gpu, args.out_json)
