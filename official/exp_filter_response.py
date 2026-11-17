"""
EXP-D: Visualise the learned spectral filter response of FraudGNAM
on tolokers (h=0.604) and yelp (h=0.774), so we can show that the
polynomial bank does NOT achieve a sharp high-pass shape on its
own (motivating the high-pass branch in §4.6).

Output:
    results/filter_response_<dataset>.json
        For each band j, samples (lambda, g_j(lambda)) with
        lambda in [0, 2] over 200 grid points.
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

from models.fraudgnam_detector import FraudGNAMDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def filter_response(spec_filter, lam_grid):
    """Given a _SpectralFilter module and a grid of lambda values in [0, 2],
    return g_j(lambda) = sum_k coeff_k * (lambda/2)^k."""
    coeff = spec_filter.coeffs.detach().cpu().numpy()  # (K+1,)
    half = lam_grid / 2.0
    K = len(coeff) - 1
    response = np.zeros_like(half)
    for k in range(K + 1):
        response += coeff[k] * (half ** k)
    return response


def run(dataset_name, gpu, out_json):
    device = f"cuda:{gpu}"
    data = Dataset(dataset_name)
    data.split(0, 0)
    seed = SEED_LIST[0]
    set_seed(seed)

    train_cfg = {"device": device, "epochs": 200, "patience": 50,
                 "metric": "AUPRC", "inductive": False, "seed": seed}
    model_cfg = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    det = FraudGNAMDetector(train_cfg, model_cfg, data)
    score = det.train()

    model = det.model
    lam_grid = np.linspace(0, 2, 200)

    bands = []
    for j, sf in enumerate(model.spectral_filters):
        bands.append({
            "j": j,
            "coeffs": sf.coeffs.detach().cpu().tolist(),
            "lambda_grid": lam_grid.tolist(),
            "response": filter_response(sf, lam_grid).tolist(),
        })

    # The high-pass branch frequency response is exactly (lambda/2)^t
    # because each iteration applies (I - D^{-1/2} A D^{-1/2}) = L_hat,
    # and we divide by 2 in the polynomial bank, but the HP branch
    # uses unnormalised L_hat without the /2 scaling.  In the spectral
    # domain, L_hat^t corresponds to (lambda)^t.
    hp_responses = []
    if model.highpass_filters is not None:
        for k, hpf in enumerate(model.highpass_filters):
            t = hpf.iters
            hp_responses.append({
                "k": k,
                "iters": int(t),
                "lambda_grid": lam_grid.tolist(),
                "response": (lam_grid ** t).tolist(),
            })

    summary = {
        "dataset": dataset_name,
        "homophily": float(det.homophily),
        "num_bands": int(model.num_bands),
        "num_highpass": int(model.num_highpass) if model.num_highpass else 0,
        "spectral_bands": bands,
        "highpass_branches": hp_responses,
        "trained_AUROC": float(score["AUROC"]),
        "trained_AUPRC": float(score["AUPRC"]),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> {out_json} saved (bands={summary['num_bands']}, "
          f"hp={summary['num_highpass']})", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()
    run(args.dataset, args.gpu, args.out_json)
