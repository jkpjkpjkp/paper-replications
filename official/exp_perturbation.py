"""
EXP-7: Perturbation stability of FraudGNAM's component-level decomposition.

Trains a FraudGNAM model (single seed/split) on a dataset, then for each
test node measures the Spearman rank correlation between the original
component contribution vector [alpha_c * psi_c(v)]_c and the vector
obtained after adding Gaussian noise N(0, sigma^2) to the input features.

Reports mean/median rho across test nodes for sigma in {0.01, 0.05, 0.10}.
Addresses rebuttal point R4-W2 and R2-comment (stability of explanations).
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
import torch.nn.functional as F
from scipy.stats import spearmanr

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


def decompose(model, graph):
    """
    Return per-node component contribution vector:
    [alpha_F*F, alpha_P*P, alpha_B*Band, alpha_E*EFI, alpha_H*HP, alpha_H*HP_EFI]
    where each entry is (num_nodes,).  If a component is not active, it is 0.
    """
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        num_nodes = x.size(0)
        device = x.device

        if model.feature_net_grouped is not None:
            total_feature = model.feature_net_grouped(x)
        else:
            total_feature = torch.zeros(num_nodes, 1, device=device)
            for i, f_net in enumerate(model.feature_nets):
                total_feature = total_feature + f_net(x[:, i : i + 1])

        total_pairwise = torch.zeros(num_nodes, 1, device=device)
        if model.use_pairwise and model.pairwise_nets is not None:
            for idx, p_net in enumerate(model.pairwise_nets):
                i, j = model.pair_indices[idx]
                total_pairwise = total_pairwise + p_net(
                    x[:, i : i + 1], x[:, j : j + 1]
                )

        L_powers = model._compute_laplacian_powers_dgl(graph, x)
        total_band = torch.zeros(num_nodes, 1, device=device)
        total_efi = torch.zeros(num_nodes, 1, device=device)
        band_weights = F.softmax(model.band_weights, dim=0) * model.num_bands
        for j, spec_filter in enumerate(model.spectral_filters):
            band_x = spec_filter(L_powers)
            total_band = total_band + model.band_nets[j](band_x) * band_weights[j]
            if model.use_efi and model.efi_nets is not None:
                total_efi = total_efi + model.efi_nets[j](band_x, x) * band_weights[j]

        total_hp = torch.zeros(num_nodes, 1, device=device)
        total_hp_efi = torch.zeros(num_nodes, 1, device=device)
        if model.highpass_filters is not None:
            hp_w = F.softmax(model.highpass_weights, dim=0) * model.num_highpass
            for k, hp_filter in enumerate(model.highpass_filters):
                hp_x = hp_filter(graph, x)
                total_hp = total_hp + model.highpass_nets[k](hp_x) * hp_w[k]
                if model.use_efi and model.highpass_efi_nets is not None:
                    total_hp_efi = total_hp_efi + model.highpass_efi_nets[k](hp_x, x) * hp_w[k]

        comps = torch.stack(
            [
                model.feature_scale * total_feature.squeeze(-1),
                model.pairwise_scale * total_pairwise.squeeze(-1),
                model.band_scale * total_band.squeeze(-1),
                model.efi_scale * total_efi.squeeze(-1),
                (model.highpass_scale * total_hp.squeeze(-1))
                if model.highpass_filters is not None
                else torch.zeros(num_nodes, device=device),
                (model.highpass_scale * total_hp_efi.squeeze(-1))
                if model.highpass_filters is not None
                else torch.zeros(num_nodes, device=device),
            ],
            dim=1,
        )  # (N, 6)

        return comps.detach().cpu().numpy()


def run(dataset_name, gpu, sigma_list, num_eval_nodes, trial, out_json):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"

    data = Dataset(dataset_name)
    data.split(0, trial)
    seed = SEED_LIST[trial]
    set_seed(seed)

    train_config = {
        "device": device, "epochs": 200, "patience": 50,
        "metric": "AUPRC", "inductive": False, "seed": seed,
    }
    model_config = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    print(f"Training FraudGNAM on {dataset_name} (trial={trial}) ...", flush=True)
    t0 = time.time()
    detector = FraudGNAMDetector(train_config, model_config, data)
    score = detector.train()
    print(
        f"  trained in {time.time()-t0:.1f}s | "
        f"AUROC={score['AUROC']:.4f} AUPRC={score['AUPRC']:.4f}",
        flush=True,
    )

    model = detector.model
    graph = detector.source_graph.to(device)
    original_features = graph.ndata["feature"].clone()
    test_mask = graph.ndata["test_mask"]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0].cpu().numpy()
    if len(test_idx) > num_eval_nodes:
        rng = np.random.default_rng(seed)
        test_idx = rng.choice(test_idx, size=num_eval_nodes, replace=False)
    test_idx = sorted(test_idx.tolist())

    # Original component contributions
    comps_orig = decompose(model, graph)  # (N, 6)

    results_per_sigma = {}
    for sigma in sigma_list:
        rng = np.random.default_rng(seed + 999)
        rhos = []
        top_match = []   # does argmax component match?
        # Use 3 noise draws for robust estimate
        for draw in range(3):
            noise = torch.tensor(
                rng.normal(0, sigma, size=original_features.shape),
                device=device, dtype=original_features.dtype,
            )
            graph.ndata["feature"] = original_features + noise
            comps_pert = decompose(model, graph)
            for v in test_idx:
                orig_v = comps_orig[v]
                pert_v = comps_pert[v]
                # Filter zero (inactive) components to avoid degenerate ties
                active = (np.abs(orig_v) + np.abs(pert_v)) > 1e-10
                if active.sum() < 2:
                    continue
                r, _ = spearmanr(orig_v[active], pert_v[active])
                if not np.isnan(r):
                    rhos.append(float(r))
                top_match.append(
                    int(np.argmax(np.abs(orig_v)) == np.argmax(np.abs(pert_v)))
                )

        graph.ndata["feature"] = original_features  # restore

        results_per_sigma[str(sigma)] = {
            "rho_mean": float(np.mean(rhos)) if rhos else None,
            "rho_median": float(np.median(rhos)) if rhos else None,
            "rho_std": float(np.std(rhos)) if rhos else None,
            "rho_q10": float(np.quantile(rhos, 0.10)) if rhos else None,
            "rho_q90": float(np.quantile(rhos, 0.90)) if rhos else None,
            "top_component_match_rate": float(np.mean(top_match)) if top_match else None,
            "n_nodes_evaluated": len(test_idx),
            "n_noise_draws": 3,
        }
        print(
            f"  sigma={sigma}: rho_mean={results_per_sigma[str(sigma)]['rho_mean']:.4f} "
            f"median={results_per_sigma[str(sigma)]['rho_median']:.4f} "
            f"top_match={results_per_sigma[str(sigma)]['top_component_match_rate']:.4f}",
            flush=True,
        )

    summary = {
        "dataset": dataset_name,
        "trial": trial,
        "num_eval_nodes": len(test_idx),
        "sigma_list": sigma_list,
        "per_sigma": results_per_sigma,
        "model_AUROC": float(score["AUROC"]),
        "model_AUPRC": float(score["AUPRC"]),
    }
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n>>> Saved {out_json}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--num_eval_nodes", type=int, default=200)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    run(
        args.dataset,
        args.gpu,
        sigma_list=[0.01, 0.05, 0.10],
        num_eval_nodes=args.num_eval_nodes,
        trial=args.trial,
        out_json=args.out_json,
    )
