"""
EXP-2: NCE gradient-flow analysis on Yelp.

Compares the learned distribution of component contributions
|alpha_c * psi_c(v)| across components, with and without NCE contrastive
loss.  Addresses reviewer concerns (R4-W1, R5-W3, R2-W5): the worry is
that NCE may concentrate predictive mass into a single interpretable
branch, effectively making the decomposition misleading.

Output per-component mean |alpha_c * psi_c(v)| on train/test, per-component
|alpha_c|, and the Gini-style concentration index of the contribution mass.
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

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fraudgnam_detector import FraudGNAMDetector
from run_ablation import AblationFraudGNAMDetector, ABLATION_CONFIGS
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def component_contributions(model, graph):
    """Return per-node contributions (N, 6) for:
    [Feature, Pairwise, Band, EFI, HP, HP_EFI] after alpha scaling."""
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        N = x.size(0)
        dev = x.device

        if model.feature_net_grouped is not None:
            tF = model.feature_net_grouped(x)
        else:
            tF = torch.zeros(N, 1, device=dev)
            for i, f in enumerate(model.feature_nets):
                tF = tF + f(x[:, i:i + 1])

        tP = torch.zeros(N, 1, device=dev)
        if model.use_pairwise and model.pairwise_nets is not None:
            for idx, p in enumerate(model.pairwise_nets):
                i, j = model.pair_indices[idx]
                tP = tP + p(x[:, i:i + 1], x[:, j:j + 1])

        Lp = model._compute_laplacian_powers_dgl(graph, x)
        tB = torch.zeros(N, 1, device=dev)
        tE = torch.zeros(N, 1, device=dev)
        bw = F.softmax(model.band_weights, dim=0) * model.num_bands
        for j, sf in enumerate(model.spectral_filters):
            bx = sf(Lp)
            tB = tB + model.band_nets[j](bx) * bw[j]
            if model.use_efi and model.efi_nets is not None:
                tE = tE + model.efi_nets[j](bx, x) * bw[j]

        tH = torch.zeros(N, 1, device=dev)
        tHE = torch.zeros(N, 1, device=dev)
        if model.highpass_filters is not None:
            hw = F.softmax(model.highpass_weights, dim=0) * model.num_highpass
            for k, hpf in enumerate(model.highpass_filters):
                hx = hpf(graph, x)
                tH = tH + model.highpass_nets[k](hx) * hw[k]
                if model.use_efi and model.highpass_efi_nets is not None:
                    tHE = tHE + model.highpass_efi_nets[k](hx, x) * hw[k]

        return torch.stack([
            model.feature_scale * tF.squeeze(-1),
            model.pairwise_scale * tP.squeeze(-1),
            model.band_scale * tB.squeeze(-1),
            model.efi_scale * tE.squeeze(-1),
            (model.highpass_scale * tH.squeeze(-1))
            if model.highpass_filters is not None else torch.zeros(N, device=dev),
            (model.highpass_scale * tHE.squeeze(-1))
            if model.highpass_filters is not None else torch.zeros(N, device=dev),
        ], dim=1).cpu().numpy()


def alpha_vector(model):
    out = {
        "feature_scale": float(model.feature_scale.item()),
        "pairwise_scale": float(model.pairwise_scale.item()),
        "band_scale": float(model.band_scale.item()),
        "efi_scale": float(model.efi_scale.item()),
        "highpass_scale": float(model.highpass_scale.item())
        if model.highpass_filters is not None else 0.0,
    }
    return out


def gini(abs_contributions_mean):
    """Gini concentration of component contributions.
    0 = perfectly equal; 1 = all mass in one component."""
    a = np.sort(np.abs(abs_contributions_mean))
    n = len(a)
    if a.sum() < 1e-10:
        return 0.0
    c = np.cumsum(a)
    return float((n + 1 - 2 * np.sum(c) / c[-1]) / n)


def run_single(dataset_name, use_nce_override, trial, gpu):
    device = f"cuda:{gpu}"
    data = Dataset(dataset_name)
    data.split(0, trial)
    seed = SEED_LIST[trial]
    set_seed(seed)

    train_config = {"device": device, "epochs": 200, "patience": 50,
                    "metric": "AUPRC", "inductive": False, "seed": seed}
    model_config = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    if use_nce_override:
        AblationFraudGNAMDetector._ablation_overrides = {}
        det = AblationFraudGNAMDetector(train_config, model_config, data)
    else:
        AblationFraudGNAMDetector._ablation_overrides = {"use_nce": False}
        det = AblationFraudGNAMDetector(train_config, model_config, data)

    t0 = time.time()
    score = det.train()
    print(f"  [{dataset_name}/use_nce={use_nce_override}] "
          f"AUROC={score['AUROC']:.4f} in {time.time()-t0:.0f}s", flush=True)

    graph = det.source_graph.to(device)
    comps = component_contributions(det.model, graph)     # (N, 6)
    labels = det.labels.cpu().numpy()
    test_mask = graph.ndata["test_mask"].cpu().numpy().astype(bool)

    fraud_mask = (labels == 1) & test_mask
    normal_mask = (labels == 0) & test_mask

    abs_per_component = np.abs(comps)   # (N, 6)
    names = ["Feature", "Pairwise", "Band", "EFI", "HP", "HP-EFI"]

    summary = {
        "AUROC": float(score["AUROC"]),
        "AUPRC": float(score["AUPRC"]),
        "RecK": float(score["RecK"]),
        "alpha": alpha_vector(det.model),
        "mean_abs_contribution_test": {
            n: float(abs_per_component[test_mask, i].mean())
            for i, n in enumerate(names)
        },
        "mean_abs_contribution_fraud": {
            n: float(abs_per_component[fraud_mask, i].mean())
            for i, n in enumerate(names)
        },
        "mean_abs_contribution_normal": {
            n: float(abs_per_component[normal_mask, i].mean())
            for i, n in enumerate(names)
        },
        "gini_test":  gini(np.array([abs_per_component[test_mask, i].mean()   for i in range(6)])),
        "gini_fraud": gini(np.array([abs_per_component[fraud_mask, i].mean()  for i in range(6)])),
    }
    del det, data
    torch.cuda.empty_cache()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="yelp")
    p.add_argument("--gpu", type=int, default=3)
    p.add_argument("--trial", type=int, default=0)
    p.add_argument("--out_json", required=True)
    args = p.parse_args()

    results = {
        "dataset": args.dataset,
        "trial": args.trial,
        "with_NCE": run_single(args.dataset, True, args.trial, args.gpu),
        "without_NCE": run_single(args.dataset, False, args.trial, args.gpu),
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)

    # Print pretty diff
    print("\n===== NCE GRADIENT FLOW (component mean |alpha·psi|) =====")
    print(f"{'Component':<10} {'w/ NCE':>10} {'w/o NCE':>10} {'Δ':>10}")
    for c in ["Feature", "Pairwise", "Band", "EFI", "HP", "HP-EFI"]:
        a = results["with_NCE"]["mean_abs_contribution_test"][c]
        b = results["without_NCE"]["mean_abs_contribution_test"][c]
        print(f"{c:<10} {a:>10.4f} {b:>10.4f} {a-b:>+10.4f}")
    print(f"\nGini (test)   w/NCE={results['with_NCE']['gini_test']:.3f}  "
          f"w/o NCE={results['without_NCE']['gini_test']:.3f}")
    print(f"Gini (fraud)  w/NCE={results['with_NCE']['gini_fraud']:.3f}  "
          f"w/o NCE={results['without_NCE']['gini_fraud']:.3f}")
    print(f"\nAUROC  w/NCE={results['with_NCE']['AUROC']:.4f}  "
          f"w/o NCE={results['without_NCE']['AUROC']:.4f}")
    print(f"AUPRC  w/NCE={results['with_NCE']['AUPRC']:.4f}  "
          f"w/o NCE={results['without_NCE']['AUPRC']:.4f}")
    print(f"\nSaved -> {args.out_json}")
