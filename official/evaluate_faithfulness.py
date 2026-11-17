"""
Faithfulness evaluation for FraudGNAM's inherent interpretability.

Metrics:
1. LOCO Faithfulness: For scalar-sum architecture, removing component c_k
   should change logit by exactly c_k. Measures exact decomposability.
2. Top-k Sufficiency: AUROC retained using only top-k components.
3. Comprehensiveness: AUROC drop when removing top-k components.

Usage:
    python evaluate_faithfulness.py --datasets amazon,yelp --gpu 0
    python evaluate_faithfulness.py --datasets all --gpu 0
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fraudgnam_detector import FraudGNAMDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))
ALL_DATASETS = [
    "amazon",
    "yelp",
    "weibo",
    "reddit",
    "tfinance",
    "questions",
    "tolokers",
]


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_component_vectors(model, graph):
    """Extract per-component contribution vectors (scaled, pre-temperature).

    Returns dict mapping component name -> (N,) tensor of per-node contributions.
    The full logit (pre-temperature) = sum of all components + output_bias.
    """
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        num_nodes = x.size(0)
        device = x.device
        components = {}

        # 1. Feature
        if model.feature_net_grouped is not None:
            raw = model.feature_net_grouped(x).squeeze(-1)
        else:
            raw = torch.zeros(num_nodes, device=device)
            for i, f_net in enumerate(model.feature_nets):
                raw = raw + f_net(x[:, i : i + 1]).squeeze(-1)
        components["feature"] = model.feature_scale * raw

        # 2. Pairwise
        raw_pw = torch.zeros(num_nodes, device=device)
        if model.use_pairwise and model.pairwise_nets is not None:
            for idx, p_net in enumerate(model.pairwise_nets):
                i, j = model.pair_indices[idx]
                raw_pw = raw_pw + p_net(x[:, i : i + 1], x[:, j : j + 1]).squeeze(-1)
        components["pairwise"] = model.pairwise_scale * raw_pw

        # 3. Spectral bands + EFI
        L_powers = model._compute_laplacian_powers_dgl(graph, x)
        bw = F.softmax(model.band_weights, dim=0) * model.num_bands

        total_band = torch.zeros(num_nodes, device=device)
        total_efi = torch.zeros(num_nodes, device=device)
        for j, sf in enumerate(model.spectral_filters):
            band_x = sf(L_powers)
            total_band = total_band + model.band_nets[j](band_x).squeeze(-1) * bw[j]
            if model.use_efi and model.efi_nets is not None:
                total_efi = total_efi + model.efi_nets[j](band_x, x).squeeze(-1) * bw[j]
        components["band"] = model.band_scale * total_band
        components["efi"] = model.efi_scale * total_efi

        # 4. High-pass + HP-EFI
        total_hp = torch.zeros(num_nodes, device=device)
        total_hp_efi = torch.zeros(num_nodes, device=device)
        if model.highpass_filters is not None:
            hw = F.softmax(model.highpass_weights, dim=0) * model.num_highpass
            for k, hf in enumerate(model.highpass_filters):
                hp_x = hf(graph, x)
                total_hp = total_hp + model.highpass_nets[k](hp_x).squeeze(-1) * hw[k]
                if model.use_efi and model.highpass_efi_nets is not None:
                    total_hp_efi = (
                        total_hp_efi
                        + model.highpass_efi_nets[k](hp_x, x).squeeze(-1) * hw[k]
                    )
        hp_scale = getattr(model, "highpass_scale", None)
        if hp_scale is not None:
            components["highpass"] = hp_scale * (total_hp + total_hp_efi)
        else:
            components["highpass"] = total_hp + total_hp_efi

        # Bias
        components["_bias"] = torch.full(
            (num_nodes,), model.output_bias.item(), device=device
        )
        components["_temperature"] = model.temperature.clamp(min=0.1)

    return components


def logit_from_components(components, exclude=None, only=None):
    """Reconstruct fraud logit from component dict.

    exclude: set of component names to zero out
    only: set of component names to keep (others zeroed)
    """
    temp = components["_temperature"]
    total = components["_bias"].clone()
    for name, vals in components.items():
        if name.startswith("_"):
            continue
        if exclude and name in exclude:
            continue
        if only is not None and name not in only:
            continue
        total = total + vals
    return total / temp


def auroc_from_logit(logit, labels, mask):
    """Compute AUROC from fraud logit (higher = more fraud)."""
    probs = torch.sigmoid(logit[mask]).cpu().numpy()
    y = labels[mask].cpu().numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    return roc_auc_score(y, probs)


# ──────────────────────────────────────────────────────────────────────
# Metric 1: LOCO Faithfulness
# ──────────────────────────────────────────────────────────────────────
def loco_faithfulness(components, test_mask):
    """Measure exact decomposability of scalar-sum architecture.

    For each component c_k:
      predicted_delta = c_k[i]  (the component's contribution)
      actual_delta = logit_full[i] - logit_{-k}[i]

    For a perfect scalar-sum, predicted_delta == actual_delta exactly.
    We report:
      - Mean absolute error (should be ~0 for scalar-sum)
      - Pearson correlation (should be ~1.0)
    """
    full_logit = logit_from_components(components)
    results = {}
    active_components = [
        n
        for n in components
        if not n.startswith("_") and components[n].abs().sum() > 1e-8
    ]

    for name in active_components:
        ablated_logit = logit_from_components(components, exclude={name})
        predicted_delta = components[name][test_mask]
        temp = components["_temperature"]
        predicted_delta_scaled = predicted_delta / temp
        actual_delta = (full_logit - ablated_logit)[test_mask]
        mae = (predicted_delta_scaled - actual_delta).abs().mean().item()
        # Pearson correlation
        pd_np = predicted_delta_scaled.cpu().numpy()
        ad_np = actual_delta.cpu().numpy()
        if pd_np.std() > 1e-10 and ad_np.std() > 1e-10:
            corr = float(np.corrcoef(pd_np, ad_np)[0, 1])
        else:
            corr = 1.0  # constant → perfectly correlated
        results[name] = {"mae": mae, "correlation": corr}

    return results


# ──────────────────────────────────────────────────────────────────────
# Metric 2: Top-k Component Sufficiency
# ──────────────────────────────────────────────────────────────────────
def topk_sufficiency(components, labels, test_mask):
    """AUROC using only top-k components ranked by contribution magnitude.

    For each node, rank components by |c_k(x_i)|.
    Keep only top-k components, zero others.
    Measure AUROC on test set.
    """
    full_logit = logit_from_components(components)
    full_auroc = auroc_from_logit(full_logit, labels, test_mask)

    active_names = [
        n
        for n in components
        if not n.startswith("_") and components[n].abs().sum() > 1e-8
    ]
    num_active = len(active_names)

    # Stack component values: (N, C)
    comp_stack = torch.stack([components[n] for n in active_names], dim=1)
    # Per-node component magnitudes
    comp_mags = comp_stack.abs()

    results = {"full_auroc": full_auroc, "num_active": num_active, "curves": {}}

    for k in range(1, num_active + 1):
        # For each node, keep only top-k components
        _, topk_idx = comp_mags.topk(k, dim=1)
        mask_k = torch.zeros_like(comp_stack, dtype=torch.bool)
        mask_k.scatter_(1, topk_idx, True)
        partial = comp_stack * mask_k.float()

        # Reconstruct logit
        temp = components["_temperature"]
        partial_logit = (components["_bias"] + partial.sum(dim=1)) / temp
        k_auroc = auroc_from_logit(partial_logit, labels, test_mask)
        results["curves"][k] = {
            "auroc": k_auroc,
            "retention": k_auroc / full_auroc if full_auroc > 0 else 0.0,
        }

    return results


# ──────────────────────────────────────────────────────────────────────
# Metric 3: Top-k Comprehensiveness
# ──────────────────────────────────────────────────────────────────────
def topk_comprehensiveness(components, labels, test_mask):
    """AUROC after removing top-k most important components.

    Higher drop = explanation is more comprehensive (captures the signal).
    """
    full_logit = logit_from_components(components)
    full_auroc = auroc_from_logit(full_logit, labels, test_mask)

    active_names = [
        n
        for n in components
        if not n.startswith("_") and components[n].abs().sum() > 1e-8
    ]
    num_active = len(active_names)

    comp_stack = torch.stack([components[n] for n in active_names], dim=1)
    comp_mags = comp_stack.abs()

    results = {"full_auroc": full_auroc, "curves": {}}

    for k in range(1, num_active + 1):
        _, topk_idx = comp_mags.topk(k, dim=1)
        keep_mask = torch.ones_like(comp_stack, dtype=torch.bool)
        keep_mask.scatter_(1, topk_idx, False)
        remaining = comp_stack * keep_mask.float()

        temp = components["_temperature"]
        rem_logit = (components["_bias"] + remaining.sum(dim=1)) / temp
        rem_auroc = auroc_from_logit(rem_logit, labels, test_mask)
        results["curves"][k] = {
            "auroc": rem_auroc,
            "drop": full_auroc - rem_auroc,
        }

    return results


# ──────────────────────────────────────────────────────────────────────
# Main driver
# ──────────────────────────────────────────────────────────────────────
def evaluate_dataset(dataset_name, device, trial_id=0):
    seed = SEED_LIST[trial_id]
    set_seed(seed)
    print(f"\n{'=' * 60}")
    print(f"  Evaluating faithfulness: {dataset_name} (trial {trial_id})")
    print(f"{'=' * 60}")

    data = Dataset(dataset_name)
    data.split(semi_supervised=False, trial_id=trial_id)

    train_config = {
        "device": device,
        "epochs": 200,
        "patience": 50,
        "metric": "AUPRC",
        "inductive": False,
        "seed": seed,
    }
    model_config = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    print("  Training FraudGNAM...")
    detector = FraudGNAMDetector(train_config, model_config, data)
    t0 = time.time()
    test_score = detector.train()
    train_sec = time.time() - t0
    print(f"  Training done in {train_sec:.1f}s  AUROC={test_score['AUROC']:.3f}")

    model = detector.model
    graph = detector.source_graph
    labels = detector.labels
    test_mask = detector.test_mask

    print("  Computing component vectors...")
    components = compute_component_vectors(model, graph)

    active = [
        n
        for n in components
        if not n.startswith("_") and components[n].abs().sum() > 1e-8
    ]
    print(f"  Active components: {active}")

    print("  LOCO faithfulness...")
    loco = loco_faithfulness(components, test_mask)

    print("  Top-k sufficiency...")
    suff = topk_sufficiency(components, labels, test_mask)

    print("  Top-k comprehensiveness...")
    comp = topk_comprehensiveness(components, labels, test_mask)

    result = {
        "dataset": dataset_name,
        "trial_id": trial_id,
        "test_score": {k: float(v) for k, v in test_score.items()},
        "active_components": active,
        "loco_faithfulness": loco,
        "sufficiency": suff,
        "comprehensiveness": comp,
    }

    del detector, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="amazon,yelp")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="./faithfulness_results.json",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.datasets == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(",")]

    all_results = {"metadata": {"datasets": datasets, "trial_id": args.trial}}

    for ds in datasets:
        result = evaluate_dataset(ds, device, trial_id=args.trial)
        all_results[ds] = result

        # Print summary
        print(f"\n  --- {ds} Summary ---")
        print(f"  AUROC: {result['test_score']['AUROC']:.3f}")
        print(f"  LOCO Faithfulness (MAE / Corr):")
        for comp_name, vals in result["loco_faithfulness"].items():
            print(
                f"    {comp_name:12s}: MAE={vals['mae']:.6f}  corr={vals['correlation']:.4f}"
            )
        print(f"  Top-k Sufficiency (AUROC retention):")
        for k in sorted(result["sufficiency"]["curves"].keys()):
            info = result["sufficiency"]["curves"][k]
            print(f"    k={k}: AUROC={info['auroc']:.3f} ({info['retention']:.1%})")
        print(f"  Top-k Comprehensiveness (AUROC drop):")
        for k in sorted(result["comprehensiveness"]["curves"].keys()):
            info = result["comprehensiveness"]["curves"][k]
            print(f"    k={k}: AUROC={info['auroc']:.3f} (drop={info['drop']:+.3f})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
