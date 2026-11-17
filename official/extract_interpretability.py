import argparse
import json
import os
import random
import sys
import time
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fraudgnam_detector import FraudGNAMDetector
from utils import Dataset


SEED_LIST = list(range(3407, 10000, 10))
DATASET_ID_TO_NAME = {2: "amazon", 3: "yelp"}


def set_seed(seed=3407):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def clone_spectral_coefficients(model):
    return [
        filt.coeffs.detach().cpu().clone().tolist() for filt in model.spectral_filters
    ]


def extract_shape_functions_for_all_features(model, features, num_points=100):
    if model.feature_nets is None:
        return None
    shape_functions = {}
    model.eval()
    with torch.no_grad():
        for i, net in enumerate(model.feature_nets):
            col = features[:, i]
            x_min = float(col.min().item())
            x_max = float(col.max().item())
            x_range = torch.linspace(
                x_min, x_max, num_points, device=features.device
            ).unsqueeze(1)
            y_range = net(x_range).squeeze(-1)
            shape_functions[str(i)] = {
                "x": x_range.squeeze(-1).detach().cpu().tolist(),
                "y": y_range.detach().cpu().tolist(),
                "min": x_min,
                "max": x_max,
            }
    return shape_functions


def extract_component_importance(model):
    component_importance = {
        "feature_scale": float(model.feature_scale.item()),
        "band_scale": float(model.band_scale.item()),
        "efi_scale": float(model.efi_scale.item()),
        "pairwise_scale": float(model.pairwise_scale.item()),
        "highpass_scale": float(model.highpass_scale.item())
        if model.highpass_filters is not None
        else 0.0,
        "band_weights": model.band_weights.detach().cpu().tolist(),
        "band_weights_normalized": (
            F.softmax(model.band_weights, dim=0) * model.num_bands
        )
        .detach()
        .cpu()
        .tolist(),
        "output_bias": float(model.output_bias.item()),
        "temperature": float(model.temperature.item()),
    }
    if model.highpass_filters is not None:
        component_importance["highpass_weights"] = (
            model.highpass_weights.detach().cpu().tolist()
        )
    return component_importance


def extract_pairwise_interactions_top3(model, features, grid_points=50):
    if model.pairwise_nets is None or len(model.pairwise_nets) == 0:
        return None
    interactions = {}
    model.eval()
    top_k = min(3, len(model.pairwise_nets))
    with torch.no_grad():
        for idx in range(top_k):
            i_feat, j_feat = model.pair_indices[idx]
            i_feat = int(i_feat.item())
            j_feat = int(j_feat.item())
            x_i = features[:, i_feat]
            x_j = features[:, j_feat]
            i_grid = torch.linspace(
                float(x_i.min().item()),
                float(x_i.max().item()),
                grid_points,
                device=features.device,
            )
            j_grid = torch.linspace(
                float(x_j.min().item()),
                float(x_j.max().item()),
                grid_points,
                device=features.device,
            )
            mesh_i, mesh_j = torch.meshgrid(i_grid, j_grid, indexing="ij")
            pair_out = model.pairwise_nets[idx](
                mesh_i.reshape(-1, 1), mesh_j.reshape(-1, 1)
            ).reshape(grid_points, grid_points)
            interactions[str(idx)] = {
                "pair_index": [i_feat, j_feat],
                "x_i": i_grid.detach().cpu().tolist(),
                "x_j": j_grid.detach().cpu().tolist(),
                "f_ij": pair_out.detach().cpu().tolist(),
            }
    return interactions


def compute_node_components(model, graph):
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        num_nodes = x.size(0)
        device = x.device

        if model.feature_net_grouped is not None:
            feature_total_raw = model.feature_net_grouped(x).squeeze(-1)
        else:
            feature_total_raw = torch.zeros(num_nodes, device=device)
            for i, f_net in enumerate(model.feature_nets):
                feature_total_raw = feature_total_raw + f_net(x[:, i : i + 1]).squeeze(
                    -1
                )

        pairwise_total_raw = torch.zeros(num_nodes, device=device)
        if model.use_pairwise and model.pairwise_nets is not None:
            for idx, p_net in enumerate(model.pairwise_nets):
                i_feat, j_feat = model.pair_indices[idx]
                pairwise_total_raw = pairwise_total_raw + p_net(
                    x[:, i_feat : i_feat + 1], x[:, j_feat : j_feat + 1]
                ).squeeze(-1)

        L_powers = model._compute_laplacian_powers_dgl(graph, x)
        band_weights = F.softmax(model.band_weights, dim=0) * model.num_bands

        per_band_raw = []
        per_efi_raw = []
        for j, spec_filter in enumerate(model.spectral_filters):
            band_x = spec_filter(L_powers)
            per_band_raw.append(
                model.band_nets[j](band_x).squeeze(-1) * band_weights[j]
            )
            if model.use_efi and model.efi_nets is not None:
                per_efi_raw.append(
                    model.efi_nets[j](band_x, x).squeeze(-1) * band_weights[j]
                )

        per_highpass_raw = []
        per_highpass_efi_raw = []
        if model.highpass_filters is not None:
            hp_weights = F.softmax(model.highpass_weights, dim=0) * model.num_highpass
            for k, hp_filter in enumerate(model.highpass_filters):
                hp_x = hp_filter(graph, x)
                per_highpass_raw.append(
                    model.highpass_nets[k](hp_x).squeeze(-1) * hp_weights[k]
                )
                if model.use_efi and model.highpass_efi_nets is not None:
                    per_highpass_efi_raw.append(
                        model.highpass_efi_nets[k](hp_x, x).squeeze(-1) * hp_weights[k]
                    )

        feature_total = model.feature_scale * feature_total_raw
        pairwise_total = model.pairwise_scale * pairwise_total_raw
        per_band = [model.band_scale * comp for comp in per_band_raw]
        per_efi = [model.efi_scale * comp for comp in per_efi_raw]
        per_highpass = [model.highpass_scale * comp for comp in per_highpass_raw]
        per_highpass_efi = [
            model.highpass_scale * comp for comp in per_highpass_efi_raw
        ]

        aggregate = feature_total + pairwise_total
        for comp in per_band:
            aggregate = aggregate + comp
        for comp in per_efi:
            aggregate = aggregate + comp
        for comp in per_highpass:
            aggregate = aggregate + comp
        for comp in per_highpass_efi:
            aggregate = aggregate + comp
        aggregate = aggregate + model.output_bias

        temp = model.temperature.clamp(min=0.1)
        aggregate_scaled = aggregate / temp
        fraud_prob = torch.sigmoid(aggregate_scaled)

        return {
            "feature_total": feature_total.detach().cpu().numpy(),
            "pairwise_total": pairwise_total.detach().cpu().numpy(),
            "bands": [comp.detach().cpu().numpy() for comp in per_band],
            "efi": [comp.detach().cpu().numpy() for comp in per_efi],
            "highpass": [comp.detach().cpu().numpy() for comp in per_highpass],
            "highpass_efi": [comp.detach().cpu().numpy() for comp in per_highpass_efi],
            "output_bias": float(model.output_bias.item()),
            "temperature": float(model.temperature.item()),
            "fraud_logit_pre_temperature": aggregate.detach().cpu().numpy(),
            "fraud_logit": aggregate_scaled.detach().cpu().numpy(),
            "fraud_probability": fraud_prob.detach().cpu().numpy(),
        }


def build_node_decomposition(
    model, graph, labels, test_mask, max_per_class=100, seed=3407
):
    contributions = compute_node_components(model, graph)
    labels_np = labels.detach().cpu().numpy()
    test_indices = np.where(test_mask.detach().cpu().numpy())[0]
    fraud_nodes = test_indices[labels_np[test_indices] == 1]
    normal_nodes = test_indices[labels_np[test_indices] == 0]

    rng = np.random.RandomState(seed)
    fraud_pick = rng.choice(
        fraud_nodes, size=min(max_per_class, len(fraud_nodes)), replace=False
    )
    normal_pick = rng.choice(
        normal_nodes, size=min(max_per_class, len(normal_nodes)), replace=False
    )

    feature_total_arr = cast(Any, contributions["feature_total"])
    pairwise_total_arr = cast(Any, contributions["pairwise_total"])
    bands_arr = cast(Any, contributions["bands"])
    efi_arr = cast(Any, contributions["efi"])
    highpass_arr = cast(Any, contributions["highpass"])
    highpass_efi_arr = cast(Any, contributions["highpass_efi"])
    pretemp_arr = cast(Any, contributions["fraud_logit_pre_temperature"])
    fraud_logit_arr = cast(Any, contributions["fraud_logit"])
    fraud_prob_arr = cast(Any, contributions["fraud_probability"])
    output_bias = float(cast(Any, contributions["output_bias"]))
    temperature = float(cast(Any, contributions["temperature"]))

    def pack_node(node_id):
        return {
            "node_id": int(node_id),
            "label": int(labels_np[node_id]),
            "feature_total": float(feature_total_arr[node_id]),
            "pairwise_total": float(pairwise_total_arr[node_id]),
            "bands": [float(comp[node_id]) for comp in bands_arr],
            "efi": [float(comp[node_id]) for comp in efi_arr],
            "highpass": [float(comp[node_id]) for comp in highpass_arr],
            "highpass_efi": [float(comp[node_id]) for comp in highpass_efi_arr],
            "output_bias": output_bias,
            "temperature": temperature,
            "fraud_logit_pre_temperature": float(pretemp_arr[node_id]),
            "fraud_logit": float(fraud_logit_arr[node_id]),
            "fraud_probability": float(fraud_prob_arr[node_id]),
        }

    return {
        "fraud_count": int(len(fraud_pick)),
        "normal_count": int(len(normal_pick)),
        "fraud_nodes": [pack_node(idx) for idx in fraud_pick],
        "normal_nodes": [pack_node(idx) for idx in normal_pick],
    }


def train_and_extract(dataset_id, trial_id=0):
    dataset_name = DATASET_ID_TO_NAME[dataset_id]
    print(f"\n{'=' * 70}")
    print(f"Dataset ID {dataset_id}: {dataset_name}")
    print(f"{'=' * 70}")

    seed = SEED_LIST[trial_id]
    set_seed(seed)
    print(f"[1/6] Loading dataset '{dataset_name}' with Dataset class")
    data = Dataset(dataset_name)
    data.split(semi_supervised=False, trial_id=trial_id)

    train_config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epochs": 200,
        "patience": 50,
        "metric": "AUPRC",
        "inductive": False,
        "seed": seed,
    }
    model_config = {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}

    print("[2/6] Initializing FraudGNAM detector")
    detector = FraudGNAMDetector(train_config, model_config, data)
    model = detector.model

    print("[3/6] Saving initial spectral coefficients (before training)")
    spectral_before = clone_spectral_coefficients(model)

    print("[4/6] Training FraudGNAM (epochs=200, patience=50)")
    start = time.time()
    test_score = detector.train()
    train_sec = time.time() - start
    print(f"Training finished in {train_sec:.1f}s, test score: {test_score}")

    print("[5/6] Extracting interpretability artifacts")
    graph = detector.source_graph
    features = graph.ndata["feature"]
    labels = detector.labels
    test_mask = detector.test_mask

    spectral_after = clone_spectral_coefficients(model)
    component_importance = extract_component_importance(model)
    node_decomposition = build_node_decomposition(
        model,
        graph,
        labels,
        test_mask,
        max_per_class=100,
        seed=seed + dataset_id,
    )

    shape_functions = None
    pairwise_interactions = None
    pair_indices = None

    if dataset_name == "amazon":
        shape_functions = extract_shape_functions_for_all_features(
            model, features, num_points=100
        )
        pairwise_interactions = extract_pairwise_interactions_top3(
            model, features, grid_points=50
        )
        if model.pairwise_nets is not None:
            pair_indices = [
                [int(i), int(j)] for i, j in model.pair_indices.detach().cpu().tolist()
            ]

    print("[6/6] Packaging extracted dataset payload")
    payload = {
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "seed": seed,
        "train_config": {
            "epochs": train_config["epochs"],
            "patience": train_config["patience"],
            "metric": train_config["metric"],
            "inductive": train_config["inductive"],
        },
        "test_score": {k: float(v) for k, v in test_score.items()},
        "model_meta": {
            "in_feats": int(model.in_feats),
            "order": int(model.order),
            "num_bands": int(model.num_bands),
            "use_pairwise": bool(model.use_pairwise),
            "use_efi": bool(model.use_efi),
            "num_highpass": int(model.num_highpass),
        },
        "training_seconds": float(train_sec),
        "component_importance": component_importance,
        "spectral_coefficients": {
            "before_training": spectral_before,
            "after_training": spectral_after,
        },
        "node_decomposition": node_decomposition,
    }
    if shape_functions is not None:
        payload["shape_functions"] = shape_functions
    if pairwise_interactions is not None:
        payload["pairwise_interactions"] = pairwise_interactions
    if pair_indices is not None:
        payload["pair_indices"] = pair_indices

    del detector, data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument(
        "--output",
        type=str,
        default="./interp_data.json",
    )
    args = parser.parse_args()

    print("Starting FraudGNAM interpretability extraction")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device index: {torch.cuda.current_device()}")

    all_data = {
        "metadata": {
            "datasets": [2, 3],
            "dataset_names": ["amazon", "yelp"],
            "trial_id": int(args.trial),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "amazon": train_and_extract(dataset_id=2, trial_id=args.trial),
        "yelp": train_and_extract(dataset_id=3, trial_id=args.trial),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved interpretability JSON to: {args.output}")
    print("Done.")


if __name__ == "__main__":
    main()
