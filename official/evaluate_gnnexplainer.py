# pyright: reportMissingTypeStubs=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnusedCallResult=false, reportPrivateImportUsage=false, reportAny=false, reportUnannotatedClassAttribute=false, reportImplicitOverride=false, reportMissingParameterType=false, reportCallInDefaultInitializer=false, reportUnnecessaryIsInstance=false, reportImplicitStringConcatenation=false
import argparse
import json
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.explain import GNNExplainer
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
DEFAULT_FraudGNAM_FILES = [
    "./faithfulness_results.json",
    "./faithfulness_results_remaining.json",
]


def set_seed(seed: int = 3407) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def safe_float(value, default: float = float("nan")) -> float:
    if isinstance(value, (int, float, np.floating)):
        return float(value)
    return default


def evaluate_scores(labels: torch.Tensor, probs: torch.Tensor) -> Dict[str, float]:
    labels_np = labels.detach().cpu().numpy()
    probs_np = probs.detach().cpu().numpy()
    if len(np.unique(labels_np)) < 2:
        auroc = 0.5
    else:
        auroc = roc_auc_score(labels_np, probs_np)
    return {
        "AUROC": float(auroc),
        "AUPRC": float(average_precision_score(labels_np, probs_np)),
    }


class ExplainableGCN(nn.Module):
    def __init__(
        self, in_feats: int, hidden_feats: int = 64, num_classes: int = 2
    ) -> None:
        super().__init__()
        self.conv1 = GraphConv(
            in_feats, hidden_feats, norm="both", allow_zero_in_degree=True
        )
        self.conv2 = GraphConv(
            hidden_feats, num_classes, norm="both", allow_zero_in_degree=True
        )

    def forward(
        self, graph, feat: torch.Tensor, eweight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.conv1(graph, feat, edge_weight=eweight)
        h = F.relu(h)
        h = self.conv2(graph, h, edge_weight=eweight)
        return h


def train_gcn(
    graph,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    seed: int,
    epochs: int = 200,
    patience: int = 50,
    lr: float = 0.01,
) -> Tuple[ExplainableGCN, Dict[str, float], Dict[str, float]]:
    set_seed(seed)

    in_feats = graph.ndata["feature"].shape[1]
    model = ExplainableGCN(in_feats=in_feats, hidden_feats=64, num_classes=2).to(
        graph.device
    )

    train_labels = labels[train_mask]
    fraud_ratio = train_labels.float().mean().item()
    if fraud_ratio <= 0:
        class_weight = 1.0
    else:
        class_weight = 1.0 / (2.0 * fraud_ratio)
    ce_weight = torch.tensor([1.0, class_weight], device=graph.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    best_state = None
    best_val_auprc = -1.0
    best_test_scores = {"AUROC": 0.5, "AUPRC": 0.0}
    patience_knt = 0

    features = graph.ndata["feature"]
    for epoch in range(epochs):
        model.train()
        logits = model(graph, features)
        loss = F.cross_entropy(logits[train_mask], train_labels, weight=ce_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            val_scores = evaluate_scores(labels[val_mask], probs[val_mask])
            test_scores = evaluate_scores(labels[test_mask], probs[test_mask])

        if val_scores["AUPRC"] > best_val_auprc:
            best_val_auprc = val_scores["AUPRC"]
            best_test_scores = test_scores
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_knt = 0
            print(
                f"Epoch {epoch:03d}, loss={loss.item():.4f}, "
                f"val_auroc={val_scores['AUROC']:.4f}, val_auprc={val_scores['AUPRC']:.4f}, "
                f"test_auroc={test_scores['AUROC']:.4f}, test_auprc={test_scores['AUPRC']:.4f}"
            )
        else:
            patience_knt += 1
            if patience_knt > patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"AUPRC": float(best_val_auprc)}, best_test_scores


def topk_binary_mask(mask_values: torch.Tensor, ratio: float) -> torch.Tensor:
    num_feats = mask_values.numel()
    k = max(1, int(np.ceil(num_feats * ratio)))
    topk_idx = torch.topk(mask_values, k=k, largest=True).indices
    binary = torch.zeros_like(mask_values)
    binary[topk_idx] = 1.0
    return binary


def explain_fidelity(
    model: ExplainableGCN,
    graph,
    test_nodes: torch.Tensor,
    topk_ratios: List[float],
    explainer_lr: float = 0.01,
    explainer_epochs: int = 100,
) -> Dict[str, object]:
    model.eval()
    features = graph.ndata["feature"]
    with torch.no_grad():
        base_logits = model(graph, features)
        base_probs = torch.softmax(base_logits, dim=1)[:, 1]

    explainer = GNNExplainer(
        model, num_hops=2, lr=explainer_lr, num_epochs=explainer_epochs
    )

    fid_plus = {ratio: [] for ratio in topk_ratios}
    fid_minus = {ratio: [] for ratio in topk_ratios}

    start = time.time()
    for node_id in tqdm(test_nodes.tolist(), desc="Explaining test nodes"):
        node_prob = base_probs[node_id].item()
        _, _, feat_mask, _ = explainer.explain_node(node_id, graph, features)
        feat_mask = feat_mask.detach()

        for ratio in topk_ratios:
            m_top = topk_binary_mask(feat_mask, ratio)

            keep_feats = features.clone()
            keep_feats[node_id] = features[node_id] * m_top
            with torch.no_grad():
                keep_prob = torch.softmax(model(graph, keep_feats), dim=1)[
                    node_id, 1
                ].item()

            remove_feats = features.clone()
            remove_feats[node_id] = features[node_id] * (1.0 - m_top)
            with torch.no_grad():
                remove_prob = torch.softmax(model(graph, remove_feats), dim=1)[
                    node_id, 1
                ].item()

            fid_plus[ratio].append(node_prob - remove_prob)
            fid_minus[ratio].append(node_prob - keep_prob)

    elapsed = time.time() - start
    num_nodes = len(test_nodes)

    def _aggregate(values: List[float]) -> Dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        return {"mean": float(arr.mean()), "std": float(arr.std())}

    ratio_results = {}
    for ratio in topk_ratios:
        ratio_key = f"top_{int(ratio * 100)}"
        plus_stats = _aggregate(fid_plus[ratio])
        minus_stats = _aggregate(fid_minus[ratio])
        ratio_results[ratio_key] = {
            "mean_fid_plus": plus_stats["mean"],
            "std_fid_plus": plus_stats["std"],
            "mean_fid_minus": minus_stats["mean"],
            "std_fid_minus": minus_stats["std"],
        }

    default_key = "top_20"
    if default_key not in ratio_results:
        first_key = list(ratio_results.keys())[0]
        default_key = first_key

    return {
        "num_explained": num_nodes,
        "time_per_node_sec": float(elapsed / max(num_nodes, 1)),
        "topk_fidelity": ratio_results,
        "mean_fid_plus": ratio_results[default_key]["mean_fid_plus"],
        "std_fid_plus": ratio_results[default_key]["std_fid_plus"],
        "mean_fid_minus": ratio_results[default_key]["mean_fid_minus"],
        "std_fid_minus": ratio_results[default_key]["std_fid_minus"],
        "reporting_topk": default_key,
    }


def load_fraudgnam_results(paths: List[str]) -> Dict[str, Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, value in data.items():
            if key == "metadata":
                continue
            merged[key] = value
    return merged


def summarize_fraudgnam(dataset_name: str, fraudgnam_data):
    if dataset_name not in fraudgnam_data:
        return {
            "auroc": None,
            "loco_mae": None,
            "loco_corr": None,
            "component_fid_plus": None,
            "component_fid_minus": None,
        }

    ds = fraudgnam_data[dataset_name]
    if not isinstance(ds, dict):
        return {
            "auroc": None,
            "loco_mae": None,
            "loco_corr": None,
            "component_fid_plus": None,
            "component_fid_minus": None,
        }

    loco = ds.get("loco_faithfulness", {})
    if isinstance(loco, dict) and loco:
        mae_vals = [
            float(v.get("mae", 0.0)) for v in loco.values() if isinstance(v, dict)
        ]
        corr_vals = [
            float(v.get("correlation", 0.0))
            for v in loco.values()
            if isinstance(v, dict)
        ]
        loco_mae = float(np.mean(mae_vals))
        loco_corr = float(np.mean(corr_vals))
    else:
        loco_mae = None
        loco_corr = None

    comprehensiveness = ds.get("comprehensiveness", {})
    sufficiency = ds.get("sufficiency", {})
    comp = (
        comprehensiveness.get("curves", {})
        if isinstance(comprehensiveness, dict)
        else {}
    )
    suff = sufficiency.get("curves", {}) if isinstance(sufficiency, dict) else {}

    if isinstance(comp, dict) and comp:
        first_k_comp = sorted(comp.keys(), key=int)[0]
        comp_fid_plus = float(comp[first_k_comp]["drop"])
    else:
        comp_fid_plus = None

    if (
        isinstance(suff, dict)
        and suff
        and isinstance(sufficiency, dict)
        and "full_auroc" in sufficiency
    ):
        first_k_suff = sorted(suff.keys(), key=int)[0]
        comp_fid_minus = float(sufficiency["full_auroc"] - suff[first_k_suff]["auroc"])
    else:
        comp_fid_minus = None

    test_score = ds.get("test_score", {})
    auroc = (
        float(test_score.get("AUROC", 0.0)) if isinstance(test_score, dict) else None
    )

    return {
        "auroc": auroc,
        "loco_mae": loco_mae,
        "loco_corr": loco_corr,
        "component_fid_plus": comp_fid_plus,
        "component_fid_minus": comp_fid_minus,
    }


def evaluate_dataset(
    dataset_name: str,
    device: str,
    trial_id: int,
    num_explain: int,
    topk_ratios: List[float],
    fraudgnam_data,
):
    seed = SEED_LIST[trial_id]
    set_seed(seed)

    print(f"\n{'=' * 70}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'=' * 70}")

    data = Dataset(dataset_name)
    data.split(semi_supervised=False, trial_id=trial_id)
    graph = data.graph.to(device)

    labels = graph.ndata["label"]
    train_mask = graph.ndata["train_mask"].bool()
    val_mask = graph.ndata["val_mask"].bool()
    test_mask = graph.ndata["test_mask"].bool()

    print("Training 2-layer GCN for GNNExplainer...")
    model, _, gcn_test_score = train_gcn(
        graph=graph,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        seed=seed,
        epochs=200,
        patience=50,
        lr=0.01,
    )

    test_nodes = torch.nonzero(test_mask, as_tuple=False).squeeze(1)
    set_seed(seed)
    if num_explain < test_nodes.numel():
        perm = torch.randperm(test_nodes.numel(), device=test_nodes.device)
        test_nodes = test_nodes[perm[:num_explain]]

    print(f"Running GNNExplainer on {test_nodes.numel()} test nodes...")
    gnnexplainer_result = explain_fidelity(
        model=model,
        graph=graph,
        test_nodes=test_nodes,
        topk_ratios=topk_ratios,
        explainer_lr=0.01,
        explainer_epochs=30,
    )

    fraudgnam_summary = summarize_fraudgnam(dataset_name, fraudgnam_data)

    result = {
        "gcn_auroc": float(gcn_test_score["AUROC"]),
        "gcn_auprc": float(gcn_test_score["AUPRC"]),
        "gnnexplainer": gnnexplainer_result,
        "fraudgnam": fraudgnam_summary,
    }

    del data
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def parse_datasets(dataset_arg: str) -> List[str]:
    if dataset_arg == "all":
        return ALL_DATASETS
    return [d.strip() for d in dataset_arg.split(",") if d.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="amazon,yelp")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--num_explain", type=int, default=200)
    parser.add_argument(
        "--output",
        type=str,
        default="./results/gnnexplainer_vs_fraudgnam.json",
    )
    parser.add_argument(
        "--fraudgnam_files",
        type=str,
        default=",".join(DEFAULT_FraudGNAM_FILES),
        help="Comma-separated paths to precomputed FraudGNAM faithfulness JSON files.",
    )
    args = parser.parse_args()

    if args.trial >= len(SEED_LIST):
        raise ValueError(
            f"trial_id {args.trial} out of range (max {len(SEED_LIST) - 1})"
        )

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    datasets = parse_datasets(args.datasets)
    topk_ratios = [0.10, 0.20, 0.50]

    fraudgnam_paths = [p.strip() for p in args.fraudgnam_files.split(",") if p.strip()]
    fraudgnam_data = load_fraudgnam_results(fraudgnam_paths)

    output = {
        "metadata": {
            "datasets": datasets,
            "trial_id": args.trial,
            "num_explain": int(args.num_explain),
            "topk_percentages": [10, 20, 50],
            "device": device,
            "fraudgnam_sources": fraudgnam_paths,
        }
    }

    for dataset_name in datasets:
        dataset_result = evaluate_dataset(
            dataset_name=dataset_name,
            device=device,
            trial_id=args.trial,
            num_explain=args.num_explain,
            topk_ratios=topk_ratios,
            fraudgnam_data=fraudgnam_data,
        )
        output[dataset_name] = dataset_result

        ds = dataset_result
        gnn = ds.get("gnnexplainer", {}) if isinstance(ds, dict) else {}
        print(f"\n{dataset_name} summary:")
        gcn_auroc = (
            ds.get("gcn_auroc", float("nan")) if isinstance(ds, dict) else float("nan")
        )
        print(f"  GCN AUROC: {gcn_auroc:.4f}")
        if isinstance(gnn, dict):
            print(
                f"  GNNExplainer ({gnn.get('reporting_topk', 'top_20')}): "
                f"Fid+={safe_float(gnn.get('mean_fid_plus')):.4f}"
                f"+-{safe_float(gnn.get('std_fid_plus')):.4f}, "
                f"Fid-={safe_float(gnn.get('mean_fid_minus')):.4f}"
                f"+-{safe_float(gnn.get('std_fid_minus')):.4f}"
            )
        fraudgnam_comp = ds.get("fraudgnam", {}) if isinstance(ds, dict) else {}
        if isinstance(fraudgnam_comp, dict):
            print(
                f"  FraudGNAM: AUROC={fraudgnam_comp.get('auroc')}, "
                f"LOCO_MAE={fraudgnam_comp.get('loco_mae')}, LOCO_CORR={fraudgnam_comp.get('loco_corr')}"
            )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved comparison results to: {args.output}")


if __name__ == "__main__":
    main()
