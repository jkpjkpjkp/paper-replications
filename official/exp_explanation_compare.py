"""A16: Side-by-side explanation comparison FraudGNAM vs GNAN on the same Yelp node."""
import argparse
import json
import os
import random
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.fraudgnam_detector import FraudGNAMDetector
from models.gnan_detector import GNANDetector
from utils import Dataset

SEED_LIST = list(range(3407, 10000, 10))


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_fraudgnam_components(det, graph):
    """Decompose Yelp prediction into 6 components."""
    model = det.model
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        N = x.size(0); dev = x.device
        # Feature
        if model.feature_net_grouped is not None:
            tF = model.feature_net_grouped(x)
        else:
            tF = sum(f(x[:, i:i+1]) for i, f in enumerate(model.feature_nets))
        # Pairwise
        tP = torch.zeros(N, 1, device=dev)
        if model.use_pairwise and model.pairwise_nets is not None:
            for idx, p in enumerate(model.pairwise_nets):
                i, j = model.pair_indices[idx]
                tP = tP + p(x[:, i:i+1], x[:, j:j+1])
        # Spectral
        Lp = model._compute_laplacian_powers_dgl(graph, x)
        tB = torch.zeros(N, 1, device=dev); tE = torch.zeros(N, 1, device=dev)
        bw = F.softmax(model.band_weights, dim=0) * model.num_bands
        for j, sf in enumerate(model.spectral_filters):
            bx = sf(Lp)
            tB = tB + model.band_nets[j](bx) * bw[j]
            if model.use_efi: tE = tE + model.efi_nets[j](bx, x) * bw[j]
        # HP
        tH = torch.zeros(N, 1, device=dev); tHE = torch.zeros(N, 1, device=dev)
        if model.highpass_filters is not None:
            hw = F.softmax(model.highpass_weights, dim=0) * model.num_highpass
            for k, hpf in enumerate(model.highpass_filters):
                hx = hpf(graph, x)
                tH = tH + model.highpass_nets[k](hx) * hw[k]
                if model.use_efi: tHE = tHE + model.highpass_efi_nets[k](hx, x) * hw[k]
        return {
            'Feature': (model.feature_scale * tF.squeeze(-1)).cpu().numpy(),
            'Pairwise': (model.pairwise_scale * tP.squeeze(-1)).cpu().numpy(),
            'Band': (model.band_scale * tB.squeeze(-1)).cpu().numpy(),
            'EFI': (model.efi_scale * tE.squeeze(-1)).cpu().numpy(),
            'HP': (model.highpass_scale * tH.squeeze(-1)).cpu().numpy() if model.highpass_filters else np.zeros(N),
            'HP-EFI': (model.highpass_scale * tHE.squeeze(-1)).cpu().numpy() if model.highpass_filters else np.zeros(N),
        }


def get_gnan_contributions(det, graph):
    """Decompose GNAN prediction into per-hop contributions."""
    model = det.model
    model.eval()
    with torch.no_grad():
        x = graph.ndata["feature"]
        feats = [x]
        for _ in range(model.K):
            feats.append(model._aggregate(graph, feats[-1]))
        hop_w = F.softmax(model.hop_weights, dim=0) * (model.K + 1)
        contribs = {}
        for k in range(model.K + 1):
            c = model._compute_feature_contributions(k, feats[k]) * hop_w[k]
            contribs[f'Hop {k}'] = c.detach().cpu().numpy()
        return contribs


def main(out_pdf, gpu, node_idx=None):
    device = f"cuda:{gpu}"
    data1 = Dataset("yelp"); data1.split(0, 0); set_seed(SEED_LIST[0])
    train_cfg = {"device": device, "epochs": 200, "patience": 50,
                 "metric": "AUPRC", "inductive": False, "seed": SEED_LIST[0]}

    print("Training FraudGNAM ...", flush=True)
    det1 = FraudGNAMDetector(train_cfg, {"model": "FraudGNAM", "lr": 0.01, "drop_rate": 0}, data1)
    s1 = det1.train()
    g1 = det1.source_graph.to(device)
    fg_comps = get_fraudgnam_components(det1, g1)
    print(f"FraudGNAM AUROC = {s1['AUROC']:.4f}", flush=True)

    data2 = Dataset("yelp"); data2.split(0, 0); set_seed(SEED_LIST[0])
    print("Training GNAN ...", flush=True)
    det2 = GNANDetector(train_cfg, {"model": "GNAN", "lr": 0.01, "drop_rate": 0.1, "K": 2, "h_feats": 32}, data2)
    s2 = det2.train()
    g2 = det2.source_graph.to(device)
    gnan_contribs = get_gnan_contributions(det2, g2)
    print(f"GNAN AUROC = {s2['AUROC']:.4f}", flush=True)

    labels = data1.graph.ndata['label'].numpy()
    test_mask = data1.graph.ndata['test_mask'].numpy().astype(bool)
    fraud_test_idx = np.where(test_mask & (labels == 1))[0]
    if node_idx is None:
        # Choose a node FraudGNAM correctly classifies as fraud with high confidence
        scores_fg = sum(fg_comps.values())
        cand = [v for v in fraud_test_idx if scores_fg[v] > 3.0]
        node_idx = int(cand[0]) if cand else int(fraud_test_idx[0])
    print(f"Selected node {node_idx} (label={labels[node_idx]})")

    fg_node = {k: float(v[node_idx]) for k, v in fg_comps.items()}
    gnan_node = {k: float(v[node_idx]) for k, v in gnan_contribs.items()}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Left: FraudGNAM
    keys1 = list(fg_node.keys()); vals1 = [fg_node[k] for k in keys1]
    cols1 = ['#e74c3c' if v > 0 else '#3498db' for v in vals1]
    axes[0].barh(keys1, vals1, color=cols1, edgecolor='black')
    axes[0].axvline(0, color='black', linewidth=0.5)
    axes[0].set_title(f"FraudGNAM components\n(node {node_idx}, sum = {sum(vals1):.2f})", fontsize=11)
    axes[0].set_xlabel("Contribution to fraud logit")
    axes[0].grid(alpha=0.3, axis='x')
    for i, v in enumerate(vals1):
        axes[0].text(v, i, f' {v:+.2f}', va='center',
                      ha='left' if v > 0 else 'right', fontsize=9)
    # Right: GNAN
    keys2 = list(gnan_node.keys()); vals2 = [gnan_node[k] for k in keys2]
    cols2 = ['#e74c3c' if v > 0 else '#3498db' for v in vals2]
    axes[1].barh(keys2, vals2, color=cols2, edgecolor='black')
    axes[1].axvline(0, color='black', linewidth=0.5)
    axes[1].set_title(f"GNAN per-hop\n(node {node_idx}, sum = {sum(vals2):.2f})", fontsize=11)
    axes[1].set_xlabel("Contribution to fraud logit")
    axes[1].grid(alpha=0.3, axis='x')
    for i, v in enumerate(vals2):
        axes[1].text(v, i, f' {v:+.2f}', va='center',
                      ha='left' if v > 0 else 'right', fontsize=9)

    fig.suptitle(f"Same Yelp node, two interpretable models — FraudGNAM provides spectral + interaction breakdown that GNAN cannot",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150, bbox_inches='tight')
    print("Saved", out_pdf, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_pdf", required=True)
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--node_idx", type=int, default=None)
    args = p.parse_args()
    main(args.out_pdf, args.gpu, args.node_idx)
