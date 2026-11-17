"""
FraudGNAM Detector for GADBench integration.
INherent Spectral Interaction Graph neTwork - Inherently Interpretable GNN.
"""

import torch
import torch.nn.functional as F
import numpy as np
from models.detector import BaseDetector
from models.gnn import FraudGNAM


def _fraudgnam_nce_loss(nce_emb, original_features, labels, train_mask):
    """NCE contrastive loss: push normal embeddings toward original features,
    pull fraud embeddings away. Modeled after SEC-GFD's NCE loss."""
    train_emb = nce_emb[train_mask]
    train_feat = original_features[train_mask]
    train_labels = labels[train_mask]

    # Cosine similarity between embeddings and original features
    sim = F.cosine_similarity(train_emb, train_feat, dim=-1)

    # Normal nodes: high similarity (positive pairs)
    normal_mask = train_labels == 0
    # Fraud nodes: low similarity (negative pairs)
    fraud_mask = train_labels == 1

    # NCE-style: log(exp(sim_pos) / (exp(sim_pos) + sum(exp(sim_neg))))
    pos_sim = sim[normal_mask]
    neg_sim = sim[fraud_mask]

    if pos_sim.numel() == 0 or neg_sim.numel() == 0:
        return torch.tensor(0.0, device=nce_emb.device)

    # For each positive, contrast against mean negative
    pos_exp = torch.exp(pos_sim).clamp(min=1e-8)
    neg_exp_sum = torch.exp(neg_sim).sum().clamp(min=1e-8)

    loss = -torch.log(
        pos_exp / (pos_exp + neg_exp_sum / max(neg_sim.numel(), 1))
    ).mean()
    return loss


class FraudGNAMDetector(BaseDetector):
    """
    FraudGNAM detector with adaptive configuration based on graph properties.
    Uses focal loss and auto-configures based on dataset characteristics.
    Configuration is driven by graph structure (degree, homophily),
    feature dimensionality, and fraud ratio.
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        in_feats = self.data.graph.ndata["feature"].shape[1]
        model_config["in_feats"] = in_feats

        # Analyze graph properties for adaptive config
        graph = self.source_graph
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        avg_degree = num_edges / num_nodes

        labels_np = self.labels.cpu().numpy()
        fraud_ratio = labels_np.mean()

        # Compute homophily (edge homophily ratio)
        src, dst = graph.edges()
        same_label = labels_np[src.cpu().numpy()] == labels_np[dst.cpu().numpy()]
        homophily = same_label.mean() if len(same_label) > 0 else 0.5
        self.homophily = homophily

        # Classification thresholds
        high_homophily = homophily > 0.9
        sparse_graph = avg_degree < 30  # Reddit(15.3), Questions(~20) are sparse
        high_dim = in_feats > 100  # Weibo(400), Questions(301), Elliptic(166)
        rare_fraud = fraud_ratio < 0.05  # Reddit(3.3%), Questions(3.0%)

        # Training fraud count (used for multiple decisions)
        train_fraud_count = labels_np[self.train_mask.cpu().numpy()].sum()

        # --- Spectral order ---
        # Heterophilic datasets get order 4 for finer spectral resolution
        if homophily < 0.85:
            model_config["order"] = 4
        elif sparse_graph:
            model_config["order"] = 3
        else:
            model_config["order"] = 2
        model_config["self_loop_weight"] = 1.0

        # --- Model capacity (h_feats) ---
        # Scale capacity with available training signal (fraud nodes)
        if train_fraud_count < 500:
            # Very limited fraud signal: keep model small to avoid overfitting
            model_config["h_feats"] = 32
        else:
            # Sufficient fraud signal: use larger capacity
            model_config["h_feats"] = 64

        # --- Feature networks ---
        # Group for mid-to-high dim features to control overparameterization
        if in_feats > 100:
            use_grouped = True
            model_config["feature_group_size"] = 8 if in_feats > 200 else 5
        elif in_feats > 40:
            # Mid-dim (e.g., Reddit 64): group with larger group size
            use_grouped = True
            model_config["feature_group_size"] = 16
        else:
            use_grouped = False
            model_config["feature_group_size"] = 0
        model_config["use_grouped_features"] = use_grouped

        # Feature hidden dims for individual feature nets (low-dim only)
        if in_feats <= 15:
            model_config["feature_hidden"] = [64, 64, 32]
        else:
            model_config["feature_hidden"] = [64, 32]

        # --- Pairwise ---
        if in_feats <= 30 and not sparse_graph:
            model_config["use_pairwise"] = True
            model_config["num_pairwise"] = 15
        else:
            model_config["use_pairwise"] = False
            model_config["num_pairwise"] = 0

        # --- EFI ---
        # Disable for very small fraud sets where extra parameters hurt
        if train_fraud_count < 500:
            model_config["use_efi"] = False
        else:
            model_config["use_efi"] = True

        # --- High-pass spectral branch ---
        # Enable for datasets with meaningful heterophilic edges (homophily <= 0.95)
        if homophily < 0.85:
            model_config["num_highpass"] = 2
        elif homophily < 0.95:
            model_config["num_highpass"] = 1
        else:
            model_config["num_highpass"] = 0

        # --- NCE contrastive loss ---
        # Enable when sufficient fraud signal AND heterophilic edges present
        if train_fraud_count >= 500 and homophily < 0.95:
            model_config["use_nce"] = True
            if fraud_ratio < 0.05:
                self.nce_lambda = 0.05
            else:
                self.nce_lambda = 0.1
        else:
            model_config["use_nce"] = False
            self.nce_lambda = 0.0

        # --- Concat+MLP integration ---
        # Disabled: breaks inherent interpretability (component contributions not decomposable)
        model_config["use_concat_integration"] = False

        # --- Dropout ---
        if high_dim and high_homophily:
            model_config["drop_rate"] = 0.2
        elif model_config.get("drop_rate", 0) == 0:
            model_config["drop_rate"] = 0.1

        # --- Focal loss ---
        # Very small fraud sets: focal loss causes overfitting to noise, use standard CE
        if train_fraud_count < 500:
            self.focal_gamma = 0.0
        elif fraud_ratio < 0.05:
            self.focal_gamma = 2.0
        elif fraud_ratio < 0.1:
            self.focal_gamma = 1.5
        else:
            self.focal_gamma = 1.0

        # --- Grad clip ---
        # Tighter clip when NCE is active to prevent instability
        self.grad_clip = 1.5 if model_config["use_nce"] else 2.0

        # --- LR warmup ---
        # When highpass+NCE are both active, warmup to prevent early instability
        self.warmup_epochs = (
            10 if (model_config["num_highpass"] > 0 and model_config["use_nce"]) else 0
        )

        print(
            f"[FraudGNAM] in_feats={in_feats}, avg_degree={avg_degree:.1f}, "
            f"homophily={homophily:.3f}, fraud_ratio={fraud_ratio:.2%}, "
            f"train_fraud={train_fraud_count:.0f}"
        )
        print(
            f"  h_feats={model_config['h_feats']}, grouped={use_grouped}, "
            f"efi={model_config['use_efi']}, order={model_config['order']}, "
            f"pairwise={model_config['use_pairwise']}, focal_gamma={self.focal_gamma}"
        )
        print(
            f"  highpass={model_config['num_highpass']}, nce={model_config['use_nce']}"
            f" (lambda={self.nce_lambda}), concat={model_config['use_concat_integration']}, "
            f"grad_clip={self.grad_clip}, warmup={self.warmup_epochs}"
        )

        self.model = FraudGNAM(**model_config).to(train_config["device"])

        # Select feature pairs by label correlation (paper §3.3, §A.5)
        if model_config.get("use_pairwise", False):
            train_features = graph.ndata["feature"][self.train_mask]
            train_labels_for_pairs = self.labels[self.train_mask]
            self.model.set_important_pairs(train_features, train_labels_for_pairs)

    def train(self):
        base_lr = self.model_config["lr"]
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=base_lr, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.train_config["epochs"] - self.warmup_epochs,
            eta_min=base_lr * 0.01,
        )

        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        use_nce = self.model_config.get("use_nce", False)
        original_features = self.train_graph.ndata["feature"] if use_nce else None

        for e in range(self.train_config["epochs"]):
            # LR warmup: linear ramp from base_lr*0.1 to base_lr
            if self.warmup_epochs > 0 and e < self.warmup_epochs:
                warmup_factor = 0.1 + 0.9 * (e / self.warmup_epochs)
                for pg in optimizer.param_groups:
                    pg["lr"] = base_lr * warmup_factor

            self.model.train()
            output = self.model(self.train_graph)

            # Handle dual return (logits, nce_emb) when NCE is active during training
            if use_nce and isinstance(output, tuple):
                logits, nce_emb = output
            else:
                logits = output
                nce_emb = None

            # Focal loss with class weights
            weights = torch.tensor([1.0, self.weight], device=self.labels.device)
            train_mask_local = self.train_graph.ndata["train_mask"]
            ce_loss = F.cross_entropy(
                logits[train_mask_local], train_labels, weight=weights, reduction="none"
            )

            if self.focal_gamma > 0:
                probs_train = F.softmax(logits[train_mask_local], dim=1)
                pt = probs_train.gather(1, train_labels.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - pt) ** self.focal_gamma
                loss = (focal_weight * ce_loss).mean()
            else:
                loss = ce_loss.mean()

            # NCE loss
            if use_nce and nce_emb is not None and self.nce_lambda > 0:
                nce_loss = _fraudgnam_nce_loss(
                    nce_emb, original_features, self.labels, train_mask_local
                )
                loss = loss + self.nce_lambda * nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
            optimizer.step()

            # Only step scheduler after warmup completes
            if e >= self.warmup_epochs:
                scheduler.step()

            if (
                self.model_config.get("drop_rate", 0) > 0
                or self.train_config["inductive"]
            ):
                self.model.eval()
                val_output = self.model(self.val_graph)
                # In eval mode, forward always returns logits only
                logits = (
                    val_output if not isinstance(val_output, tuple) else val_output[0]
                )

            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata["val_mask"]])

            if val_score[self.train_config["metric"]] > self.best_score:
                if self.train_config["inductive"]:
                    src_output = self.model(self.source_graph)
                    logits = (
                        src_output
                        if not isinstance(src_output, tuple)
                        else src_output[0]
                    )
                    probs = logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config["metric"]]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print(
                    "Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, "
                    "test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}".format(
                        e,
                        loss,
                        val_score["AUROC"],
                        val_score["AUPRC"],
                        val_score["RecK"],
                        test_score["AUROC"],
                        test_score["AUPRC"],
                        test_score["RecK"],
                    )
                )
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config["patience"]:
                    break
        return test_score
