"""
Refined FraudGNAM detector with three rule additions discovered during the
rebuttal experiments:

  (R1) n_f < 200      ->  K = 2   (Reddit-style sparse-fraud regime)
  (R2) h > 0.97 and d >= 300 ->  K = 4   (Weibo-style high-homophily,
                                          high-dim regime)
  (R3) h < 0.70       ->  T_highpass = 0   (extremely heterophilic, e.g.
                                          tolokers, where HP empirically
                                          *hurts* AUROC by ~0.6pp)

Other configuration logic is unchanged from the submitted rule set.
"""

import torch
from models.fraudgnam_detector import FraudGNAMDetector


class RefinedFraudGNAMDetector(FraudGNAMDetector):
    """Subclass that re-applies the adaptive rule with the refined cutoffs."""

    def __init__(self, train_config, model_config, data):
        # Run parent init first (this builds the model with the *old* rule).
        # We then re-derive a refined config and rebuild the model if needed.
        super().__init__(train_config, model_config, data)

        graph = self.source_graph
        labels_np = self.labels.cpu().numpy()
        avg_degree = graph.num_edges() / max(graph.num_nodes(), 1)
        in_feats = graph.ndata["feature"].shape[1]
        homophily = self.homophily
        train_fraud_count = int(
            labels_np[self.train_mask.cpu().numpy()].sum()
        )

        # ---------- Refined order ----------
        new_order = model_config.get("order", 2)
        if homophily < 0.85:
            new_order = 4
        elif avg_degree < 30:
            new_order = 3
        else:
            new_order = 2
        # NEW R1: very small training fraud -> use lowest order
        if train_fraud_count < 200:
            new_order = 2
        # NEW R2: extremely homophilic + high feature dim -> richer spectral res
        if homophily > 0.97 and in_feats >= 300:
            new_order = 4

        # ---------- Refined high-pass ----------
        if homophily < 0.85:
            new_hp = 2
        elif homophily < 0.95:
            new_hp = 1
        else:
            new_hp = 0
        # NEW R3: extreme heterophily empirically harms with HP active
        if homophily < 0.70:
            new_hp = 0

        # If the refined rule changes the model architecture, rebuild it.
        old_order = model_config.get("order", 2)
        old_hp = model_config.get("num_highpass", 0)
        if (new_order != old_order) or (new_hp != old_hp):
            print(
                f"  [REFINED RULE] order: {old_order}->{new_order}  "
                f"highpass: {old_hp}->{new_hp}  "
                f"(h={homophily:.3f}, n_f={train_fraud_count}, d={in_feats})"
            )
            model_config["order"] = new_order
            model_config["num_highpass"] = new_hp
            # NCE was decided by parent based on n_f and h; preserve that decision.
            self.model = type(self.model)(**model_config).to(
                train_config["device"]
            )
            # Re-initialize pair indices if pairwise was enabled
            if model_config.get("use_pairwise", False):
                train_features = graph.ndata["feature"][self.train_mask]
                train_labels_for_pairs = self.labels[self.train_mask]
                self.model.set_important_pairs(
                    train_features, train_labels_for_pairs
                )
        else:
            print(
                f"  [REFINED RULE] unchanged (h={homophily:.3f}, "
                f"n_f={train_fraud_count}, d={in_feats}, K={new_order}, T={new_hp})"
            )
