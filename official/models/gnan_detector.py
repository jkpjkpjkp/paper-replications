"""
GNAN (Graph Neural Additive Network) baseline.

Adapted from Bechler-Speicher et al. 2024 "Graph Neural Additive Networks".
Combines NAM-style per-feature shape functions with multi-hop message passing.

Architecture:
    1. NAM step: per-feature MLP   f_i: R -> R^h
    2. K-hop graph aggregation:    h^(k) = D^{-1} A · h^(k-1)
    3. Per-hop NAM-style sum:      logit = sum_k alpha_k · sum_i g_k(f_i(x_i)^(k))
                                          = sum_k sum_i alpha_k · b_{k,i}(x_i)
    where b_{k,i} = MLP per (hop, feature) pair.

This preserves NAM-style feature decomposability while incorporating graph
structure. Compatible with GADBench training loop.
"""

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.detector import BaseDetector


class _FeatureMLP(nn.Module):
    def __init__(self, h_feats=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, h_feats), nn.LayerNorm(h_feats),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h_feats, h_feats), nn.LayerNorm(h_feats),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.Linear(h_feats, 1),  # scalar contribution
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GNAN(nn.Module):
    """Graph Neural Additive Network.

    For each (hop, feature) pair we learn an independent shape function.
    The final logit is a sum of all (hop, feature) contributions, preserving
    additive interpretability.
    """

    def __init__(self, in_feats, K=2, h_feats=32, dropout=0.1,
                 use_grouped=False, group_size=10):
        super().__init__()
        self.in_feats = in_feats
        self.K = K
        self.use_grouped = use_grouped
        if use_grouped:
            # Reduce parameter count for high-dim features by grouping
            self.group_size = group_size
            self.num_groups = (in_feats + group_size - 1) // group_size
            self.feature_nets = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(group_size if g < self.num_groups - 1
                                   else in_feats - g * group_size, h_feats),
                        nn.LayerNorm(h_feats), nn.LeakyReLU(0.1),
                        nn.Dropout(dropout),
                        nn.Linear(h_feats, 1)
                    )
                    for g in range(self.num_groups)
                ])
                for _ in range(K + 1)  # K+1 hops including hop 0
            ])
        else:
            self.feature_nets = nn.ModuleList([
                nn.ModuleList([_FeatureMLP(h_feats, dropout)
                               for _ in range(in_feats)])
                for _ in range(K + 1)
            ])
        self.hop_weights = nn.Parameter(torch.ones(K + 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _aggregate(self, graph, h):
        # Mean aggregation: D^{-1} A · h
        with graph.local_scope():
            deg = graph.in_degrees().float().clamp(min=1).unsqueeze(-1).to(h.device)
            graph.ndata['h'] = h / deg
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h')

    def _compute_feature_contributions(self, hop_idx, x):
        """For a given hop, compute sum_i f_i(x_i) returning (N, 1)."""
        if self.use_grouped:
            total = 0
            for g, net in enumerate(self.feature_nets[hop_idx]):
                start, end = g * self.group_size, min((g + 1) * self.group_size, self.in_feats)
                total = total + net(x[:, start:end]).squeeze(-1)
            return total
        else:
            total = 0
            for i, net in enumerate(self.feature_nets[hop_idx]):
                total = total + net(x[:, i:i + 1])
            return total

    def forward(self, graph):
        x = graph.ndata['feature']
        # Pre-compute K-hop aggregated features
        feats_per_hop = [x]
        for k in range(self.K):
            feats_per_hop.append(self._aggregate(graph, feats_per_hop[-1]))
        # Per-hop NAM contribution
        logit = self.bias
        hop_w = F.softmax(self.hop_weights, dim=0) * (self.K + 1)
        for k in range(self.K + 1):
            contribution = self._compute_feature_contributions(k, feats_per_hop[k])
            logit = logit + hop_w[k] * contribution
        return torch.stack([-logit / 2, logit / 2], dim=1)


class GNANDetector(BaseDetector):
    """GNAN detector adapted to GADBench training loop."""

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        in_feats = self.data.graph.ndata['feature'].shape[1]
        K = model_config.get('K', 2)
        h_feats = model_config.get('h_feats', 32)
        drop_rate = model_config.get('drop_rate', 0.1)
        use_grouped = in_feats > 100  # Match FraudGNAM convention

        self.model = GNAN(in_feats=in_feats, K=K, h_feats=h_feats,
                          dropout=drop_rate,
                          use_grouped=use_grouped).to(train_config['device'])

        print(f"[GNAN] in_feats={in_feats}, K={K}, h_feats={h_feats}, "
              f"grouped={use_grouped}")

    def train(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                       lr=self.model_config['lr'],
                                       weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.train_config['epochs']
        )
        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        weights = torch.tensor([1.0, self.weight], device=self.labels.device)

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(
                logits[self.train_graph.ndata['train_mask']],
                train_labels, weight=weights
            )
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            optimizer.step(); scheduler.step()

            self.model.eval()
            with torch.no_grad():
                if self.train_config['inductive']:
                    logits_val = self.model(self.val_graph)
                else:
                    logits_val = logits
                probs = logits_val.softmax(1)[:, 1]
            val_score = self.eval(
                val_labels, probs[self.val_graph.ndata['val_mask']]
            )
            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    src_logits = self.model(self.source_graph)
                    probs = src_logits.softmax(1)[:, 1]
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score
