"""
GAGA Detector for GADBench integration.
GAGA: Label Information Enhanced Fraud Detection against Low Homophily in Graphs (WWW 2023).

Core idea: Convert graph to feature sequences via label-based group aggregation,
then classify with a Transformer encoder. Group aggregation splits neighbors by
training labels (benign/fraud/unknown), mean-aggregates each group's features per hop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
from models.detector import BaseDetector


class GAGA(nn.Module):
    """
    GAGA model: Group Aggregation + Transformer Encoder.

    The group aggregation is performed as a preprocessing step within forward().
    For each node, for each hop k (1..K), neighbors are split into 3 groups
    based on training labels: benign (0), fraud (1), unknown (2).
    Mean features of each group form a sequence fed to a Transformer.
    """

    def __init__(self, in_feats, h_feats=32, n_hops=2, n_heads=4,
                 n_layers=2, ff_dim=64, drop_rate=0.1, agg_type='cat',
                 **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.n_hops = n_hops
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_groups = 3  # benign, fraud, unknown
        self.agg_type = agg_type

        # Sequence length per node: center_feat + n_hops * 3 groups
        self.seq_len = 1 + n_hops * self.n_groups

        # Feature projection
        self.feat_proj = nn.Linear(in_feats, h_feats)

        # Learnable encodings
        self.hop_emb = nn.Embedding(n_hops + 1, h_feats)  # hop 0 = center
        self.group_emb = nn.Embedding(self.n_groups + 1, h_feats)  # group 0,1,2 + center(3)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_feats, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=drop_rate,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Classifier
        self.classifier = nn.Linear(h_feats, 2)
        self.dropout = nn.Dropout(drop_rate)

        # Will be set by detector
        self._train_mask = None
        self._labels = None
        self._seq_cache = None

    def set_label_info(self, train_mask, labels):
        """Store training labels for group aggregation."""
        self._train_mask = train_mask
        self._labels = labels

    def _precompute_group_sequences(self, graph, features):
        """
        Precompute group-aggregated feature sequences (cached, no gradients).

        For hop k, we aggregate from k-hop neighbors by iteratively propagating
        through the graph, then splitting by label group at each hop level.

        For hop 1: aggregate 1-hop group features directly
        For hop 2: aggregate from 1-hop neighbors using their 1-hop-propagated features
        (This avoids materializing the dense A^k matrix.)

        Returns: (N, seq_len, in_feats) tensor
        """
        device = features.device
        n = features.shape[0]
        labels = self._labels
        train_mask = self._train_mask

        # Build label assignment: 0=benign_train, 1=fraud_train, 2=unknown
        label_assign = torch.full((n,), 2, device=device, dtype=torch.long)
        train_idx = train_mask.nonzero(as_tuple=True)[0]
        label_assign[train_idx] = labels[train_idx].long()

        src, dst = graph.edges()
        ones = torch.ones(src.shape[0], device=device)
        src_labels = label_assign[src]

        # Build group-specific sparse adjacency matrices
        group_adjs = []
        group_degs = []
        for g in range(self.n_groups):
            mask = (src_labels == g)
            if mask.any():
                g_adj = torch.sparse_coo_tensor(
                    torch.stack([dst[mask], src[mask]]),
                    ones[mask], size=(n, n)
                ).coalesce()
            else:
                g_adj = torch.sparse_coo_tensor(
                    torch.zeros(2, 0, device=device, dtype=torch.long),
                    torch.zeros(0, device=device), size=(n, n)
                ).coalesce()
            group_adjs.append(g_adj)
            deg = torch.sparse.sum(g_adj, dim=1).to_dense().clamp(min=1e-8)
            group_degs.append(deg)

        # Full adjacency for multi-hop feature propagation
        full_adj = torch.sparse_coo_tensor(
            torch.stack([dst, src]), ones, size=(n, n)
        ).coalesce()
        full_deg = torch.sparse.sum(full_adj, dim=1).to_dense().clamp(min=1e-8)

        # Initialize sequence
        sequences = torch.zeros(n, self.seq_len, self.in_feats, device=device)
        sequences[:, 0, :] = features

        # current_features starts as original features, gets propagated each hop
        current_features = features
        for hop in range(self.n_hops):
            base_idx = 1 + hop * self.n_groups

            # For each group, aggregate current_features from group neighbors
            for g_idx in range(self.n_groups):
                agg = torch.sparse.mm(group_adjs[g_idx], current_features)
                agg = agg / group_degs[g_idx].unsqueeze(1)
                sequences[:, base_idx + g_idx, :] = agg

            # Propagate features through full graph for next hop
            if hop < self.n_hops - 1:
                current_features = torch.sparse.mm(full_adj, current_features)
                current_features = current_features / full_deg.unsqueeze(1)

        return sequences

    def _build_group_sequences(self, graph, features):
        """Build or return cached group sequences."""
        if self._seq_cache is not None and not self.training:
            return self._seq_cache

        with torch.no_grad():
            seq = self._precompute_group_sequences(graph, features)

        if not self.training:
            self._seq_cache = seq
        return seq

    def forward(self, graph):
        features = graph.ndata['feature']
        device = features.device

        # Step 1: Group aggregation -> (N, S, F)
        sequences = self._build_group_sequences(graph, features)

        # Step 2: Project features -> (N, S, H)
        proj = self.feat_proj(sequences)

        # Step 3: Add positional encodings (hop + group)
        hop_ids = torch.zeros(self.seq_len, device=device, dtype=torch.long)
        group_ids = torch.zeros(self.seq_len, device=device, dtype=torch.long)

        hop_ids[0] = 0  # center
        group_ids[0] = 3  # center uses special group id
        for hop in range(self.n_hops):
            base = 1 + hop * self.n_groups
            for g in range(self.n_groups):
                hop_ids[base + g] = hop + 1
                group_ids[base + g] = g

        hop_enc = self.hop_emb(hop_ids)   # (S, H)
        grp_enc = self.group_emb(group_ids)  # (S, H)
        proj = proj + hop_enc.unsqueeze(0) + grp_enc.unsqueeze(0)  # broadcast over N

        # Step 4: Transformer expects (S, N, H)
        proj = proj.transpose(0, 1)

        # Step 5: Transformer encoding
        out = self.transformer(proj)  # (S, N, H)

        # Step 6: Use center token (position 0) for classification
        center = out[0]  # (N, H)
        center = self.dropout(center)

        # Step 7: Classify
        logits = self.classifier(center)  # (N, 2)
        return logits


class GAGADetector(BaseDetector):
    """
    GAGA detector for GADBench.
    Uses label-based group aggregation + Transformer encoder.
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        in_feats = self.data.graph.ndata['feature'].shape[1]
        model_config['in_feats'] = in_feats

        # Default hyperparameters
        h_feats = model_config.get('h_feats', 32)
        n_hops = model_config.get('n_hops', 2)
        n_heads = model_config.get('n_heads', 4)
        n_layers = model_config.get('n_layers', 2)
        ff_dim = model_config.get('ff_dim', 64)
        drop_rate = model_config.get('drop_rate', 0.1)

        self.model = GAGA(
            in_feats=in_feats,
            h_feats=h_feats,
            n_hops=n_hops,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            drop_rate=drop_rate,
        ).to(train_config['device'])

        # Provide label info for group aggregation
        self.model.set_label_info(self.train_mask, self.labels)

        print(f"[GAGA] in_feats={in_feats}, h_feats={h_feats}, "
              f"n_hops={n_hops}, n_heads={n_heads}, n_layers={n_layers}")

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config['lr'],
            weight_decay=self.model_config.get('weight_decay', 1e-4)
        )
        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits = self.model(self.train_graph)
            loss = F.cross_entropy(
                logits[self.train_graph.ndata['train_mask']], train_labels,
                weight=torch.tensor([1., self.weight], device=self.labels.device)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            # Guard against NaN (can occur on very high-degree graphs)
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.5)
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])

            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
                    if torch.isnan(probs).any():
                        probs = torch.nan_to_num(probs, nan=0.5)
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(test_labels, probs[self.test_mask])
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, '
                      'test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
        return test_score
