"""
PMP Detector for GADBench integration.
PMP: Partitioned Message Passing for Graph Fraud Detection (ICLR 2024).

Core idea: Partition neighbors by pseudo-label (benign/fraud/unknown from training set),
apply node-specific transformations to each partition separately, and use adaptive gating
for unknown-label neighbors. Single-layer architecture with projection head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.detector import BaseDetector


class LILinear(nn.Module):
    """
    Label-Imbalance-aware Linear layer with node-specific weights.
    The weight matrix is conditioned on the center node's original features.
    Uses element-wise modulation (efficient version) instead of full outer product.
    """

    def __init__(self, in_features, out_features, origin_infeat):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.trans_src = nn.Linear(origin_infeat, out_features)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, src):
        """
        input: aggregated neighbor features (N, in_features)
        src: center node's original features (N, origin_infeat)
        Returns: (N, out_features)
        """
        t_src = self.trans_src(src)  # (N, out_features)
        # Element-wise modulation: project input, then modulate by node-specific features
        out = F.linear(input, self.weight, self.bias)  # (N, out_features)
        out = out * torch.sigmoid(t_src)  # node-specific gating
        return out


class LIMLP(nn.Module):
    """Label-Imbalance MLP: stack of LILinear layers with activation."""

    def __init__(self, in_features, out_features, origin_infeat, n_layers=1,
                 activation=nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        for i in range(n_layers):
            in_f = in_features if i == 0 else out_features
            self.layers.append(LILinear(in_f, out_features, origin_infeat))

    def forward(self, input, src):
        h = input
        for layer in self.layers:
            h = layer(h, src)
            h = self.activation(h)
        return h


class PMPConv(nn.Module):
    """
    PMP Convolution layer: partitioned message passing.

    Splits neighbors into fraud/benign/unknown groups based on training labels,
    applies separate node-specific transformations to each partition.
    """

    def __init__(self, in_feats, out_feats, origin_infeat, n_trans=1,
                 activation=nn.ReLU(), drop_rate=0.0):
        super().__init__()

        # Node-specific MLPs for each partition
        self.fc_neigh_fraud = LIMLP(in_feats, out_feats, origin_infeat, n_trans, activation)
        self.fc_neigh_benign = LIMLP(in_feats, out_feats, origin_infeat, n_trans, activation)

        # Balance gate for unknown neighbors
        self.fc_balance = nn.Sequential(
            nn.Linear(origin_infeat, out_feats),
            activation,
        )
        self.balance_gate = nn.Sigmoid()

        # Self-transform
        self.fc_self = nn.Linear(in_feats, out_feats)

        self.activation = activation
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, features, adj_fraud, adj_benign, adj_unknown,
                original_features):
        """
        features: (N, in_feats) - current node features
        adj_fraud: sparse (N,N) - adjacency restricted to fraud source nodes
        adj_benign: sparse (N,N) - adjacency restricted to benign source nodes
        adj_unknown: sparse (N,N) - adjacency restricted to unknown source nodes
        original_features: (N, origin_infeat) - raw features for node-specific conditioning
        """
        # Aggregate from each partition (sum aggregation)
        neigh_fr = torch.sparse.mm(adj_fraud, features)    # (N, in_feats)
        neigh_be = torch.sparse.mm(adj_benign, features)   # (N, in_feats)
        neigh_unk = torch.sparse.mm(adj_unknown, features)  # (N, in_feats)

        # Node-specific transformations
        h_fr = self.fc_neigh_fraud(neigh_fr, original_features)    # (N, out_feats)
        h_be = self.fc_neigh_benign(neigh_be, original_features)   # (N, out_feats)

        # Unknown: adaptive gating between fraud and benign transformations
        balance = self.balance_gate(self.fc_balance(original_features))  # (N, out_feats)
        h_unk_fr = self.fc_neigh_fraud(neigh_unk, original_features)
        h_unk_be = self.fc_neigh_benign(neigh_unk, original_features)
        h_unk = balance * h_unk_fr + (1 - balance) * h_unk_be

        # Combine partitions + self
        h_neigh = h_fr + h_be + h_unk
        h_self = self.fc_self(features)

        out = self.activation(h_self + h_neigh)
        out = self.dropout(out)
        return out


class PMP(nn.Module):
    """
    PMP model for full-graph training in GADBench.

    Architecture: 1 PMPConv layer + projection to 2 classes.
    Uses training labels to partition neighbors during forward pass.
    """

    def __init__(self, in_feats, h_feats=64, drop_rate=0.0, n_trans=1,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.in_feats = in_feats
        self.h_feats = h_feats

        act = nn.ReLU() if activation == 'ReLU' else nn.ELU()

        self.conv = PMPConv(
            in_feats=in_feats,
            out_feats=h_feats,
            origin_infeat=in_feats,
            n_trans=n_trans,
            activation=act,
            drop_rate=drop_rate,
        )

        # Projection head (equivalent to 1D conv in original)
        self.proj = nn.Linear(h_feats, 2)

        # Will be set by detector
        self._train_mask = None
        self._labels = None
        self._adj_partitions = None

    def set_label_info(self, train_mask, labels):
        """Store training labels for partitioned aggregation."""
        self._train_mask = train_mask
        self._labels = labels

    def _build_partitioned_adj(self, graph):
        """Build partitioned adjacency matrices based on source node labels."""
        if self._adj_partitions is not None:
            return self._adj_partitions

        device = graph.device
        n = graph.num_nodes()
        src, dst = graph.edges()

        # Build label assignment
        label_assign = torch.full((n,), 2, device=device, dtype=torch.long)
        train_idx = self._train_mask.nonzero(as_tuple=True)[0]
        label_assign[train_idx] = self._labels[train_idx].long()

        # Partition edges by source label
        src_labels = label_assign[src]
        ones = torch.ones(src.shape[0], device=device)

        adj_fraud = torch.sparse_coo_tensor(
            torch.stack([dst, src])[:, src_labels == 1],
            ones[src_labels == 1],
            size=(n, n)
        ).coalesce()

        adj_benign = torch.sparse_coo_tensor(
            torch.stack([dst, src])[:, src_labels == 0],
            ones[src_labels == 0],
            size=(n, n)
        ).coalesce()

        adj_unknown = torch.sparse_coo_tensor(
            torch.stack([dst, src])[:, src_labels == 2],
            ones[src_labels == 2],
            size=(n, n)
        ).coalesce()

        self._adj_partitions = (adj_fraud, adj_benign, adj_unknown)
        return self._adj_partitions

    def forward(self, graph):
        features = graph.ndata['feature']

        # Build partitioned adjacency (cached after first call)
        adj_fraud, adj_benign, adj_unknown = self._build_partitioned_adj(graph)

        # Single-layer PMP convolution
        h = self.conv(features, adj_fraud, adj_benign, adj_unknown, features)

        # Project to class logits
        logits = self.proj(h)
        return logits


class PMPDetector(BaseDetector):
    """
    PMP detector for GADBench.
    Uses partitioned message passing with label-based neighbor grouping.
    """

    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        in_feats = self.data.graph.ndata['feature'].shape[1]
        model_config['in_feats'] = in_feats

        h_feats = model_config.get('h_feats', 64)
        drop_rate = model_config.get('drop_rate', 0.0)
        n_trans = model_config.get('n_trans', 1)
        activation = model_config.get('activation', 'ReLU')

        self.model = PMP(
            in_feats=in_feats,
            h_feats=h_feats,
            drop_rate=drop_rate,
            n_trans=n_trans,
            activation=activation,
        ).to(train_config['device'])

        # Provide label info for partitioned aggregation
        self.model.set_label_info(self.train_mask, self.labels)

        print(f"[PMP] in_feats={in_feats}, h_feats={h_feats}, "
              f"drop_rate={drop_rate}, n_trans={n_trans}")

    def train(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.model_config['lr'],
            weight_decay=self.model_config.get('weight_decay', 0.0)
        )
        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            # Clear cached adj partitions if graph changes (inductive)
            if self.train_config['inductive']:
                self.model._adj_partitions = None
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
                if self.train_config['inductive']:
                    self.model._adj_partitions = None
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            if torch.isnan(probs).any():
                probs = torch.nan_to_num(probs, nan=0.5)
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])

            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    self.model._adj_partitions = None
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
