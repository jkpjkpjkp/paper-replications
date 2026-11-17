"""
SEC-GFD Detector for GADBench integration.
Adapted from SEC-GFD (AAAI 2024)
"Revisiting Graph-Based Fraud Detection in Sight of Heterophily and Spectrum"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import dgl
import dgl.function as fn
from dgl.nn import GraphConv
import sympy
import scipy
import numpy as np
from models.detector import BaseDetector


def _calculate_theta(d):
    """Compute Bernstein basis polynomial coefficients of degree d."""
    thetas = []
    x = sympy.symbols('x')
    for i in range(d + 1):
        f = sympy.poly((x / 2) ** i * (1 - x / 2) ** (d - i) /
                        (scipy.special.beta(i + 1, d + 1 - i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for j in range(d + 1):
            inv_coeff.append(float(coeff[d - j]))
        thetas.append(inv_coeff)
    return thetas


class _BandConv(nn.Module):
    """Spectral band-pass convolution using Bernstein polynomial basis."""
    def __init__(self, in_feats, out_feats, theta, lin=False):
        super().__init__()
        self._theta = theta
        self._k = len(self._theta)
        self.linear = nn.Linear(in_feats, out_feats, bias=False)
        self.lin = lin

    def forward(self, graph, feat):
        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1),
                                  -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0] * feat
            for k in range(1, self._k):
                # Unnormalized Laplacian: feat - D^{-1/2} A D^{-1/2} feat
                graph.ndata['h'] = feat * D_invsqrt
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                feat = feat - graph.ndata.pop('h') * D_invsqrt
                h += self._theta[k] * feat
        if self.lin:
            h = F.leaky_relu(self.linear(h))
        return h


class _HighConv(nn.Module):
    """High-pass iterative Laplacian filter."""
    def __init__(self, num_layer=1):
        super().__init__()
        self.num_layer = num_layer

    def forward(self, graph, feat):
        D_invsqrt = torch.pow(graph.in_degrees().float().clamp(min=1),
                              -0.5).unsqueeze(-1).to(feat.device)
        for _ in range(self.num_layer):
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            feat = feat - graph.ndata.pop('h') * D_invsqrt
        return feat


class _NCE_GCN(nn.Module):
    """Simple 2-layer GCN for NCE embedding branch."""
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hid_dim, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hid_dim, out_dim, allow_zero_in_degree=True)
        self.act = nn.ReLU()

    def forward(self, g, feat):
        h = self.act(self.conv1(g, feat))
        h = self.conv2(g, h)
        return h


class SECGFD(nn.Module):
    """
    SEC-GFD: Spectral-Enhanced Contrastive Graph Fraud Detection.
    Combines band-pass spectral convolutions, high-order Laplacian filters,
    and a GCN branch for NCE contrastive loss.
    """
    def __init__(self, in_feats, h_feats, num_classes=2, order=2, high_order=2,
                 drop_rate=0.0, **kwargs):
        super().__init__()
        thetas = _calculate_theta(d=order)
        convs = []
        for i in range(len(thetas)):
            convs.append(_BandConv(h_feats, h_feats, thetas[i], lin=False))
        for j in range(high_order):
            convs.append(_HighConv(num_layer=j + 1))
        self.convs = nn.ModuleList(convs)
        num_filters = len(thetas) + high_order

        self.linear1 = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats * num_filters, h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.drop_rate = drop_rate

        self.nce_gcn = _NCE_GCN(in_feats, h_feats, in_feats)

    def forward(self, graph):
        feat = graph.ndata['feature']
        # Remove self-loops for NCE GCN branch
        g_nsl = dgl.remove_self_loop(graph)

        h = self.act(self.linear1(feat))
        if self.drop_rate > 0:
            h = F.dropout(h, p=self.drop_rate, training=self.training)
        h = self.act(self.linear2(h))

        h_parts = []
        for conv in self.convs:
            h_parts.append(conv(graph, h))
        h_cat = torch.cat(h_parts, dim=-1)

        h_out = self.act(self.linear3(h_cat))
        if self.drop_rate > 0:
            h_out = F.dropout(h_out, p=self.drop_rate, training=self.training)
        logits = self.linear4(h_out)

        emb = self.nce_gcn(g_nsl, feat)
        return logits, emb


def _cal_nceloss(emb, features, labels, train_mask):
    """NCE contrastive loss: push normal embeddings close to features,
    pull anomaly embeddings away."""
    train_normal_idx = torch.where((train_mask) & (labels == 0))[0]
    train_anomaly_idx = torch.where((train_mask) & (labels == 1))[0]
    sim_score = F.cosine_similarity(features, emb)
    nor = sim_score[train_normal_idx].mean()
    abn = sim_score[train_anomaly_idx].mean()
    nce_loss = -torch.log(nor / (abn + 1e-8))
    return nce_loss


class SECGFDDetector(BaseDetector):
    """
    SEC-GFD detector with adaptive configuration for GADBench.
    """
    def __init__(self, train_config, model_config, data):
        super().__init__(train_config, model_config, data)
        in_feats = self.data.graph.ndata['feature'].shape[1]
        model_config['in_feats'] = in_feats

        # Adaptive config
        h_feats = model_config.get('h_feats', 64)
        order = model_config.get('order', 2)
        high_order = model_config.get('high_order', 2)
        drop_rate = model_config.get('drop_rate', 0)
        nce_lambda = model_config.get('nce_lambda', 0.2)

        self.nce_lambda = nce_lambda
        self.model = SECGFD(
            in_feats=in_feats,
            h_feats=h_feats,
            num_classes=2,
            order=order,
            high_order=high_order,
            drop_rate=drop_rate,
        ).to(train_config['device'])

        print(f"[SEC-GFD] in_feats={in_feats}, h_feats={h_feats}, "
              f"order={order}, high_order={high_order}, "
              f"nce_lambda={nce_lambda}, drop_rate={drop_rate}")

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.model_config['lr'])

        train_labels = self.labels[self.train_mask]
        val_labels = self.labels[self.val_mask]
        test_labels = self.labels[self.test_mask]

        for e in range(self.train_config['epochs']):
            self.model.train()
            logits, emb = self.model(self.train_graph)

            # CE loss with class weights
            weights = torch.tensor([1., self.weight], device=self.labels.device)
            ce_loss = F.cross_entropy(
                logits[self.train_graph.ndata['train_mask']],
                train_labels, weight=weights)

            # NCE contrastive loss
            features = self.train_graph.ndata['feature']
            nce_loss = _cal_nceloss(
                emb, features, self.labels,
                self.train_graph.ndata['train_mask'].bool())
            loss = ce_loss + self.nce_lambda * nce_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            optimizer.step()

            if self.model_config.get('drop_rate', 0) > 0 or self.train_config['inductive']:
                self.model.eval()
                logits, _ = self.model(self.val_graph)

            probs = logits.softmax(1)[:, 1]
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']])

            if val_score[self.train_config['metric']] > self.best_score:
                if self.train_config['inductive']:
                    logits, _ = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]
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
