"""Equations 2–24 of Hermes. Unspecified choices are recorded in README.txt."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalMixing(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(t, t), nn.GELU(), nn.Linear(t, 1))
            for t in range(1, length + 1)
        ])

    def forward(self, x):
        # x: batch, stock, time, feature. Each output uses a prefix only.
        x = x.transpose(-1, -2)
        return torch.cat([m(x[..., :t + 1]) for t, m in enumerate(self.layers)], -1).transpose(-1, -2)


def moving_aggregation(edges, window):
    """Eq. 10–14: endpoints query all industries and times in their past window."""
    if window < 1 or window > edges.shape[-2]:
        raise ValueError("lead-lag window must lie within the scale length")
    # [B,K,L,d] -> [B,P,K,w,d]
    patches = edges.unfold(2, window, 1).permute(0, 2, 1, 4, 3)
    b, p, k, w, d = patches.shape
    sources = patches.reshape(b, p, k * w, d)
    queries = patches[..., -1, :]
    scores = torch.einsum('bpud,bpkd->bpuk', sources, queries)
    weights = scores.softmax(dim=-2)
    fused = torch.einsum('bpuk,bpud->bpkd', weights, sources).transpose(1, 2)
    return torch.cat([edges[:, :, :window - 1], fused], dim=2)


class Hermes(nn.Module):
    def __init__(self, incidence, lookback=16, features=5, hidden=8,
                 scales=(1, 2, 4), windows=(3, 2, 1), ablation='full', distance_eps=1e-6):
        super().__init__()
        h = torch.as_tensor(incidence, dtype=torch.bool)
        if h.ndim != 2 or not h.any(0).all() or not h.any(1).all():
            raise ValueError("every stock and hyperedge must have a membership")
        if ablation == 'single_scale':
            scales, windows = (1,), (windows[0],)
        if len(scales) != len(windows):
            raise ValueError("one lead-lag window is required per scale")
        self.register_buffer('incidence', h)
        self.lookback, self.ablation = lookback, ablation
        self.distance_eps = distance_eps
        self.scales, self.windows = tuple(scales), tuple(windows)
        lengths = [lookback // s for s in scales]
        if any(w < 1 or w > l for w, l in zip(windows, lengths)):
            raise ValueError("invalid scale or window")
        self.downsample = nn.ModuleList([
            nn.Conv1d(features, features, kernel_size=s, stride=s) for s in scales
        ])
        self.mixers = nn.ModuleList([CausalMixing(l) for l in lengths])
        self.projections = nn.ModuleList([nn.Linear(features, hidden) for _ in scales])
        self.scores = nn.ModuleList([nn.Linear(l * hidden, 1) for l in lengths])
        self.upsample = nn.ModuleList([nn.Linear(l, lookback) for l in lengths])
        self.metric_factor = nn.Parameter(torch.eye(hidden))
        self.edge_projection = nn.Linear(hidden, hidden, bias=False)
        self.initial_heads = nn.ModuleList([nn.Linear(l * hidden, 1) for l in lengths])
        self.graph_heads = nn.ModuleList([nn.Linear(lookback * hidden, 1) for _ in scales])

    def fuse(self, edges):
        # [B,S,K,T,d] -> [B,T,SK,d], independent metric per timestamp.
        b, s, k, t, d = edges.shape
        flat = edges.permute(0, 3, 1, 2, 4).reshape(b, t, s * k, d)
        z = flat @ self.metric_factor.T  # Q=A.T@A, Eq. 17
        norm = z.square().sum(-1, keepdim=True)
        dist = (norm + norm.transpose(-1, -2) - 2 * z @ z.transpose(-1, -2)).clamp_min(self.distance_eps)
        affinity = dist.reciprocal()
        diagonal = torch.eye(s * k, device=edges.device, dtype=torch.bool)
        affinity = affinity.masked_fill(diagonal, 0.)  # zero logits, not excluded self-loops
        # Literal Eq. 18 column softmax, then Eq. 19 B @ E @ W.
        adjacency = affinity.softmax(dim=-2)
        out = self.edge_projection(adjacency @ flat)
        return out.reshape(b, t, s, k, d).permute(0, 2, 3, 1, 4)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.incidence.shape[0] or x.shape[2] != self.lookback:
            raise ValueError("expected [batch, stock, lookback, feature]")
        b, n, t, f = x.shape
        originals, edges, memberships = [], [], []
        for conv, mix, proj, score, up, window in zip(
                self.downsample, self.mixers, self.projections, self.scores, self.upsample, self.windows):
            down = conv(x.reshape(b * n, t, f).transpose(1, 2)).transpose(1, 2)
            nodes = proj(mix(down.reshape(b, n, -1, f)))
            logits = score(nodes.flatten(-2)).expand(-1, -1, self.incidence.shape[1])
            membership = logits.masked_fill(~self.incidence.unsqueeze(0), -torch.inf).softmax(dim=1)
            edge = torch.einsum('bnk,bntd->bktd', membership, nodes)
            if self.ablation != 'no_lead_lag':
                edge = moving_aggregation(edge, window)
            edges.append(up(edge.transpose(-1, -2)).transpose(-1, -2))
            originals.append(nodes)
            memberships.append(membership)
        edges = torch.stack(edges, dim=1)
        if self.ablation not in ('no_fusion', 'single_scale', 'initial_only'):
            edges = self.fuse(edges)
        predictions = []
        for i, (nodes, membership) in enumerate(zip(originals, memberships)):
            initial = self.initial_heads[i](nodes.flatten(-2)).squeeze(-1)
            if self.ablation == 'initial_only':
                predictions.append(initial)
                continue
            stock = torch.einsum('bnk,bktd->bntd', membership, edges[:, i])
            graph = self.graph_heads[i](stock.flatten(-2)).squeeze(-1)
            predictions.append(graph if self.ablation == 'no_skip' else initial + graph)
        return torch.stack(predictions).sum(0)


def ranking_loss(prediction, target, mask=None, alpha=1., reduction='sum'):
    """Eq. 24: MSE plus summed pairwise ranking hinge, averaged across days."""
    if mask is None:
        mask = torch.ones_like(target, dtype=torch.bool)
    mse = ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1)
    pred_diff = prediction.unsqueeze(-1) - prediction.unsqueeze(-2)
    true_diff = target.unsqueeze(-1) - target.unsqueeze(-2)
    pairs = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    pair = F.relu(-pred_diff * true_diff) * pairs
    if reduction == 'sum':
        rank = pair.sum() / prediction.shape[0]
    elif reduction == 'mean':
        rank = pair.sum() / pairs.sum().clamp_min(1)
    else:
        raise ValueError(reduction)
    return mse + alpha * rank, mse.detach(), rank.detach()


class LinearBaseline(nn.Module):
    def __init__(self, lookback=16, features=5):
        super().__init__()
        self.head = nn.Linear(lookback * features, 1)

    def forward(self, x):
        return self.head(x.flatten(-2)).squeeze(-1)
