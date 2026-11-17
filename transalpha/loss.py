"""Section 3.3, including literal hard ordinal ranks.

Hard ranks have zero derivative almost everywhere. The Spearman factor is
therefore detached, but the Pearson factor in their product remains trainable.
No unreported soft-rank approximation is silently substituted.
"""
import torch
from torch.nn import functional as F


def pearson(x, y, eps=1e-12):
    x, y = x - x.mean(), y - y.mean()
    return (x * y).sum() / (x.square().sum() * y.square().sum()).clamp_min(eps).sqrt()


def ordinal_rank(x):
    # Stable sorting makes ties deterministic; the paper specifies ordinal ranks.
    order = torch.argsort(x.detach(), stable=True)
    return torch.argsort(order).to(x.dtype) + 1


def cross_section_loss(prediction, target):
    if prediction.numel() < 5:
        raise ValueError("At least five labelled stocks required")
    k = max(1, int(target.numel() * 0.2))
    order = torch.argsort(target, stable=True)
    bottom, top = prediction[order[:k]], prediction[order[-k:]]
    # Paper does not specify how top/bottom stocks are paired: use every pair.
    rank_loss = F.relu(0.1 - (top[:, None] - bottom[None, :])).mean()
    huber = F.huber_loss(prediction, target, delta=1.0)
    ic = pearson(prediction, target)
    rank_ic = pearson(ordinal_rank(prediction), ordinal_rank(target)).detach()
    return {"rank": rank_loss, "huber": huber, "corr": 1 - ic,
            "prod": 1 - ic * rank_ic, "pearson": ic, "spearman": rank_ic}


def hybrid_loss(prediction, target, parameters, mask=None):
    if prediction.ndim == 1:
        prediction, target = prediction[None], target[None]
        mask = None if mask is None else mask[None]
    valid = torch.isfinite(target)
    if mask is not None:
        valid = valid & mask
    counts = valid.sum(-1)
    if torch.any(counts < 5):
        raise ValueError("At least five labelled stocks required per cross section")
    # Vectorized across dates, with every reduction confined to its stock axis.
    p = torch.where(valid, prediction, 0.)
    y = torch.where(valid, target, 0.)
    count = counts.to(p.dtype)
    pc = (p - (p.sum(-1) / count)[:, None]) * valid
    yc = (y - (y.sum(-1) / count)[:, None]) * valid
    ic = (pc * yc).sum(-1) / (pc.square().sum(-1) * yc.square().sum(-1)).clamp_min(1e-12).sqrt()
    order_y = torch.argsort(torch.where(valid, target, float("inf")), dim=-1, stable=True)
    order_p = torch.argsort(torch.where(valid, prediction.detach(), float("inf")), dim=-1, stable=True)
    ranks_y = torch.argsort(order_y, dim=-1).to(p.dtype) + 1
    ranks_p = torch.argsort(order_p, dim=-1).to(p.dtype) + 1
    rp = (ranks_p - (count + 1)[:, None] / 2) * valid
    ry = (ranks_y - (count + 1)[:, None] / 2) * valid
    ric = ((rp * ry).sum(-1) / (rp.square().sum(-1) * ry.square().sum(-1)).clamp_min(1e-12).sqrt()).detach()
    k = (counts // 5).clamp_min(1)
    max_k = int(k.max())
    offsets = torch.arange(max_k, device=p.device)[None]
    bottom_ids = order_y[:, :max_k]
    top_ids = order_y.gather(-1, (counts - k)[:, None].add(offsets).clamp_max(p.shape[-1] - 1))
    bottom = p.gather(-1, bottom_ids)
    top = p.gather(-1, top_ids)
    pair_valid = (offsets < k[:, None])
    pairs = pair_valid[:, :, None] & pair_valid[:, None, :]
    rank_loss = (F.relu(.1 - (top[:, :, None] - bottom[:, None, :])) * pairs).sum((-2, -1)) / k.square()
    huber = (F.huber_loss(p, y, delta=1., reduction="none") * valid).sum(-1) / count
    parts = {"rank": rank_loss.mean(), "huber": huber.mean(), "corr": (1 - ic).mean(),
             "prod": (1 - ic * ric).mean(), "pearson": ic.mean(), "spearman": ric.mean()}
    parts["reg"] = 1e-5 * sum(p.square().sum() for p in parameters)
    total = (0.4 * parts["rank"] + 0.05 * parts["huber"] + 0.4 * parts["corr"]
             + 0.1 * parts["prod"] + 0.05 * parts["reg"])
    return total, parts
