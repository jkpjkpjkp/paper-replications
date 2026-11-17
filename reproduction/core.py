"""Paper equations 6--15, with date-wise masks and differentiable look-ahead."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import ExitStack
import math

import numpy as np
import torch
from torch import nn
from torch.func import functional_call


class Forecaster(nn.Module):
    def __init__(self, features=66, hidden_size=64, layers=1, dropout=0.0):
        super().__init__()
        self.rnn = nn.LSTM(features, hidden_size, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        encoded, _ = self.rnn(x)
        return self.head(encoded[:, -1]).squeeze(-1)


def standardized_mse(pred, labels, eps=1e-12):
    """One date, many horizons; normalize predictions on each label's mask.

    Returns a loss vector and usable-horizon flags. Undefined horizons get
    zero loss and are explicitly removed from the horizon-weight denominator.
    Population variance gives MSE = 2 - 2*IC for nonconstant observations.
    """
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError('Nonfinite model predictions')
    if labels.ndim == 1:
        labels = labels[:, None]
    good = torch.isfinite(labels) & torch.isfinite(pred[:, None])
    n = good.sum(0).clamp_min(1)
    y = torch.where(good, labels, 0.0)
    p = torch.where(good, pred[:, None], 0.0)
    dy = torch.where(good, y - y.sum(0) / n, 0.0)
    dp = torch.where(good, p - p.sum(0) / n, 0.0)
    vy = dy.square().sum(0) / n
    vp = dp.square().sum(0) / n
    zy = dy / vy.clamp_min(eps).sqrt()
    zp = dp / vp.clamp_min(eps).sqrt()
    usable = (good.sum(0) >= 3) & (vy > eps)
    loss = ((zp - zy).square() * good).sum(0) / n
    return torch.where(usable, loss, 0.0), usable


def standardized_labels(labels, eps=1e-12):
    good = torch.isfinite(labels)
    n = good.sum(0).clamp_min(1)
    y = torch.where(good, labels, 0.0)
    dy = torch.where(good, y - y.sum(0) / n, 0.0)
    variance = dy.square().sum(0) / n
    usable = (good.sum(0) >= 3) & (variance > eps)
    return torch.where(good & usable[None], dy / variance.clamp_min(eps).sqrt(),
                       torch.full_like(labels, float('nan')))


def warmup_label(labels):
    """Equation 12: mean standardized levels, without differencing returns."""
    if not bool(torch.isfinite(labels).all()):
        raise ValueError('Equation 12 requires a complete candidate-price path')
    return standardized_labels(labels).mean(dim=1)


def losses_for_days(model, days, *, params=None, horizon=None, warmup=False):
    losses, flags = [], []
    for x, y in days:
        pred = model(x) if params is None else functional_call(model, params, (x,))
        target = warmup_label(y) if warmup else (y if horizon is None else y[:, horizon])
        loss, usable = standardized_mse(pred, target)
        losses.append(loss)
        flags.append(usable)
    if not losses:
        raise ValueError('An empty date batch cannot be used for training')
    flags = torch.stack(flags)
    count = flags.sum(0)
    # Equations 6 and 10 sum over dates. Keeping the sum is material at the
    # specified fixed inner rate and entropy coefficient.
    return torch.stack(losses).sum(0), count > 0


def weighted_loss(model, days, weights, *, params=None):
    loss, usable = losses_for_days(model, days, params=params)
    effective = weights * usable
    if not bool(usable.any()):
        raise ValueError('No usable candidate labels in support dates')
    return (loss * effective).sum() / effective.sum().clamp_min(1e-12)


def target_loss(model, days, horizon, *, params=None, warmup=False):
    loss, usable = losses_for_days(model, days, params=params, horizon=horizon, warmup=warmup)
    if not bool(usable[0]):
        raise ValueError('No usable target labels in date batch')
    return loss[0]

def supervised_step(model, days, optimizer, horizon, *, warmup=False):
    """Sum date gradients; one AdamW step per batch, one retained date graph."""
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for day in days:
        loss = target_loss(model, [day], horizon, warmup=warmup)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite supervised loss')
        loss.backward()
        total += float(loss.detach())
    optimizer.step()
    return total


def lookahead(model, support, logits, inner_lr):
    """Equation 13, M=1. Retain the graph linking theta* to horizon logits."""
    params = OrderedDict(model.named_parameters())
    inner = weighted_loss(model, support, logits.softmax(0), params=params)
    grads = torch.autograd.grad(inner, tuple(params.values()), create_graph=True)
    return OrderedDict((name, value - inner_lr * grad)
                       for (name, value), grad in zip(params.items(), grads)), inner


def entropy(logits):
    return -(logits.softmax(0) * logits.log_softmax(0)).sum()


def second_order_context():
    # Fused cuDNN/MKLDNN RNN backward does not provide every required double
    # derivative in the pinned torch version. Native kernels retain the graph.
    stack = ExitStack()
    stack.enter_context(torch.backends.cudnn.flags(enabled=False))
    stack.enter_context(torch.backends.mkldnn.flags(enabled=False))
    return stack


def bilevel_step(model, support, query, logits, optimizer, inner_lr, gamma, target):
    optimizer.zero_grad(set_to_none=True)
    with second_order_context():
        ahead, inner = lookahead(model, support, logits, inner_lr)
        outer = target_loss(model, query, target, params=ahead) - gamma * entropy(logits)
        gradient, = torch.autograd.grad(outer, logits)
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError('Nonfinite horizon hypergradient')
    logits.grad = gradient
    optimizer.step()
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(ahead[name].detach())
    return {'inner_loss': float(inner.detach()), 'outer_loss': float(outer.detach()),
            'hypergradient_norm': float(gradient.norm())}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3 or np.std(a[good]) == 0 or np.std(b[good]) == 0:
        return float('nan')
    return float(np.corrcoef(a[good], b[good])[0, 1])


def daily_metrics(pred, target):
    from scipy.stats import rankdata
    good = np.isfinite(pred) & np.isfinite(target)
    p, y = np.asarray(pred)[good], np.asarray(target)[good]
    if len(y) < 3:
        raise ValueError('Too few observations for daily financial metrics')
    # Stable tie handling: a tied score at the cutoff includes its whole tie
    # group, with equal fractional allocation to the remaining decile slots.
    k = max(1, math.ceil(len(y) * 0.1))
    cutoff = np.partition(p, len(p) - k)[len(p) - k]
    above, tied = p > cutoff, p == cutoff
    top = (y[above].sum() + (k - above.sum()) * y[tied].mean()) / k
    return {'ic': corr(p, y), 'rank_ic': corr(rankdata(p), rankdata(y)),
            'top_return': float(top), 'stocks': len(y)}


def summarize_daily(rows):
    def ratio(values, annualize=1.0):
        values = np.asarray(values, float)
        values = values[np.isfinite(values)]
        sd = values.std(ddof=1) if len(values) > 1 else 0.0
        return float(values.mean() / sd * annualize) if sd > 0 else None
    ic = [r['ic'] for r in rows]
    rank = [r['rank_ic'] for r in rows]
    top = [r['top_return'] for r in rows]
    return {'ic': float(np.nanmean(ic)), 'icir': ratio(ic),
            'rank_ic': float(np.nanmean(rank)), 'rank_icir': ratio(rank),
            'top_return': float(np.mean(top)), 'sharpe': ratio(top, math.sqrt(252)),
            'days': len(rows)}


def paired_block_bootstrap(differences, seed=2026, draws=10000, block=10):
    """Circular moving-block bootstrap on common test dates, seed-averaged IC.

    The interval concerns date sampling conditional on these fitted seeds.
    It does not estimate all model-training uncertainty.
    """
    d = np.asarray(differences, float)
    if not np.isfinite(d).all() or len(d) < block:
        raise ValueError('Bootstrap needs finite paired dates and a full block')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(d), size=(draws, math.ceil(len(d) / block)))
    indices = ((starts[..., None] + np.arange(block)) % len(d)).reshape(draws, -1)[:, :len(d)]
    means = d[indices].mean(1)
    return {'mean_difference': float(d.mean()),
            'ci_95': np.quantile(means, [0.025, 0.975]).tolist(),
            'block_days': block, 'draws': draws, 'seed': seed,
            'interpretation': 'date uncertainty conditional on fitted seeds'}
