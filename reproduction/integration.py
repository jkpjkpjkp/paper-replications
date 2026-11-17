"""Auditable local corrections; the pinned upstream files stay unchanged."""
from functools import wraps

import numpy as np
from scipy.stats import rankdata
import torch

from alphagen.data import expression as expressions
from alpha_gfn.alpha_pool import AlphaPoolGFN


class ForwardReturn(expressions.Expression):
    """Use formation membership and full adjusted endpoint prices."""
    def __init__(self, horizon=20):
        self.horizon = horizon

    def evaluate(self, data, period=slice(0, 1)):
        return data.forward_returns(self.horizon, period)

    @property
    def is_featured(self):
        return True

    def __str__(self):
        return f'FormationMemberForwardReturn({self.horizon})'


def install_expression_guards():
    """Mask each operation before another operation can consume its output.

    In particular Sign(NaN), TsRank(all-NaN), and NaN**0 can otherwise create
    finite values for future-only names and enter a nested cross-sectional Rank.
    Only LocalStockData opts into this behavior via its eligibility method.
    """
    for cls in vars(expressions).values():
        if not isinstance(cls, type) or not issubclass(cls, expressions.Expression):
            continue
        original = cls.__dict__.get('evaluate')
        if original is None or getattr(original, '__isabstractmethod__', False):
            continue
        if getattr(original, '_membership_guard', False):
            continue

        def guard(function):
            @wraps(function)
            def evaluate(self, data, period=slice(0, 1)):
                value = function(self, data, period)
                if hasattr(data, 'eligibility'):
                    value = value.masked_fill(~data.eligibility(period) | ~torch.isfinite(value), torch.nan)
                return value
            evaluate._membership_guard = True
            return evaluate
        cls.evaluate = guard(original)

    # Retain the released zero-based fractional scale, but average tied ranks.
    # This is linear-memory ranking and invariant to the ordering of columns.
    def rank(self, operand):
        values = operand.detach().cpu().numpy()
        finite = np.isfinite(values)
        ranks = rankdata(np.where(finite, values, np.nan), axis=1,
                         method='average', nan_policy='omit') - 1
        result = ranks / np.maximum(finite.sum(1, keepdims=True), 1)
        return torch.as_tensor(result, dtype=operand.dtype, device=operand.device)
    expressions.Rank._apply = rank


class LocalAlphaPoolGFN(AlphaPoolGFN):
    def __init__(self, capacity, stock_data, target, **kwargs):
        super().__init__(capacity, stock_data, target, **kwargs)
        # A missing future outcome must not become a zero training label.
        self.target = stock_data.normalize(target.evaluate(stock_data), fill_missing=False)

    def _normalize_by_day(self, value):
        return self.data.normalize(value)

    def _compute_consistency_loss(self, query_value, neighbor_indices, weights):
        loss = 0.
        for index, weight in zip(neighbor_indices, weights):
            other = self.values[index]
            valid = torch.isfinite(query_value) & torch.isfinite(other)
            squared = torch.where(valid, (query_value - other).square(), 0.)
            daily = squared.sum(1) / valid.sum(1).clamp_min(1)
            loss += weight.item() * daily.mean().item()
        return loss
