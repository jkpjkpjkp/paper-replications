# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Dict, Sequence
import numpy as np


def ndcg_at_k_binary(items_ranked: np.ndarray, relevant_set: set, k: int) -> float:
    if k <= 0 or items_ranked.size == 0:
        return 0.0
    gains = np.array([1.0 if (str(it) in relevant_set) else 0.0 for it in items_ranked[:k]], dtype=float)
    if gains.sum() <= 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * discounts).sum())
    R = int(min(k, len(relevant_set)))
    if R <= 0:
        return 0.0
    idcg = float((np.ones(R, dtype=float) * (1.0 / np.log2(np.arange(2, R + 2)))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def roi_from_prices(
    items: Sequence[str],
    w: np.ndarray,
    start_close: Dict[str, float],
    end_close: Dict[str, float],
    fee_bps_buy: float = 0.0,
    fee_bps_sell: float = 0.0,
    slip_bps: float = 0.0,
) -> float:
    gross = 0.0
    for i, it in enumerate(items):
        ps = float(start_close.get(str(it), np.nan))
        pe = float(end_close.get(str(it), np.nan))
        if not np.isfinite(ps) or ps <= 0.0 or not np.isfinite(pe):
            continue
        gross += float(w[i]) * (pe / ps - 1.0)
    fee = (fee_bps_buy + fee_bps_sell + slip_bps) / 10000.0
    return float(gross - fee * float(np.sum(w)))

