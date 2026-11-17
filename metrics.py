"""Auditable top-five metrics; corrected NDCG is not the released metric."""
import numpy as np


def evaluate(prediction, truth, mask, k=5):
    prediction, truth, mask = map(np.asarray, (prediction, truth, mask))
    if prediction.shape != truth.shape or prediction.shape != mask.shape or prediction.ndim != 2:
        raise ValueError("Expected equal [stock, day] arrays")
    daily, reciprocal, ndcg = [], [], []
    discounts = 1 / np.log2(np.arange(k) + 2)
    for day in range(prediction.shape[1]):
        valid = (mask[:, day] > 0.5) & np.isfinite(prediction[:, day]) & np.isfinite(truth[:, day])
        if valid.sum() < k:
            raise ValueError(f"Fewer than {k} valid stocks on day {day}")
        p, y = prediction[valid, day], truth[valid, day]
        top = np.argsort(-p, kind="stable")[:k]
        daily.append(float(y[top].mean()))
        # Average ranks make ground-truth ties independent of arbitrary IDs.
        rank = 1 + (y > y[top[0]]).sum() + 0.5 * ((y == y[top[0]]).sum() - 1)
        reciprocal.append(1 / rank)
        # Nonnegative linear gains from cross-sectional percentile ranks.
        from scipy.stats import rankdata
        gains = rankdata(y, method="average") / len(y)
        ideal = np.sort(gains)[::-1][:k] @ discounts
        ndcg.append(float(gains[top] @ discounts / ideal))
    daily = np.asarray(daily)
    std = daily.std(ddof=0)
    return {"sharpe5": float(np.sqrt(252) * daily.mean() / std) if std > 0 else None,
            "mrr": float(np.mean(reciprocal)), "ndcg5_percentile": float(np.mean(ndcg)),
            "downside_deviation_daily": float(np.sqrt(np.mean(np.minimum(daily, 0)**2))),
            "downside_deviation_annual": float(np.sqrt(252 * np.mean(np.minimum(daily, 0)**2))),
            "simple_return": float(daily.sum()), "compound_return": float(np.prod(1+daily)-1),
            "days": len(daily), "daily_returns": daily.tolist()}
