"""Auditable metrics for eventual prediction/weight exports.

These do not train a model or reproduce a result by themselves. Unspecified
paper conventions are explicit arguments (annualization, short definition,
first-day turnover treatment). Arrays must already be aligned causally.
"""
import numpy as np


def portfolio_metrics(weights, next_returns, market, *, annualization,
                      initial_weights, short_definition, net=False):
    w, y, m = map(lambda a: np.asarray(a, dtype=float), (weights, next_returns, market))
    if w.ndim != 2 or y.shape != w.shape or m.shape != (len(w),) or len(w) < 2:
        raise ValueError("Expected aligned [time, asset] weights/returns and [time] market")
    if not all(np.isfinite(a).all() for a in [w,y,m]) or annualization <= 0:
        raise ValueError("Inputs must be finite and annualization positive")
    if not np.allclose(np.abs(w).sum(axis=1), 1, atol=1e-6):
        raise ValueError("Weights must have unit L1 exposure")
    initial = np.asarray(initial_weights, dtype=float)
    if initial.shape != (w.shape[1],) or not np.isfinite(initial).all():
        raise ValueError("Supply explicit portfolio held before first evaluation day")
    previous = np.vstack([initial, w[:-1]])
    turnover = np.abs(w - previous).sum(axis=1)
    short_exposure = np.maximum(-w, 0).sum(axis=1)
    cost = .0005 * turnover + .0001 * short_exposure
    r = np.einsum("ta,ta->t", w, y) - (cost if net else 0)
    vol = r.std(ddof=0)
    if vol == 0 or m.var(ddof=0) == 0:
        raise ValueError("Sharpe/beta undefined with zero variance")
    if short_definition == "exposure":
        short = short_exposure.mean()
    elif short_definition == "asset_fraction":
        short = (w < 0).mean()
    else:
        raise ValueError("short_definition must be exposure or asset_fraction")
    return dict(sharpe=float(np.sqrt(annualization)*r.mean()/vol),
        annual_return_pct=float(100*annualization*r.mean()),
        annual_volatility_pct=float(100*np.sqrt(annualization)*vol),
        beta=float(((r-r.mean())*(m-m.mean())).mean()/m.var(ddof=0)),
        daily_turnover=float(turnover.mean()), short_fraction=float(short),
        mean_daily_cost=float(cost.mean()))


def binary_auc(labels, scores):
    """Rank-sum AUC with half credit for ties, O(n log n)."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Expected aligned finite score vectors")
    if not np.isin(labels, [0,1]).all():
        raise ValueError("Labels must be binary")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(labels)]
    ranks = np.empty(len(labels), dtype=float)
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + 1 + end) / 2
    return float((ranks[labels.astype(bool)].sum() - positives*(positives+1)/2) / (positives*negatives))


def mortgage_metrics(initial_states, next_states, probabilities, *, min_count=10):
    initial, target = np.asarray(initial_states), np.asarray(next_states)
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or initial.shape != (len(p),) or target.shape != initial.shape or not len(p):
        raise ValueError("Expected aligned states and [observation, class] probabilities")
    nclass = p.shape[1]
    if not all(np.issubdtype(a.dtype, np.integer) for a in [initial,target]):
        raise ValueError("State codes must be integers")
    if any(((a < 0) | (a >= nclass)).any() for a in [initial,target]):
        raise ValueError("Remove inactive/masked observations before scoring")
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(1),1):
        raise ValueError("Supply finite strictly positive normalized probabilities")
    if min_count < 1:
        raise ValueError("min_count must be positive")
    transitions = []
    for start in range(nclass):
        mask = initial == start
        for end in range(nclass):
            labels = target[mask] == end
            count = int(labels.sum())
            if count >= min_count:
                value = binary_auc(labels, p[mask,end])
                if value is not None:
                    transitions.append(dict(initial_state=start,end_state=end,count=count,auc=value))
    return dict(cross_entropy=float(-np.log(p[np.arange(len(p)), target]).mean()),
        average_transition_auc=float(np.mean([x["auc"] for x in transitions])) if transitions else None,
        transitions=transitions)
