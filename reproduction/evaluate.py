"""Adaptive expanding OLS using the released screening/label lag, with sufficient statistics.

The upstream evaluator recomputes SVDs over millions of rows each day. This
version accumulates X'X/X'y in float64 and solves only the selected small system.
It deliberately retains observations, including when factors are rank deficient.
"""
import argparse
import json
import time
from pathlib import Path

from common import ROOT, LocalStockData, initialize, load_expressions, daily_correlations, metrics, fresh_output
import numpy as np
import pandas as pd
import torch
from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType
from alphagen.utils.pytorch_utils import normalize_by_day


def solve_stats(gram, rhs, indices):
    sub = gram[np.ix_(indices, indices)]
    # Moore-Penrose OLS solution, no ridge penalty. Float64 rank tolerance is explicit.
    return np.linalg.lstsq(sub, rhs[indices], rcond=1e-12)[0]


def adaptive_predictions(x, y, rank_ics, start_day, n_factors=20,
                         threshold_ric=0.015, threshold_ricir=0.15, horizon=20):
    days, stocks, factors = x.shape
    gram = np.zeros((factors+1, factors+1), dtype=np.float64)
    rhs = np.zeros(factors+1, dtype=np.float64)
    cum = np.vstack([np.zeros((1, factors)), rank_ics.cumsum(0)])
    cum_sq = np.vstack([np.zeros((1, factors)), (rank_ics**2).cumsum(0)])
    predictions = np.full((days-start_day, stocks), np.nan, dtype=np.float32)
    selected = []
    last = 0
    max_fit_day = []
    for cur in range(start_day, days):
        stop = cur-horizon-1  # Exclusive upper bound: exactly upstream's cur-shift.
        for day in range(last, stop):
            good = np.isfinite(y[day]) & np.isfinite(x[day]).all(1)
            a = np.column_stack([x[day, good].astype(np.float64), np.ones(good.sum())])
            b = y[day, good].astype(np.float64)
            gram += a.T @ a
            rhs += a.T @ b
        last = stop
        mean = cum[stop]/stop
        variance = np.maximum((cum_sq[stop]-cum[stop]**2/stop)/(stop-1), 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(mean, np.sqrt(variance))
        ratio = np.where(np.isnan(ratio), 0., ratio)
        eligible = np.flatnonzero((np.abs(mean) > threshold_ric) & (np.abs(ratio) > threshold_ricir))
        if not len(eligible):
            eligible = np.array([np.argmax(np.abs(ratio))])
        chosen = eligible[:n_factors].tolist()
        indices = chosen+[factors]
        coef = solve_stats(gram, rhs, indices)
        to_pred = np.column_stack([np.nan_to_num(x[cur, :, chosen].T, nan=0.), np.ones(stocks)])
        predictions[cur-start_day] = to_pred @ coef
        selected.append(chosen)
        max_fit_day.append(stop-1)
        if (cur-start_day+1) % 100 == 0:
            print(json.dumps(dict(event='adaptive_progress', completed_days=cur-start_day+1,
                                  total_days=days-start_day, selected_factors=len(chosen))), flush=True)
    return predictions, selected, np.asarray(max_fit_day)


def portfolio_proxy(pred, target):
    """Released evaluator's top-20% forward-return/20 proxy, not an executable daily P&L."""
    returns = []
    for p, y in zip(pred, target):
        good = np.isfinite(p) & np.isfinite(y)
        count = max(1, int(good.sum()*0.2))
        if not good.any():
            returns.append(0.)
            continue
        score, outcome = p[good], y[good]
        chosen = np.argsort(score, kind='stable')[-count:]
        returns.append(float(outcome[chosen].mean()/20))
    returns = np.asarray(returns)
    wealth = np.cumprod(1+returns)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    return dict(annualized_arithmetic_return=float(252*returns.mean()),
                sharpe=float(np.sqrt(252)*returns.mean()/returns.std(ddof=1)) if returns.std() else 0.,
                maximum_drawdown=float(np.min(wealth/peak-1)),
                interpretation='top-20% 20-day forward return divided by 20; overlapping labels; selects among finite future targets; not a self-financing backtest'), returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--n-factors', type=int, default=20)
    parser.add_argument('--legacy-return-proxy', action='store_true',
                        help='Diagnostic only: reproduce the biased overlapping-label return proxy')
    args = parser.parse_args()
    args.output = fresh_output(args.output)
    torch.set_num_threads(args.threads)
    initialize()
    started = time.monotonic()
    expressions, weights = load_expressions(args.pool)
    data = LocalStockData('csi300', '2010-01-01', '2024-12-31')
    dates = data._dates[data.max_backtrack_days:-data.max_future_days]
    with torch.no_grad():
        y = data.forward_returns(20).cpu().numpy()
        x = np.stack([data.normalize(expr.evaluate(data)).cpu().numpy() for expr in expressions], axis=-1)
    if np.isinf(x).any():
        raise ValueError('Normalized features contain infinities')
    rank_ics = np.column_stack([daily_correlations(x[:, :, i], y)[1] for i in range(len(expressions))])
    start = int(dates.searchsorted(pd.Timestamp('2021-01-01')))
    pred, selected, fit_day = adaptive_predictions(x, y, rank_ics, start, n_factors=args.n_factors)
    pred[~data.eligibility()[start:].cpu().numpy()] = np.nan
    eval_dates, eval_targets = dates[start:], y[start:]
    validation = eval_dates.year == 2021
    test = eval_dates.year >= 2022
    valid_result = metrics(pred[validation], eval_targets[validation])
    test_result = metrics(pred[test], eval_targets[test])
    valid_proxy = test_proxy = None
    returns = np.array([], dtype=np.float64)
    if args.legacy_return_proxy:
        valid_proxy, _ = portfolio_proxy(pred[validation], eval_targets[validation])
        test_proxy, returns = portfolio_proxy(pred[test], eval_targets[test])
    observed_members = np.isfinite(data.data[data.max_backtrack_days:-data.max_future_days, 1, :].numpy())
    evaluation = dict(status='complete', method='released adaptive screening and expanding OLS via sufficient statistics',
                      validation=valid_result, test=test_result,
                      validation_return_proxy=valid_proxy, test_return_proxy=test_proxy,
                      pool_size=len(expressions), n_factors=args.n_factors,
                      selected_factor_count=dict(min=min(map(len, selected)), max=max(map(len, selected)), mean=float(np.mean(list(map(len, selected))))),
                      data=dict(days=data.n_days, stocks=data.n_stocks, start=str(dates[0].date()), end=str(dates[-1].date()),
                                daily_member_count_min=int(observed_members.sum(1).min()), daily_member_count_max=int(observed_members.sum(1).max())),
                      information_timing=dict(horizon=20, shift=21, last_training_index_exclusive='cur-21',
                                              latest_label_observation_precedes_prediction=bool(np.all(fit_day+20 < np.arange(start, data.n_days)))),
                      checkpoint_selection='fixed final training episode; no test-based model selection',
                      solver=dict(dtype='float64', rcond=1e-12, ridge_penalty=0, removes_training_rows=False),
                      wall_seconds=time.monotonic()-started,
                      limitations=['Community Tushare/Qlib snapshot, not the authors exact data',
                                   'This result covers one AlphaSAGE seed and CSI300 only; baselines are saved separately',
                                   'Training endpoint is recorded in the pool protocol; purged and nominal modes differ',
                                   'Sufficient-statistics OLS differs from released rank-deficiency row/column heuristics',
                                   'No executable portfolio backtest; legacy return proxy is opt-in and not evidence of trading performance'])
    np.savez_compressed(args.output/'predictions.npz', predictions=pred, targets=eval_targets,
                        dates=eval_dates.to_numpy(), stocks=np.asarray(data._stock_ids, dtype=str),
                        max_fit_index=fit_day, test_proxy_returns=returns)
    daily_ic, daily_ric = daily_correlations(pred, eval_targets)
    pd.DataFrame(dict(date=eval_dates, ic=daily_ic, rank_ic=daily_ric,
                      selected_factors=list(map(len, selected)))).to_csv(args.output/'daily_metrics.csv', index=False)
    (args.output/'selection.json').write_text(json.dumps(selected))
    (args.output/'data-provenance.json').write_text(json.dumps(data.provenance, indent=2))
    (args.output/'summary.json').write_text(json.dumps(evaluation, indent=2, allow_nan=False))
    print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == '__main__':
    main()
