"""Ranking, an independent ridge reference, and an explicit delayed portfolio.

Portfolio numbers are a reconstruction of TopK/DropN with next-close execution,
fractional shares, equal weights, and proportional costs. They are not the
unreleased author portfolio backtest or a full Qlib broker simulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from runtime import ROOT, Panel, rankic, ranking_metrics
from build_data import save_json, sha


def topk_drop(scores, held, k=30, drop=5):
    """Qlib candidate/combined-ranking replacement rule with forced universe exits."""
    scores = scores.sort_values(ascending=False, kind='stable')
    held = set(held) & set(scores.index)
    today = [s for s in scores.index if s not in held][:drop+k-len(held)]
    combined = scores.loc[sorted(held) + today].sort_values(ascending=False, kind='stable')
    effective_drop = min(drop, len(today))
    sell = held & set(combined.index[-effective_drop:]) if effective_drop else set()
    buy = today[:len(sell)+k-len(held)]
    return (held-sell) | set(buy)


def equal_weight_trade(values, selected, wealth, open_cost=.0005, close_cost=.0015):
    """Solve a self-financing equal-weight rebalance after proportional fees."""
    all_ids = sorted(set(values) | set(selected))
    target_wealth = wealth
    for _ in range(30):
        target = target_wealth / len(selected) if selected else 0
        diffs = np.array([(target if i in selected else 0)-values.get(i, 0) for i in all_ids])
        buys, sells = np.maximum(diffs, 0).sum(), np.maximum(-diffs, 0).sum()
        cost = open_cost*buys + close_cost*sells
        new = wealth-cost
        if abs(new-target_wealth) < 1e-13*max(wealth, 1):
            break
        target_wealth = new
    allocation = {i: new/len(selected) for i in selected} if selected else {}
    assert new >= 0 and cost >= 0
    assert abs(sum(allocation.values()) + (new if not selected else 0) + cost - wealth) < 1e-9*max(wealth, 1)
    return allocation, new, float(cost), float(buys), float(sells)


def backtest(panel, predictions, out):
    out.mkdir(parents=True, exist_ok=True)
    sym_index = {str(s): i for i, s in enumerate(panel.symbols)}
    date_index = {str(d): i for i, d in enumerate(panel.dates)}
    # Historical-only marks for missing quotations; every fallback is counted.
    marks = pd.DataFrame(np.asarray(panel.close).T).ffill().to_numpy().T
    values, cash, previous_wealth, previous_exec = {}, 1., 1., None
    records = []
    fallback_count = 0
    skipped_untradable = 0
    terminal_settlements = 0
    terminal = getattr(panel, 'terminal_events', None)
    for decision, f in predictions.groupby('date', sort=True):
        t = date_index[str(decision)]
        execution = t+1
        if execution >= len(panel.dates) or str(panel.dates[execution]) > '2024-12-31':
            continue
        if previous_exec is not None:
            for i in list(values):
                end = execution
                ended = False
                if terminal is not None:
                    events = np.flatnonzero(terminal[i, previous_exec+1:execution+1])
                    if len(events):
                        end = previous_exec+1+int(events[0])
                        ended = True
                returns = np.asarray(panel.totalret[i, previous_exec+1:end+1], dtype=float).copy()
                missing = ~np.isfinite(returns)
                if missing.any():
                    a = marks[i, previous_exec:end]
                    b = marks[i, previous_exec+1:end+1]
                    alternate = b/a-1
                    if not np.isfinite(alternate[missing]).all():
                        raise ValueError(f'Cannot mark held {panel.symbols[i]} at {panel.dates[execution]}')
                    returns[missing] = alternate[missing]
                    fallback_count += int(missing.sum())
                values[i] *= float(np.prod(1+returns))
                if ended:
                    cash += values.pop(i)
                    terminal_settlements += 1
        wealth = cash+sum(values.values())
        scores = pd.Series(f.prediction.to_numpy(), index=[sym_index[s] for s in f.symbol])
        if scores.index.duplicated().any() or not np.isfinite(scores).all():
            raise ValueError('Invalid portfolio scores')
        can_trade = np.array(panel.tradable[:, execution], copy=True)
        if terminal is not None:
            can_trade &= ~terminal[:, execution]
        skipped_untradable += int((~can_trade[scores.index]).sum())
        eligible_scores = scores.loc[can_trade[scores.index]]
        # Untradable positions remain held, with no fabricated sale.
        locked = {i for i in values if not can_trade[i]}
        free_values = {i: v for i, v in values.items() if i not in locked}
        desired = topk_drop(eligible_scores, free_values, k=max(0, 30-len(locked)), drop=5)
        free_wealth = wealth-sum(values[i] for i in locked)
        allocation, net, cost, buys, sells = equal_weight_trade(free_values, desired, free_wealth)
        values = {i: values[i] for i in locked} | allocation
        cash = net if not desired else 0.
        after = cash+sum(values.values())
        daily = after/previous_wealth-1
        if not np.isfinite(daily) or daily <= -1:
            raise ValueError('Invalid self-financing daily return')
        records.append(dict(date=str(panel.dates[execution]), signal_date=str(decision),
            net_return=daily, wealth=after, transaction_cost=cost/previous_wealth,
            buys=buys/wealth, sells=sells/wealth, holdings=len(values), locked_holdings=len(locked)))
        previous_wealth, previous_exec = after, execution
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError('No portfolio sessions')
    frame.to_csv(out / 'portfolio_daily.csv', index=False)
    g = np.log1p(frame.net_return.to_numpy())
    wealth = np.r_[1., frame.wealth.to_numpy()]
    stats = dict(AR=float(np.expm1(252*g.mean())),
        MDD=float(np.max(1-wealth/np.maximum.accumulate(wealth))),
        SR=float(np.sqrt(252)*g.mean()/g.std(ddof=0)),
        cumulative_return=float(wealth[-1]-1), days=len(g),
        mean_one_way_turnover=float((frame.buys+frame.sells).mean()/2),
        missing_return_price_mark_fallbacks=fallback_count,
        dated_terminal_event_settlements=terminal_settlements,
        unavailable_execution_stock_days=skipped_untradable,
        implementation='reconstructed equal-weight Top30/Drop5, next-session close, fractional shares; costs buy5bp/sell15bp; total returns for CRSP',
        paper_portfolio_exact=False)
    save_json(out / 'portfolio_metrics.json', stats)
    return stats


def ridge_reference(universe):
    """Training-only sufficient statistics; ridge penalty chosen on validation RankIC."""
    out = ROOT / f'results/{universe}/ridge'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        metrics = json.loads((out / 'metrics.json').read_text())
        metrics.update(backtest(Panel(universe), pd.read_parquet(out / 'predictions.parquet'), out))
        save_json(out / 'metrics.json', metrics)
        return
    panel = Panel(universe)
    gram, xy = np.zeros((159, 159)), np.zeros(159)
    for t in panel.plans['train']:
        ids = panel.ids(t, 'train')
        raw = panel.close[ids, t+5]/panel.close[ids, t+1]-1
        y = (pd.Series(raw).rank(pct=True).to_numpy()-.5)*3.46
        x = np.c_[np.asarray(panel.features[ids, t], dtype=float), np.ones(len(ids))]
        gram += x.T@x
        xy += x.T@y
    candidates = []
    for alpha in [0.1, 1., 10., 100., 1000.]:
        penalty = np.eye(159)*alpha
        penalty[-1, -1] = 0
        beta = np.linalg.solve(gram+penalty, xy)
        daily = []
        for t in panel.plans['valid']:
            ids = panel.ids(t, 'valid')
            pred = np.asarray(panel.features[ids, t])@beta[:-1]+beta[-1]
            y = panel.close[ids, t+5]/panel.close[ids, t+1]-1
            daily.append(rankic(pred, y))
        candidates.append((ranking_metrics(daily)['RankIC'], alpha, beta))
    valid_rankic, alpha, beta = max(candidates, key=lambda x: x[0])
    np.save(out / 'coefficients.npy', beta)
    frames, daily = [], []
    for t in panel.plans['test']:
        ids = panel.ids(t, 'test')
        pred = np.asarray(panel.features[ids, t])@beta[:-1]+beta[-1]
        y = panel.close[ids, t+5]/panel.close[ids, t+1]-1
        daily.append(rankic(pred, y))
        frames.append(pd.DataFrame(dict(date=str(panel.dates[t]), symbol=panel.symbols[ids], prediction=pred, label=y)))
    predictions = pd.concat(frames, ignore_index=True)
    predictions.to_parquet(out / 'predictions.parquet', index=False)
    pd.DataFrame(dict(date=panel.dates[panel.plans['test']], RankIC=daily)).to_csv(out / 'daily_rankic.csv', index=False)
    metrics = dict(**ranking_metrics(daily), model='independent current-Alpha158 ridge reference',
        alpha=alpha, validation_RankIC=valid_rankic,
        validation_grid={str(a): v for v, a, _ in candidates}, predictions_sha256=sha(out / 'predictions.parquet'))
    metrics['ranking_days'] = metrics.pop('days')
    metrics.update(backtest(panel, predictions, out))
    save_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics), flush=True)


def ensemble(universe):
    out = ROOT / f'results/{universe}/ensemble'
    out.mkdir(parents=True, exist_ok=True)
    folders = [ROOT / f'results/{universe}/stage2/seed{s}' for s in range(5)]
    if not all((f / 'metrics.json').exists() for f in folders):
        return None
    arrays = [pd.read_parquet(f / 'predictions.parquet').set_index(['date', 'symbol']).sort_index() for f in folders]
    first = arrays[0]
    for a in arrays[1:]:
        pd.testing.assert_index_equal(a.index, first.index)
        np.testing.assert_allclose(a.label, first.label, rtol=0, atol=0, equal_nan=True)
    output = first[['label']].copy()
    output['prediction'] = sum(a.prediction for a in arrays)/5
    output = output.reset_index()
    output.to_parquet(out / 'predictions.parquet', index=False)
    daily = [rankic(f.prediction.to_numpy(), f.label.to_numpy()) for _, f in output.groupby('date', sort=True)]
    per_seed = [json.loads((f / 'metrics.json').read_text()) for f in folders]
    metrics = dict(seed_mean_RankIC=float(np.mean([m['RankIC'] for m in per_seed])),
        seed_std_RankIC=float(np.std([m['RankIC'] for m in per_seed], ddof=1)),
        seed_mean_RankICIR=float(np.mean([m['RankICIR'] for m in per_seed])),
        ensemble_ranking=ranking_metrics(daily), seeds=5, exact_paper_reproduction=False,
        predictions_sha256=sha(out / 'predictions.parquet'))
    metrics.update(backtest(Panel(universe), output, out))
    save_json(out / 'metrics.json', metrics)
    return metrics


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--universe', choices=['csi300', 'sp500'], required=True)
    p.add_argument('--ridge', action='store_true')
    a = p.parse_args()
    if a.ridge:
        ridge_reference(a.universe)
    else:
        print(ensemble(a.universe))
