"""Read-only audit of saved runs; writes diagnostics, never changes models or returns.

Run with .venv/bin/python reproduction/audit_causality.py.
Snapshot invariance is not verification of the vendor's point-in-time data history.
"""
import copy
import hashlib
import json
from datetime import datetime, timezone

from common import ROOT, LocalStockData, initialize, load_expressions, daily_correlations
from evaluate import adaptive_predictions, portfolio_proxy
import numpy as np
import pandas as pd
import torch
from alphagen.utils.pytorch_utils import normalize_by_day


def top_indices(scores, eligible):
    indices = np.flatnonzero(eligible)
    return indices[np.argsort(scores[indices], kind='stable')[-max(1, int(len(indices)*.2)):]]


def main():
    torch.set_num_threads(2)
    initialize()
    data = LocalStockData('csi300', '2010-01-01', '2024-12-31')
    dates = data._dates[100:-30]
    close = data.data[100:-30, 1].numpy()
    expressions, _ = load_expressions(ROOT/'results/csi300-seed0/pool.json')
    gp, _ = load_expressions(ROOT/'results/gp-csi300-seed0/pool.json')
    selection = json.loads((ROOT/'results/csi300-seed0/adaptive/selection.json').read_text())
    with np.load(ROOT/'results/csi300-seed0/adaptive/predictions.npz') as saved:
        prediction_dates = pd.DatetimeIndex(saved['dates'])
    probes = []
    with torch.no_grad():
        for date in ['2021-01-04', '2022-01-04', '2023-06-01', '2024-06-03']:
            day = int(dates.searchsorted(pd.Timestamp(date)))
            block = copy.copy(data)
            # 100 history days + one signal day + 30 future context days.
            block.data = data.data[day:day+131].clone()
            changed = copy.copy(block)
            changed.data = block.data.clone()
            changed.data[101:] = 12345.
            same_day = copy.copy(block)
            same_day.data = block.data.clone()
            generator = torch.Generator().manual_seed(900 + day)
            same_day.data[100] *= .8 + .4*torch.rand(same_day.data[100].shape, generator=generator)
            known = torch.isfinite(data.data[:100+day+1, 1]).any(0)
            reduced = copy.copy(block)
            reduced.data = block.data[:, :, known]
            current = torch.isfinite(block.data[100, 1])
            universe_differences, contemporaneous = [], []
            for i, expr in enumerate(expressions + gp[:1]):
                base = expr.evaluate(block)
                future = expr.evaluate(changed)
                torch.testing.assert_close(base, future, rtol=0, atol=0, equal_nan=True)
                now = expr.evaluate(same_day)
                if not torch.allclose(base[:, current], now[:, current], equal_nan=True):
                    contemporaneous.append(i)
                if i < len(expressions):
                    full_norm = normalize_by_day(base)[:, known]
                    reduced_norm = normalize_by_day(expr.evaluate(reduced))
                    difference = torch.abs(full_norm[:, current[known]] - reduced_norm[:, current[known]])
                    maximum = float(difference.max())
                    if maximum > 1e-5:
                        universe_differences.append(dict(factor=i, expression=str(expr), max_change=maximum))
            probes.append(dict(date=str(dates[day].date()), expressions_tested=len(expressions)+1,
                               future_values_and_missingness_invariant=True,
                               same_day_sensitive_factors=contemporaneous,
                               future_only_names_removed=int((~known).sum()),
                               selected_factors_with_change_above_001=[item['factor'] for item in universe_differences
                                   if item['max_change'] > .01 and item['factor'] in selection[prediction_dates.get_loc(dates[day])]],
                               future_universe_sensitive_factors=universe_differences))

    # Isolate a structural example, avoiding differences caused solely by float32
    # reduction order or unstable ordinal-rank ties when tensor width changes.
    day = int(dates.searchsorted('2022-01-04'))
    block = copy.copy(data)
    block.data = data.data[day:day+131].clone()
    known = torch.isfinite(data.data[:100+day+1, 1]).any(0)
    raw = expressions[22].evaluate(block)  # TsRank($high,20), in the saved selected set.
    assert str(expressions[22]) == 'TsRank($high,20)'
    assert (raw[:, ~known] == 0).all()
    reduced = copy.copy(block)
    reduced.data = block.data[:, :, known]
    reduced_raw = expressions[22].evaluate(reduced)
    torch.testing.assert_close(raw[:, known], reduced_raw, rtol=0, atol=0)
    current = torch.isfinite(block.data[100, 1, known])
    normalized_difference = (normalize_by_day(raw)[:, known][:, current] -
                             normalize_by_day(reduced_raw)[:, current]).abs()
    structural_example = dict(date='2022-01-04', factor=22, expression=str(expressions[22]),
        absent_future_stock_raw_values_all_zero=True, current_stock_raw_values_identical=True,
        max_current_normalized_value_change=float(normalized_difference.max()),
        explanation='TsRank converts all-NaN histories to zero; those future-only columns enter daily normalization.')

    # Independent poison tests at multiple dates, including missingness and selection stats.
    rng = np.random.default_rng(20261117)
    x = rng.normal(size=(100, 31, 6)).astype(np.float32)
    y = rng.normal(size=(100, 31)).astype(np.float32)
    def rankics(a, b):
        return np.column_stack([daily_correlations(a[:, :, i], b)[1] for i in range(a.shape[-1])])
    rank = rankics(x, y)
    rolling_checks = []
    for cur in [45, 65, 85]:
        baseline, chosen, fit = adaptive_predictions(x[:cur+1], y[:cur+1], rank[:cur+1], cur)
        poisoned_x, poisoned_y = x.copy(), y.copy()
        poisoned_y[cur-21:] = rng.normal(100, 10, poisoned_y[cur-21:].shape)
        poisoned_y[cur-21::2] = np.nan
        poisoned_x[cur+1:] *= 1000
        rerank = rankics(poisoned_x, poisoned_y)
        other, other_chosen, _ = adaptive_predictions(poisoned_x[:cur+1], poisoned_y[:cur+1], rerank[:cur+1], cur)
        np.testing.assert_array_equal(baseline, other)
        assert chosen == other_chosen
        rolling_checks.append(dict(prediction_index=cur, last_fit_index=int(fit[0]),
                                   latest_used_label_end=int(fit[0]+20), poison_invariant=True))

    cases = [
        ('AlphaSAGE', 'results/csi300-seed0/adaptive/predictions.npz'),
        ('GP', 'results/gp-csi300-seed0/test_predictions.npz'),
        ('XGBoost 100', 'results/xgboost-csi300-seed0/intended_100_rounds_test_predictions.npz'),
        ('XGBoost 10', 'results/xgboost-csi300-seed0/literal_notebook_10_rounds_test_predictions.npz'),
    ]
    diagnostics = []
    for name, path in cases:
        with np.load(ROOT/path, allow_pickle=False) as saved:
            keep = saved['dates'] >= np.datetime64('2022-01-01')
            p, y = saved['predictions'][keep], saved['targets'][keep]
            ds = pd.DatetimeIndex(saved['dates'][keep])
            stock_ids = pd.Index(saved['stocks'])
            rows = dates.get_indexer(ds)
            cols = data._stock_ids.get_indexer(stock_ids)
            assert (rows >= 0).all() and (cols >= 0).all()
            present = np.isfinite(close[np.ix_(rows, cols)])
            if 'max_fit_index' in saved:
                last_fit = saved['max_fit_index'][keep]
                assert np.all(last_fit+20 < rows)
            known_candidates = present & np.isfinite(p)
            missing = known_candidates & ~np.isfinite(y)
            changed_days = missing_selected = selected_total = 0
            examples = []
            for day in range(len(p)):
                causal = top_indices(p[day], known_candidates[day])
                original = top_indices(p[day], np.isfinite(p[day]) & np.isfinite(y[day]))
                missing_selected += int((~np.isfinite(y[day, causal])).sum())
                selected_total += len(causal)
                if set(causal) != set(original):
                    changed_days += 1
                    if len(examples) < 3:
                        examples.append(dict(date=str(ds[day].date()),
                                             excluded_from_causal_top=stock_ids[causal[~np.isfinite(y[day, causal])]].tolist(),
                                             causal_count=len(causal), original_count=len(original)))
            proxy, returns = portfolio_proxy(p, y)
            yearly = {}
            for year in sorted(set(ds.year)):
                m = ds.year == year
                ic, ric = daily_correlations(p[m], y[m])
                yearly[str(year)] = dict(days=int(m.sum()), ic=float(ic.mean()), rank_ic=float(ric.mean()),
                                        proxy_annualized_return=float(252*returns[m].mean()))
            diagnostics.append(dict(method=name, test_days=len(p),
                current_candidates=int(known_candidates.sum()), candidates_missing_future_target=int(missing.sum()),
                fraction_missing_future_target=float(missing.sum()/known_candidates.sum()),
                days_selection_changed_by_future_target_filter=changed_days,
                causal_selected_positions=selected_total, causal_selected_positions_missing_future_target=missing_selected,
                examples=examples, original_proxy=proxy, yearly=yearly,
                proxy_return_lag1_autocorrelation=float(np.corrcoef(returns[:-1], returns[1:])[0, 1])))

    # A future missing outcome changes today's selection with identical scores.
    scores = np.arange(10, dtype=np.float32)[None, :]
    outcomes = np.linspace(-.2, .2, 10, dtype=np.float32)[None, :]
    before = top_indices(scores[0], np.isfinite(outcomes[0])).tolist()
    outcomes[0, -1] = np.nan
    after = top_indices(scores[0], np.isfinite(outcomes[0])).tolist()
    assert before != after
    train_last = int(dates.searchsorted('2020-11-30', side='right')-1)
    valid_last = int(dates.searchsorted('2021-12-31', side='right')-1)
    manifest = json.loads((ROOT/'sources/artifact-manifest.json').read_text())
    mismatches = []
    for entry in manifest['files']:
        path = ROOT/entry['path']
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            mismatches.append(entry['path'])
    result = dict(audit_date_utc=datetime.now(timezone.utc).isoformat(),
        verdict='Not certified free of forward-looking bias; return-selection leakage confirmed.',
        scope='Saved CSI300 seed-0 runs, pinned source, snapshot-level causal probes; no retraining.',
        training_last_signal_date=str(dates[train_last].date()),
        training_latest_label_end=str(dates[train_last+20].date()),
        validation_last_label_end=str(dates[valid_last+20].date()),
        expression_probes=probes, adaptive_poison_checks=rolling_checks,
        future_universe_structural_example=structural_example,
        return_mask_counterexample=dict(before=before, after=after),
        saved_run_diagnostics=diagnostics,
        original_manifest=dict(files_checked=len(manifest['files']), mismatches=mismatches),
        limitations=['No independent point-in-time vendor vintages or corporate-action audit.',
                     'No executable trade ledger or costs; no corrected return estimate claimed.',
                     'Multiple-testing and research-selection history cannot be reconstructed from saved runs.',
                     'Future-universe probes remove names never yet observed in the index snapshot; this is an end-date dependence test, not evidence those companies were unknown.'])
    destination = ROOT/'results/causality-audit'
    destination.mkdir(exist_ok=True)
    (destination/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
