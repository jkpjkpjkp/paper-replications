"""Meaningful checks for integration, numerical equivalence and label information timing."""
import json
import tempfile
from pathlib import Path

from common import LocalStockData, initialize, daily_correlations, load_expressions
from evaluate import adaptive_predictions, solve_stats
import numpy as np
import pandas as pd
import torch
from alphagen.data.expression import Feature, Ref
from alphagen_qlib.stock_data import FeatureType
from alphagen.utils.correlation import batch_pearsonr, batch_spearmanr


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(1907)
    x = rng.integers(-3, 4, size=(10, 29)).astype(np.float32)
    y = rng.integers(-2, 5, size=x.shape).astype(np.float32)
    x[1, 3] = y[2, 5] = np.nan
    ic, ric = daily_correlations(x, y)
    tx, ty = torch.tensor(x), torch.tensor(y)
    ic_error = float(np.max(np.abs(ic-batch_pearsonr(tx, ty).numpy())))
    rank_error = float(np.max(np.abs(ric-batch_spearmanr(tx, ty).numpy())))
    assert ic_error < 1e-6, ic_error
    assert rank_error < 1e-6, rank_error
    tiny_x = np.arange(1,6,dtype=np.float32)[None,:]*1e-6
    tiny_y = np.arange(1,6,dtype=np.float32)[None,:]
    standard_tiny = daily_correlations(tiny_x,tiny_y)[0][0]
    released_tiny = daily_correlations(tiny_x,tiny_y,released_epsilon=True)[0][0]
    expected_tiny = batch_pearsonr(torch.tensor(tiny_x),torch.tensor(tiny_y)).item()
    assert abs(standard_tiny-1)<1e-10
    assert abs(released_tiny-expected_tiny)<1e-10
    design = rng.normal(size=(173, 5))
    outcome = rng.normal(size=173)
    coef = solve_stats(design.T@design, design.T@outcome, list(range(5)))
    direct = np.linalg.lstsq(design, outcome, rcond=None)[0]
    ols_error = float(np.max(np.abs(coef-direct)))
    assert ols_error < 1e-10, ols_error
    factors = rng.normal(size=(64, 13, 3)).astype(np.float32)
    labels = rng.normal(size=(64, 13)).astype(np.float32)
    rankics = np.column_stack([daily_correlations(factors[:, :, i], labels)[1] for i in range(3)])
    predictions, selections, fit_day = adaptive_predictions(factors, labels, rankics, 48)
    direct_errors = []
    for row, cur in enumerate(range(48, 64)):
        idx = selections[row]
        fit = factors[:cur-21][:, :, idx].reshape(-1, len(idx))
        fit = np.column_stack([fit, np.ones(len(fit))])
        target = labels[:cur-21].ravel()
        coef = np.linalg.lstsq(fit, target, rcond=None)[0]
        expected = np.column_stack([factors[cur][:, idx], np.ones(13)]) @ coef
        direct_errors.append(float(np.max(np.abs(expected-predictions[row]))))
    assert max(direct_errors) < 1e-6, direct_errors
    assert np.all(fit_day+20 < np.arange(48, 64))
    # Causal test: changing labels unavailable at day 48 cannot change that day's prediction.
    altered = labels.copy()
    altered[27:] += 1000
    altered_rank = np.column_stack([daily_correlations(factors[:, :, i], altered)[1] for i in range(3)])
    second, _, _ = adaptive_predictions(factors, altered, altered_rank, 48)
    np.testing.assert_array_equal(predictions[0], second[0])
    initialize()
    data = LocalStockData('csi300', '2010-01-01', '2010-06-30')
    calendar = pd.DatetimeIndex(__import__('qlib.data', fromlist=['D']).D.calendar())
    assert data.n_days == int(((calendar >= '2010-01-01') & (calendar <= '2010-06-30')).sum())
    assert torch.isnan(data.data[:100]).all(), 'Missing pre-2010 history must remain NaN'
    assert str(data._dates[100].date()) == '2010-01-04'
    close = Feature(FeatureType.CLOSE)
    target = data.forward_returns(20)
    expected = data.full_close[120:120+data.n_days]/data.full_close[100:100+data.n_days]-1
    expected = expected.masked_fill(~data.eligibility(), torch.nan)
    torch.testing.assert_close(target, expected, equal_nan=True)
    result = dict(status='passed', correlation_max_error=ic_error, rank_correlation_max_error=rank_error,
                  normal_equation_ols_max_error=ols_error, rolling_ols_prediction_max_error=max(direct_errors),
                  unavailable_future_labels_do_not_affect_prediction=True,
                  missing_early_history_preserved=True, date_tensor_alignment=True,
                  twenty_day_target_matches_direct_indexing=True)
    result['tiny_scale_correlation'] = dict(standard=float(standard_tiny), released=float(released_tiny), torch_reference=expected_tiny)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
