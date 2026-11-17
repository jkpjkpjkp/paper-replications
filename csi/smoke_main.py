#!/usr/bin/env python3
"""Exercise the complete CSI main path on disposable synthetic inputs.

This checks fitting, held-out evaluation, aggregation, bootstrap and report
generation. All synthetic summaries/predictions/reports are deleted; the
retained artifact records only verification status, never synthetic metrics
presented as financial results.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import tempfile
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

import train

HERE = Path(__file__).resolve().parent


def build_fixture(folder):
    cache = folder / 'cache/csi300'
    cache.mkdir(parents=True)
    dates = {'train': pd.date_range('2020-01-01', periods=40).strftime('%Y-%m-%d'),
             'validation': pd.date_range('2023-07-03', periods=3).strftime('%Y-%m-%d'),
             'test': pd.date_range('2024-07-01', periods=25).strftime('%Y-%m-%d')}
    rng = np.random.default_rng(20261103)
    records = []
    for split, values in dates.items():
        for index, date in enumerate(values):
            stocks = 12 + index % 4
            x = rng.normal(size=(stocks, 158)).astype(np.float32)
            x[0, index % 158] = np.nan
            signal = 0.007 * np.nan_to_num(x[:, 0, None]) * (1 - np.exp(-np.arange(1, 11) / 3))
            y = (signal + rng.normal(scale=0.004, size=(stocks, 10)).cumsum(1)).astype(np.float32)
            y[::7, :2] = np.nan
            path = cache / f'{date}.npz'
            np.savez_compressed(path, X=x, Y=y,
                                instrument=np.array([f'SYNTHETIC-{i:03}' for i in range(stocks)]),
                                date=np.datetime64(date, 'D'))
            records.append({'date': date, 'split': split, 'file': path.name,
                            'sha256': train.sha(path), 'bytes': path.stat().st_size,
                            'stocks': stocks})
    train.save_json(cache / 'manifest.json', {'complete': True, 'records': records,
                                            'evidence_type': 'synthetic main-path fixture'})


def run():
    started = time.monotonic()
    settings = copy.deepcopy(json.loads((HERE / 'protocol.json').read_text()))
    settings['model'] = {'hidden': [8, 4], 'dropout': 0.0}
    settings['maximum_epochs'] = 1
    settings['patience'] = 1
    settings['warmup_epochs'] = 3
    assert settings['seeds'] == [0, 1, 2, 3, 4]
    original_fit_all, original_evaluate = train.fit_all, train.evaluate
    fitted = False
    evaluations = 0

    def fit_then_mark(*args, **kwargs):
        nonlocal fitted
        result = original_fit_all(*args, **kwargs)
        assert len(result[0]) == 50 and len(result[1]) == 5 and len(result[2]) == 5
        fitted = True
        return result

    def guarded_evaluate(*args, **kwargs):
        nonlocal evaluations
        if not fitted:
            raise AssertionError('Main accessed held-out evaluation before every fit completed')
        evaluations += 1
        return original_evaluate(*args, **kwargs)

    with tempfile.TemporaryDirectory(prefix='song26l-csi-main-synthetic-') as temporary:
        root = Path(temporary)
        folder = root / 'csi'
        folder.mkdir()
        for name in ['train.py', 'data.py']:
            shutil.copyfile(HERE / name, folder / name)
        (root / 'reproduction').mkdir()
        shutil.copyfile(HERE.parent / 'reproduction/core.py', root / 'reproduction/core.py')
        train.save_json(folder / 'protocol.json', settings)
        build_fixture(folder)
        with patch.object(train, 'HERE', folder), \
             patch.object(train, 'fit_all', fit_then_mark), \
             patch.object(train, 'evaluate', guarded_evaluate), \
             contextlib.redirect_stdout(io.StringIO()):
            train.run(argparse.Namespace(universe='csi300', device='cpu', threads=2))
        summary = json.loads((folder / 'results/csi300/summary.json').read_text())
        assert summary['complete'] and summary['seeds'] == [0, 1, 2, 3, 4]
        assert len(summary['horizon_sweep']) == 10
        assert set(summary['primary']) == {'standard', 'selected'}
        assert len(summary['selected_horizons']) == 5
        assert len(summary['lambda_test_ic_rank_correlation']['per_seed']) == 5
        assert summary['paired_ic_bootstrap']['block_days'] == 20
        assert summary['paired_ic_bootstrap']['draws'] == settings['bootstrap']['replicates']
        assert evaluations == 55
        assert len(list((folder / 'results/csi300/predictions').glob('*.parquet'))) == 55
        per_seed = pd.read_csv(folder / 'results/csi300/per_seed_metrics.csv')
        assert len(per_seed) == 55
        daily = pd.read_csv(folder / 'results/csi300/daily_metrics.csv')
        assert len(daily) == 55 * 25
        assert np.isfinite(daily[['ic', 'rank_ic', 'top_return_10sessions']]).all().all()
        markdown = (folder / 'REPORT.md').read_text()
        html = (folder / 'REPORT.html').read_text()
        assert 'CSI300' in markdown and 'standard' in markdown and 'selected' in markdown
        assert '<table>' in html and '<img' in html
        assert (folder / 'results/horizon_curves.png').stat().st_size > 1000
        status = json.loads((folder / 'results/csi300/status.json').read_text())
        assert status['phase'] == 'complete'
    result = {'status': 'passed', 'evidence_type': 'fully synthetic main-path verification; no financial inputs or retained synthetic metrics',
              'stages': ['five-seed 10-horizon fit', 'five BLO selections and retrains',
                         'held-out evaluation after all fits', '55 prediction exports',
                         'five-seed aggregation', '20-date 10000-draw bootstrap',
                         'weight/IC rank correlation', 'Markdown and HTML report', 'horizon plot'],
              'elapsed_seconds': time.monotonic() - started,
              'temporary_artifacts_deleted': True,
              'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    train.save_json(HERE / 'synthetic_main_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
