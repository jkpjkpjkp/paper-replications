#!/usr/bin/env python3
"""Temporary synthetic five-seed runs verify all three isolated result paths.

The raw-audit metadata is an explicit fixture. No licensed market data is read,
and all generated checkpoints, predictions and synthetic summaries are deleted.
"""
import argparse
import contextlib
import datetime as dt
import io
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
import pandas as pd

import train
from data import save_json, sha

HERE = Path(__file__).resolve().parent


def main():
    started = time.monotonic()
    rng = np.random.default_rng(20261110)
    checks = {}
    with tempfile.TemporaryDirectory(prefix='song26l-all-backbones-synthetic-') as temporary:
        root = Path(temporary)
        folder = root / 'reproduction'
        cache = folder / 'cache'
        cache.mkdir(parents=True)
        for filename in ['train.py', 'data.py', 'core.py', 'build_tensors.py', 'backbones.py', 'paper_reference.json']:
            shutil.copy2(HERE / filename, folder / filename)
        shutil.copytree(HERE / 'vendor/qlib_backbones', folder / 'vendor/qlib_backbones')
        config = json.loads((HERE / 'protocol.json').read_text())
        config.update(maximum_epochs=1, patience=1, warmup_epochs=3)
        config['model']['hidden_size'] = 4
        save_json(folder / 'protocol.json', config)
        records = []
        for split, start, count in [('train', '2019-01-02', 40),
                                     ('validation', '2023-07-03', 2),
                                     ('test', '2024-07-01', 12)]:
            for date in pd.date_range(start, periods=count, freq='B'):
                day = str(date.date())
                following = str((date + pd.offsets.BDay()).date())
                x = rng.normal(size=(24, 26, 66)).astype(np.float32)
                y = (rng.normal(size=(24, 390)).cumsum(1) / 1000 +
                     x[:, -1, 0, None] * 0.01).astype(np.float32)
                path = cache / (day + '.npz')
                np.savez_compressed(path, X=x, Y=y, permno=np.arange(24),
                                    date=np.datetime64(day, 'D'), label_date=np.datetime64(following, 'D'))
                records.append({'date': day, 'label_date': following, 'split': split,
                                'sha256': sha(path), 'bytes': path.stat().st_size,
                                'stocks': 24, 'config_sha256': 'synthetic-fixture'})
        save_json(folder / 'tensor_status.json', {'complete': True, 'config_sha256': 'synthetic-fixture', 'cached': records})
        save_json(root / 'validation.json', {'complete': True, 'evidence_type': 'synthetic fixture only',
            'summary': {'expected_sessions': 1655, 'verified_sessions': 1655, 'pending_sessions': 0, 'integrity_error_count': 0}})
        for name in ['LSTM', 'GRU', 'TCN']:
            args = argparse.Namespace(root=root, device='cpu', threads=2, backbone=name, cached_days=1400)
            with contextlib.redirect_stdout(io.StringIO()):
                train.run(args)
            output = folder / 'results'
            if name != 'LSTM':
                output = output / 'backbones' / name.lower()
            result = json.loads((output / 'summary.json').read_text())
            assert result['complete'] and result['backbone'] == name
            assert set(result['selected_horizons']) == {str(s) for s in range(5)}
            assert len(list((output / 'predictions').glob('*.parquet'))) == 10
            assert len(pd.read_csv(output / 'per_seed_metrics.csv')) == 10
            assert len(pd.read_csv(output / 'daily_metrics.csv')) == 120
            assert np.isfinite(result['five_seed_metrics']['standard']['ic']['mean'])
            assert np.isfinite(result['five_seed_metrics']['selected']['ic']['mean'])
            frozen = json.loads((output / 'manifest.json').read_text())
            assert frozen['inputs']['effective_model']['name'] == name
            checks[name] = {'status': 'passed', 'five_seed_metrics_generated_in_temporary_fixture': True,
                            'predictions': 10, 'per_seed_metric_rows': 10, 'daily_rows': 120}
    result = {'status': 'passed', 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'evidence_type': 'synthetic orchestration test; no market inputs or financial evidence',
              'temporary_outputs_deleted': True, 'seconds': time.monotonic() - started, 'backbones': checks}
    save_json(HERE / 'integration_checks.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
