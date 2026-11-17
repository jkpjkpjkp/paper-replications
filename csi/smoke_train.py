#!/usr/bin/env python3
"""Synthetic end-to-end CSI fitting and checkpoint-recovery smoke check.

No actual CSI inputs or financial test metrics are read or generated. Model
checkpoints are created in a temporary directory and deleted after the check.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

import train

HERE = Path(__file__).resolve().parent


class SyntheticDataset:
    def __init__(self, device='cpu'):
        self.device = device
        self.splits = {'train': [f'train-{i:02}' for i in range(40)],
                       'validation': [f'valid-{i:02}' for i in range(3)],
                       'test': [f'held-out-{i:02}' for i in range(12)]}
        self.items = {}
        self.accessed = []
        rng = np.random.default_rng(20261103)
        for split in ['train', 'validation']:
            for index, date in enumerate(self.splits[split]):
                stocks = 12 + index % 3
                x = rng.normal(size=(stocks, 158)).astype(np.float32)
                noise = rng.normal(scale=0.004, size=(stocks, 10)).cumsum(1)
                signal = 0.007 * x[:, 0, None] * (1 - np.exp(-np.arange(1, 11) / 3))
                y = (signal + noise).astype(np.float32)
                y[::7, :2] = np.nan
                self.items[date] = x, y

    def days(self, dates):
        if any(date.startswith('held-out') for date in dates):
            raise AssertionError('Actual fitting accessed the held-out split')
        self.accessed.extend(dates)
        pairs = [self.items[date] for date in dates]
        x = np.concatenate([pair[0] for pair in pairs])
        y = np.concatenate([pair[1] for pair in pairs])
        ids = np.concatenate([np.full(len(pair[0]), index, dtype=np.int64)
                              for index, pair in enumerate(pairs)])
        return (torch.as_tensor(x, device=self.device),
                torch.as_tensor(y, device=self.device),
                torch.as_tensor(ids, device=self.device))


def run():
    torch.set_num_threads(2)
    settings = copy.deepcopy(json.loads((HERE / 'protocol.json').read_text()))
    settings['model'] = {'hidden': [8, 4], 'dropout': 0.0}
    settings['maximum_epochs'] = 2
    settings['patience'] = 2
    settings['warmup_epochs'] = 3
    settings['seeds'] = [0]
    dataset = SyntheticDataset()
    stages = []
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='song26l-csi-synthetic-') as temporary:
        folder = Path(temporary)
        fitted, selected, retrained = train.fit_all(
            dataset, settings, folder, 'synthetic-integration-only', stages.append)
        assert set(fitted) == {(h, 0) for h in range(1, 11)}
        assert 1 <= selected[0]['selected_horizon'] <= 10
        assert retrained[0]['horizon'] == selected[0]['selected_horizon']
        weights = np.asarray(selected[0]['weights'])
        assert np.isfinite(weights).all() and np.isclose(weights.sum(), 1)
        assert all(np.isfinite(record['hypergradient_norm'])
                   for record in selected[0]['history'])
        def reused_progress(_):
            raise AssertionError('Completed fit unexpectedly retrained')
        train.supervised_fit(dataset, settings, 0, 10, folder / 'h10-seed0.pt',
                             'synthetic-integration-only', reused_progress)
        train.select_horizon(dataset, settings, 0, folder / 'blo-seed0.pt',
                             'synthetic-integration-only', reused_progress)
        path = folder / 'h10-seed0.pt'
        with path.open('ab') as stream:
            stream.write(b'corrupted-synthetic-checkpoint')
        try:
            train.checkpoint_load(path, 'synthetic-integration-only')
        except RuntimeError as error:
            assert 'checksum' in str(error)
        else:
            raise AssertionError('Corrupted checkpoint was accepted')
    result = {'status': 'passed', 'evidence_type': 'synthetic integration only; no financial inputs or metrics',
              'device': dataset.device, 'elapsed_seconds': time.monotonic() - started,
              'stages': ['10-horizon fitting', '3-epoch warmup', 'differentiable BLO',
                         'selected-horizon retraining', 'checkpoint reuse', 'corruption rejection',
                         'held-out split access guard'],
              'temporary_artifacts_deleted': True,
              'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    train.save_json(HERE / 'synthetic_training_check.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    run()
