"""Verified daily tensors and streaming training-only feature preprocessing."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from build_tensors import FEATURES


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


class DailyStore:
    def __init__(self, root, device='cpu', *, require_complete=True, max_cached_days=40):
        self.root = Path(root)
        self.folder = self.root / 'reproduction/cache'
        self.status = json.loads((self.root / 'reproduction/tensor_status.json').read_text())
        if require_complete and not self.status.get('complete'):
            raise RuntimeError('Financial training requires a complete tensor manifest')
        self.receipts = {r['date']: r for r in self.status['cached']}
        self.splits = {s: sorted(d for d, r in self.receipts.items() if r['split'] == s)
                       for s in ['train', 'validation', 'test']}
        self.device = device
        self.max_cached_days = max_cached_days
        self.cache = OrderedDict()
        self.verified = set()
        self.mean = self.scale = None

    def raw(self, date):
        if date in self.cache:
            self.cache.move_to_end(date)
            return self.cache[date]
        receipt = self.receipts[date]
        path = self.folder / (date + '.npz')
        if date not in self.verified:
            if path.stat().st_size != receipt['bytes'] or sha(path) != receipt['sha256']:
                raise ValueError('Tensor receipt mismatch: ' + date)
            if receipt['config_sha256'] != self.status['config_sha256']:
                raise ValueError('Mixed tensor protocols: ' + date)
            self.verified.add(date)
        with np.load(path, allow_pickle=False) as z:
            item = {k: z[k] for k in z.files}
        if str(item['date']) != date or str(item['label_date']) != receipt['label_date']:
            raise ValueError('Tensor date does not match receipt: ' + date)
        x, y = item['X'], item['Y']
        if x.shape[1:] != (26, len(FEATURES)) or y.shape != (len(x), 390):
            raise ValueError('Unexpected tensor shape: ' + date)
        if not np.isfinite(y).all():
            raise ValueError('All candidate prices must be available: ' + date)
        self.cache[date] = item
        while len(self.cache) > self.max_cached_days:
            self.cache.popitem(last=False)
        return item

    def fit_scaler(self, output):
        dates = self.splits['train']
        if not dates:
            raise ValueError('No training dates; validation/test cannot fit preprocessing')
        # Stable parallel Welford aggregation over finite feature observations.
        n = np.zeros(len(FEATURES), np.int64)
        mean = np.zeros(len(FEATURES), np.float64)
        m2 = np.zeros(len(FEATURES), np.float64)
        for date in dates:
            x = self.raw(date)['X'].reshape(-1, len(FEATURES)).astype(np.float64)
            good = np.isfinite(x)
            count = good.sum(0)
            local = np.where(good, x, 0).sum(0) / np.maximum(count, 1)
            centered = np.where(good, x - local, 0)
            local_m2 = (centered * centered).sum(0)
            total = n + count
            delta = local - mean
            mean += delta * count / np.maximum(total, 1)
            m2 += local_m2 + delta * delta * n * count / np.maximum(total, 1)
            n = total
        scale = np.sqrt(m2 / np.maximum(n, 1))
        self.mean = mean
        self.scale = np.where(scale > 1e-8, scale, 1)
        save_json(output, {'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                           'finite_count': n.tolist(), 'fit_dates': dates,
                           'config_sha256': self.status['config_sha256']})

    def day(self, date):
        if self.mean is None:
            raise RuntimeError('Fit the training-only feature scaler first')
        item = self.raw(date)
        x = np.where(np.isfinite(item['X']), item['X'], self.mean)
        x = (x - self.mean) / self.scale
        return (torch.as_tensor(x, device=self.device, dtype=torch.float64),
                torch.as_tensor(item['Y'], device=self.device, dtype=torch.float64))

    def days(self, dates):
        return [self.day(d) for d in dates]
