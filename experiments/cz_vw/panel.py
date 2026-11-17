"""Daily PRISM batches with point-in-time monthly CZ characteristics."""
import json
import numpy as np
import pandas as pd
import torch
from common import DATA, settings
from runtime import Panel as OriginalPanel


class Panel(OriginalPanel):
    def __init__(self, universe='sp500'):
        assert universe == 'sp500'
        self.folder = DATA
        self.manifest = json.loads((DATA / 'manifest.json').read_text())
        for name in ['dates', 'symbols', 'close', 'priors', 'member', 'tradable', 'valid_priors',
                     'totalret', 'terminal_events', 'monthly_features', 'month_index', 'source_months',
                     'lagged_cap', 'price_return']:
            setattr(self, name, np.load(DATA / f'{name}.npy', mmap_mode='r'))
        p = settings()
        self.plans = {}
        for split in ['train', 'valid', 'test']:
            start, stop = p[split]
            self.plans[split] = np.array([t for t in range(p['lookback'] - 1, len(self.dates) - 10)
                if start <= self.dates[t] <= stop and self.valid_priors[t]
                and (split == 'test' or self.dates[t + 10] <= stop)
                and len(self.ids(t, split)) >= 30], dtype=np.int64)

    def batch(self, t, split, device='cpu'):
        ids = self.ids(t, split)
        months = self.month_index[t - 19:t + 1]
        x = np.array(self.monthly_features[ids[:, None], months[None, :]], copy=True)
        prior = np.repeat(self.priors[t:t + 1], len(ids), axis=0).copy()
        with np.errstate(invalid='ignore', divide='ignore'):
            labels = np.asarray(self.close[ids, t + 1:t + 11] / self.close[ids, t + 1:t + 2] - 1)
        labels[~np.isfinite(labels)] = np.nan
        if split != 'test':
            labels = ((pd.DataFrame(labels).rank(pct=True) - .5) * 3.46).to_numpy(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(prior).to(device), torch.from_numpy(labels.copy()).to(device), ids
