from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd


def labels_from_market(close, volume, high, low, vwap):
    """Return y[t] realized at t+2, following Section II exactly."""
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.where(volume > 0, volume, np.nan))
        prior = pd.DataFrame(logv.T).rolling(5, min_periods=5).mean().to_numpy().T
        y = np.full((*close.shape, 3), np.nan, dtype=np.float32)
        y[:, :-2, 0] = close[:, 2:] / close[:, 1:-1] - 1
        y[:, :-2, 1] = logv[:, 2:] - prior[:, 1:-1]
        y[:, :-2, 2] = (high[:, 2:] - low[:, 2:]) / vwap[:, 2:]
    y[~np.isfinite(y)] = np.nan
    return y


def cs_zscore(y):
    return (y - np.nanmean(y, axis=0)) / np.maximum(np.nanstd(y, axis=0, ddof=1), 1e-8)


def fill_window(x):
    """ffill then within-observed-window mean; never read beyond signal t."""
    x = x.copy()
    if np.isfinite(x).all():
        return x
    for t in range(1, x.shape[1]):
        missing = ~np.isfinite(x[:, t])
        x[:, t][missing] = x[:, t-1][missing]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(x, axis=1, keepdims=True)
    return np.nan_to_num(np.where(np.isfinite(x), x, mean), nan=0., posinf=0., neginf=0.)


class Panel:
    def __init__(self, folder, lookback=21):
        folder = Path(folder)
        self.folder, self.lookback = folder, lookback
        self.manifest = json.loads((folder / "manifest.json").read_text())
        for key in ["features", "labels", "member", "market", "dates", "symbols"]:
            setattr(self, key, np.load(folder / f"{key}.npy", mmap_mode="r"))
        # Inference eligibility uses information observable at t, never t+2.
        self.eligible = self.member & np.isfinite(self.market[:, :, 0]) & (self.market[:, :, 1] > 0)

    def split(self, split):
        bounds = dict(train=("2010-01-01", "2014-12-31"),
                      valid=("2015-01-01", "2016-12-31"),
                      test=("2017-01-01", "2020-08-01"))
        a, b = bounds[split]
        # Purge labels whose realization date t+2 is outside this split.
        idx = np.arange(self.lookback - 1, len(self.dates) - 2)
        return idx[(self.dates[idx] >= a) & (self.dates[idx + 2] <= b) &
                   (self.eligible[:, idx].sum(axis=0) >= 50)]

    def day(self, t):
        ids = np.flatnonzero(self.eligible[:, t])
        x = fill_window(self.features[ids, t-self.lookback+1:t+1])
        y = np.array(self.labels[ids, t])
        return ids, x, cs_zscore(y).astype(np.float32), y
