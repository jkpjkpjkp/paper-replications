"""Cache author-equivalent signatures once per distinct window.

Uses the unmodified Dataset_Sig and StandardScaler; rejects unequal original
samples before and after scaling. Does not change splits, features or returns.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_provider.data_loader import Dataset_Sig


class CachedWindows(Dataset_Sig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        last = int(self.valid_idx[-1]) + self.H
        self.window_x = np.empty((last, self.D, 2), dtype=np.float32)
        self.window_cross = np.empty((last, self.D, self.D, 1), dtype=np.float32)
        for i in range(last):
            # Last axis contiguous: same 1D reduction order as author's np.sum.
            p = np.ascontiguousarray(self.prices[i:i+self.W].T)
            delta = np.diff(p, axis=1, prepend=p[:, :1])
            x = self.window_x[i]
            x[:, 0] = p[:, -1] - p[:, 0]
            x[:, 1] = np.sum(p[:, :-1] * np.diff(p, axis=1), axis=1)
            c = np.sum(np.cumsum(delta, axis=1)[:, None, :-1]
                       * delta[None, :, 1:], axis=-1)
            np.fill_diagonal(c, 0)
            self.window_cross[i, :, :, 0] = c

    def _build_sample(self, start_idx):
        a = start_idx
        x, c = self._scale_sigs(self.window_x[a:a+self.H].copy(),
                                self.window_cross[a:a+self.H].copy())
        h = a + self.W
        now, nxt = self.raw_prices[h:h+self.H], self.raw_prices[h+1:h+self.H+1]
        dates = self.dates[h:h+self.H].strftime('%Y-%m-%d').tolist()
        from utils.timefeatures import time_features
        return dict(x_sigs=x, cross_sigs=c,
                    future_return_unscaled=((nxt-now)/(now+1e-8)).astype(np.float32),
                    dates_horizon=dates,
                    date_feats=time_features(pd.to_datetime(dates), freq='B').T.astype(np.float32))


def check(original, cached, indices):
    errors = {}
    for i in indices:
        a, b = original[int(i)], cached[int(i)]
        for k in a:
            if k == 'dates_horizon':
                assert a[k] == b[k]
            else:
                err = float(np.max(np.abs(a[k]-b[k])))
                errors[k] = max(errors.get(k, 0.), err)
                assert np.array_equal(a[k], b[k]), (i, k, err)
    return errors


def prepare(pool, verify_only=False):
    tick = time.perf_counter()
    df = pd.read_csv(ROOT/'asset_data/full_dataset.csv', parse_dates=['Date']).set_index('Date').iloc[:, :pool]
    context = (pd.Timestamp('2020-01-01') - BDay(61)).strftime('%Y-%m-%d')
    frames = {'train': df.loc['2000':'2016'], 'val': df.loc['2017':'2019'],
              'test': df.loc[context:'2024-12-31']}
    target = ROOT/'signature_cache_6020'/f'pool_{pool}'
    target.mkdir(parents=True, exist_ok=True)
    report = {'pool': pool, 'numpy': np.__version__, 'pandas': pd.__version__,
              'csv_sha256': hashlib.sha256((ROOT/'asset_data/full_dataset.csv').read_bytes()).hexdigest(),
              'split_checks': {}}
    scaler = None
    for split, frame in frames.items():
        kw = dict(pred_start='2020-01-01', pred_end='2024-12-31') if split == 'test' else {}
        old = Dataset_Sig(frame, frame, 60, 20, **kw)
        ds = CachedWindows(frame, frame, 60, 20, **kw)
        indices = sorted(set([0, 1, len(ds)//2, len(ds)-1] + np.random.default_rng(918).integers(0, len(ds), 3).tolist()))
        raw_errors = check(old, ds, indices)
        print(pool, split, 'raw equivalence', raw_errors, flush=True)
        if verify_only:
            continue
        N, H, D = len(ds), ds.H, ds.D
        if split == 'train':
            flat = np.empty((N, H*(D*2+D*D)), dtype=np.float32)
            for i, a in enumerate(ds.valid_idx):
                flat[i] = np.concatenate([ds.window_x[a:a+H].ravel(), ds.window_cross[a:a+H].ravel()])
            scaler = StandardScaler().fit(flat)
            del flat
            joblib.dump(scaler, target/'signature_scaler.pkl')
        old.set_sig_scaler(scaler)
        ds.set_sig_scaler(scaler)
        scaled_errors = check(old, ds, indices)
        report['split_checks'][split] = dict(N=N, H=H, D=D,
                                             indices=indices, raw=raw_errors, scaled=scaled_errors,
                                             first_date=ds[0]['dates_horizon'][0],
                                             last_date=ds[N-1]['dates_horizon'][-1])
        arrays = {k: np.memmap(target/f'{split}_{k}.npy', dtype='float32', mode='w+', shape=shape)
                  for k, shape in [('x', (N,H,D,2)), ('cross', (N,H,D,D,1)), ('ret', (N,H,D))]}
        dates = []
        for i in range(N):
            sample = ds[i]
            for k, source in [('x','x_sigs'), ('cross','cross_sigs'), ('ret','future_return_unscaled')]:
                arrays[k][i] = sample[source]
            dates.append(sample['dates_horizon'])
        for mm in arrays.values():
            mm.flush()
        np.save(target/f'{split}_dates.npy', np.array(dates, dtype=object))
        (target/f'{split}_meta.json').write_text(json.dumps(dict(N=N,H=H,D=D)))
        print(pool, split, 'saved', N, flush=True)
    report['seconds'] = time.perf_counter()-tick
    report['verify_only'] = verify_only
    (ROOT/'reproduction'/f'cache_equivalence_{pool}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--pools', nargs='+', type=int, default=[40,50])
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    for pool in args.pools:
        prepare(pool, args.verify_only)
