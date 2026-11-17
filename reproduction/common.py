"""Explicit community-data contract and local integration for the pinned release."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream'
DATA = Path(os.environ.get('ALPHASAGE_DATA', ROOT.parent / 'csi/qlib/cn_data'))
CACHE = Path(os.environ.get('ALPHASAGE_CACHE', ROOT / 'cache'))
ADAPTER_VERSION = 3
UPSTREAM_COMMIT = '517467a34909512a92d6e3139df54891957560de'
sys.path[:0] = [str(UPSTREAM / 'src'), str(UPSTREAM)]

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from alphagen_qlib.stock_data import StockData as OriginalStockData


def initialize():
    import qlib
    qlib.init(provider_uri=str(DATA), region='cn', kernels=2,
              joblib_backend='threading', expression_cache=None, dataset_cache=None)
    OriginalStockData._qlib_initialized = True
    from integration import install_expression_guards
    install_expression_guards()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fresh_output(path):
    """Refuse partial as well as completed runs, and writes through archive links."""
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f'Output must not traverse a symlink: {path}')
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output already contains artifacts; choose a fresh directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def adjusted_values(frame, features):
    """Snapshot stores raw prices, volume in 100-share lots, and cumulative factors.

    Keep its stored adjustment anchor explicit. Do not divide by a future price
    or normalize to a test-date factor. This is not the authors' exact snapshot.
    """
    factor = frame['$factor'].to_numpy(dtype=np.float64)
    close = frame['$close'].to_numpy(dtype=np.float64)
    bad = np.isfinite(close) & (close > 0) & (~np.isfinite(factor) | (factor <= 0))
    if bad.any():
        raise ValueError(f'Missing/invalid adjustment factors for {int(bad.sum())} observed prices')
    factor = np.where(np.isfinite(factor) & (factor > 0), factor, np.nan)
    columns = []
    for feature in features:
        raw = frame['$' + feature.name.lower()].to_numpy(dtype=np.float64)
        if feature.name == 'VOLUME':
            adjusted = raw * 100.0 / factor
        else:
            adjusted = raw * factor
            adjusted[raw <= 0] = np.nan
        adjusted[~np.isfinite(adjusted)] = np.nan
        columns.append(adjusted)
    full_close = close * factor
    full_close[(close <= 0) | ~np.isfinite(full_close)] = np.nan
    return np.column_stack(columns).astype(np.float32), full_close.astype(np.float32)


def normalize_members(value, eligible, *, fill_missing=True):
    """Normalize only observed current members; never add future columns as zeros."""
    good = eligible & torch.isfinite(value)
    clean = torch.where(good, value, 0.)
    count = good.sum(1).clamp_min(1)
    mean = clean.sum(1) / count
    centered = torch.where(good, value - mean[:, None], 0.)
    std = (centered.square().sum(1) / count).sqrt()
    normalized = centered / torch.where(std > 0, std, 1.)[:, None]
    keep = eligible if fill_missing else good
    return normalized.masked_fill(~keep, torch.nan)


class LocalStockData(OriginalStockData):
    """Adjusted member features plus uncensored close history for forward labels."""
    def __init__(self, instrument, start_time, end_time, qlib_path=None,
                 device=torch.device('cpu'), **kwargs):
        super().__init__(instrument, start_time, end_time, qlib_path=str(DATA),
                         device=device, **kwargs)

    def _get_data(self):
        from qlib.data import D
        if self.raw or self.freq != 'day':
            raise ValueError('This adapter explicitly adjusts the raw daily snapshot; raw=True is unsupported')
        calendar = pd.DatetimeIndex(D.calendar(freq=self.freq))
        start = calendar.searchsorted(pd.Timestamp(self._start_time))
        stop = calendar.searchsorted(pd.Timestamp(self._end_time), side='right')
        first = max(0, start - self.max_backtrack_days)
        last = stop + self.max_future_days
        if stop <= start or last > len(calendar):
            raise ValueError('Empty requested period or insufficient future context in source calendar')
        real_dates = calendar[first:last]
        dates = real_dates
        missing = max(0, self.max_backtrack_days - start)
        if missing:
            placeholders = pd.bdate_range(end=real_dates[0] - pd.Timedelta(days=1), periods=missing)
            dates = placeholders.append(real_dates)
        if isinstance(self._instrument, str):
            intervals = D.list_instruments(D.instruments(self._instrument),
                start_time=real_dates[0], end_time=real_dates[-1], freq=self.freq, as_list=False)
            membership_file = DATA / 'instruments' / (self._instrument + '.txt')
        else:
            intervals = {s: [(real_dates[0], real_dates[-1])] for s in self._instrument}
            membership_file = None
        stocks = pd.Index(sorted(intervals))
        if not len(stocks):
            raise ValueError('No instruments in the requested period')
        features = ['$' + f.name.lower() for f in self._features]
        fields = list(dict.fromkeys([*features, '$close', '$factor']))
        hashes = {f'calendars/{self.freq}.txt': sha256(DATA / 'calendars' / (self.freq + '.txt'))}
        if membership_file:
            hashes[str(membership_file.relative_to(DATA))] = sha256(membership_file)
        for stock in stocks:
            for field in fields:
                relative = f'features/{stock.lower()}/{field[1:]}.{self.freq}.bin'
                hashes[relative] = sha256(DATA / relative)
        config = dict(instrument=self._instrument, start=self._start_time,
                      end=self._end_time, back=self.max_backtrack_days,
                      future=self.max_future_days, features=features, freq=self.freq,
                      adapter=ADAPTER_VERSION, data_directory=str(DATA.resolve()),
                      input_sha256=hashes,
                      prices='raw OHLC/VWAP multiplied by stored cumulative factor',
                      volume='raw 100-share lots multiplied by 100, divided by factor',
                      adjustment_anchor='stored snapshot factor; no endpoint rebasing',
                      membership='features and labels masked at formation; future close independent of membership',
                      data_revision='ee0fa6b2299cb5e927d2df491888c84f6c9c027a')
        key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:20]
        cache = CACHE / f'{key}.npz'
        cache.parent.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (cache.parent, *cache.parent.parents)):
            raise ValueError('Use a fresh writable cache, not an archive cache symlink')
        if cache.exists():
            with np.load(cache, allow_pickle=False) as f:
                values = f['values']
                dates = pd.DatetimeIndex(f['dates'])
                stocks = pd.Index(f['stocks'])
                full_close = f['full_close']
                eligible = f['eligible']
        else:
            # A list requests full price histories, not membership-censored rows.
            frame = D.features(list(stocks), fields, real_dates[0], real_dates[-1], freq=self.freq)
            index = pd.MultiIndex.from_product([stocks, dates], names=['instrument', 'datetime'])
            frame = frame.reindex(index)
            flat, close = adjusted_values(frame, self._features)
            values = flat.reshape(len(stocks), len(dates), len(features)).transpose(1, 2, 0).copy()
            full_close = close.reshape(len(stocks), len(dates)).T.copy()
            eligible = np.zeros((len(dates), len(stocks)), dtype=bool)
            for column, stock in enumerate(stocks):
                for low, high in intervals[stock]:
                    eligible[:, column] |= (dates >= pd.Timestamp(low)) & (dates <= pd.Timestamp(high))
            eligible &= np.isfinite(full_close)
            values = np.where(eligible[:, None, :], values, np.nan)
            with tempfile.NamedTemporaryFile(dir=cache.parent, suffix='.npz', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, values=values, full_close=full_close, eligible=eligible,
                                    dates=dates.to_numpy(), stocks=stocks.to_numpy(dtype=str))
            temporary.replace(cache)
            (cache.with_suffix('.json')).write_text(json.dumps(config, indent=2))
        self.full_close = torch.as_tensor(full_close, dtype=torch.float32, device=self.device)
        self.eligible = torch.as_tensor(eligible, dtype=torch.bool, device=self.device)
        self.provenance = dict(config, cache_key=key, cache_path=str(cache))
        self.df_bak = None
        tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        print(json.dumps(dict(event='data_loaded', instrument=self._instrument,
                              start=self._start_time, end=self._end_time,
                              days=len(dates)-self.max_backtrack_days-self.max_future_days,
                              stocks=len(stocks), shape=list(tensor.shape),
                              finite_fraction=float(np.isfinite(values).mean()), cache=str(cache))), flush=True)
        return tensor, dates, stocks

    def eligibility(self, period=slice(0, 1)):
        start = self.max_backtrack_days + period.start
        stop = self.max_backtrack_days + self.n_days + period.stop - 1
        return self.eligible[start:stop]

    def normalize(self, value, *, fill_missing=True):
        return normalize_members(value, self.eligibility(), fill_missing=fill_missing)

    def forward_returns(self, horizon=20, period=slice(0, 1)):
        if horizon <= 0 or period.start < -self.max_backtrack_days or period.stop - 1 + horizon > self.max_future_days:
            raise ValueError('Forward-return horizon outside available data context')
        start = self.max_backtrack_days + period.start
        stop = self.max_backtrack_days + self.n_days + period.stop - 1
        result = self.full_close[start+horizon:stop+horizon] / self.full_close[start:stop] - 1
        return result.masked_fill(~self.eligibility(period) | ~torch.isfinite(result), torch.nan)


def daily_correlations(x, y, released_epsilon=False):
    """Paired finite mask and average ties, without a quadratic rank allocation.

    released_epsilon reproduces upstream's scale-dependent 1e-3 std clamp;
    ordinary Pearson is the default and is reported explicitly as such.
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    def corr(a, b):
        n = valid.sum(1)
        a = np.where(valid, a, 0)
        b = np.where(valid, b, 0)
        am = np.divide(a.sum(1), n, out=np.zeros(len(n)), where=n > 0)
        bm = np.divide(b.sum(1), n, out=np.zeros(len(n)), where=n > 0)
        a = np.where(valid, a-am[:, None], 0)
        b = np.where(valid, b-bm[:, None], 0)
        aa, bb = (a*a).sum(1), (b*b).sum(1)
        denominator = np.sqrt(aa * bb)
        if released_epsilon:
            safe_n = np.maximum(n, 1)
            tiny = (np.sqrt(aa/safe_n) < 1e-3) | (np.sqrt(bb/safe_n) < 1e-3)
            denominator = np.where(tiny, n, denominator)
        return np.divide((a*b).sum(1), denominator, out=np.zeros(len(n)), where=denominator > 0)
    ic = corr(x, y)
    rx = rankdata(np.where(valid, x, np.nan), axis=1, method='average', nan_policy='omit')
    ry = rankdata(np.where(valid, y, np.nan), axis=1, method='average', nan_policy='omit')
    ric = corr(rx, ry)
    return ic, ric


def metrics(predictions, targets):
    ic, ric = daily_correlations(predictions, targets)
    released_ic, _ = daily_correlations(predictions, targets, released_epsilon=True)
    pred_array = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.asarray(predictions)
    target_array = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    target_count = int(np.isfinite(target_array).sum())
    paired_count = int((np.isfinite(pred_array) & np.isfinite(target_array)).sum())
    return dict(ic=float(ic.mean()), icir=float(ic.mean()/ic.std(ddof=1)) if ic.std() else 0.,
                rank_ic=float(ric.mean()), rank_icir=float(ric.mean()/ric.std(ddof=1)) if ric.std() else 0.,
                ic_released_evaluator=float(released_ic.mean()),
                icir_released_evaluator=float(released_ic.mean()/released_ic.std(ddof=1)) if released_ic.std() else 0.,
                target_finite_pairs=target_count, prediction_target_finite_pairs=paired_count,
                prediction_coverage_of_finite_targets=float(paired_count/target_count) if target_count else 0.,
                days=len(ic))


def load_expressions(path):
    import alphagen.data.expression as expressions
    from alphagen_qlib.stock_data import FeatureType
    namespace = {name: getattr(expressions, name) for name in dir(expressions) if not name.startswith('_')}
    namespace.update({f.name.lower(): expressions.Feature(f) for f in FeatureType})
    pool = json.loads(Path(path).read_text())
    # This parser accepts only locally generated upstream expressions.
    result = [eval(expr.replace('$', ''), {'__builtins__': {}}, namespace) for expr in pool['exprs']]
    return result, np.asarray(pool['weights'])
