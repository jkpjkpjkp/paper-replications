"""Build disk-backed Alpha158 panels with dated membership and lagged JKP priors.

CSI is a 2010-start community Qlib adaptation. US uses CRSP, with unavailable
VWAP0 filled as missing (zero after train-only normalization), never proxied.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile
import warnings
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CN = ROOT.parent / 'csi/qlib/cn_data'
CRSP = ROOT.parent / 'crsp-daily/data/crsp_a_stock/dsf_v2'
THEMES = ['accruals', 'debt_issuance', 'investment', 'low_leverage', 'low_risk',
          'momentum', 'profit_growth', 'profitability', 'quality', 'seasonality',
          'short_term_reversal', 'size', 'value']


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def status(folder, phase, **kwargs):
    obj = dict(phase=phase, updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    save_json(folder / 'build_status.json', obj)
    print(json.dumps(obj), flush=True)


def write_bin(folder, field, values):
    folder.mkdir(parents=True, exist_ok=True)
    # Common support is required by Qlib Corr for newly listed/delisted names.
    # Preserve missing leading/trailing observations instead of trimming each field.
    np.r_[np.float32(0), values].astype('<f4').tofile(folder / f'{field}.day.bin')


def us_provider(folder):
    """Create only local derivative Qlib bins; preserve original CRSP files."""
    provider = folder / 'provider'
    if (provider / 'manifest.json').exists():
        prior = json.loads((provider / 'manifest.json').read_text())
        if prior.get('provider_builder_version') == 2:
            return provider, prior
    membership = pd.read_parquet(ROOT / 'sources/sp500_membership.parquet')
    ids = sorted(membership.permno.unique().tolist())
    frames, hashes = [], {}
    columns = ['permno', 'dlycaldt', 'dlyopen', 'dlyhigh', 'dlylow', 'dlyclose',
               'dlyvol', 'dlyret', 'dlycumfacpr', 'dlycumfacshr', 'tradingstatusflg']
    for year in range(2008, 2026):
        path = CRSP / f'{year}.parquet'
        receipt = json.loads(path.with_suffix('.done.json').read_text())
        digest = sha(path)
        if digest != receipt['sha256']:
            raise ValueError(f'CRSP transfer checksum mismatch: {path}')
        hashes[str(path)] = digest
        f = pd.read_parquet(path, columns=columns, filters=[('permno', 'in', ids)])
        for col in columns[2:-1]:
            f[col] = f[col].astype(float)
        f['dlycaldt'] = pd.to_datetime(f.dlycaldt)
        frames.append(f)
        status(folder, 'reading_crsp', year=year, rows=len(f))
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.dlycaldt <= '2025-01-31'].copy()
    if frame.duplicated(['permno', 'dlycaldt']).any():
        raise ValueError('Duplicate CRSP security/date')
    dates = pd.DatetimeIndex(sorted(frame.dlycaldt.unique()))
    (provider / 'calendars').mkdir(parents=True, exist_ok=True)
    (provider / 'instruments').mkdir(exist_ok=True)
    (provider / 'calendars/day.txt').write_text('\n'.join(dates.strftime('%Y-%m-%d')) + '\n')
    lines = []
    for r in membership.itertuples():
        lines.append(f'P{r.permno}\t{r.mbrstartdt}\t{r.mbrenddt}')
    (provider / 'instruments/sp500.txt').write_text('\n'.join(lines) + '\n')
    (provider / 'instruments/all.txt').write_text('\n'.join(
        f'P{i}\t{dates[0].date()}\t{dates[-1].date()}' for i in ids) + '\n')
    for permno, f in frame.groupby('permno', sort=True):
        f = f.set_index('dlycaldt').reindex(dates)
        dest = provider / 'features' / f'p{permno}'
        factor = f.dlycumfacpr.where(f.dlycumfacpr > 0)
        share = f.dlycumfacshr.where(f.dlycumfacshr > 0)
        for field in ['open', 'high', 'low', 'close']:
            values = f['dly' + field].where(f['dly' + field] > 0) / factor
            write_bin(dest, field, values.to_numpy())
        write_bin(dest, 'volume', (f.dlyvol.where(f.dlyvol >= 0) * share).to_numpy())
        write_bin(dest, 'vwap', np.full(len(dates), np.nan))
        write_bin(dest, 'totalret', f.dlyret.to_numpy())
        write_bin(dest, 'active', (f.tradingstatusflg == 'A').to_numpy(dtype=float))
    manifest = dict(provider_builder_version=2, source='WRDS CRSP CIZ dsf_v2', source_hashes=hashes,
                    membership_sha256=sha(ROOT / 'sources/sp500_membership.parquet'),
                    adjustment='OHLC / dlycumfacpr; volume * dlycumfacshr',
                    missing_feature='VWAP0: CRSP daily has no transaction VWAP; leave NaN, normalize then Fillna(0)',
                    dates=[str(dates[0].date()), str(dates[-1].date())], instruments=len(ids))
    save_json(provider / 'manifest.json', manifest)
    return provider, manifest


def membership_map(provider, universe, dates):
    spans = {}
    for line in (provider / f'instruments/{universe}.txt').read_text().splitlines():
        s, a, b = line.split()
        if a <= '2024-12-31' and b >= '2009-01-01':
            spans.setdefault(s.upper(), []).append((a, b))
    symbols = sorted(spans)
    active = np.zeros((len(symbols), len(dates)), dtype=bool)
    for i, s in enumerate(symbols):
        for a, b in spans[s]:
            active[i] |= (dates >= a) & (dates <= b)
    return symbols, active


def expressions(universe):
    from qlib.contrib.data.loader import Alpha158DL
    expr, names = Alpha158DL.get_feature_config()
    if universe == 'csi300':
        expr = [re.sub(r'\$(open|high|low|close|vwap)\b', r'($\1*$factor)', e) for e in expr]
    assert len(expr) == len(names) == 158
    return expr, names


def init_worker(provider, dates, universe):
    global WORK_DATES, WORK_EXPRS, WORK_EXTRA
    import qlib
    qlib.init(provider_uri=provider, region='cn' if universe == 'csi300' else 'us',
              kernels=1, logging_level=40, expression_cache=None, dataset_cache=None)
    WORK_DATES = pd.DatetimeIndex(dates)
    WORK_EXPRS = expressions(universe)[0]
    WORK_EXTRA = (['$close*$factor', '$volume', '$factor'] if universe == 'csi300'
                  else ['$close', '$volume', '$totalret', '$active'])


def extract(symbol):
    from qlib.data import D
    from qlib.data.cache import H
    f = D.features([symbol], WORK_EXPRS + WORK_EXTRA,
                   start_time=str(WORK_DATES[0].date()), end_time=str(WORK_DATES[-1].date()))
    if len(f):
        f = f.xs(symbol, level='instrument').reindex(WORK_DATES)
        a = f.to_numpy(dtype=np.float32)
    else:
        a = np.full((len(WORK_DATES), len(WORK_EXPRS + WORK_EXTRA)), np.nan, np.float32)
    a[~np.isfinite(a)] = np.nan
    H.clear()
    return a


def factor_inputs(country, dates):
    path = ROOT / f'sources/{country}-themes.zip'
    with zipfile.ZipFile(path) as z:
        assert len(z.namelist()) == 1
        raw = pd.read_csv(z.open(z.namelist()[0]), parse_dates=['date'])
    assert set(raw.name.unique()) == set(THEMES)
    raw = raw[(raw.location == country) & (raw.freq == 'daily') & (raw.weighting == 'vw_cap')]
    if raw.duplicated(['date', 'name']).any():
        raise ValueError('Duplicate JKP theme/day')
    returns = raw.pivot(index='date', columns='name', values='ret').reindex(pd.DatetimeIndex(dates))[THEMES]
    priors = np.expm1(np.log1p(returns.shift(1)).rolling(20, min_periods=20).sum())
    return priors.to_numpy(dtype=np.float32), dict(url=f'https://jkpfactors-data.s3.amazonaws.com/public/%5B{country}%5D_%5Ball_themes%5D_%5Bdaily%5D_%5Bvw_cap%5D.zip',
        sha256=sha(path), names=THEMES, units='decimal returns', construction='product(1+r[t-20:t])-1; current date excluded')


def robust_fit(values):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        scale = (np.nanmedian(np.abs(values - median), axis=0) + 1e-12) * 1.4826
    absent = ~np.isfinite(median) | ~np.isfinite(scale)
    median[absent], scale[absent] = 0, 1
    return median, scale, absent


def build(universe, workers):
    folder = ROOT / 'data' / universe
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        print(f'{universe}: existing completed dataset', flush=True)
        return
    provider, source = (us_provider(folder) if universe == 'sp500' else (CN, {}))
    dates = np.array((provider / 'calendars/day.txt').read_text().splitlines(), dtype='U10')
    dates = dates[(dates >= '2008-01-01') & (dates <= '2025-01-31')]
    symbols, member = membership_map(provider, universe, dates)
    np.save(folder / 'dates.npy', dates)
    np.save(folder / 'symbols.npy', np.array(symbols))
    np.save(folder / 'member.npy', member)
    if universe == 'csi300':
        transfer_path = CN.parents[1] / 'download_manifest.json'
        transfer = json.loads(transfer_path.read_text())
        files = ['calendars/day.txt', 'instruments/csi300.txt']
        files += [f'features/{s.lower()}/{f}.day.bin' for s in symbols
                  for f in ['open', 'high', 'low', 'close', 'vwap', 'volume', 'factor']]
        hashes = {}
        for rel in files:
            digest = sha(CN / rel)
            if digest != transfer['files'][rel]['sha256']:
                raise ValueError('CSI transfer checksum mismatch: ' + rel)
            hashes[rel] = digest
        source = dict(source_page=transfer['source_page'], revision=transfer['revision'],
                      transfer_sha256=sha(transfer_path), verified_files=hashes,
                      adjustment='all Alpha158 price references and close labels use raw price * factor; source volume retained',
                      caveat='Community Qlib/Tushare snapshot; vendor data and constituent accuracy unverified; 2009 absent')
    shape = (len(symbols), len(dates), 158)
    cache = folder / ('instrument_cache' if universe == 'csi300' else 'instrument_cache_v2')
    cache.mkdir(exist_ok=True)
    needed = [s for s in symbols if not (cache / f'{s}.npy').exists()]
    status(folder, 'alpha158', instruments=len(symbols), pending=len(needed))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(str(provider), dates.tolist(), universe)) as pool:
        for i, (s, a) in enumerate(zip(needed, pool.map(extract, needed, chunksize=1))):
            np.save(cache / f'{s}.npy', a)
            if i % 25 == 0 or i == len(needed)-1:
                status(folder, 'alpha158', completed=len(symbols)-len(needed)+i+1, instruments=len(symbols))
    features = np.lib.format.open_memmap(folder / 'features.npy', mode='w+', dtype=np.float32, shape=shape)
    close = np.full(shape[:2], np.nan, dtype=np.float32)
    volume, totalret = close.copy(), close.copy()
    tradable = np.zeros(shape[:2], dtype=bool)
    for i, s in enumerate(symbols):
        a = np.load(cache / f'{s}.npy')
        features[i] = a[:, :158]
        close[i], volume[i] = a[:, 158], a[:, 159]
        if universe == 'sp500':
            totalret[i] = a[:, 160]
            tradable[i] = a[:, 161] == 1
        else:
            tradable[i] = np.isfinite(a[:, 160]) & (a[:, 160] > 0)
    tradable &= np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
    if universe == 'csi300':
        totalret[:, 1:] = close[:, 1:] / close[:, :-1] - 1
    priors, factor_source = factor_inputs('chn' if universe == 'csi300' else 'usa', dates)
    train_dates = (dates >= '2009-01-01') & (dates <= '2019-12-31')
    # Fit on decision-date constituents; only raw feature/prior observations.
    si, ti = np.nonzero(member & train_dates[None, :])
    status(folder, 'fitting_train_normalization', feature_rows=len(si))
    median, scale, absent = robust_fit(features[si, ti])
    pmedian, pscale, pabsent = robust_fit(priors[ti])
    if np.any(pabsent):
        raise ValueError('A JKP theme is absent from the training period')
    for i in range(len(symbols)):
        x = np.clip((features[i] - median) / scale, -3, 3)
        features[i] = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    features.flush()
    valid_priors = np.isfinite(priors).all(1)
    np.save(folder / 'priors_raw.npy', priors)
    priors = np.nan_to_num(np.clip((priors-pmedian)/pscale, -3, 3), nan=0)
    np.save(folder / 'priors.npy', priors)
    np.save(folder / 'valid_priors.npy', valid_priors)
    np.save(folder / 'close.npy', close)
    np.save(folder / 'volume.npy', volume)
    np.save(folder / 'totalret.npy', totalret)
    np.save(folder / 'tradable.npy', tradable)
    np.savez(folder / 'normalization.npz', median=median, scale=scale, absent=absent,
             prior_median=pmedian, prior_scale=pscale)
    manifest = dict(universe=universe, shape=list(shape), dates=[str(dates[0]), str(dates[-1])],
        source=source, factors=factor_source, feature_names=expressions(universe)[1],
        normalization='Qlib RobustZScoreNorm: train membership median, (MAD+1e-12)*1.4826, clip[-3,3], Fillna(0)',
        missing_features=[expressions(universe)[1][i] for i in np.flatnonzero(absent)],
        label='close[t+h+1]/close[t+1]-1, h=0..9; target index4 equals close[t+5]/close[t+1]-1',
        split_policy='train2009-2019; validation2020-2021; test2022-2024; purge train/validation labels crossing boundary',
        train_start_actual=str(dates[np.flatnonzero(train_dates & valid_priors)[0]]),
        train_only_normalization=True, lookback=20, builder_sha256=sha(__file__),
        created_utc=datetime.now(timezone.utc).isoformat())
    manifest['output_hashes'] = {p.name: sha(p) for p in folder.glob('*.npy')}
    manifest['output_hashes']['normalization.npz'] = sha(folder / 'normalization.npz')
    save_json(folder / 'manifest.json', manifest)
    status(folder, 'complete', shape=list(shape), missing_features=manifest['missing_features'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--universe', choices=['csi300', 'sp500'], required=True)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()
    build(args.universe, args.workers)
