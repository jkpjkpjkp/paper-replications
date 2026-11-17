"""Build an explicitly labelled CRSP substitute for the released WIKI loader."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, default=ROOT.parent)
    args = ap.parse_args()
    out = ROOT / 'data'
    out.mkdir(exist_ok=True)
    membership = ROOT / 'sources/sp500_membership.parquet'
    m = pd.read_parquet(membership)
    m['mbrstartdt'] = pd.to_datetime(m.mbrstartdt)
    m['mbrenddt'] = pd.to_datetime(m.mbrenddt)
    universe = sorted(m.loc[(m.mbrstartdt <= '2004-01-02') &
                            (m.mbrenddt >= '2004-01-02'), 'permno'].unique())
    columns = ['permno', 'dlycaldt', 'dlyret', 'dlyprc', 'dlyvol', 'ticker']
    parts, sources = [], []
    for year in range(2004, 2017):
        path = args.data_root / f'crsp-daily/parquet/{year}.parquet'
        d = pq.read_table(path, columns=columns,
                          filters=[('permno', 'in', universe)]).to_pandas()
        for c in ['dlyret', 'dlyprc', 'dlyvol']:
            d[c] = pd.to_numeric(d[c], errors='coerce')
        parts.append(d)
        sources.append({'path': str(path.resolve()), 'sha256': sha256(path), 'rows_selected': len(d)})
        print(year, len(d), flush=True)
    raw = pd.concat(parts, ignore_index=True)
    raw['dlycaldt'] = pd.to_datetime(raw.dlycaldt)
    assert not raw.duplicated(['dlycaldt', 'permno']).any()
    dates = np.sort(raw.dlycaldt.unique())
    ret = raw.pivot(index='dlycaldt', columns='permno', values='dlyret').reindex(dates)
    # Same complete-history restriction as the release, disclosed as survivorship conditioning.
    eligible = ret.columns[ret.notna().all() & (ret > -1).all()].to_numpy()
    # Execute only the pure feature function, avoiding the upstream API-key import.
    src = ROOT / 'stock_portfolio/data_utils.py'
    tree = ast.parse(src.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_monthly_cols')
    ns = {'pd': pd, 'np': np}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(src), 'exec'), ns)
    features, targets, realized, symbols = [], [], [], []
    future = [f'next{i}_return' for i in range(1, 11)]
    for permno in eligible:
        d = raw[raw.permno == permno].set_index('dlycaldt').reindex(dates)
        # Total-return adjusted price index, anchored to the first observed price.
        close = (1 + d.dlyret).cumprod()
        close = close / close.iloc[0] * abs(d.dlyprc.iloc[0])
        f = ns['compute_monthly_cols'](pd.DataFrame({'Close': close}))
        f['Close'] = close.loc[f.index]
        feature_cols = [c for c in f if c not in future]
        assert len(feature_cols) == 28
        assert np.isfinite(f.to_numpy()).all()
        if features:
            assert f.index.equals(feature_dates)
        feature_dates = f.index
        features.append(f[feature_cols].to_numpy(np.float32))
        targets.append(f.next1_return.to_numpy(np.float32))
        realized.append(d.dlyret.shift(-1).loc[f.index].to_numpy(np.float32))
        symbols.append(str(d.ticker.dropna().iloc[0]))
    x = np.stack(features, axis=1)
    y = np.stack(targets, axis=1)
    assert np.allclose(y, np.stack(realized, axis=1), atol=1e-7)
    n = len(y)
    train_end, val_end = int(n*.7), int(n*.7)+int(n*.1)
    np.savez_compressed(out / 'crsp_panel.npz', x=x, y=y,
                        dates=feature_dates.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        target_dates=pd.Series(dates, index=dates).shift(-1).loc[feature_dates].dt.strftime('%Y-%m-%d').to_numpy(dtype='U10'),
                        permnos=eligible, symbols=np.array(symbols), feature_names=np.array(feature_cols))
    receipt = {
        'status': 'alternative_data_replication_not_exact_WIKI',
        'source': 'CRSP CIZ daily total returns, 2026 local snapshot',
        'membership': {'path': str(membership.resolve()), 'sha256': sha256(membership), 'asof': '2004-01-02'},
        'input_files': sources, 'raw_members': len(universe), 'complete_history_members': len(eligible),
        'shape': list(x.shape), 'feature_names': feature_cols,
        'date_range': [str(feature_dates[0].date()), str(feature_dates[-1].date())],
        'split_counts_before_one_day_embargo': [train_end, val_end-train_end, n-val_end],
        'feature_function_sha256': sha256(src),
        'panel_sha256': sha256(out / 'crsp_panel.npz'),
        'limitations': [
            'CRSP total-return index replaces Quandl dividend/split-adjusted close; exact source not supplied.',
            'S&P 500 membership fixed at 2004-01-02; original selected securities are unpublished.',
            'Complete-history selection conditions on survival through 2016, as in released loader; not a point-in-time investable-universe backtest.',
            'Released features are 28 per stock (1400 for 50 stocks), without PCA or volume; paper describes 28 total with PCA/volume.',
            'Date endpoint is exclusive 2017-01-01, following released code; trailing 365-day warmup and final 10-day trim retained.',
        ]}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k != 'input_files'}, indent=2))


if __name__ == '__main__':
    main()
