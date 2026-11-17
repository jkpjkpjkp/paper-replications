"""Prepare the cited ESTIMATE S&P500 release without fitting on validation/test."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'upstream/estimate/src/data/US/sp500')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/sp500.npz')
    args = parser.parse_args()
    raw_path = args.source / 'baseline_data_sp500.npy'
    if sha256(raw_path) != 'd16a37a8f4c761cf6a9f2b4850e6f480c699a92d72853410a72a434109cad685':
        raise ValueError('raw S&P500 checksum differs from the pinned release')
    if sha256(args.source / 'sp500_ticker.csv') != 'f820d36fc38be430c5dea4711acfef3a5b83110e26ec95ddeb0553839e61efe8':
        raise ValueError('sector file checksum differs from the pinned release')
    # This upstream dataset is a pickled dict of pandas frames; never accept arbitrary untrusted pickle files.
    raw = np.load(raw_path, allow_pickle=True).item()
    tickers = sorted(raw)
    sector_table = pd.read_csv(args.source / 'sp500_ticker.csv').set_index('Symbol')
    sectors = sorted(sector_table.loc[tickers, 'Sector'].unique())
    frames = {s: raw[s].set_axis(pd.to_datetime(raw[s].index)).sort_index() for s in tickers}
    dates = pd.DatetimeIndex(sorted(set().union(*[set(f.index) for f in frames.values()])))
    dates = dates[(dates >= '2016-01-04') & (dates <= '2022-05-25')]
    if len(dates) != 1611 or len(tickers) != 474 or len(sectors) != 11:
        raise ValueError(f'dataset differs from paper: {len(dates)}, {len(tickers)}, {len(sectors)}')
    # Five OHLCV inputs, unchanged split-adjusted OHLC as provided by the upstream release.
    columns = ['open', 'high', 'low', 'close', 'volume']
    prices = np.stack([frames[s].reindex(dates)[columns].to_numpy(float) for s in tickers], axis=1)
    observed = np.isfinite(prices).all(-1) & (prices[..., :4] > 0).all(-1)
    valid_return = observed.copy()
    valid_return[0] = False
    valid_return[1:] &= observed[:-1]
    returns = np.zeros(prices.shape[:2], np.float32)
    np.divide(prices[1:, :, 3], prices[:-1, :, 3], out=returns[1:], where=valid_return[1:])
    returns[1:] -= valid_return[1:]
    returns[~valid_return] = 0
    # Forward fill only; initial missing values use training-only medians.
    values = np.stack([pd.DataFrame(prices[:, i]).ffill().to_numpy() for i in range(len(tickers))], axis=1)
    values[..., 4] = np.log1p(np.maximum(values[..., 4], 0))
    median = np.nanmedian(values[:1006], axis=0)
    values = np.where(np.isfinite(values), values, median[None])
    mean, std = values[:1006].mean(0), values[:1006].std(0)
    features = ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)
    incidence = np.array([[sector_table.loc[s, 'Sector'] == c for c in sectors] for s in tickers])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, returns=returns, valid=valid_return,
                        observed=observed, incidence=incidence, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),
                        tickers=np.array(tickers), sectors=np.array(sectors), mean=mean, std=std)
    manifest = {
        'preparation_versions': {'numpy':np.__version__,'pandas':pd.__version__},
        'source': 'https://github.com/thanhtrunghuynh93/estimate',
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT / 'upstream/estimate'), 'rev-parse', 'HEAD'], text=True).strip(),
        'raw_sha256': sha256(raw_path), 'sector_sha256': sha256(args.source / 'sp500_ticker.csv'),
        'prepared_sha256': sha256(args.output), 'shape': list(features.shape), 'industries': len(sectors),
        'features': columns, 'split_boundaries': [1006, 1259, 1611],
        'dates': {k: str(dates[i].date()) for k, i in [('start', 0), ('train_end', 1005), ('val_start', 1006), ('val_end', 1258), ('test_start', 1259), ('end', 1610)]},
        'missing_stock_days': int((~observed).sum()), 'valid_target_fraction': float(valid_return[1:].mean()),
        'preprocessing': 'per-stock per-feature training z-score; volume log1p; forward-fill only; initial train median',
        'target': 'close[t]/close[t-1]-1; inputs end at t-1; first 16 dates have no training sample',
    }
    manifest_path = ROOT / 'data/manifest.json' if args.output.resolve() == (ROOT / 'data/sp500.npz').resolve() else args.output.with_suffix('.manifest.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
