"""Recover two companion benchmarks from RSR; explicitly differs from Hermes Table 4."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/Temporal_Relational_Stock_Ranking'


def main():
    for market in ['NASDAQ', 'NYSE']:
        folder = SOURCE / 'data'
        tickers = np.loadtxt(folder / f'{market}_tickers_qualify_dr-0.98_min-5_smooth.csv', dtype=str)
        with tarfile.open(folder / 'relation.tar.gz') as tar:
            industry = json.load(tar.extractfile(f'relation/sector_industry/{market}_industry_ticker.json'))
        sectors = sorted(industry)
        incidence = np.array([[s in industry[k] for k in sectors] for s in tickers])
        arrays = []
        digest = hashlib.sha256()
        for ticker in tickers:
            path = folder / '2013-01-01' / f'{market}_{ticker}_1.csv'
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
            arrays.append(np.loadtxt(path, delimiter=',')[:, 1:])
        raw = np.stack(arrays, axis=1)
        dates = pd.to_datetime(np.loadtxt(folder / f'{market}_aver_line_dates.csv', delimiter='\t', dtype=str)[29:29 + len(raw)])
        keep = (dates >= '2013-01-02') & (dates <= '2017-12-08 23:59:59')
        raw, dates = raw[keep], dates[keep]
        observed = (raw > 0).all(-1) & np.isfinite(raw).all(-1)
        valid = observed.copy()
        valid[0] = False
        valid[1:] &= observed[:-1]
        returns = np.zeros(raw.shape[:2], dtype=np.float32)
        np.divide(raw[1:, :, -1], raw[:-1, :, -1], out=returns[1:], where=valid[1:])
        returns[1:] -= valid[1:]
        returns[~valid] = 0
        values = np.where(observed[..., None], raw, np.nan)
        values = np.stack([pd.DataFrame(values[:, i]).ffill().to_numpy() for i in range(len(tickers))], axis=1)
        medians = np.nanmedian(values[:756], axis=0)
        values = np.where(np.isfinite(values), values, medians[None])
        mean, std = values[:756].mean(0), values[:756].std(0)
        features = ((values - mean) / np.maximum(std, 1e-6)).astype(np.float32)
        assert features.shape[0] == 1245
        assert np.isfinite(features).all() and incidence.any(0).all() and incidence.any(1).all()
        dest = ROOT / 'data' / f'{market.lower()}.npz'
        np.savez_compressed(dest, features=features, returns=returns, valid=valid, observed=observed,
                            incidence=incidence, dates=dates.strftime('%Y-%m-%d').to_numpy(dtype=str),
                            tickers=tickers, sectors=np.array(sectors), mean=mean, std=std)
        manifest = {
            'status': 'companion reconstruction; not the exact Table 4 protocol',
            'source': 'https://github.com/fulifeng/Temporal_Relational_Stock_Ranking',
            'commit': subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
            'source_csv_combined_sha256': digest.hexdigest(),
            'prepared_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(),
            'shape': list(features.shape), 'industries': len(sectors),
            'features': ['close_MA5', 'close_MA10', 'close_MA20', 'close_MA30', 'close'],
            'split_days': [756,252,len(raw)-1008], 'paper_split_days': [756,252,273],
            'dates': [str(dates[0].date()),str(dates[755].date()),str(dates[756].date()),
                      str(dates[1007].date()),str(dates[1008].date()),str(dates[-1].date())],
            'missing_stock_days': int((~observed).sum()),
            'normalization': 'training-only per-stock feature z-score; forward fill and initial training medians',
            'source_scale_note': 'source uses full-period price-max scale; per-stock train z-score and close ratios cancel that multiplicative scale, except CSV rounding',
        }
        (ROOT / 'data' / f'{market.lower()}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(json.dumps(manifest,indent=2))


if __name__ == '__main__':
    main()
