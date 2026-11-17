"""Validate prepared panels and expose sparse-coverage trading days."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
out = {}
for market in ['NASDAQ', 'NYSE']:
    z = np.load(ROOT / f'data/{market}.npz')
    x, price, mask, edges = (z[k] for k in ['features', 'price', 'mask', 'edges'])
    counts = np.array([mask[:, t-4:t+1].all(1).sum() for t in range(4, mask.shape[1])])
    assert x.shape[-1] == 9 and np.isfinite(x).all()
    assert np.isfinite(price).all() and edges.min() >= 0 and edges.max() < len(x)
    assert len(np.unique(z['tickers'])) == len(x)
    assert counts.min() >= 5
    assert (z['dates'][1:] > z['dates'][:-1]).all()
    sparse = np.flatnonzero(counts < len(x) / 2) + 4
    out[market] = {'shape': list(x.shape), 'valid_stocks_min_per_target': int(counts.min()),
        'valid_stocks_mean_per_target': float(counts.mean()), 'test_days': x.shape[1]-1008,
        'finite_features': True, 'unique_tickers': True, 'ordered_dates': True,
        'valid_graph_indices': True, 'less_than_half_universe_valid': [
            {'date': str(z['dates'][t]), 'valid_windows': int(counts[t-4]),
             'valid_observations_on_date': int(mask[:, t].sum())} for t in sparse]}
(ROOT / 'results/data_audit.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
