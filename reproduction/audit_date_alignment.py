"""Sensitivity audit of SHIPPED SIT returns against a FRESH official EW baseline.

Date labels differ: SIT labels a forward return by its start date; the
traditional baseline labels a backward return by its end date. Align economic
return end dates before comparing. This is not new SIT training.
"""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from audit_results import metrics

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--sit-results', type=Path, default=ROOT / 'results')
parser.add_argument('--output', type=Path, default=ROOT / 'reproduction')
parser.add_argument('--sit-label', default='SIT_shipped')
args = parser.parse_args()
prices = pd.read_csv(ROOT / 'asset_data/full_dataset.csv', parse_dates=['Date']).set_index('Date')
rows = []
periods = []
for pool in (40, 50):
    ew = pd.read_csv(ROOT / f'reproduction/fresh_traditional/results_trad/EW_{pool}/test_positions.csv', parse_dates=['Date'])
    ew_ret = pd.Series(ew.DailyPnL.values / np.r_[10000., ew.CumulativeCapital.values[:-1]], index=ew.Date)
    for path in sorted(args.sit_results.glob(f'dp{pool}_*test_positions.csv')):
        sit = pd.read_csv(path, parse_dates=['Date'])
        ix = prices.index.get_indexer(sit.Date)
        assert (ix >= 0).all() and (ix + 1 < len(prices)).all()
        sit_ret = pd.Series(sit.DailyPnL.values / np.r_[10000., sit.CumulativeCapital.values[:-1]], index=prices.index[ix + 1])
        common = sit_ret.index.intersection(ew_ret.index)
        periods.append(dict(pool=pool, run=path.name, sit_days=len(sit_ret), ew_days=len(ew_ret),
                            common_days=len(common), sit_first_return_end=str(sit_ret.index.min().date()),
                            ew_first_return_end=str(ew_ret.index.min().date()),
                            common_last_return_end=str(common.max().date())))
        for label, series in [(args.sit_label + '_original', sit_ret), ('EW_fresh_original', ew_ret),
                              (args.sit_label + '_common_dates', sit_ret.loc[common]), ('EW_fresh_common_dates', ew_ret.loc[common])]:
            rows.append(dict(pool=pool, run=path.name, label=label, days=len(series), **metrics(series.values)))
out = args.output
out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out / 'date_alignment_sensitivity_runs.csv', index=False)
means = pd.DataFrame(rows).groupby(['pool', 'label'], sort=False)[['days', 'Sharpe', 'Sortino', 'MaxDrawdown', 'FinalWealthFactor']].mean()
means.to_csv(out / 'date_alignment_sensitivity_means.csv')
(out / 'date_alignment_periods.json').write_text(json.dumps(periods, indent=2))
print(means.to_string())
