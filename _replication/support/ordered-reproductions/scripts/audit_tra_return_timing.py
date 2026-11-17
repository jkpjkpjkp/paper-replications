"""Independently compare released TRA outcomes with underlying adjusted prices."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gap-resolution-20261113'
RAW = ROOT / 'data/crowd_data'
cal = pd.DatetimeIndex(pd.to_datetime((RAW / 'calendars/day.txt').read_text().splitlines()))
label = pd.read_pickle(ROOT / 'data/TRA/kdd21/label.pkl').squeeze().loc['2018-09-21':'2020-06-30']
ret = pd.read_pickle(ROOT / 'data/TRA/kdd21/ret.pkl').squeeze().reindex(label.index)
frame = pd.DataFrame(dict(released_label=label, released_return=ret)).sort_index()
frame['date'] = frame.index.get_level_values('datetime')
frame['stock'] = frame.index.get_level_values('instrument')
frame['group'] = ''
for name in ['csi300', 'csi500']:
    members = pd.read_csv(RAW / 'instruments' / (name+'.txt'), sep='\t', header=None, names=['stock','start','end'])
    for row in members.itertuples(index=False):
        mask = (frame.stock == row.stock) & (frame.date >= row.start) & (frame.date <= row.end)
        assert (frame.loc[mask, 'group'] == '').all(), 'Index membership overlaps'
        frame.loc[mask, 'group'] = name
missing = []
specs = [(kind, a, b) for kind in ['close','vwap'] for a,b in [(0,20),(0,21),(1,21),(1,22)]]
for kind, a, b in specs:
    frame[f'{kind}_{a}_{b}'] = np.nan
for lag in [0,1,2]:
    frame[f'daily_close_lag{lag}'] = np.nan

for stock, rows in frame.groupby('stock').groups.items():
    pos = cal.get_indexer(frame.loc[rows, 'date'])
    assert (pos >= 0).all()
    for field in ['close','vwap']:
        path = RAW / 'features' / stock.lower() / (field+'.day.bin')
        if not path.exists():
            missing.append(str(path))
            continue
        blob = np.fromfile(path, dtype='<f4')
        values = np.full(len(cal), np.nan)
        start = int(blob[0]); values[start:start+len(blob)-1] = blob[1:]
        # Carry the last observed price through a suspension; never backfill from the future.
        carried = pd.Series(values).ffill().to_numpy()
        for kind, a, b in specs:
            if kind == field:
                frame.loc[rows, f'{kind}_{a}_{b}'] = carried[pos+b]/carried[pos+a]-1
        if field == 'close':
            for lag in [0,1,2]:
                frame.loc[rows, f'daily_close_lag{lag}'] = np.clip(carried[pos+lag]/carried[pos+lag-1]-1, -.1, .1)

daily = {}
for lag in [0,1,2]:
    x = frame[f'daily_close_lag{lag}'];good=x.notna()
    delta=(x-frame.released_return).abs()
    daily[str(lag)] = dict(pairs=int(good.sum()), mean_absolute_difference=float(delta.mean()),
                           differences_over_1pp=int((delta>.01).sum()),
                           differences_over_10bp=int((delta>.001).sum()),
                           correlation=float(x.corr(frame.released_return)))
label_checks = {}
for kind, a, b in specs:
    name=f'{kind}_{a}_{b}'
    ranked=frame.groupby(level='datetime')[name].rank(pct=True)
    subrank=frame.groupby(['date','group'])[name].rank(pct=True)
    joined=pd.DataFrame(dict(label=frame.released_label, all800=ranked, by_index=subrank))
    correlations=joined.groupby(level='datetime').apply(lambda g: pd.Series(dict(all800=g.label.corr(g.all800),by_index=g.label.corr(g.by_index))))
    label_checks[name]=dict(finite=int(frame[name].notna().sum()),
        mean_daily_label_correlation_with_all800_ranks=float(correlations.all800.mean()),
        mean_daily_label_correlation_with_subindex_ranks=float(correlations.by_index.mean()))

# Test the discrete rank lattice itself, independent of the price archive.
rank_position=frame.released_label/3.46+.5
lattice={}
for group,n in [('csi300',300),('csi500',500)]:
    values=rank_position[frame.group==group]*n*2  # average-tie ranks may be half integers
    lattice[group]=dict(rows=len(values),fraction_on_half_integer_rank_lattice=float((abs(values-values.round())<.001).mean()))
result=dict(rows=len(frame),missing_price_files=missing,unknown_subindex_rows=int((frame.group=='').sum()),
    source_note='Paper section6.1.2 specifies next21 trading-day return ranks. Alternative endpoint checks are diagnostics of the released data, not model selection or altered evaluation labels.',
    daily_return_alignment=daily,label_endpoint_checks=label_checks,rank_lattice_by_independent_historical_membership=lattice,
    caution='The independent archive is a different vendor/vintage. High agreement can corroborate timing; residual differences alone do not prove corruption. No labels or predictions are changed.')
(OUT/'tra-return-timing.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
frame.to_parquet(OUT/'tra-independent-price-comparison.parquet')
print(json.dumps(result,indent=2))
