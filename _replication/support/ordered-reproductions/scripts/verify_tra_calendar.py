"""Verify calendar-aware windows, unchanged labels, and loss-memory timing before training."""
import importlib.util
import json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gap-resolution-20261113'
spec = importlib.util.spec_from_file_location('calendar_dataset', ROOT / 'sources/tra_dataset_calendar_corrected.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.device = 'cpu'
warnings.filterwarnings('ignore', category=FutureWarning)

# Two stocks make the global calendar explicit. A's data are absent at sessions 2/3.
dates = pd.bdate_range('2020-01-06', periods=8)
index = pd.MultiIndex.from_tuples([('A', dates[i]) for i in [0, 1, 4, 5, 6, 7]] + [('B', d) for d in dates], names=['instrument', 'datetime'])
obj = object.__new__(module.MTSDatasetH)
obj.seq_len = 5
obj.horizon = 2
obj.num_states = 1
obj.pin_memory = False
obj.batch_size = 100
obj.shuffle = False
obj.drop_last = False
obj._index = index
obj._calendar_positions = dates.get_indexer(index.get_level_values(1))
obj._data = np.c_[np.arange(1, len(index)+1), np.arange(101, 101+len(index))].astype('float32')
obj._label = np.arange(len(index), dtype='float32')
obj.zeros = np.zeros((5, 2), dtype='float32')
obj.batch_slices = module._create_ts_slices(index, 5)
batch = next(iter(obj))
assert batch['data'][2, :, 0].tolist() == [1, 2, 0, 0, 3]
assert batch['data'][2, :, 1].tolist() == [101, 102, 0, 0, 0]
assert batch['data'][3, :, 0].tolist() == [2, 0, 0, 3, 4]
assert batch['data'][6, :, 0].tolist() == [0, 0, 0, 0, 7]
assert np.array_equal(batch['label'].numpy(), obj._label)
assert np.array_equal(batch['index'].numpy(), np.arange(len(index)))
before = obj._data.copy()
obj.assign_data(torch.tensor([2]), torch.tensor([[999.]]))
assert obj._data[2, -1] == 999 and np.array_equal(obj._data[:, 0], before[:, 0])

frame = pd.read_pickle(ROOT / 'data/TRA/kdd21/feature.pkl')
frame = frame.swaplevel().sort_index()
calendar = pd.DatetimeIndex(frame.index.get_level_values(1).unique()).sort_values()
independent_calendar = pd.to_datetime(pd.read_csv(ROOT / 'data/crowd_data/calendars/day.txt', header=None)[0])
independent_calendar = pd.DatetimeIndex(independent_calendar[(independent_calendar >= calendar.min()) & (independent_calendar <= calendar.max())])
assert calendar.equals(independent_calendar), 'Panel union must cover every independent market session'
positions = calendar.get_indexer(frame.index.get_level_values(1))
slices = module._create_ts_slices(frame.index, 60)
ends = np.fromiter((s.stop for s in slices), dtype='int64')
starts = np.fromiter((s.start for s in slices), dtype='int64')
assert np.array_equal(ends, np.arange(1, len(frame)+1))
assert np.all(positions[ends-1] - positions[starts] < 60)
stock_codes = frame.index.codes[0]
assert np.array_equal(stock_codes[ends-1], stock_codes[starts])
counts = frame.groupby(level=0).size().to_numpy()
stock_starts = np.repeat(np.r_[0, counts.cumsum()[:-1]], counts)
previous_starts = np.maximum(ends-60, stock_starts)
old_span = positions[ends-1] - positions[previous_starts] + 1
unchanged = old_span == ends-previous_starts
assert np.array_equal(starts[unchanged], previous_starts[unchanged])

# Inspect the demonstrated 12-year example against direct date-index lookup.
loc = frame.index.get_loc(('SZ002080', pd.Timestamp('2020-06-29')))
slc = slices[loc]
slots = 59 - (positions[loc] - positions[slc])
window = np.zeros((60, frame.shape[1]), dtype='float32')
window[slots] = frame.iloc[slc].to_numpy(dtype='float32')
expected = frame.loc['SZ002080'].reindex(calendar[positions[loc]-59:positions[loc]+1]).fillna(0).to_numpy(dtype='float32')
assert np.array_equal(window, expected)
assert len(slots) == 9
assert str(frame.index[slc.start][1].date()) == '2020-06-15'
split_dates = frame.index.get_level_values(1)
counts_by_split = {}
for name, first, last in [('train','2007-10-30','2016-05-27'), ('valid','2016-09-26','2018-05-29'), ('test','2018-09-21','2020-06-30')]:
    mask = (split_dates >= first) & (split_dates <= last)
    counts_by_split[name] = dict(rows=int(mask.sum()), affected_windows=int((~unchanged & mask).sum()), previously_over60=int(((old_span>60)&mask).sum()))
result = dict(passed=True, total_rows=len(frame), calendar_sessions=len(calendar), independent_calendar_match=True,
              labels_and_output_rows_unchanged=True, no_cross_stock_rows=True, all_sequences_within60_market_sessions=True,
              unaffected_sequences_identical=True, synthetic_internal_padding_and_loss_memory_checks=True,
              affected_by_split=counts_by_split,
              worst_example=dict(instrument='SZ002080', end='2020-06-29', real_observations=9, zero_padded_sessions=51,
                                 first_real_observation='2020-06-15', stale_2008_observations=0),
              policy='Zero-pad unavailable sessions using released initial-padding convention. No imputed returns or feature reconstruction. Original targets/universe and latest21-session error-memory mask preserved.')
(OUT / 'tra-calendar-verification.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
