"""Calendar alignment, split provenance and data leakage regression checks."""
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runtime/scripts'))
from data_processing import Dataset


def config(tmp_path, **overrides):
    values = dict(data_dir=str(ROOT / 'results'), datafile='synthetic.dat', train=True,
                  min_unrollings=5, max_unrollings=5, stride=12, forecast_n=12,
                  forecast_steps=1, start_date=199401, end_date=199912,
                  validation_size=.3, seed=521, financial_fields='saleq_ttm-atq_mrq',
                  aux_fields='rel_mom1m', dont_scale_fields=None, target_field='oiadpq_ttm',
                  scale_field='mrkcap', key_field='gvkey', date_field='date', active_field='active',
                  model_dir='test', experiments_dir=str(tmp_path), scalesfile=None,
                  data_scaler='StandardScaler', log_squasher=False, aux_masking=False,
                  nn_type='RNNPointEstimate')
    return SimpleNamespace(**(values | overrides))


def test_split_metadata_calendar_and_target_boundaries(tmp_path):
    d = Dataset(config(tmp_path)); d.generate_dataset()
    assert set(d._train_gvkeys).isdisjoint(d._valid_gvkeys)
    for kind, keys in [('train', d._train_gvkeys), ('valid', d._valid_gvkeys)]:
        for inp, tar, meta in getattr(d, kind + '_set').as_numpy_iterator():
            key = meta[1].decode()
            assert key in keys
            assert key == d.data_values[inp[1], d._gvkey_idx]
            assert d.data_values[tar[1], d._date_idx] <= d.end_date
            assert d.data_values[tar[1], d._date_idx] == d.data_values[inp[1], d._date_idx] + pd.DateOffset(months=12)
            dates = d.data_values[inp[0]:inp[1]+1:12, d._date_idx]
            assert all(b == a + pd.DateOffset(years=1) for a,b in zip(dates, dates[1:]))


def test_marketcap_normalization_and_scaling_roundtrip(tmp_path):
    d = Dataset(config(tmp_path)); d.generate_dataset()
    inp, tar, meta = next(iter(d.train_set.batch(8)))
    x, y, metadata = d.get_batch(inp, tar, meta)
    for n,(start,end,pad) in enumerate(inp.numpy()):
        expected = d.data_values[end, d.fin_col_ids].astype(float)
        restored = (x.numpy()[n,-1,:3] * d.scaling_params['scale'][:3] + d.scaling_params['center'][:3]) * float(metadata[n,2])
        np.testing.assert_allclose(restored, expected, rtol=1e-5)


@pytest.mark.parametrize('damage', ['gap', 'duplicate'])
def test_reject_ambiguous_month_offsets(tmp_path, damage):
    df = pd.read_csv(ROOT/'results/synthetic.dat', sep=' ', dtype={'gvkey':str})
    df = df.drop(index=100) if damage=='gap' else pd.concat([df,df.iloc[[100]]])
    df.to_csv(tmp_path/'broken.dat', sep=' ', index=False)
    with pytest.raises(ValueError, match='contiguous|Duplicate'):
        Dataset(config(tmp_path, data_dir=str(tmp_path), datafile='broken.dat'))


def test_nonannual_target_cutoff(tmp_path):
    d = Dataset(config(tmp_path, forecast_n=18)); d.generate_dataset()
    for kind in ['train', 'valid']:
        for start,end,pad in d._dataset[kind+'_Y']:
            assert d.data_values[end,d._date_idx] <= d.end_date
