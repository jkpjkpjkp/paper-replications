import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['MPLBACKEND'] = 'Agg'
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import numpy as np
import pandas as pd
import torch
from build_data import factor_inputs, robust_fit
from runtime import ROOT, Panel, config, temporal_model, encode, from_codes, rankic
from train import seed_all


def test_target_horizon_and_boundary_purge():
    panel = Panel('csi300')
    for split, stop in [('train', '2019-12-31'), ('valid', '2021-12-31')]:
        days = panel.plans[split]
        assert all(panel.dates[t+10] <= stop for t in days)
        t = int(days[len(days)//2])
        _, _, y, ids = panel.batch(t, split)
        target = panel.close[ids, t+5] / panel.close[ids, t+1] - 1
        expected = (pd.Series(target).rank(pct=True).to_numpy()-.5)*3.46
        np.testing.assert_allclose(y[:, 4].numpy(), expected, atol=5e-7)


def test_test_universe_uses_current_information_only():
    p = Panel('csi300')
    for t in p.plans['test'][::100]:
        expected = np.flatnonzero(p.member[:, t] & p.tradable[:, t])
        np.testing.assert_array_equal(p.ids(t, 'test'), expected)


def test_jkp_lag_matches_twenty_prior_sessions():
    import zipfile
    p = Panel('csi300')
    raw, _ = factor_inputs('chn', p.dates)
    with zipfile.ZipFile(ROOT / 'sources/chn-themes.zip') as z:
        f = pd.read_csv(z.open(z.namelist()[0]), parse_dates=['date'])
    f = f.pivot(index='date', columns='name', values='ret')
    t = int(p.plans['test'][100])
    observations = f.reindex(pd.to_datetime(p.dates[t-20:t]))
    expected = ((1+observations).prod(axis=0)-1).to_numpy()
    np.testing.assert_allclose(raw[t], expected, rtol=2e-6, atol=1e-8)
    assert str(observations.index[-1].date()) < str(p.dates[t])


def test_saved_normalization_fits_train_only():
    p = Panel('csi300')
    saved = np.load(p.folder / 'normalization.npz')
    # Reconstruct an independent sample of full training columns from raw cache.
    cols = [0, 20, 50, 100, 157]
    values = []
    train = (p.dates >= '2009-01-01') & (p.dates <= '2019-12-31')
    for i, s in enumerate(p.symbols):
        x = np.load(p.folder / f'instrument_cache/{s}.npy', mmap_mode='r')
        values.append(x[p.member[i] & train][:, cols])
    median, scale, absent = robust_fit(np.concatenate(values))
    np.testing.assert_allclose(saved['median'][cols], median)
    np.testing.assert_allclose(saved['scale'][cols], scale)
    assert not absent.any()


def test_frozen_stage1_and_cached_native_agreement(tmp_path):
    seed_all(0)
    torch.set_num_threads(2)
    ckpt = ROOT / 'results/smoke/best.pt'
    model = temporal_model(config('csi300', stage1=ckpt), 10)
    p = Panel('csi300')
    t = int(p.plans['valid'][0])
    x, prior, y, ids = p.batch(t, 'valid')
    before = {k: v.clone() for k, v in model.quantizer.state_dict().items()}
    model.train()
    assert not model.encoder.training and not model.quantizer.training and not model.revin.training
    pred, _, _, _, aux = model(x, prior)
    loss = ((pred-y[:, 4])**2).mean()+model.aux_weight*aux
    loss.backward()
    assert model.latent_value_head.head[0].weight.grad is not None
    assert all(p.grad is None for p in model.encoder.parameters())
    for k, v in model.quantizer.state_dict().items():
        torch.testing.assert_close(v, before[k], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        native = model(x, prior)[0]
        z, _ = encode(model, x)
        cached, _ = from_codes(model, x, prior, z)
    torch.testing.assert_close(native, cached, rtol=0, atol=0)
    from train import build_code_cache
    disk_cache = build_code_cache(model, p, tmp_path, 'cpu', {'valid': [t]})
    np.testing.assert_array_equal(disk_cache[ids, t], z.numpy())
    reused = build_code_cache(model, p, tmp_path, 'cpu', {'valid': [t]})
    np.testing.assert_array_equal(reused[ids, t], z.numpy())


def test_rankic_known_ordering():
    assert abs(rankic(np.array([1, 2, 3, 4]), np.array([9, 7, 4, 2]))+1) < 1e-12


def test_portfolio_replacements_and_self_financing_fees():
    from evaluate import topk_drop, equal_weight_trade
    scores = pd.Series([10, 9, 8, 7], index=['d', 'a', 'b', 'c'])
    assert topk_drop(scores, ['a', 'b', 'c'], k=3, drop=1) == {'a', 'b', 'd'}
    assert topk_drop(scores, ['d', 'a', 'b'], k=3, drop=1) == {'a', 'b', 'd'}
    values, wealth, cost, buy, sell = equal_weight_trade({}, {1, 2}, 1)
    np.testing.assert_allclose(wealth, 1/1.0005, atol=1e-12)
    np.testing.assert_allclose(sum(values.values())+cost, 1, atol=1e-12)
    assert sell == 0


def test_portfolio_signal_is_executed_next_close(tmp_path):
    from types import SimpleNamespace
    from evaluate import backtest
    panel = SimpleNamespace(symbols=np.array(['A']),
        dates=np.array(['2022-01-03', '2022-01-04', '2022-01-05']),
        close=np.array([[1., 2., 3.]]), totalret=np.array([[np.nan, 1., .5]]),
        tradable=np.ones((1, 3), dtype=bool))
    predictions = pd.DataFrame(dict(date=['2022-01-03', '2022-01-04'], symbol=['A', 'A'], prediction=[1., 1.]))
    result = backtest(panel, predictions, tmp_path)
    daily = pd.read_csv(tmp_path / 'portfolio_daily.csv')
    np.testing.assert_allclose(daily.net_return.iloc[0], 1/1.0005-1, atol=1e-12)
    np.testing.assert_allclose(daily.net_return.iloc[1], .5, atol=1e-12)


def test_terminal_return_settles_to_cash_without_stale_position(tmp_path):
    from types import SimpleNamespace
    from evaluate import backtest
    panel = SimpleNamespace(symbols=np.array(['A']),
        dates=np.array(['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06']),
        close=np.array([[1., 2., np.nan, np.nan]]),
        totalret=np.array([[np.nan, 1., .1, np.nan]]),
        tradable=np.array([[True, True, False, False]]),
        terminal_events=np.array([[False, False, True, False]]))
    pred = pd.DataFrame(dict(date=panel.dates[:3], symbol=['A']*3, prediction=[1.]*3))
    stats = backtest(panel, pred, tmp_path)
    daily = pd.read_csv(tmp_path / 'portfolio_daily.csv')
    assert daily.holdings.tolist() == [1, 0, 0]
    np.testing.assert_allclose(daily.net_return.iloc[1], .1, atol=1e-12)
    np.testing.assert_allclose(daily.net_return.iloc[2], 0, atol=1e-12)
    assert stats['dated_terminal_event_settlements'] == 1
    assert stats['missing_return_price_mark_fallbacks'] == 0
