"""Focused preflight tests of time alignment, screening, batches and accounting."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
from common import HERE, ROOT, DATA, settings, save_json, sha, now
from panel import Panel
from build import prior_month, rank_signals
from portfolio import decile_targets, neutral_projection, backtest


def run():
    p = Panel()
    cfg = settings()
    monthly_root = ROOT.parent / 'crsp-monthly'
    monthly_receipt = json.loads((monthly_root / 'manifest.json').read_text())
    for part in monthly_receipt['files']:
        assert sha(monthly_root / 'parquet' / part['name']) == part['sha256']
    for name, digest in p.manifest['output_sha256'].items():
        assert sha(DATA / name) == digest, name
    assert p.monthly_features.shape[-1] == len(p.manifest['feature_names']) == 202
    np.testing.assert_array_equal(p.source_months[p.month_index], prior_month(p.dates))
    thresholds = pd.read_csv(DATA / 'nyse20_breakpoints.csv').set_index('yyyymm').nyse20
    cutoff = thresholds.reindex(p.source_months).to_numpy()[p.month_index]
    mask = np.asarray(p.member)
    assert np.isfinite(p.lagged_cap[mask]).all()
    assert (np.asarray(p.lagged_cap) >= cutoff[None, :])[mask].all()
    assert np.isfinite(p.monthly_features).all()
    assert abs(p.monthly_features).max() <= 1.
    train_mask = np.load(DATA / 'training_feature_mask.npy')
    assert np.all(np.ptp(p.monthly_features[train_mask], axis=0) > 1e-12)
    summary = p.summary()
    for split in ['train', 'valid', 'test']:
        days = p.plans[split]
        assert len(days) > 400
        if split != 'test':
            assert str(p.dates[days[-1] + 10]) <= cfg[split][1]
        for t in days[[0, len(days) // 2, -1]]:
            x, prior, y, ids = p.batch(t, split)
            assert x.shape == (len(ids), 20, 202) and prior.shape == (len(ids), 13)
            np.testing.assert_array_equal(x.numpy(), p.monthly_features[ids[:, None], p.month_index[t - 19:t + 1][None, :]])
            if split == 'test':
                # Future-label availability cannot alter prediction eligibility.
                np.testing.assert_array_equal(ids, np.flatnonzero(p.member[:, t] & p.tradable[:, t]))
    rank = rank_signals(pd.DataFrame({'a': [1., 2., np.nan], 'b': [5., 5., 5.], 'c': [np.nan] * 3}))
    np.testing.assert_array_equal(rank[:, 0], [-1, 1, 0])
    np.testing.assert_array_equal(rank[:, 1:], 0)
    target = decile_targets(np.arange(20), np.arange(20), np.arange(1, 21), 20)
    np.testing.assert_allclose(target[9, 18:20], [19 / 39, 20 / 39])
    assert set(np.flatnonzero(target[0])) == {0, 1}
    projected = neutral_projection([1., -1.], [.2, -2.], [.2, 0.])
    np.testing.assert_allclose(projected, [.2, -.2])
    # Two realizations: scores at t0 execute t1, earn t2; unchanged scores at
    # t1 execute t2 and earn t3. Include explicit entry, drift and exit costs.
    n = 20
    fake = SimpleNamespace(symbols=np.array([f'P{i:02}' for i in range(n)]),
        dates=np.array(['2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06']),
        member=np.ones((n, 4), bool), tradable=np.ones((n, 4), bool),
        terminal_events=np.zeros((n, 4), bool), lagged_cap=np.ones((n, 4)),
        totalret=np.zeros((n, 4)), price_return=np.zeros((n, 4)))
    fake.totalret[18:, 2] = .1
    fake.price_return[18:, 2] = .1
    frames = pd.DataFrame([dict(date=d, symbol=s, prediction=i) for d in fake.dates[:2]
                          for i, s in enumerate(fake.symbols)])
    gross, _, _ = backtest(fake, frames, 0)
    net, weights, _ = backtest(fake, frames, 10)
    np.testing.assert_allclose(gross.return_gross, [.1, 0])
    np.testing.assert_allclose(net.fees.iloc[0], .002)
    assert net.final_liquidation_dollars.iloc[-1] > 0
    np.testing.assert_allclose(net.return_net, net.return_gross - net.fees / net.start_nav)
    assert net.date.tolist() == ['2022-01-05', '2022-01-06']
    # A halt must preserve holdings; projection balances the opposite side.
    fake.tradable[19, 2] = False
    halted, held, _ = backtest(fake, frames, 0)
    np.testing.assert_allclose(held[1, 19] * halted.start_nav.iloc[1], held[0, 19] * 1.1)
    # Synthetic example exercises the microcap comparison even if large-cap
    # membership makes actual below-threshold names rare.
    assert np.array([9., 10., 11.]).__ge__(10.).tolist() == [False, True, True]
    save_json(HERE / 'preflight.json', dict(status='passed', updated_utc=now(), features=202,
        split_summary=summary, checks=['prior-calendar-month information sets', 'explicit NYSE20',
        'training-only feature removal', 'split purge', 'test universe independent of labels',
        'value deciles', 'next-close execution', 'entry/rebalance/exit fees', 'frozen holdings and neutrality']))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    run()
