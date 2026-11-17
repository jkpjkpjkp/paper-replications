"""Economic invariants: causal information, weighted spreads and cash accounting."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

from experiments.standard_characteristics.data import Panel, prior_month, rank_signals, nonmicro_flags
from experiments.standard_characteristics.portfolio import backtest, decile_targets
from experiments.standard_characteristics.model import Model


def fixture():
    symbols = np.array([f"P{i}" for i in range(20)])
    dates = np.array(pd.bdate_range("2020-01-01", periods=8).strftime("%Y-%m-%d"))
    panel = SimpleNamespace(symbols=symbols, dates=dates, member=np.ones((20, 8), bool),
        capitalization=np.broadcast_to(np.arange(1, 21)[:, None], (20, 8)).copy(),
        market=np.ones((20, 8, 7)), total_return=np.zeros((20, 8)), price_return=np.zeros((20, 8)),
        terminal=np.zeros((20, 8), bool), execution_tradable=np.ones((20, 8), bool))
    panel.market[:, :, 1] = np.arange(20, 0, -1)[:, None]
    pred = pd.DataFrame(dict(date=dates[2], realized_date=dates[4], symbol=symbols, pred_return=np.arange(20)))
    return panel, pred


def test_month_lag_on_month_end_year_boundary_and_leap_day():
    np.testing.assert_array_equal(prior_month(["2020-01-31", "2020-02-01", "2020-02-29", "2020-03-02"]),
        [201912, 202001, 202001, 202002])


def test_daily_window_preserves_each_historical_month_information_set():
    panel = object.__new__(Panel)
    panel.lookback = 3
    panel.eligible = np.ones((2, 5), bool)
    panel.month_index = np.array([0, 0, 1, 1, 2])
    panel.month_features = np.array([[[10], [20], [999]], [[30], [40], [888]]], np.float32)
    panel.labels = np.ones((2, 5, 3), np.float32)
    _, x, _, _ = panel.day(3)
    np.testing.assert_array_equal(x[:, :, 0], [[10, 20, 20], [30, 40, 40]])
    panel.month_features[:, 2] = -12345
    np.testing.assert_array_equal(panel.day(3)[1], x)


def test_ranks_ties_singletons_and_missing_are_neutral():
    frame = pd.DataFrame(dict(a=[1, 2, 3], b=[1, 1, 1], c=[np.nan, 4, np.nan]))
    np.testing.assert_allclose(rank_signals(frame), [[-1, 0, 0], [0, 0, 0], [1, 0, 0]])


def test_nyse_breakpoint_uses_nyse_only():
    frame = pd.DataFrame(dict(permno=np.arange(6), yyyymm=202001, mthcap=[1, 2, 3, 4, 5, 1000],
        primaryexch=["N"] * 5 + ["Q"], securitytype="EQTY", securitysubtype="COM", sharetype="NS",
        issuertype="CORP", usincflg="Y", conditionaltype="RW", tradingstatusflg="A", mthdelflg="N"))
    selected, bp = nonmicro_flags(frame)
    np.testing.assert_allclose(bp.iloc[0], 1.8)
    assert not selected.nonmicro.iloc[0] and selected.nonmicro.iloc[1:].all()


def test_weighted_deciles_spread_scale_and_timing():
    panel, pred = fixture()
    panel.total_return[18:, 4] = [.1, .2]
    panel.total_return[:2, 4] = [.03, .06]
    panel.total_return[:, 3] = 99  # Execution-day payoff must not enter P&L.
    result, weights, _ = backtest(panel, pred, "value", save_weights=True)
    expected = (.1 * 19 + .2 * 20) / 39 - (.03 * 1 + .06 * 2) / 3
    np.testing.assert_allclose(result.return_net, expected)
    np.testing.assert_allclose(abs(weights).sum(1), 2)
    np.testing.assert_allclose(weights.sum(1), 0, atol=1e-10)
    assert result.execution_date.iloc[0] == panel.dates[3]
    result_volume, w_volume, _ = backtest(panel, pred, "volume", save_weights=True)
    expected_volume = (.1 * 2 + .2 * 1) / 3 - (.03 * 20 + .06 * 19) / 39
    np.testing.assert_allclose(result_volume.return_net, expected_volume)
    assert not np.allclose(weights, w_volume)


def test_volume_weights_cannot_read_future_volume():
    panel, pred = fixture()
    _, before, _ = backtest(panel, pred, "volume", save_weights=True)
    panel.market[:, 3:, 1] = np.arange(1000, 1020)[:, None]
    _, after, _ = backtest(panel, pred, "volume", save_weights=True)
    np.testing.assert_array_equal(before, after)


def test_missing_payoff_keeps_selection_and_adverse_stress():
    panel, pred = fixture()
    panel.total_return[19, 4] = np.nan
    result, w, _ = backtest(panel, pred, "value", save_weights=True)
    stress, z, _ = backtest(panel, pred, "value", adverse_missing=True, save_weights=True)
    np.testing.assert_array_equal(w, z)
    np.testing.assert_allclose(stress.return_net, -20 / 39)
    assert result.missing_held_count.iloc[0] == 1


def test_all_trade_costs_include_both_legs_and_final_liquidation():
    panel, pred = fixture()
    result, _, _ = backtest(panel, pred, "value", 10)
    np.testing.assert_allclose(result.return_net, -.004)  # 2x opening + 2x closing notional.


def test_blocked_fill_remains_neutral_and_does_not_reselect_deciles():
    panel, pred = fixture()
    panel.execution_tradable[19, 3] = False
    result, weights, _ = backtest(panel, pred, "value", save_weights=True)
    assert weights[0, 19] == 0
    np.testing.assert_allclose(weights.sum(1), 0, atol=1e-9)
    assert result.target_deviation_l1.iloc[0] > 0


def test_input_projection_supports_cz_and_all_three_gradients():
    torch.set_num_threads(1)
    model = Model(209, d_model=32, cross_heads=4, temporal_heads=2, experts=2, dropout=0)
    out = model(torch.randn(12, 3, 209))
    assert out.shape == (12, 3) and torch.isfinite(out).all()
    out.square().sum().backward()
    assert torch.isfinite(model.net.feature_layer[0].weight.grad).all()


def test_ties_are_deterministic_under_row_permutation():
    ids = np.arange(20)
    first = decile_targets(ids, np.zeros(20), ids + 1, 20)
    order = np.random.default_rng(3).permutation(20)
    second = decile_targets(ids[order], np.zeros(20), ids[order] + 1, 20)
    np.testing.assert_array_equal(first, second)
