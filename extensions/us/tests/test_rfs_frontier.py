"""Financial invariants for the new RFS evaluation, independent of model fitting."""
import numpy as np
from experiments.rfs_frontier.engine import rank_target, rank_backtest, utility_flow
from test_us_transfer import fixture


def test_rank_ties_and_neutrality():
    scores = np.array([0., 0., 1., 2., 2.])
    weights = rank_target(scores)
    np.testing.assert_allclose(weights, [-.25, -.25, 0, .25, .25])
    np.testing.assert_allclose(rank_target(np.ones(5)), np.zeros(5))
    np.testing.assert_allclose(rank_target(-scores), -weights)


def test_rank_timing_payoff_and_terminal_liquidation():
    panel, pred = fixture()
    panel.total_return[:, 3] = 999.  # Cannot earn the execution day's move.
    panel.total_return[:, 4] = np.linspace(-.02, .02, 20)
    result, weights = rank_backtest(panel, pred, save_weights=True)
    expected = rank_target(pred.pred_return) @ panel.total_return[:, 4]
    np.testing.assert_allclose(result.return_net, expected)
    np.testing.assert_allclose(weights.sum(1), 0, atol=1e-12)
    np.testing.assert_allclose(abs(weights).sum(1), 1, atol=1e-12)
    assert result.liquidation_buy.iloc[0] > 0 and result.liquidation_sell.iloc[0] > 0


def test_rank_missing_payoff_stress_does_not_reselect():
    panel, pred = fixture()
    panel.total_return[19, 4] = np.nan
    ordinary, weights = rank_backtest(panel, pred, save_weights=True)
    stressed, stress_weights = rank_backtest(panel, pred, adverse_missing=True, save_weights=True)
    np.testing.assert_array_equal(weights, stress_weights)
    np.testing.assert_allclose(stressed.return_net - ordinary.return_net, -weights[:, 19])


def test_rank_frozen_holding_and_utility_scaling():
    panel, pred = fixture()
    panel.execution_tradable[19, 3] = False
    result, weights = rank_backtest(panel, pred, save_weights=True)
    assert weights[0, 19] == 0
    np.testing.assert_allclose(weights.sum(1), 0, atol=1e-10)
    x = np.array([.01, -.02, .03, .01])
    gamma, scale = 10, 2.
    observed = utility_flow(x * scale, gamma).mean()
    expected = scale * x.mean() - gamma / 2 * scale**2 * x.var(ddof=0)
    np.testing.assert_allclose(observed, expected)
