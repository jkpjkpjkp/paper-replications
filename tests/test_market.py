import numpy as np
from stco.market import (transaction_fraction, predict, backtest, ubah_backtest,
                         fixed_weight_backtest)
from stco.solver import PSGAConfig


def test_transaction_equation_and_one_way_purchase():
    w = np.array([.3, .7])
    np.testing.assert_allclose(transaction_fraction(np.zeros(2), w, .005), 1/1.005)
    for h in [np.array([.5, .5]), np.array([1., 0.])]:
        for gamma in [0., .0025, .01]:
            theta = transaction_fraction(h, w, gamma)
            assert abs(theta+gamma*np.abs(h-theta*w).sum()-1) < 2e-14
    theta = transaction_fraction(np.array([1., 0]), np.array([0., 1]), .01)
    np.testing.assert_allclose(theta, .99/1.01)


def test_prediction_startup_and_history():
    x = np.array([[2., 4.], [4., 2.], [2., 2.], [3., 3.]])
    np.testing.assert_allclose(predict(x, 0, 2, 4), (1+1/x[0])/2)
    np.testing.assert_allclose(predict(x, 2, 2, 4),
                               (1+1/x[2]+1/(x[2]*x[1])+1/(x[2]*x[1]*x[0]))/4)


def test_causal_backtest_and_self_financing():
    rng = np.random.default_rng(53)
    x = np.exp(rng.normal(0, .03, (35, 6)))
    cfg = PSGAConfig(k=3)
    result = backtest(x, .0025, config=cfg)
    altered = x.copy(); altered[20:] = 1.8
    other = backtest(altered, .0025, config=cfg)
    # Day 20's choice uses observations only through day 19 (zero-based).
    np.testing.assert_allclose(result["weights"][:21], other["weights"][:21])
    np.testing.assert_allclose(result["wealth"],
                               np.cumprod(result["gross"]*result["theta"]))
    np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=1e-10)
    assert result["converged"].all()


def test_buy_and_hold_and_single_asset_baseline():
    x = np.array([[1.1, .9], [.95, 1.2], [1.05, 1.1]])
    result = ubah_backtest(x, .0025)
    np.testing.assert_allclose(result["wealth"], np.cumprod(x, axis=0).mean(axis=1)/1.0025)
    fixed = fixed_weight_backtest(x, np.array([1., 0]), .0025)
    np.testing.assert_allclose(fixed["wealth"], np.cumprod(x[:, 0])/1.0025)
