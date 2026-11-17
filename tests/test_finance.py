import numpy as np
import pandas as pd

from finance.data import stock_features
from finance.portfolio import equal_long_short, portfolio_path
from finance.statistics import hac_mean, holm, performance, stationary_indices
from finance.train import split_masks


def test_equal_long_short_deterministic_ties_and_zero_net_exposure():
    ids = np.arange(100)
    scores = np.zeros(100)
    weights = equal_long_short(ids, scores, 0.1)
    np.testing.assert_allclose(weights.sum(), 0, atol=1e-15)
    np.testing.assert_allclose(abs(weights).sum(), 2)
    assert (weights > 0).sum() == (weights < 0).sum() == 10
    np.testing.assert_allclose(weights[:10], -0.1)
    np.testing.assert_allclose(weights[-10:], 0.1)


def test_initial_entry_cost_counts_both_legs_and_borrow():
    holdings = pd.DataFrame({"month": [1, 1], "permno": [1, 2], "weight": [1., -1.],
                              "realized_return": [0., 0.], "terminal": [False, False]})
    result = portfolio_path(holdings, 10, 0.012).iloc[0]
    np.testing.assert_allclose(result.turnover, 2)
    np.testing.assert_allclose(result.transaction_cost, 0.002)
    np.testing.assert_allclose(result.borrow_cost, 0.001)
    np.testing.assert_allclose(result["return"], -0.003)


def test_drifted_holdings_require_rebalancing_even_when_names_do_not_change():
    holdings = pd.DataFrame({"month": [1, 1, 2, 2], "permno": [1, 2, 1, 2],
                              "weight": [1., -1., 1., -1.], "realized_return": [0.1, 0, 0, 0],
                              "terminal": [False] * 4})
    result = portfolio_path(holdings, 0, 0)
    np.testing.assert_allclose(result.iloc[0]["return"], 0.1)
    np.testing.assert_allclose(result.iloc[1].turnover, 1 / 11)


def test_terminal_cash_conversion_does_not_create_fictitious_exit_trade():
    holdings = pd.DataFrame({"month": [1, 1, 2, 2], "permno": [1, 2, 3, 2],
                              "weight": [1., -1., 1., -1.], "realized_return": [0.1, 0, 0, 0],
                              "terminal": [True, False, False, False]})
    result = portfolio_path(holdings, 0, 0)
    np.testing.assert_allclose(result.iloc[1].turnover, 1 + 1 / 11)


def test_missing_return_is_retained_and_adverse_stress_changes_pnl():
    holdings = pd.DataFrame({"month": [1] * 4, "permno": [1, 2, 3, 4],
                              "weight": [.5, .5, -.5, -.5], "realized_return": [np.nan, 0, 0, 0],
                              "terminal": [False] * 4})
    normal = portfolio_path(holdings, 0, 0).iloc[0]
    stress = portfolio_path(holdings, 0, 0, adverse_missing=True).iloc[0]
    assert normal.missing_count == 1 and normal.n_long == 2
    np.testing.assert_allclose(normal["return"], 0)
    np.testing.assert_allclose(stress["return"], -.5)


def test_hac_standard_error_matches_independent_bartlett_covariance():
    y = np.random.default_rng(2).normal(0.01, 0.04, 100)
    lag, n = 6, len(y)
    residual = y - y.mean()
    variance = residual @ residual
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (residual[k:] @ residual[:-k])
    se = np.sqrt(variance / n**2 * n / (n - 1))
    result = hac_mean(y, lag)
    np.testing.assert_allclose(result["mean_se_annual"], 12 * se, rtol=1e-12)


def test_drawdown_includes_initial_nav_and_holm_adjusts_family():
    result = performance(np.r_[-0.1, np.repeat(0.001, 23)])
    np.testing.assert_allclose(result["max_drawdown"], -0.1)
    np.testing.assert_allclose(holm([.01, .04, .2]), [.03, .08, .2])


def test_chronological_boundaries_have_one_month_label_embargo():
    months = np.arange(1990 * 12, 2010 * 12)
    frame = pd.DataFrame({"month": months, "y_rank": np.zeros(len(months))})
    masks, boundaries = split_masks(frame, 2005, 2009,
                                    {"train_months": 120, "validation_months": 24, "label_embargo_months": 1})
    assert boundaries["val_end"] == 2004 * 12 + 10  # November 2004
    assert frame.loc[masks["train"], "month"].max() < frame.loc[masks["val"], "month"].min()
    assert not (masks["train"] | masks["val"] | masks["test"])[frame.month.eq(2004 * 12 + 11)].any()


def test_finite_partial_delisting_payoff_is_included_in_adverse_stress():
    holdings = pd.DataFrame({"month": [1] * 6, "permno": np.arange(6),
                              "weight": [1 / 3] * 3 + [-1 / 3] * 3,
                              "realized_return": [0.] * 6, "terminal": [True] + [False] * 5,
                              "unresolved_payoff": [True] + [False] * 5})
    normal = portfolio_path(holdings, 0, 0).iloc[0]
    stress = portfolio_path(holdings, 0, 0, adverse_missing=True).iloc[0]
    assert normal.missing_count == 0 and normal.unresolved_payoff_count == 1
    np.testing.assert_allclose(stress["return"], -1 / 3)


def test_future_source_returns_cannot_change_past_features():
    n = 50
    raw = pd.DataFrame({"permno": np.ones(n, dtype=int), "month": np.arange(1000, 1000 + n),
                        "ret": .03 * np.sin(np.arange(n)), "volume": np.arange(n) * 100 + 10000,
                        "shrout": np.repeat(1000., n), "price": np.arange(n) + 20.,
                        "cap": np.arange(n) * 1000 + 100000., "dollar_volume": np.arange(n) * 10000 + 1000000.})
    ff = pd.DataFrame({"month": raw.month, "mkt_rf": .02 * np.cos(np.arange(n)), "rf": .001})
    first, names = stock_features(raw, ff)
    changed = raw.copy()
    changed.loc[changed.month >= 1035, "ret"] = 0.9
    second, _ = stock_features(changed, ff)
    np.testing.assert_allclose(first.loc[first.month < 1035, names],
                               second.loc[second.month < 1035, names], rtol=0, atol=0)


def test_stationary_bootstrap_is_seeded_and_uses_valid_time_indices():
    indices = stationary_indices(24, 100, 12, 7)
    np.testing.assert_array_equal(indices, stationary_indices(24, 100, 12, 7))
    assert indices.min() >= 0 and indices.max() < 24
    continuation = indices[:, 1:] == ((indices[:, :-1] + 1) % 24)
    assert continuation.mean() > .85
