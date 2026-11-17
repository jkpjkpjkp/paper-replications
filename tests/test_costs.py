import numpy as np
from hermes.costs import buy_only_costs


def test_buy_only_round_trip_charges_long_entry_and_short_cover():
    weights = np.array([[.5,-.5]])
    result = buy_only_costs(weights,np.zeros_like(weights))
    rate = .0005
    entry_fee = rate*.5/(1+rate*.5)
    expected = (1-entry_fee)*(1-rate*.5)-1
    np.testing.assert_allclose(result['net_portfolio_return'],[expected],atol=1e-15)
    np.testing.assert_allclose(result['transaction_fees'],rate*result['buy_turnover'],atol=1e-15)
    # Free final long sale and initial short sale; only two 50% NAV buys pay.
    assert .00049 < -expected < .00051


def test_holding_unchanged_books_without_price_moves_has_no_rebalance_fee():
    weights = np.tile([.5,-.5],(3,1))
    result = buy_only_costs(weights,np.zeros_like(weights),liquidate=False)
    np.testing.assert_allclose(result['transaction_fees'][1:],0,atol=1e-15)


def test_weight_drift_is_charged_and_zero_cost_matches_gross_return():
    weights = np.tile([.5,-.5],(3,1))
    returns = np.array([[.1,.1],[.01,-.02],[.03,.01]])
    result = buy_only_costs(weights,returns,liquidate=False)
    assert result['buy_turnover'][1] > 0
    free = buy_only_costs(weights,returns,buy_cost_bps=0)
    np.testing.assert_allclose(free['net_portfolio_return'],(weights*returns).sum(1),atol=1e-15)


def test_dividend_cash_does_not_change_share_holdings():
    weights=np.tile([.5,-.5],(2,1))
    total=np.array([[.01,.01],[0.,0.]])
    result=buy_only_costs(weights,total,price_returns=np.zeros_like(total),liquidate=False)
    # Matching dividend receipts/obligations cancel; unchanged prices need no rebalance.
    np.testing.assert_allclose(result['buy_turnover'][1],0,atol=1e-15)
