"""Self-financing daily P&L with the user's buy-transactions-only cost convention."""
import numpy as np


def buy_only_costs(weights, asset_returns, buy_cost_bps=5., liquidate=True, price_returns=None):
    """Charge purchases and short covers; sales and new shorts are free.

    The target weights refer to NAV after paying rebalancing fees. A fixed
    point accounts for that fee's small effect on the dollars traded. Pretrade
    weights drift with asset returns, so turnover includes price moves. Start
    flat; when liquidate=True, cover the final short book and sell the long
    book at the end, charging only the buy-to-cover orders.
    """
    weights, asset_returns = np.asarray(weights,float), np.asarray(asset_returns,float)
    if weights.ndim != 2 or weights.shape != asset_returns.shape or not len(weights):
        raise ValueError('weights and asset returns must have equal nonempty [day, stock] shapes')
    if not np.isfinite(weights).all() or buy_cost_bps < 0 or buy_cost_bps >= 100:
        raise ValueError('invalid weights or buy cost')
    if np.any(~np.isfinite(asset_returns) & (weights != 0)):
        raise ValueError('a held position has a missing realized return')
    asset_returns = np.where(np.isfinite(asset_returns),asset_returns,0.)
    price_returns = asset_returns if price_returns is None else np.asarray(price_returns,float)
    if price_returns.shape != weights.shape or np.any(~np.isfinite(price_returns)&(weights!=0)):
        raise ValueError('a held position has a missing price return')
    price_returns = np.where(np.isfinite(price_returns),price_returns,0.)
    rate = buy_cost_bps / 10000.
    old = np.zeros(weights.shape[1])
    n = len(weights)
    net, gross, fees, buys, sells, entry_fees, exit_fees = [np.zeros(n) for _ in range(7)]
    for day, (target, realized, price_move) in enumerate(zip(weights,asset_returns,price_returns)):
        fee = 0.
        for _ in range(50):
            purchases = np.maximum((1-fee)*target-old,0.).sum()
            updated = rate * purchases
            if abs(updated-fee) < 1e-15:
                fee = updated
                break
            fee = updated
        else:
            raise ArithmeticError('rebalancing fee calculation did not converge')
        trades = (1-fee)*target-old
        buys[day], sells[day] = np.maximum(trades,0.).sum(), np.maximum(-trades,0.).sum()
        gross[day] = target @ realized
        if 1+gross[day] <= 0 or fee >= 1:
            raise ValueError('portfolio NAV is nonpositive')
        factor = (1-fee)*(1+gross[day])
        net[day] = factor-1
        fees[day], entry_fees[day] = fee, fee
        # Dividends affect cash/NAV, while share holdings drift with prices only.
        old = target*(1+price_move)/(1+gross[day])
        if liquidate and day == n-1:
            cover = np.maximum(-old,0.).sum()
            sales = np.maximum(old,0.).sum()
            exit_fee = rate*cover
            buys[day] += cover*factor
            sells[day] += sales*factor
            fees[day] += exit_fee*factor
            exit_fees[day] = exit_fee*factor
            net[day] = factor*(1-exit_fee)-1
    np.testing.assert_allclose(fees, rate*buys,rtol=1e-12,atol=1e-15)
    return {'net_portfolio_return':net,'buy_turnover':buys,'sell_turnover':sells,
            'transaction_fees':fees,'entry_fee_fraction':entry_fees,'exit_fee_fraction':exit_fees}
