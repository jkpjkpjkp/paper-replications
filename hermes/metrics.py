import numpy as np
from .costs import buy_only_costs


PORTFOLIO_PROTOCOL = {
    'name': 'dollar_neutral_top_bottom_equal_weight',
    'rebalance': 'daily',
    'long_exposure': 0.5, 'short_exposure': -0.5,
    'gross_exposure': 1.0, 'net_exposure': 0.0,
    'selection': 'highest top_k scores long; lowest top_k scores short; disjoint books',
    'small_universe': 'use min(top_k, valid_stocks//2) stocks on each side',
    'ties': 'stable descending score order, then stored ticker order; bottom book uses the end of the same ordering',
    'buy_cost_bps': 5.0, 'sell_cost_bps': 0.0,
    'cost_convention': 'buys only: long purchases and short covers; sells and new shorts free',
    'cost_turnover': 'actual traded dollars after drift; initial entry and final liquidation included',
    'position_drift': 'price returns mark holdings; total returns determine P&L when supplied',
    'annualization_days': 252,
    'definition': 'dollar neutrality at each daily rebalance; beta neutrality is not imposed',
}


def _validated_arrays(prediction, target, mask):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.ndim != 2 or target.shape != prediction.shape:
        raise ValueError('prediction and target must have equal [day, stock] shapes')
    valid = np.ones(target.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if valid.shape != target.shape:
        raise ValueError('mask must have the same shape as target')
    valid = valid & np.isfinite(prediction) & np.isfinite(target)
    if not len(target) or np.any(valid.sum(1) < 2):
        raise ValueError('each evaluated day needs at least two valid stocks for disjoint long/short books')
    return prediction, target, valid


def market_neutral_portfolio(prediction, target, mask=None, top_k=5):
    """50% NAV long the top scores, 50% NAV short the bottom scores.

    Return = 0.5 * (mean(long underlying returns) - mean(short underlying
    returns)); long plus absolute short exposure is 100% NAV. Ranking uses
    predictions only. Long/short weights are audited and saved with results.
    """
    prediction, target, valid = _validated_arrays(prediction, target, mask)
    if not isinstance(top_k, (int, np.integer)) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    weights = np.zeros(target.shape, dtype=np.float64)
    positions = np.empty(len(target), dtype=np.int64)
    for day, (scores, eligible) in enumerate(zip(prediction, valid)):
        stocks = np.flatnonzero(eligible)
        order = stocks[np.argsort(-scores[stocks], kind='stable')]
        k = min(top_k, len(stocks) // 2)
        weights[day, order[:k]] = 0.5 / k
        weights[day, order[-k:]] = -0.5 / k
        positions[day] = k
    safe_target = np.where(valid, target, 0.).astype(np.float64)
    long_pnl = (np.maximum(weights, 0.) * safe_target).sum(1)
    short_pnl = (np.minimum(weights, 0.) * safe_target).sum(1)
    return {
        'portfolio_weights': weights, 'portfolio_return': long_pnl + short_pnl,
        'long_pnl': long_pnl, 'short_pnl': short_pnl, 'positions_per_leg': positions,
    }


def evaluate(prediction, target, mask=None, top_k=5, pnl_target=None):
    """Daily Pearson IC and dollar-neutral, daily-rebalanced top/bottom portfolio."""
    prediction, target, valid = _validated_arrays(prediction, target, mask)
    pnl_target = target if pnl_target is None else np.asarray(pnl_target)
    if pnl_target.shape != target.shape or np.any(valid & ~np.isfinite(pnl_target)):
        raise ValueError('P&L returns must be finite for every eligible stock')
    book = market_neutral_portfolio(prediction, pnl_target, valid, top_k)
    costs = buy_only_costs(book['portfolio_weights'],pnl_target,PORTFOLIO_PROTOCOL['buy_cost_bps'],price_returns=target)
    ics = []
    for p, y, m in zip(prediction, target, valid):
        p, y = p[m], y[m]
        if np.std(p) > 0 and np.std(y) > 0:
            ics.append(float(np.corrcoef(p, y)[0, 1]))
    ic = np.asarray(ics)
    returns = book['portfolio_return']
    ratio = lambda mean, std: float(mean / std) if std > 0 else None
    sr = ratio(returns.mean(), returns.std())
    net = costs['net_portfolio_return']
    net_sr = ratio(net.mean(), net.std())
    return {
        'IC': float(ic.mean()) if len(ic) else None,
        'ICIR': ratio(ic.mean(), ic.std()) if len(ic) else None,
        'SR_daily': sr, 'SR_annualized': None if sr is None else float(sr * np.sqrt(252)),
        'mean_daily_return': float(returns.mean()),
        'cumulative_return': float(np.prod(1 + returns) - 1),
        'mean_long_pnl': float(book['long_pnl'].mean()),
        'mean_short_pnl': float(book['short_pnl'].mean()),
        'mse': float(np.mean((prediction[valid] - target[valid]) ** 2)),
        'days': len(returns), 'IC_days': len(ic), 'min_stocks': int(valid.sum(1).min()),
        'top_k': int(top_k), 'min_positions_per_leg': int(book['positions_per_leg'].min()),
        'long_exposure': 0.5, 'short_exposure': -0.5,
        'gross_exposure': 1.0, 'net_exposure': 0.0,
        'net_SR_daily':net_sr, 'net_SR_annualized':None if net_sr is None else float(net_sr*np.sqrt(252)),
        'mean_net_daily_return':float(net.mean()), 'net_cumulative_return':float(np.prod(1+net)-1),
        'mean_buy_turnover':float(costs['buy_turnover'].mean()),
        'mean_sell_turnover':float(costs['sell_turnover'].mean()),
        'mean_transaction_fee':float(costs['transaction_fees'].mean()),
        'buy_cost_bps':5.0, 'sell_cost_bps':0.0,
    }, ic, returns
