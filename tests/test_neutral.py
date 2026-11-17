import numpy as np
import pytest
import pandas as pd
from limt.neutral import neutral_projection,neutral_backtest
from tests.test_accounting import fixture


def test_neutral_projection_respects_asymmetric_capacity():
    result=neutral_projection([10.,-10.],[0.,-10.],[5.,0.])
    np.testing.assert_allclose(result,[5.,-5.],atol=1e-6)


def test_infeasible_neutrality_is_reported():
    with pytest.raises(ValueError,match='infeasible'):
        neutral_projection([1.,-1.],[2.,0.],[3.,1.])


def test_directional_fees_and_equal_executed_legs():
    panel,pred=fixture();panel.market[54:,22,0]=110.
    daily,metrics=neutral_backtest(panel,pred,buy_fee=.0005,sell_fee=.0015,capital=1000.)
    row=daily.iloc[0]
    np.testing.assert_allclose([row.long_notional,row.short_notional],[1000.,1000.])
    np.testing.assert_allclose([row.buy_value,row.sell_value],[1000.,1000.])
    np.testing.assert_allclose(row.fees,2.)
    np.testing.assert_allclose(row.return_net,(100.-2.)/2000.)
    assert metrics['max_absolute_net_exposure']<1e-10


def test_float32_quotes_and_drifting_positions_preserve_neutrality():
    panel,pred=fixture();rng=np.random.default_rng(23)
    panel.market=panel.market.astype(np.float32)
    prices=(100*np.cumprod(1+rng.uniform(-.03,.03,(60,25)),axis=1)).astype(np.float32)
    for k in [0,3,6]:panel.market[:,:,k]=prices
    panel.market[:,:,4]=prices*1.02;panel.market[:,:,5]=prices*.98
    panel.market[:,:,1]=rng.uniform(300,3000,(60,25)).astype(np.float32)
    pred=pd.concat([pred.assign(date=panel.dates[t]) for t in [20,21,22]],ignore_index=True)
    daily,metrics=neutral_backtest(panel,pred,buy_fee=.0005,sell_fee=.0015,capital=1e9)
    assert len(daily)==3
    np.testing.assert_allclose(daily.long_notional,daily.short_notional,atol=.2,rtol=0)
    assert metrics['max_absolute_net_exposure']<1e-8
