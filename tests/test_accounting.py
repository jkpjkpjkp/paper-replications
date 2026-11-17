from types import SimpleNamespace
import numpy as np
import pandas as pd
from limt.portfolio import backtest


def fixture():
    market=np.full((60,25,7),100.,dtype=np.float64)
    market[:,:,1]=1e6
    market[:,:,2]=1.
    market[:,:,4]=101.
    market[:,:,5]=99.
    panel=SimpleNamespace(market=market,member=np.ones((60,25),bool),
        dates=np.array(pd.bdate_range('2020-01-01',periods=25).strftime('%Y-%m-%d')),
        symbols=np.array([str(i) for i in range(60)]))
    pred=pd.DataFrame(dict(date=panel.dates[20],symbol=panel.symbols,
        pred_return=np.arange(60),pred_volume=0.,pred_volatility=0.))
    return panel,pred


def test_return_and_fee_use_t_plus_one_execution_and_total_capital():
    panel,pred=fixture()
    # Only the long decile appreciates, after the execution close.
    panel.market[54:,22,0]=110.
    daily,metric=backtest(panel,pred,strategy='equal',capital=1000.,fee=.001)
    assert len(daily)==1
    np.testing.assert_allclose(daily.iloc[0].fees,2.)
    np.testing.assert_allclose(daily.iloc[0].return_net,(100.-2.)/2000.)
    np.testing.assert_allclose(daily.iloc[0].gross_exposure,1.)
    assert daily.iloc[0].execution_date==panel.dates[21]
    assert daily.iloc[0].date==panel.dates[22]


def test_suspension_blocks_order_and_keeps_unspent_cash():
    panel,pred=fixture()
    panel.market[54,21,1]=0.  # Not observable when candidates were chosen at t.
    panel.market[54:,22,0]=110.
    daily,metric=backtest(panel,pred,strategy='equal',capital=1000.,fee=.001)
    requested=2000.
    filled=requested-1000./6
    expected=(1000.*5/6*.1-filled*.001)/2000.
    np.testing.assert_allclose(daily.iloc[0].filled_value,filled)
    np.testing.assert_allclose(daily.iloc[0].return_net,expected)
    np.testing.assert_allclose(metric['order_success_rate'],11/12)
