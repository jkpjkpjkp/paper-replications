from types import SimpleNamespace
import numpy as np
import pandas as pd
from portfolio import backtest
from usdata import fill_window


def fixture():
    symbols=np.array([f'P{i}' for i in range(20)])
    dates=np.array(pd.bdate_range('2020-01-01',periods=8).strftime('%Y-%m-%d'))
    panel=SimpleNamespace(symbols=symbols,dates=dates,member=np.ones((20,8),bool),
        market=np.ones((20,8,7))*100,total_return=np.zeros((20,8)),price_return=np.zeros((20,8)),
        terminal=np.zeros((20,8),bool),execution_tradable=np.ones((20,8),bool))
    frame=pd.DataFrame(dict(date=dates[2],realized_date=dates[4],symbol=symbols,
        pred_return=np.arange(20),pred_volume=0.,pred_volatility=0.))
    return panel,frame


def test_execution_delay_total_return_and_equal_legs():
    p,f=fixture()
    p.total_return[-2:,4]=.10;p.total_return[:2,4]=.04
    p.total_return[-2:,3]=99.  # Execution-session return cannot be earned.
    result,w=backtest(p,f,save_weights=True)
    np.testing.assert_allclose(result.return_net,.03)
    np.testing.assert_allclose(w.sum(1),0,atol=1e-12)
    np.testing.assert_allclose(abs(w).sum(1),1,atol=1e-12)
    assert result.execution_date.iloc[0]==p.dates[3]


def test_buys_include_terminal_short_covers_sells_free():
    p,f=fixture()
    result,_=backtest(p,f,buy_bp=5.)
    # Long opening buy=0.5*NAV, final short cover=0.5*NAV.
    np.testing.assert_allclose(result.fees,1e6)
    np.testing.assert_allclose(result.return_net,-.0005)


def test_missing_future_payoff_preserves_stock_selection_and_is_stressed():
    p,f=fixture();p.total_return[19,4]=np.nan
    original,w=backtest(p,f,save_weights=True)
    stressed,z=backtest(p,f,adverse_missing=True,save_weights=True)
    np.testing.assert_array_equal(w,z)
    np.testing.assert_allclose(original.missing_return_gross_weight,.25)
    np.testing.assert_allclose(stressed.return_net,-.25)


def test_blocked_execution_remains_dollar_neutral():
    p,f=fixture();p.execution_tradable[19,3]=False
    result,w=backtest(p,f,save_weights=True)
    assert w[0,19]==0
    np.testing.assert_allclose(w.sum(1),0,atol=1e-10)
    np.testing.assert_allclose(result.long_value,result.short_value,atol=1.)


def test_window_fill_does_not_read_future_rows():
    past=np.array([[[1.,np.nan],[2.,3.],[np.nan,4.]]],np.float32)
    out=fill_window(past)
    np.testing.assert_array_equal(out,[[[1.,3.5],[2.,3.],[2.,4.]]])
    assert np.isnan(past[0,0,1])
