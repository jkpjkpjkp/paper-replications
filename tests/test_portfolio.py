import numpy as np
import pytest
from hermes.metrics import evaluate, market_neutral_portfolio


def test_dollar_neutral_books_and_short_pnl_by_hand():
    scores = np.array([[6.,5.,4.,3.,2.,1.]])
    returns = np.array([[.04,.02,0.,0.,.01,-.03]])
    book = market_neutral_portfolio(scores,returns,top_k=2)
    np.testing.assert_array_equal(book['portfolio_weights'],[[.25,.25,0.,0.,-.25,-.25]])
    np.testing.assert_allclose(book['long_pnl'],[.015])
    np.testing.assert_allclose(book['short_pnl'],[.005])
    np.testing.assert_allclose(book['portfolio_return'],[.020])


def test_common_market_move_cancels_and_ranking_uses_only_scores():
    scores = np.array([[3.,1.,4.,2.],[1.,2.,3.,4.]])
    returns = np.array([[.02,-.01,.04,.03],[-.02,.01,.05,.06]])
    first = market_neutral_portfolio(scores,returns,top_k=2)
    shifted = market_neutral_portfolio(scores,returns+np.array([[.10],[-.20]]),top_k=2)
    np.testing.assert_array_equal(first['portfolio_weights'],shifted['portfolio_weights'])
    np.testing.assert_allclose(first['portfolio_return'],shifted['portfolio_return'],atol=1e-16)
    constant = market_neutral_portfolio(scores,np.full_like(scores,.05),top_k=2)
    np.testing.assert_allclose(constant['portfolio_return'],0,atol=1e-16)


def test_masks_missing_values_small_universes_and_ties_preserve_neutrality():
    scores = np.array([[1.,1.,np.nan,1.,1.,100.,2.]])
    returns = np.array([[.01,.02,.03,.04,.05,.06,np.nan]])
    mask = np.array([[True,True,True,True,True,False,True]])
    book = market_neutral_portfolio(scores,returns,mask,top_k=5)
    weights = book['portfolio_weights']
    np.testing.assert_array_equal(weights,[[.25,.25,0.,-.25,-.25,0.,0.]])
    np.testing.assert_allclose(np.maximum(weights,0).sum(1),.5)
    np.testing.assert_allclose(np.minimum(weights,0).sum(1),-.5)
    np.testing.assert_allclose(weights.sum(1),0,atol=1e-15)
    np.testing.assert_allclose(np.abs(weights).sum(1),1)
    assert np.isfinite(book['portfolio_return']).all()


@pytest.mark.parametrize('top_k',[0,-1,1.5,True])
def test_invalid_book_sizes_are_rejected(top_k):
    with pytest.raises(ValueError,match='positive integer'):
        market_neutral_portfolio(np.array([[1.,2.]]),np.array([[.1,.2]]),top_k=top_k)


def test_insufficient_valid_stocks_are_rejected_instead_of_using_one_sided_book():
    with pytest.raises(ValueError,match='at least two valid stocks'):
        evaluate(np.array([[1.,np.nan]]),np.array([[.1,.2]]))
