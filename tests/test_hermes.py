import numpy as np
import pytest
import torch
from hermes.model import CausalMixing, Hermes, moving_aggregation, ranking_loss
from hermes.metrics import evaluate


def test_causal_mixing_cannot_see_future():
    torch.manual_seed(2)
    module = CausalMixing(8)
    x = torch.randn(2, 4, 8, 5)
    altered = x.clone()
    altered[:, :, 4:] += 100
    torch.testing.assert_close(module(x)[:, :, :4], module(altered)[:, :, :4])


def test_lead_lag_matches_explicit_loop_and_uses_no_future():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 7, 4, dtype=torch.float64)
    actual = moving_aggregation(x, 3)
    expected = x.clone()
    for b in range(2):
        for t in range(2, 7):
            sources = x[b, :, t - 2:t + 1].reshape(9, 4)
            weight = (sources @ x[b, :, t].T).softmax(0)
            expected[b, :, t] = weight.T @ sources
    torch.testing.assert_close(actual, expected)
    future = x.clone()
    future[:, :, 4:] *= 100
    torch.testing.assert_close(actual[:, :, :4], moving_aggregation(future, 3)[:, :, :4])


def test_metric_fusion_matches_literal_equations():
    model = Hermes(np.eye(2), lookback=2, features=1, hidden=1, scales=(1,), windows=(1,)).double()
    model.edge_projection.weight.data.fill_(1)
    edges = torch.tensor([[[[[1.], [1.]], [[3.], [3.]]]]], dtype=torch.float64)
    logits = torch.tensor([[0., .25], [.25, 0.]], dtype=torch.float64)
    expected = (logits.softmax(0) @ torch.tensor([[1.], [3.]], dtype=torch.float64)).view(1, 1, 2, 1, 1).expand_as(edges)
    torch.testing.assert_close(model.fuse(edges), expected)


def test_loss_matches_hand_computed_pairs_and_mask():
    p = torch.tensor([[1., -1., 900.]])
    y = torch.tensor([[-1., 1., -900.]])
    mask = torch.tensor([[True, True, False]])
    loss, mse, rank = ranking_loss(p, y, mask)
    assert mse == 4 and rank == 8 and loss == 12
    assert ranking_loss(y, y, mask)[0] == 0


def test_model_all_ablations_finite_and_stock_permutation_equivariant():
    h = torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.bool)
    for ablation in ['full', 'no_fusion', 'single_scale', 'no_lead_lag', 'no_skip', 'initial_only']:
        model = Hermes(h, lookback=8, hidden=3, scales=(1, 2), windows=(3, 2), ablation=ablation)
        x = torch.randn(2, 4, 8, 5)
        prediction = model(x)
        assert prediction.shape == (2, 4)
        prediction.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        perm = torch.tensor([2, 0, 3, 1])
        model.incidence = h[perm]
        torch.testing.assert_close(model(x[:, perm]), prediction[:, perm], atol=2e-6, rtol=2e-5)


def test_metrics_have_known_correlation_and_portfolio():
    target = np.array([[.01, .02, -.01], [-.02, .01, .02]])
    metrics, daily_ic, portfolio = evaluate(target, target, top_k=1)
    np.testing.assert_allclose(daily_ic, 1)
    np.testing.assert_allclose(portfolio, [.015, .02])
    assert metrics['mse'] == 0


def test_total_multiscale_ablation_also_removes_fusion():
    h = torch.tensor([[1,0],[1,0],[0,1],[0,1]],dtype=torch.bool)
    single = Hermes(h,lookback=8,hidden=3,scales=(1,2),windows=(3,2),ablation='single_scale')
    no_fusion = Hermes(h,lookback=8,hidden=3,scales=(1,),windows=(3,),ablation='no_fusion')
    no_fusion.load_state_dict(single.state_dict())
    x = torch.randn(2,4,8,5)
    torch.testing.assert_close(single(x), no_fusion(x))


def test_real_data_alignment_and_train_only_statistics():
    from pathlib import Path
    d = np.load(Path(__file__).resolve().parents[1] / 'data/sp500.npz')
    assert d['features'].shape == (1611, 474, 5)
    assert d['incidence'].shape == (474, 11)
    assert d['incidence'].sum(1).min() == 1
    np.testing.assert_allclose(d['features'][:1006].mean(0), 0, atol=2e-5)
    np.testing.assert_allclose(d['features'][:1006].std(0), 1, atol=2e-5)
    assert str(d['dates'][1259]) == '2021-01-04'
    # Undo feature normalization to independently verify target t uses close t/t-1.
    close = d['features'][..., 3] * d['std'][..., 3] + d['mean'][..., 3]
    # Inverting float32 standardized prices introduces sub-micro return rounding.
    np.testing.assert_allclose(d['returns'][1:], close[1:] / close[:-1] - 1, atol=1e-6)


def test_companion_releases_have_explicit_shorter_test_sets_and_correct_returns():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    if not all((root / 'data' / f'{market}.npz').exists() for market in ['nasdaq','nyse']):
        pytest.skip('optional companion sources not prepared')
    for market, stocks, sectors in [('nasdaq',1026,113),('nyse',1737,130)]:
        d = np.load(root / 'data' / f'{market}.npz')
        assert d['features'].shape == (1245,stocks,5)
        assert d['incidence'].shape == (stocks,sectors)
        assert len(d['returns'][1008:]) == 237
        assert str(d['dates'][1008]) == '2017-01-03'
        np.testing.assert_allclose(d['features'][:756].mean(0),0,atol=2e-5)
        ticker = d['tickers'][0]
        raw = np.loadtxt(root / 'upstream/Temporal_Relational_Stock_Ranking/data/2013-01-01' / f'{market.upper()}_{ticker}_1.csv',delimiter=',')[:1245,-1]
        mask = d['valid'][1:,0]
        expected = raw[1:][mask] / raw[:-1][mask] - 1
        np.testing.assert_allclose(d['returns'][1:,0][mask],expected,atol=1e-7)
