import json
from pathlib import Path

import numpy as np
import pytest
import torch

from transalpha.model import CrossSectionalDenoising, TransAlpha
from transalpha.loss import hybrid_loss, cross_section_loss, ordinal_rank, pearson
from transalpha.data import split_indices, window_eligibility
from transalpha.evaluation import evaluate, rebalance_fee, liquidation_notional


def test_denoising_threshold_preserves_small_residuals():
    csd = CrossSectionalDenoising(2, threshold=.001)
    with torch.no_grad():
        csd.alpha_scale[:] = 3
        csd.beta_scale[:] = 2
    x = torch.tensor([[[1.0001, 2.]], [[.9999, 0.]]])
    expected = torch.tensor([[[2.0001, 5.]], [[1.9999, -1.]]])
    torch.testing.assert_close(csd(x), expected)


def test_cross_sections_are_independent_and_padding_is_ignored():
    torch.manual_seed(0)
    model = TransAlpha(7, hidden=8, dropout=0).eval()
    with torch.no_grad():
        model.csd.alpha_scale[:] = 1.8
        model.csd.beta_scale[:] = .2
    x = torch.randn(2, 12, 16, 7)
    p, a = model(x, return_attention=True)
    torch.testing.assert_close(p[0], model(x[0]))
    torch.testing.assert_close(a.sum(-2), torch.ones(2, 12, 1))
    mask = torch.ones(2, 15, dtype=torch.bool)
    mask[:, 12:] = False
    padded = torch.cat([x, torch.randn(2, 3, 16, 7) * 1000], dim=1)
    torch.testing.assert_close(model(padded, mask)[:, :12], p)
    permutation = torch.randperm(12)
    torch.testing.assert_close(model(x[:, permutation]), p[:, permutation])
    # Arbitrary stock chunking changes beta and thus predictions.
    assert not torch.allclose(model(x[0, :6]), p[0, :6], atol=1e-5)


def test_literal_hard_rank_and_trainable_product_loss():
    p = torch.tensor([-.1, .3, .5, .2, -.4, .6, .0, .8, -.2, .4], requires_grad=True)
    y = torch.tensor([.01, .02, .03, .04, .05, .06, .07, .08, .09, .10])
    loss, parts = hybrid_loss(p, y, [p])
    assert not ordinal_rank(p).requires_grad
    assert not parts["spearman"].requires_grad
    expected = sum(parts[k] * w for k, w in zip(["rank", "huber", "corr", "prod", "reg"], [.4, .05, .4, .1, .05]))
    torch.testing.assert_close(loss, expected)
    parts["prod"].backward(retain_graph=True)
    assert p.grad.abs().sum() > 0
    p.grad.zero_()
    loss.backward()
    assert torch.isfinite(p.grad).all()


def test_rank_loss_uses_all_top_bottom_pairs():
    y = torch.arange(10.)
    p = torch.tensor([0., 1., .1, .2, .3, .4, .5, .6, -.5, 2.])
    expected = np.mean([max(0, .1 - (a - b)) for a in [-.5, 2.] for b in [0., 1.]])
    assert float(cross_section_loss(p, y)["rank"]) == pytest.approx(expected)


def test_vectorized_loss_matches_independent_dates_and_gradients():
    torch.manual_seed(17)
    p = torch.randn(3, 31, requires_grad=True)
    y = torch.randn_like(p)
    y[0, 9:] = torch.nan
    y[1, 25:] = torch.nan
    loss, batched = hybrid_loss(p, y, [p])
    separate = [cross_section_loss(p[i][torch.isfinite(y[i])], y[i][torch.isfinite(y[i])]) for i in range(3)]
    for key in separate[0]:
        torch.testing.assert_close(batched[key], torch.stack([s[key] for s in separate]).mean())
    reference = sum(torch.stack([s[k] for s in separate]).mean() * w for k, w in zip(["rank", "huber", "corr", "prod"], [.4, .05, .4, .1])) + .05e-5 * p.square().sum()
    actual_grad, = torch.autograd.grad(loss, p, retain_graph=True)
    reference_grad, = torch.autograd.grad(reference, p)
    torch.testing.assert_close(actual_grad, reference_grad)


def test_lookback_uses_identity_and_requires_every_date():
    x = np.ones((20, 2, 3))
    x[3, 0, 1] = np.nan
    mask = window_eligibility(x, np.ones((20, 2), bool), 16)
    assert not mask[:15].any()
    assert not mask[15:19, 0].any()
    assert mask[19, 0] and mask[15, 1]


def test_split_boundary_purge_uses_label_exit():
    dates = np.array(["2023-09-25", "2023-09-26", "2023-09-27", "2023-09-28", "2023-10-09", "2023-10-10"])
    config = {"train": ["2023-09-25", "2023-09-28"], "validation": ["2023-10-09", "2023-10-10"], "test": ["2024-01-02", "2024-12-31"]}
    assert split_indices(dates, config)["train"] == [0, 1]


def test_cost_liquidations_include_short_covers_and_drift():
    old = np.array([.5, -.5])
    assert liquidation_notional(old, np.zeros(2)) == 1
    assert liquidation_notional(old, -old) == 1
    assert rebalance_fee(np.zeros(2), old, .001) == 0
    assert rebalance_fee(old, -old, .001) == pytest.approx(.001)
    assert rebalance_fee(np.array([.6, -.4]), old, .001) > .0001


def test_portfolio_has_half_long_half_short_and_terminal_cost():
    prediction = np.array([[0., 1., 2., 3.], [0., 1., 2., 3.]])
    returns = np.array([[.02, .04, .06, .08], [.02, .04, .06, .08]])
    metrics, daily = evaluate(prediction, returns, ["a", "b"], list("abcd"), topk=1)
    assert np.allclose(daily.gross_return, .5 * (.08 - .02))
    assert daily.rebalance_fee.iloc[0] == 0
    assert daily.terminal_fee.iloc[-1] > 0
    assert metrics["net"]["total_return"] < metrics["gross"]["total_return"]
    assert daily.traded_notional_over_nav.iloc[0] == 1


def test_future_missing_return_does_not_change_selection():
    p = np.arange(100.)[None]
    y = np.zeros_like(p)
    y[0, 99] = np.nan
    with pytest.raises(ValueError, match="unknown future return"):
        evaluate(p, y, ["2024-01-02"], np.arange(100))


def test_paper_dimension_and_all_ablation_gradients():
    for variant in ["transalpha", "no_csd", "no_tag", "no_sp"]:
        model = TransAlpha(400, variant=variant)
        p = model(torch.randn(25, 16, 400))
        assert p.shape == (25,)
        loss, _ = hybrid_loss(p, torch.randn(25), model.parameters())
        loss.backward()
        assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)


def test_paper_pooling_preserves_last_step_but_not_earlier_order():
    torch.manual_seed(24)
    model = TransAlpha(8, hidden=16, dropout=0).eval()
    x = torch.randn(12, 16, 8)
    earlier_permutation = torch.cat([torch.randperm(15), torch.tensor([15])])
    torch.testing.assert_close(model(x), model(x[:, earlier_permutation]))
    assert not torch.allclose(model(x), model(x.flip(1)), atol=1e-5)


def test_suspended_holding_is_retained_and_cash_ledger_reconciles():
    from scripts.audit import cash_share_replay
    predictions = np.array([[0., 1., 2., 3., 4., 5.], [0., 1., 2., 3., 5., 4.], [0., 1., 2., 3., 4., 5.]])
    prices = np.array([[10., 10., 10., 10., 10., 10.], [10., 10., 10., 10., 10., 11.],
                       [10., 10., 10., 10., 12., 11.], [10., 10., 10., 10., 12., 12.]])
    tradable = np.ones_like(predictions, bool)
    tradable[1, 5] = False
    returns = prices[1:] / prices[:-1] - 1
    metrics, daily = evaluate(predictions, returns, ["a", "b", "c"], np.arange(6), topk=1,
                              portfolio_returns=returns, execution_tradable=tradable)
    ledger = cash_share_replay(predictions, prices[:-1], prices[1:], tradable, topk=1)
    np.testing.assert_allclose(ledger, daily.net_return, atol=1e-12)
    assert daily.frozen_positions.iloc[1] == 1
    assert daily.blocked_forecasts.iloc[1] == 1
    # The signal switches to stock 4 while stock 5 is frozen, consuming long budget.
    assert daily.gross_return.iloc[1] == pytest.approx(0.)
    assert metrics["frozen_position_days"] == 1


def test_built_panels_have_purged_splits_and_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = list((root / "data").glob("*/manifest.json"))
    if not paths:
        pytest.skip("Raw panels not built")
    config = json.loads((root / "config.json").read_text())
    for path in paths:
        manifest = json.loads(path.read_text())
        dates = np.load(path.parent / "dates.npy")
        y = np.load(path.parent / "target.npy")
        close = np.load(path.parent / "adjusted_close.npy")
        for split in ["train", "validation"]:
            assert max(dates[np.array(manifest["indices"][split]) + 2]) <= config[split][1]
        assert manifest["days"]["test"] == 242
        np.testing.assert_allclose(y[:-2], close[2:] / close[1:-1] - 1, rtol=1e-5, atol=1e-7, equal_nan=True)
