"""Meaningful small checks for specialist extraction and the added fee rule."""
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from limt.model import LiMT
from portfolio import backtest as original_backtest
from experiments.separate_three_task_portfolio.portfolio_engine import backtest
from experiments.separate_three_task_portfolio.train_volume import VolumeOnlyModel


def fixture():
    rng = np.random.default_rng(33)
    n, length = 20, 10
    market = np.ones((n, length, 7)) * 100
    market[:, :, 1] = rng.uniform(1e9, 3e9, size=(n, length))
    market[:, :, 4] = rng.uniform(101, 120, size=(n, length))
    market[:, :, 5] = rng.uniform(80, 99, size=(n, length))
    panel = SimpleNamespace(symbols=np.array([f"P{i}" for i in range(n)]),
        dates=np.array(pd.bdate_range("2020-01-01", periods=length).strftime("%Y-%m-%d")),
        member=np.ones((n, length), bool), market=market,
        total_return=np.zeros((n, length)), price_return=np.zeros((n, length)),
        terminal=np.zeros((n, length), bool), execution_tradable=np.ones((n, length), bool))
    parts = [pd.DataFrame(dict(date=panel.dates[t], realized_date=panel.dates[t + 2],
        symbol=panel.symbols, pred_return=np.arange(n), pred_volume=rng.normal(size=n),
        pred_volatility=rng.normal(size=n))) for t in [5, 6, 7]]
    return panel, pd.concat(parts, ignore_index=True)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    original = LiMT(d_model=32, experts=2, dropout=0.)
    torch.manual_seed(19)
    volume = VolumeOnlyModel(d_model=32, experts=2, dropout=0.)
    x = torch.randn(7, 5, 158)
    torch.testing.assert_close(volume(x)[:, 0], original(x)[:, 1], rtol=1e-6, atol=1e-7)
    assert volume.tasks == ("volume",)
    assert not any("return" in name or "volatility" in name or "fusion_gate" in name or "feature_gate" in name
                   for name, _ in volume.named_parameters())
    volume(x).square().mean().backward()
    assert volume.feature_layer[0].weight.grad.abs().sum() > 0
    for strategy in ["equal", "apo"]:
        for cost in [0., 5.]:
            panel, predictions = fixture()
            old, old_weights = original_backtest(panel, predictions, strategy=strategy, buy_bp=cost, save_weights=True)
            new, new_weights = backtest(panel, predictions, strategy=strategy, buy_bp=cost, save_weights=True)
            pd.testing.assert_frame_equal(new[old.columns], old, rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(new_weights, old_weights)
    panel, predictions = fixture()
    predictions = predictions[predictions.date == panel.dates[5]]
    all_trades, _ = backtest(panel, predictions, buy_bp=10, sell_bp=10)
    np.testing.assert_allclose(all_trades.fees, 4e6, atol=1.)
    np.testing.assert_allclose(all_trades.return_net, -.002)
    assert all_trades.liquidation_buy.iloc[0] > 0 and all_trades.liquidation_sell.iloc[0] > 0
    print("PASS: volume branch matches the released head, true single-task parameters/gradients,")
    print("APO/equal legacy ledger parity, and 10bp fees on every opening and final liquidation trade.")


if __name__ == "__main__":
    main()
