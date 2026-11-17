"""Scientific invariants and finite-difference checks (synthetic data only)."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from build_tensors import FEATURES, eligible_date, features_from_grid, grid_for
from core import (Forecaster, bilevel_step, daily_metrics, entropy, lookahead,
                  paired_block_bootstrap, second_order_context, standardized_mse,
                  target_loss, warmup_label)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        self.rng = np.random.default_rng(2026)

    def test_mse_ic_identity_with_per_horizon_missingness(self):
        p = torch.randn(31, dtype=torch.float64, requires_grad=True)
        y = torch.randn(31, 4, dtype=torch.float64)
        y[:8, 1] = float('nan')
        y[:, 2] = float('nan')
        y[:, 3] = 3
        loss, good = standardized_mse(p, y)
        self.assertEqual(good.tolist(), [True, True, False, False])
        for h in [0, 1]:
            ok = torch.isfinite(y[:, h])
            ic = np.corrcoef(p.detach().numpy()[ok], y.numpy()[ok, h])[0, 1]
            self.assertAlmostEqual(float(loss[h]), 2 - 2 * ic, places=12)
        loss.sum().backward()
        self.assertTrue(bool(torch.isfinite(p.grad).all()))

    def test_warmup_is_mean_of_standardized_levels(self):
        y = torch.tensor([[0., 0.], [1., 100.], [2., 200.]])
        expected = torch.tensor([-np.sqrt(1.5), 0., np.sqrt(1.5)], dtype=torch.float32)
        torch.testing.assert_close(warmup_label(y), expected)

    def test_hypergradient_matches_central_finite_difference(self):
        model = nn.Linear(3, 1, bias=False).double()
        # Forward must return one scalar per stock, like the sequence model.
        class Squeeze(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear
            def forward(self, x):
                return self.linear(x).squeeze(-1)
        model = Squeeze(model)
        support = [(torch.randn(17, 3, dtype=torch.float64), torch.randn(17, 4, dtype=torch.float64))]
        query = [(torch.randn(19, 3, dtype=torch.float64), torch.randn(19, 4, dtype=torch.float64))]
        logits = torch.randn(4, dtype=torch.float64, requires_grad=True)
        def objective(value):
            ahead, _ = lookahead(model, support, value, 0.02)
            return target_loss(model, query, 3, params=ahead) - 0.001 * entropy(value)
        analytic, = torch.autograd.grad(objective(logits), logits)
        eps = 1e-5
        numeric = []
        for h in range(4):
            step = torch.zeros_like(logits)
            step[h] = eps
            numeric.append(float((objective(logits + step) - objective(logits - step)) / (2 * eps)))
        np.testing.assert_allclose(analytic.detach().numpy(), numeric, atol=1e-8, rtol=1e-5)
        self.assertGreater(float(analytic.norm()), 1e-7)

    def test_lstm_second_order_and_student_writeback(self):
        model = Forecaster(features=5, hidden_size=4).double()
        support = [(torch.randn(11, 3, 5, dtype=torch.float64), torch.randn(11, 4, dtype=torch.float64))]
        query = [(torch.randn(13, 3, 5, dtype=torch.float64), torch.randn(13, 4, dtype=torch.float64))]
        logits = nn.Parameter(torch.zeros(4, dtype=torch.float64))
        optimizer = torch.optim.Adam([logits], lr=0.01)
        with second_order_context():
            expected, _ = lookahead(model, support, logits, 0.001)
            expected = {k: v.detach().clone() for k, v in expected.items()}
        result = bilevel_step(model, support, query, logits, optimizer, 0.001, 0.001, 3)
        self.assertGreater(result['hypergradient_norm'], 0)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, expected[name])
        self.assertGreater(float(logits.detach().abs().sum()), 0)

    def test_split_boundary_and_short_day_purge(self):
        cal = {
            '2023-06-30': {'next_session': '2023-07-03', 'session_split': 'train', 'expected_minutes': 390},
            '2023-07-03': {'next_session': '2023-07-05', 'session_split': 'validation', 'expected_minutes': 210},
            '2023-07-05': {'next_session': '2023-07-06', 'session_split': 'validation', 'expected_minutes': 390},
            '2023-07-06': {'next_session': '2023-07-07', 'session_split': 'validation', 'expected_minutes': 390},
        }
        self.assertFalse(eligible_date('2023-06-30', cal)[0])
        self.assertFalse(eligible_date('2023-07-03', cal)[0])
        self.assertTrue(eligible_date('2023-07-05', cal)[0])
        cal['2023-07-03']['expected_minutes'] = 390
        self.assertEqual(eligible_date('2023-06-30', cal), (False, 'split_boundary_purge'))

    def test_features_and_missingness(self):
        grid = np.ones((4, 390, 7), dtype=np.float64)
        grid[:, :, :4] *= 100
        grid[:, :, 4] *= 10000
        grid[:, :, 5] *= 100
        grid[0, :5] = np.nan
        features = features_from_grid(grid)
        self.assertEqual(features.shape, (4, 26, 66))
        self.assertEqual(len(set(FEATURES)), 66)
        self.assertFalse(np.isinf(features).any())
        self.assertTrue(np.isfinite(features[0, 0]).all())
        self.assertTrue(np.isfinite(features[1:]).all())

    def test_grid_aligns_by_security_and_minute(self):
        import pyarrow as pa
        raw = {'permno': [20, 10, 99], 'minute_index': [389, 0, 1]}
        for field in ['open', 'high', 'low', 'close', 'amount', 'volume', 'trade_count']:
            raw[field] = [20., 10., 99.]
        grid = grid_for(pa.table(raw), np.array([10, 20]))
        self.assertEqual(grid[0, 0, 3], 10)
        self.assertEqual(grid[1, -1, 3], 20)
        self.assertTrue(np.isnan(grid[1, 0, 3]))

    def test_metric_units_ties_and_block_bootstrap(self):
        p = np.arange(20, dtype=float)
        y = p / 1000
        m = daily_metrics(p, y)
        self.assertAlmostEqual(m['ic'], 1)
        self.assertAlmostEqual(m['rank_ic'], 1)
        self.assertAlmostEqual(m['top_return'], 0.0185)
        tied = daily_metrics(np.ones(20), y)
        self.assertAlmostEqual(tied['top_return'], y.mean())
        b = paired_block_bootstrap(np.repeat(0.01, 40), draws=100, block=10)
        np.testing.assert_allclose(b['ci_95'], [0.01, 0.01])

    def test_training_scaler_ignores_validation_and_test(self):
        from data import DailyStore
        store = object.__new__(DailyStore)
        store.splits = {'train': ['a', 'b'], 'validation': ['v'], 'test': ['t']}
        store.status = {'config_sha256': 'synthetic'}
        values = {'a': np.ones((3, 26, 66)), 'b': np.full((2, 26, 66), 3),
                  'v': np.full((2, 26, 66), 1e9), 't': np.full((2, 26, 66), -1e9)}
        values['a'][0, 0, 0] = np.nan
        accessed = []
        def raw(date):
            accessed.append(date)
            return {'X': values[date]}
        store.raw = raw
        with tempfile.TemporaryDirectory() as tmp:
            store.fit_scaler(Path(tmp) / 'scaler.json')
        self.assertEqual(accessed, ['a', 'b'])
        for h in [0, 1, 65]:
            joined = np.concatenate([values['a'].reshape(-1, 66)[:, h], values['b'].reshape(-1, 66)[:, h]])
            self.assertAlmostEqual(float(store.mean[h]), np.nanmean(joined), places=6)
            self.assertAlmostEqual(float(store.scale[h]), np.nanstd(joined), places=6)

    def test_incomplete_data_refuses_financial_training(self):
        import argparse
        import json
        from train import run
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'validation.json').write_text(json.dumps({'complete': False, 'summary': {'integrity_error_count': 0}}))
            with self.assertRaisesRegex(RuntimeError, 'audit must pass'):
                run(argparse.Namespace(root=root, device='cpu', threads=2))
            self.assertFalse((root / 'reproduction/results/summary.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
