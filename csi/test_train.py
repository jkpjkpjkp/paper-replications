"""Synthetic numerical invariants for the CSI daily training adaptation.

No source CSI data or held-out financial observations are accessed here.
"""
from __future__ import annotations

from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.func import functional_call

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def load_us_core():
    spec = importlib.util.spec_from_file_location(
        'checked_us_core', HERE.parent / 'reproduction/core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_us_core()


class CSITrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import train
        cls.training = train
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(1949)
        self.dtype = torch.float64

    def fixture(self):
        # Interleaving rows catches accidental contiguous-day assumptions.
        ids = torch.tensor([4] * 9 + [0] * 13 + [8] * 7, dtype=torch.long)
        pred = torch.randn(len(ids), dtype=self.dtype, requires_grad=True)
        labels = torch.randn(len(ids), 10, dtype=self.dtype)
        labels[1:4, 2] = float('nan')
        labels[ids == 4, 3] = float('nan')
        labels[ids == 8, 4] = 7
        labels[:, 8] = float('nan')
        labels[:, 9] = 1
        permutation = torch.randperm(len(ids))
        return pred[permutation], labels[permutation], ids[permutation]

    def loop_loss(self, pred, labels, ids):
        values, flags = [], []
        for date in torch.unique(ids):
            loss, available = core.standardized_mse(pred[ids == date], labels[ids == date])
            values.append(loss)
            flags.append(available)
        flags = torch.stack(flags)
        counts = flags.sum(0)
        return torch.stack(values).sum(0) / counts.clamp_min(1), counts > 0

    def test_vector_loss_matches_date_loop_on_each_horizon_mask(self):
        pred, labels, ids = self.fixture()
        observed, usable = self.training.daily_vector_loss(pred, labels, ids)
        expected, flags = self.loop_loss(pred, labels, ids)
        torch.testing.assert_close(observed, expected, rtol=1e-10, atol=1e-11)
        self.assertEqual(usable.tolist(), flags.tolist())
        self.assertFalse(bool(usable[-2:].any()))

    def test_loss_normalizes_dates_separately(self):
        pred, labels, ids = self.fixture()
        base, flags = self.training.daily_vector_loss(pred, labels, ids)
        scales = {0: 0.3, 4: 20.0, 8: 4.0}
        shifts = {0: 50.0, 4: -100.0, 8: 2.0}
        p = pred.clone()
        y = labels.clone()
        for date in torch.unique(ids).tolist():
            mask = ids == date
            p[mask] = p[mask] * scales[date] + shifts[date]
            # Each horizon may also have its own positive scale and offset.
            y[mask] = y[mask] * (scales[date] * torch.arange(1, 11, dtype=self.dtype)) + shifts[date]
        shifted, shifted_flags = self.training.daily_vector_loss(p, y, ids)
        torch.testing.assert_close(shifted, base, rtol=1e-9, atol=1e-10)
        self.assertEqual(flags.tolist(), shifted_flags.tolist())

    def test_masked_loss_has_finite_prediction_gradients(self):
        pred, labels, ids = self.fixture()
        pred.retain_grad()
        losses, usable = self.training.daily_vector_loss(pred, labels, ids)
        losses[usable].mean().backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()))
        self.assertGreater(float(pred.grad.abs().sum()), 0)

    def test_warmup_matches_mean_of_standardized_label_levels(self):
        pred, labels, ids = self.fixture()
        expected = []
        for date in torch.unique(ids):
            mask = ids == date
            target = core.warmup_label(labels[mask])
            loss, usable = core.standardized_mse(pred[mask], target)
            self.assertTrue(bool(usable[0]))
            expected.append(loss[0])
        observed = self.training.warmup_loss(pred, labels, ids)
        torch.testing.assert_close(observed, torch.stack(expected).mean(), rtol=1e-10, atol=1e-11)

    def test_logits_hypergradient_matches_finite_difference(self):
        model = self.training.MLP(in_dim=3, hidden=(6, 4), dropout=0.0).double()
        xs = torch.randn(26, 3, dtype=self.dtype)
        xq = torch.randn(29, 3, dtype=self.dtype)
        ys = torch.randn(26, 10, dtype=self.dtype)
        yq = torch.randn(29, 10, dtype=self.dtype)
        ys[:2, 3] = float('nan')
        ids = torch.tensor([0] * 11 + [1] * 15)
        idq = torch.tensor([0] * 16 + [1] * 13)
        logits = torch.randn(10, dtype=self.dtype, requires_grad=True)

        def objective(values):
            params = OrderedDict(model.named_parameters())
            losses, usable = self.training.daily_vector_loss(model(xs), ys, ids)
            weights = values.softmax(0) * usable
            inner = (losses * weights).sum() / weights.sum()
            gradients = torch.autograd.grad(inner, tuple(params.values()), create_graph=True)
            fast = OrderedDict((name, param - 0.01 * gradient)
                               for (name, param), gradient in zip(params.items(), gradients))
            query, flags = self.training.daily_vector_loss(functional_call(model, fast, (xq,)), yq, idq)
            self.assertTrue(bool(flags[-1]))
            return query[-1] - 0.001 * core.entropy(values)

        analytic, = torch.autograd.grad(objective(logits), logits)
        eps = 1e-5
        numerical = []
        for h in range(10):
            delta = torch.zeros_like(logits)
            delta[h] = eps
            numerical.append(float((objective(logits + delta) - objective(logits - delta)) / (2 * eps)))
        np.testing.assert_allclose(analytic.detach().numpy(), numerical, rtol=2e-5, atol=1e-8)
        self.assertGreater(float(analytic.norm()), 1e-6)

    def test_public_demo_model_dimensions(self):
        model = self.training.MLP()
        linears = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
        self.assertEqual([(m.in_features, m.out_features) for m in linears], [(158, 256), (256, 128), (128, 1)])
        self.assertEqual(model(torch.randn(17, 158)).shape, (17,))

    def test_checkpoint_checksum_and_signature_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            value = {'signature': 'synthetic-v1', 'seed': 0,
                     'horizon': 10, 'state_dict': {'weight': torch.arange(3)}}
            self.training.checkpoint_save(path, value)
            loaded = self.training.checkpoint_load(path, 'synthetic-v1')
            torch.testing.assert_close(loaded['state_dict']['weight'], value['state_dict']['weight'])
            with self.assertRaisesRegex(RuntimeError, 'signature'):
                self.training.checkpoint_load(path, 'synthetic-v2')
            with path.open('ab') as stream:
                stream.write(b'corrupted-synthetic-fixture')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                self.training.checkpoint_load(path, 'synthetic-v1')

    def test_fit_stage_never_accesses_held_out_test_data(self):
        accesses, calls = [], []
        class GuardDataset:
            splits = {'train': ['train1', 'train2'], 'validation': ['valid1'],
                      'test': ['held-out1', 'held-out2']}
            def days(self, dates):
                if any(date.startswith('held-out') for date in dates):
                    raise AssertionError('Fit stage read held-out data')
                accesses.extend(dates)
        dataset = GuardDataset()
        config = {'seeds': [0, 1], 'candidate_horizons': list(range(1, 11))}
        def fit(data, settings, seed, horizon, path, signature, progress):
            data.days(data.splits['train'])
            data.days(data.splits['validation'])
            calls.append(('supervised', seed, horizon))
            return {'horizon': horizon, 'seed': seed, 'signature': signature}
        def choose(data, settings, seed, path, signature, progress):
            data.days(data.splits['train'])
            data.days(data.splits['validation'])
            calls.append(('select', seed, None))
            return {'selected_horizon': 3, 'seed': seed}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(self.training, 'supervised_fit', fit), \
                 patch.object(self.training, 'select_horizon', choose), \
                 patch.object(self.training, 'evaluate', side_effect=AssertionError('Premature test evaluation')):
                fitted, selected, retrained = self.training.fit_all(
                    dataset, config, Path(directory), 'synthetic', lambda _: None)
        self.assertEqual(set(fitted), {(h, seed) for h in range(1, 11) for seed in [0, 1]})
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(set(retrained), {0, 1})
        self.assertEqual(len([c for c in calls if c[0] == 'select']), 2)
        self.assertTrue(accesses)


if __name__ == '__main__':
    unittest.main(verbosity=2)
