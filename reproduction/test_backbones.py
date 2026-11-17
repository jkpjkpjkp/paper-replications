"""CPU-only shape, fidelity and true mixed-hypergradient backbone checks."""
from collections import OrderedDict
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from backbones import effective_model_config, make_forecaster
from core import (Forecaster, bilevel_step, entropy, lookahead,
                  second_order_context, target_loss)
from vendor.qlib_backbones import GRUModel, TCNModel


HERE = Path(__file__).resolve().parent


def settings(name, *, features=3, patches=6):
    return {'input_features': features, 'patches': patches,
            'model': {'name': name, 'hidden_size': 4, 'layers': 1,
                      'dropout': 0., 'tcn_layers': 3, 'tcn_kernel_size': 3}}


class BackboneTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(728)

    def test_effective_tcn_defaults_cover_us_sequence(self):
        effective = effective_model_config({'model': {'name': 'TCN'}})
        self.assertEqual(effective['tcn_layers'], 3)
        self.assertEqual(effective['tcn_kernel_size'], 3)
        self.assertEqual(effective['receptive_field_patches'], 29)
        self.assertGreaterEqual(effective['receptive_field_patches'], effective['patches'])
        model = make_forecaster({'model': {'name': 'TCN'}})
        self.assertEqual(model.backbone_config, effective)
        self.assertEqual(len(model.backbone.tcn.network), 3)

    def test_output_shapes_singletons_and_variable_patch_lengths(self):
        for name in ['LSTM', 'GRU', 'TCN']:
            with self.subTest(backbone=name):
                model = make_forecaster(settings(name)).double()
                for batch, patches in [(1, 6), (7, 6), (3, 11)]:
                    out = model(torch.randn(batch, patches, 3, dtype=torch.float64))
                    self.assertEqual(tuple(out.shape), (batch,))
                    self.assertTrue(bool(torch.isfinite(out).all()))
                    self.assertEqual(out.device.type, 'cpu')

    def test_existing_lstm_initialization_and_state_are_preserved(self):
        torch.manual_seed(17)
        original = Forecaster(features=3, hidden_size=4, layers=1, dropout=0.)
        torch.manual_seed(17)
        adapted = make_forecaster(settings('LSTM'))
        self.assertIsInstance(adapted, Forecaster)
        self.assertEqual(original.state_dict().keys(), adapted.state_dict().keys())
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(5, 6, 3)
        torch.testing.assert_close(original(x), adapted(x), rtol=0, atol=0)

    def test_feature_major_input_adapter_matches_direct_sequence(self):
        x = torch.arange(5 * 6 * 3, dtype=torch.float64).reshape(5, 6, 3) / 17
        gru = make_forecaster(settings('GRU')).double()
        encoded, _ = gru.backbone.rnn(x)
        expected = gru.backbone.fc_out(encoded[:, -1]).reshape(-1)
        torch.testing.assert_close(gru(x), expected, rtol=0, atol=0)
        tcn = make_forecaster(settings('TCN')).double()
        expected = tcn.backbone.linear(tcn.backbone.tcn(x.transpose(1, 2))[:, :, -1]).reshape(-1)
        torch.testing.assert_close(tcn(x), expected, rtol=0, atol=0)

    def test_functional_call_honors_replaced_parameters(self):
        for name in ['GRU', 'TCN']:
            with self.subTest(backbone=name):
                model = make_forecaster(settings(name)).double()
                x = torch.randn(9, 6, 3, dtype=torch.float64)
                before = {k: v.detach().clone() for k, v in model.named_parameters()}
                replaced = OrderedDict((k, v * 1.07 + 0.004) for k, v in model.named_parameters())
                clone = make_forecaster(settings(name)).double()
                with torch.no_grad():
                    for key, value in clone.named_parameters():
                        value.copy_(replaced[key])
                expected = clone(x)
                actual = functional_call(model, replaced, (x,))
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                for key, value in model.named_parameters():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                self.assertFalse(torch.allclose(model(x), expected))

    def test_tcn_registration_adapter_matches_original_function_and_first_gradient(self):
        torch.manual_seed(31)
        original = TCNModel(num_input=3, output_size=1, num_channels=[4, 4, 4], kernel_size=3, dropout=0.).double()
        torch.manual_seed(31)
        adapted = make_forecaster(settings('TCN')).double()
        x = torch.randn(11, 6, 3, dtype=torch.float64)
        expected = original(x.transpose(1, 2).contiguous().flatten(1)).reshape(-1)
        actual = adapted(x)
        torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-8)
        expected.square().mean().backward()
        actual.square().mean().backward()
        adapted_params = dict(adapted.backbone.named_parameters())
        for name, value in original.named_parameters():
            modern = name.replace('.conv1.', '.net.0.').replace('.conv2.', '.net.4.')
            torch.testing.assert_close(value.grad, adapted_params[modern].grad, rtol=2e-6, atol=1e-8)

    def test_mixed_hypergradients_match_central_finite_differences(self):
        # No entropy term here: a nonzero gradient must flow THROUGH the real
        # student's inner update, including TCN's legacy weight-normalization.
        for name in ['LSTM', 'GRU', 'TCN']:
            with self.subTest(backbone=name):
                torch.manual_seed(19)
                model = make_forecaster(settings(name)).double()
                support = [(torch.randn(13, 6, 3, dtype=torch.float64),
                            torch.randn(13, 4, dtype=torch.float64))]
                query = [(torch.randn(15, 6, 3, dtype=torch.float64),
                          torch.randn(15, 4, dtype=torch.float64))]
                support[0][1][0, 1] = float('nan')
                logits = torch.tensor([.2, -.1, .1, -.2], dtype=torch.float64, requires_grad=True)
                def objective(value):
                    with second_order_context():
                        ahead, _ = lookahead(model, support, value, 0.003)
                        return target_loss(model, query, 3, params=ahead)
                analytic, = torch.autograd.grad(objective(logits), logits)
                numeric = []
                epsilon = 1e-5
                for horizon in range(4):
                    step = torch.zeros_like(logits)
                    step[horizon] = epsilon
                    numeric.append(float((objective(logits + step) - objective(logits - step)) / (2 * epsilon)))
                np.testing.assert_allclose(analytic.detach().numpy(), numeric, atol=2e-7, rtol=2e-4)
                self.assertGreater(float(analytic.norm()), 1e-7)
                self.assertTrue(bool(torch.isfinite(analytic).all()))

    def test_repeated_bilevel_steps_write_back_and_keep_gradients_finite(self):
        for name in ['GRU', 'TCN']:
            with self.subTest(backbone=name):
                torch.manual_seed(27)
                model = make_forecaster(settings(name)).double()
                support = [(torch.randn(13, 6, 3, dtype=torch.float64),
                            torch.randn(13, 4, dtype=torch.float64))]
                query = [(torch.randn(15, 6, 3, dtype=torch.float64),
                          torch.randn(15, 4, dtype=torch.float64))]
                logits = nn.Parameter(torch.zeros(4, dtype=torch.float64))
                optimizer = torch.optim.Adam([logits], lr=1e-3)
                initial = {k: v.detach().clone() for k, v in model.named_parameters()}
                for _ in range(3):
                    with second_order_context():
                        expected, _ = lookahead(model, support, logits, 1e-4)
                        expected = {k: v.detach().clone() for k, v in expected.items()}
                    result = bilevel_step(model, support, query, logits, optimizer, 1e-4, 1e-3, 3)
                    self.assertTrue(np.isfinite(list(result.values())).all())
                    self.assertGreater(result['hypergradient_norm'], 0)
                    for key, value in model.named_parameters():
                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                self.assertGreater(float(logits.detach().abs().sum()), 0)
                self.assertTrue(any(not torch.equal(value, initial[key]) for key, value in model.named_parameters()))

    def test_invalid_names_and_kernel_settings_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model name'):
            make_forecaster({'model': {'name': 'unreleased-paper-backbone'}})
        with self.assertRaisesRegex(ValueError, 'tcn_kernel_size'):
            make_forecaster({'model': {'name': 'TCN', 'tcn_kernel_size': 1}})
        with self.assertRaisesRegex(ValueError, 'Expected N x T'):
            make_forecaster(settings('GRU'))(torch.randn(5, 6, 2))

    def test_vendored_file_checksums_and_mit_notices(self):
        folder = HERE / 'vendor/qlib_backbones'
        manifest = json.loads((folder / 'PROVENANCE.json').read_text())
        self.assertEqual(manifest['version'], '0.9.7')
        self.assertIn('not original Song', manifest['scope'])
        for record in manifest['files']:
            actual = hashlib.sha256((folder / record['vendored_file']).read_bytes()).hexdigest()
            self.assertEqual(actual, record['vendored_file_sha256'])
        for name in ['LICENSE_MICROSOFT', 'LICENSE_CMU_LOCUS_LAB']:
            text = (folder / name).read_text()
            self.assertIn('Permission is hereby granted', text)
            self.assertIn('SOFTWARE IS PROVIDED "AS IS"', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
