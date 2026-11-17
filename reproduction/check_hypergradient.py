#!/usr/bin/env python3
"""Independent, training-only finite differences at a frozen real checkpoint.

Does not alter the primary run, choose a hyperparameter, or evaluate test data.
The finite-difference objective independently implements standardized MSE and
the single SGD step; it does not call core.lookahead or core.target_loss.
"""
from collections import OrderedDict
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from core import entropy, lookahead, second_order_context, target_loss
from data import DailyStore, save_json, sha
from train import new_model

HERE = Path(__file__).resolve().parent


def snapshot_checkpoint(path, destination):
    """Read a consistent receipt/content pair while training publishes epochs."""
    receipt_path = path.with_suffix('.receipt.json')
    for _ in range(5):
        before = receipt_path.read_bytes()
        content = path.read_bytes()
        after = receipt_path.read_bytes()
        receipt = json.loads(before)
        if (before == after and len(content) == receipt['bytes']
                and hashlib.sha256(content).hexdigest() == receipt['sha256']):
            destination.write_bytes(content)
            destination.with_suffix('.receipt.json').write_bytes(before)
            saved = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
            if saved['signature'] != receipt['signature']:
                raise ValueError('Checkpoint signature disagrees with receipt')
            return saved, receipt
    raise RuntimeError('Checkpoint changed during snapshot; retry later')


def independent_mse(pred, labels):
    if labels.ndim == 1:
        labels = labels[:, None]
    p = pred - pred.mean()
    y = labels - labels.mean(dim=0)
    p = p / p.square().mean().clamp_min(1e-12).sqrt()
    y = y / y.square().mean(dim=0).clamp_min(1e-12).sqrt()
    return (p[:, None] - y).square().mean(dim=0)


def independent_objective(model, support, query, logits, config):
    params = OrderedDict(model.named_parameters())
    weights = logits.detach().softmax(0)
    # Accumulate first derivatives date by date, using no higher-order graph.
    accumulated = [torch.zeros_like(p) for p in params.values()]
    for x, y in support:
        loss = (independent_mse(model(x), y) * weights).sum()
        gradients = torch.autograd.grad(loss, tuple(params.values()))
        for total, gradient in zip(accumulated, gradients):
            total.add_(gradient)
    ahead = {name: (p - config['inner_learning_rate'] * gradient).detach()
             for (name, p), gradient in zip(params.items(), accumulated)}
    with torch.no_grad():
        query_loss = sum(float(independent_mse(functional_call(model, ahead, (x,)), y[:, -1])[0])
                         for x, y in query)
        regularizer = config['entropy_weight'] * float((weights * weights.log()).sum())
    return query_loss + regularizer


def main(args):
    torch.set_num_threads(2)
    folder = args.output.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    # Reuse the snapshot on repeat invocation, making the diagnostic reproducible.
    frozen = folder / 'checkpoint.pt'
    checkpoint, receipt = snapshot_checkpoint(frozen if frozen.exists() else args.checkpoint, frozen)
    manifest = json.loads((HERE / 'results/manifest.json').read_text())
    if receipt['signature'] != manifest['signature']:
        raise ValueError('Snapshot is from another run')
    for filename, digest in manifest['inputs']['source'].items():
        if sha(HERE / filename) != digest:
            raise ValueError('Primary source changed: ' + filename)
    config = manifest['config']
    store = DailyStore(HERE.parent, 'cpu', max_cached_days=20)
    scaler = json.loads((HERE / 'results/feature_scaler.json').read_text())
    if (scaler['fit_dates'] != store.splits['train']
            or scaler['config_sha256'] != store.status['config_sha256']):
        raise ValueError('Scaler provenance does not match training dates')
    store.mean, store.scale = np.array(scaler['mean']), np.array(scaler['scale'])
    dates = store.splits['train'][:20]
    order = np.random.default_rng(20261111).permutation(20)
    support_dates = [dates[i] for i in order[:10]]
    query_dates = [dates[i] for i in order[10:]]
    support, query = store.days(support_dates), store.days(query_dates)
    if not all(torch.isfinite(x).all() and torch.isfinite(y).all() for x, y in support + query):
        raise ValueError('Finite, common cross sections required')
    model = new_model(config, 0, 'cpu')
    model.load_state_dict(checkpoint['current_model'])
    model.eval()
    logits = checkpoint['current_logits'].detach().clone().requires_grad_()
    with second_order_context():
        ahead, inner = lookahead(model, support, logits, config['inner_learning_rate'])
        outer_task = target_loss(model, query, config['horizons'] - 1, params=ahead)
        task_gradient, = torch.autograd.grad(outer_task, logits)
        regularization = -config['entropy_weight'] * entropy(logits)
        regularizer_gradient, = torch.autograd.grad(regularization, logits)
        gradient = (task_gradient + regularizer_gradient).detach()
        analytic_objective = float(outer_task.detach() + regularization.detach())
        inner_value = float(inner.detach())
        del ahead, inner, outer_task, regularization
        gc.collect()
        reference = independent_objective(model, support, query, logits, config)
        np.testing.assert_allclose(reference, analytic_objective, rtol=0, atol=1e-11)
        g = torch.Generator().manual_seed(20261111)
        random_direction = torch.randn(len(logits), generator=g, dtype=torch.double)
        random_direction -= random_direction.mean()
        directions = {'gradient': gradient / gradient.norm(),
                      'random': random_direction / random_direction.norm()}
        checks = []
        for name, direction in directions.items():
            expected = float(gradient @ direction)
            for epsilon in [0.01, 0.003]:
                plus = independent_objective(model, support, query, logits.detach() + epsilon * direction, config)
                minus = independent_objective(model, support, query, logits.detach() - epsilon * direction, config)
                observed = (plus - minus) / (2 * epsilon)
                np.testing.assert_allclose(observed, expected, rtol=0.002, atol=2e-11)
                checks.append({'direction': name, 'epsilon': epsilon, 'autograd': expected,
                               'finite_difference': observed, 'absolute_error': abs(observed - expected)})
    # Same gradient, different optimizer: numerical sensitivity, not model fits.
    steps = {}
    for name, kind in [('SGD', torch.optim.SGD), ('AdamW', torch.optim.AdamW)]:
        value = torch.nn.Parameter(logits.detach().clone())
        kwargs = {'lr': config['outer_learning_rate']}
        if name == 'AdamW':
            kwargs.update(weight_decay=0.0, eps=1e-8)
        optimizer = kind([value], **kwargs)
        value.grad = gradient.clone()
        optimizer.step()
        weights = value.detach().softmax(0)
        steps[name] = {'logit_delta_norm': float((value.detach() - logits.detach()).norm()),
                       'weight_delta_norm': float((weights - logits.detach().softmax(0)).norm()),
                       'maximum_weight_over_uniform': float(weights.max() * len(weights)),
                       'assumptions': kwargs, 'optimizer_state': 'fresh, single diagnostic step'}
    weights = logits.detach().softmax(0)
    values, indices = weights.sort(descending=True)
    result = {'status': 'passed', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Independent hypergradient check on 20 training dates; no validation/test metrics or fitted primary writes',
              'signature': receipt['signature'], 'checkpoint_sha256': receipt['sha256'],
              'checkpoint_epoch': len(checkpoint['history']), 'protocol_version': config['protocol_version'],
              'support_dates': support_dates, 'query_dates': query_dates,
              'tensor_receipts': {d: store.receipts[d]['sha256'] for d in dates},
              'script_sha256': sha(Path(__file__)),
              'scaler_sha256': sha(HERE / 'results/feature_scaler.json'),
              'torch_version': torch.__version__, 'dtype': 'float64', 'threads': 2,
              'inner_loss': inner_value, 'outer_loss': analytic_objective,
              'independent_outer_loss': reference,
              'task_hypergradient_norm': float(task_gradient.norm()),
              'entropy_gradient_norm': float(regularizer_gradient.norm()),
              'checks': checks, 'optimizer_sensitivity': steps,
              'current_weights': {'argmax_horizon': int(indices[0]) + 1,
                                  'runner_up_horizon': int(indices[1]) + 1,
                                  'top_two_gap': float(values[0] - values[1]),
                                  'maximum_weight_over_uniform': float(weights.max() * len(weights))},
              'interpretation': 'Passing checks supports the implemented SGD hypergradient; it does not identify the authors optimizer or establish feature equivalence or financial replication.'}
    save_json(folder / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=HERE / 'results/selection-seed0.resume.pt')
    p.add_argument('--output', type=Path, default=HERE / 'diagnostics/hypergradient-20261111')
    main(p.parse_args())
