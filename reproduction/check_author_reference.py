#!/usr/bin/env python3
"""CPU-only numerical checks against the unmodified EFundAI reference code.

These checks use synthetic inputs and produce no financial performance result.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import numpy as np
import torch
from torch.func import functional_call

HERE = Path(__file__).resolve().parent
AUTHOR = (HERE.parents[1] / 'song26l-us/author-demo').resolve()
sys.path.insert(0, str(AUTHOR))
from src.model import AlphaMLP, daily_ic, weighted_daily_ic
from src.blo_trainer import BLOTrainer


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    assert not torch.cuda.is_available(), 'Reference check must not access the busy GPU'
    checks = {}
    p = torch.randn(96, dtype=torch.float64)
    y = torch.randn(96, 10, dtype=torch.float64)
    days = torch.arange(4).repeat_interleave(24)
    actual = float(daily_ic(p, y[:, -1], days))
    expected = np.mean([np.corrcoef(p[days == d], y[days == d, -1])[0, 1] for d in range(4)])
    assert abs(actual - expected) < 1e-7
    checks['official_daily_ic_against_numpy'] = {'absolute_error': abs(actual - expected)}

    model = AlphaMLP(158, dropout=0).double()
    xi, xo = torch.randn(48, 158, dtype=torch.float64), torch.randn(48, 158, dtype=torch.float64)
    yi, yo = torch.randn(48, 10, dtype=torch.float64), torch.randn(48, dtype=torch.float64)
    ids = torch.arange(2).repeat_interleave(24)
    logits = torch.linspace(-0.2, 0.2, 10, dtype=torch.float64, requires_grad=True)

    def lookahead(z):
        weights = z.softmax(0)
        params = dict(model.named_parameters())
        inner = 1 - weighted_daily_ic(model(xi), yi, ids, weights)
        grads = torch.autograd.grad(inner, tuple(params.values()), create_graph=True)
        updated = {key: value - 0.05 * grad for (key, value), grad in zip(params.items(), grads)}
        entropy = -(weights * weights.log()).sum()
        return 1 - daily_ic(functional_call(model, updated, (xo,)), yo, ids) - 1e-4 * entropy

    analytic = torch.autograd.grad(lookahead(logits), logits)[0]
    direction = torch.randn(10, dtype=torch.float64)
    direction /= direction.norm()
    epsilon = 1e-5
    numeric = float((lookahead(logits + epsilon * direction) - lookahead(logits - epsilon * direction)) / (2 * epsilon))
    directional = float(analytic @ direction)
    assert np.isclose(directional, numeric, rtol=2e-5, atol=1e-8), (directional, numeric)
    checks['official_loss_hypergradient_finite_difference'] = {
        'analytic_directional_derivative': directional, 'numeric_directional_derivative': numeric,
        'absolute_error': abs(directional - numeric)}

    # Exercise the actual public warmup, chronological sampling, Adam teacher,
    # differentiable SGD with momentum, and student writeback without replacing
    # the source loss or trainer implementation.
    x = torch.randn(32 * 24, 158)
    labels = torch.randn(32 * 24, 10)
    dates = torch.arange(32).repeat_interleave(24)
    trainer = BLOTrainer(158, device='cpu', batch_days=10, gap_days=10)
    trainer.warmup(x, labels.mean(1), dates, n_epochs=2)
    before = {key: value.detach().clone() for key, value in trainer.student.state_dict().items()}
    with contextlib.redirect_stdout(io.StringIO()):
        history = trainer.train(x, labels, dates, target_idx=9, n_iters=2)
    weights = trainer.get_lambda()
    assert len(history) == 2 and np.isfinite(weights).all()
    assert np.isclose(weights.sum(), 1) and np.max(np.abs(weights - 0.1)) > 1e-6
    assert any(not torch.equal(before[key], value) for key, value in trainer.student.state_dict().items())
    checks['unmodified_official_blo_cpu_path'] = {
        'updates': len(history), 'teacher_updated': True, 'student_writeback': True,
        'warmup_input': 'synthetic scalar fixture; not a claim about the full diff-z data pipeline'}

    result = {'status': 'passed', 'evidence_type': 'synthetic engineering check; no financial test results',
              'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'device': 'cpu',
              'repository': 'https://github.com/EFundAI/label-horizon-paradox',
              'commit': subprocess.check_output(['git', '-C', str(AUTHOR), 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {name: sha(AUTHOR / name) for name in
                                ['src/model.py', 'src/blo_trainer.py', 'scripts/run_search.py', 'scripts/run_blo.py']},
              'checks': checks}
    (HERE / 'author_reference_checks.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
