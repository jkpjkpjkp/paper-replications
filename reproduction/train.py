"""Run the released training loop with paper Chinese dates and explicit local integration fixes."""
import argparse
import difflib
import importlib.metadata
import json
import os
import random
import subprocess
import time
import types
from pathlib import Path

from common import ROOT, UPSTREAM, UPSTREAM_COMMIT, LocalStockData, initialize, metrics, fresh_output, sha256
from integration import ForwardReturn, LocalAlphaPoolGFN
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--update-freq', type=int, default=64)
    parser.add_argument('--log-freq', type=int, default=500)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-end', choices=['2020-11-30', '2020-12-31'], default='2020-11-30',
                        help='Purged default; nominal paper end includes labels reaching January 2021')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = fresh_output(args.output)
    torch.set_num_threads(args.threads)
    initialize()
    source = (UPSTREAM / 'train_gfn.py').read_text()
    patched = source.replace('from src.', 'from ')
    # The source hard-codes US dates even for CSI300. Use the paper's CN train/validation split.
    patched = patched.replace("start_time='2010-01-01', end_time='2016-12-31'", f"start_time='2010-01-01', end_time='{args.train_end}'")
    patched = patched.replace("start_time='2018-01-01', end_time='2020-12-31'", "start_time='2021-01-01', end_time='2021-12-31'")
    patched = patched.replace("torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')", "torch.device('cpu')")
    patched = patched.replace('target = Ref(close, -20) / close - 1', 'target = ForwardReturn(20)')
    patched = patched.replace('        weight_scheduler.step()', '        weight_scheduler.step()\n        logger.on_episode(episode, loss, optimizer, loss_fn, backbone, pb_head)')
    (out / 'training-integration.patch').write_text(''.join(difflib.unified_diff(source.splitlines(True), patched.splitlines(True), fromfile='upstream/train_gfn.py', tofile='runtime/train_gfn.py')))
    runtime = out / 'runtime'
    runtime.mkdir(exist_ok=True)
    (runtime / 'train_gfn.py').write_text(patched)
    module = types.ModuleType('portable_train_gfn')
    exec(compile(patched, str(runtime / 'train_gfn.py'), 'exec'), module.__dict__)
    module.StockData = LocalStockData
    module.ForwardReturn = ForwardReturn
    module.AlphaPoolGFN = LocalAlphaPoolGFN
    started = time.monotonic()
    config = dict(seed=args.seed, instrument='csi300', pool_capacity=args.capacity,
                  encoder_type='gnn', entropy_coef=0.01, entropy_temperature=1.,
                  mask_dropout_prob=1., ssl_weight=1., nov_weight=0.3,
                  n_episodes=args.episodes, update_freq=args.update_freq,
                  log_freq=args.log_freq, weight_decay_type='linear', final_weight_ratio=0.)
    (out / 'protocol.json').write_text(json.dumps(dict(training=config, device='cpu',
        upstream_commit=UPSTREAM_COMMIT,
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [UPSTREAM/'train_gfn.py', ROOT/'reproduction/common.py', ROOT/'reproduction/integration.py', Path(__file__)]},
        nominal_train=['2010-01-01', '2020-12-31'], actual_train=['2010-01-01', args.train_end],
        validation=['2021-01-01', '2021-12-31'], test=['2022-01-01', '2024-12-31'],
        purge=('Last month of training excluded so 20-day training labels precede validation' if args.train_end == '2020-11-30' else 'Nominal paper dates; final training labels reach January 2021'),
        checkpoint_selection='fixed final episode, validation diagnostic only',
        packages={name: importlib.metadata.version(name) for name in ['torch','numpy','pandas','pyqlib','torchgfn','torch-geometric']}
    ), indent=2))
    class Logger:
        def __init__(self, model, pool, log_dir, test_data, target):
            self.model, self.pool, self.valid, self.target = model, pool, test_data, target
            self.writer = types.SimpleNamespace(add_scalar=lambda *a, **kw: None)
            self.episode = -1
            self.finite_losses = self.nonfinite_losses = 0
            self.history = []
            (out/'data-provenance.json').write_text(json.dumps(dict(training=pool.data.provenance, validation=test_data.provenance), indent=2))
        def on_episode(self, episode, loss, optimizer, loss_fn, backbone, pb_head):
            self.episode = episode
            finite = loss is not None and bool(torch.isfinite(loss))
            self.finite_losses += int(finite)
            self.nonfinite_losses += int(not finite)
            if (episode + 1) % args.update_freq == 0:
                status = dict(status='running', completed_episodes=episode+1, episodes=args.episodes,
                              pool_size=self.pool.size, finite_losses=self.finite_losses,
                              nonfinite_losses=self.nonfinite_losses, wall_seconds=time.monotonic()-started)
                (out / 'progress.json').write_text(json.dumps(status, indent=2))
        def log_metrics(self, episode):
            if self.pool.size:
                with torch.no_grad():
                    pred = sum(self.valid.normalize(self.pool.exprs[i].evaluate(self.valid))*self.pool.weights[i] for i in range(self.pool.size))
                    result = metrics(pred, self.target.evaluate(self.valid))
                self.history.append(dict(episode=episode+1, **result))
                print(json.dumps(dict(event='validation', episode=episode+1, pool_size=self.pool.size, **result)), flush=True)
        def save_checkpoint(self, episode):
            torch.save(self.model.state_dict(), out / 'model.pt')
            (out / 'pool.json').write_text(json.dumps(self.pool.to_dict(), indent=2))
        def show_pool_state(self):
            pass
        def close(self):
            self.save_checkpoint(self.episode)
            self.log_metrics(self.episode)
            result = dict(status='complete', completed_episodes=self.episode+1,
                          pool_size=self.pool.size, capacity=args.capacity, finite_losses=self.finite_losses,
                          nonfinite_losses=self.nonfinite_losses, validation_history=self.history,
                          train_days=self.pool.data.n_days, train_stocks=self.pool.data.n_stocks,
                          wall_seconds=time.monotonic()-started)
            (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            (out / 'progress.json').write_text(json.dumps(result, indent=2, allow_nan=False))
            print(json.dumps(result, indent=2), flush=True)
    module.GFNLogger = Logger
    module.train(argparse.Namespace(**config))


if __name__ == '__main__':
    main()
