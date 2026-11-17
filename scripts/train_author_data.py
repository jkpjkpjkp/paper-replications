"""Run the released LiMT model/training steps on the released 2008-start samplers.

The wrapper adds input checks, epoch checkpoints and progress records. Model,
sampler, loss, day ordering, optimizer and gradient clipping come from upstream.
Test inference is a separate command, used only after validation selection.
"""
from __future__ import annotations
import argparse
import ast
import gc
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def upstream_modules():
    sys.path.insert(0, str(ROOT / 'upstream'))
    base = importlib.import_module('base_model')
    utils = importlib.import_module('utils')
    # Only this unused import is missing from the release. Class bodies are
    # executed unchanged, including the original attention/dropout operations.
    tree = ast.parse((ROOT / 'upstream/limt.py').read_text())
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and n.module == 'Embed')]
    scope = {'__name__': 'released_limt'}
    exec(compile(tree, str(ROOT / 'upstream/limt.py'), 'exec'), scope)
    return base, utils, scope


class AuthorUnpickler(pickle.Unpickler):
    allowed = {
        ('pandas.core.frame', 'DataFrame'),
        ('pandas.core.internals.managers', 'BlockManager'),
        ('pandas._libs.internals', '_unpickle_block'),
        ('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
        ('numpy', 'dtype'), ('builtins', 'slice'),
        ('pandas.core.indexes.base', '_new_Index'),
        ('pandas.core.indexes.base', 'Index'),
        ('pandas.core.indexes.multi', 'MultiIndex'),
        ('pandas.core.indexes.datetimes', '_new_DatetimeIndex'),
        ('pandas.core.indexes.datetimes', 'DatetimeIndex'),
        ('pandas.core.arrays.datetimes', 'DatetimeArray'),
        ('pandas._libs.arrays', '__pyx_unpickle_NDArrayBacked'),
        ('pandas.core.indexes.numeric', 'Int64Index'),
        ('numpy.core.multiarray', 'scalar'),
    }

    def find_class(self, module, name):
        if (module, name) == ('__main__', 'TSDataSampler'):
            return importlib.import_module('utils').TSDataSampler
        if (module, name) not in self.allowed:
            raise pickle.UnpicklingError(f'Unexpected global: {module}.{name}')
        return getattr(importlib.import_module(module), name)


def load_split(config, split):
    import numpy as np
    inventory = json.loads(Path(config['inventory']).read_text())
    suffix = f"R_Vol2_Vola2_{config['universe']}_Alpha158/dl_{split}.pkl"
    rec, = [r for r in inventory['files'] if r['path'].endswith(suffix)]
    path = Path(rec['path'])
    if path.stat().st_size != rec['bytes'] or sha(path) != rec['sha256']:
        raise ValueError(f'Author input changed: {path}')
    with path.open('rb') as f:
        data = AuthorUnpickler(f).load()
    index = data.get_index()
    dates = index.get_level_values('datetime')
    actual = (str(dates.min().date()), str(dates.max().date()))
    if actual != (rec['sample_start'], rec['sample_end']):
        raise ValueError(f'Split dates changed: {split}, {actual}')
    assert data.step_len == 21 and data.data.shape[1] == 161
    assert data.fillna_type == 'ffill+zero'
    assert index.is_monotonic_increasing and index.is_unique
    labels = data.data.iloc[:, -3:]
    assert list(labels.columns) == ['LABEL0', 'LABEL1', 'LABEL2']
    assert np.isfinite(labels.to_numpy()).all(), 'Unexpected nonfinite labels'
    summary = dict(path=str(path), sha256=rec['sha256'], rows=len(index),
                   dates=dates.nunique(), first=actual[0], last=actual[1],
                   fillna_type=data.fillna_type,
                   label_daily_mean_abs_max=labels.groupby(level='datetime').mean().abs().max().tolist(),
                   label_daily_std_median=labels.groupby(level='datetime').std().median().tolist())
    return data, summary


def make_runner(config):
    import torch
    base, utils, scope = upstream_modules()
    runner = base.SequenceModel(
        n_epochs=config['epochs'], lr=config['lr'], GPU=0, seed=config['seed'],
        mode='r_vol2_vola2', drop_extreme=False, loss_type='mse_r_vol2_vola2',
        weight1=config['weights'][0], weight2=config['weights'][1],
        weight3=config['weights'][2], stop_standard='IC')
    # Match UnifiedModel.__init__, which reseeds immediately before the model.
    utils.seed_everything(config['seed'])
    runner.model = scope['LiMT_v2_MMoE_RVol2Vola2'](
        d_feat=158, d_model=config['hidden'], t_nhead=config['temporal_heads'],
        s_nhead=config['spatial_heads'],
        T_dropout_rate=config['dropout'], S_dropout_rate=config['dropout'],
        gate_input_start_index=158, gate_input_end_index=221,
        beta=5, n_experts=config['experts'], return_gate_weight=0)
    runner.init_model()
    torch.set_num_threads(config['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return runner


def save_checkpoint(path, obj):
    import torch
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


class ProgressLoader:
    def __init__(self, loader, output, epoch, limit=0):
        self.loader, self.output, self.epoch, self.limit = loader, output, epoch, limit

    def __iter__(self):
        import numpy as np
        started = time.perf_counter()
        total = len(self.loader.sampler.daily_count)
        for i, batch in enumerate(self.loader):
            if self.limit and i >= self.limit:
                break
            if i % 25 == 0:
                write_json(self.output / 'progress.json', dict(
                    phase='training', epoch=self.epoch, completed_days=i,
                    total_days=min(total, self.limit) if self.limit else total,
                    seconds=time.perf_counter()-started, updated_utc=now(), pid=os.getpid()))
            if not np.isfinite(batch.numpy()).all():
                raise FloatingPointError('Released sampler produced nonfinite values')
            yield batch


def choose_checkpoint(best, valid_ic):
    import math
    if not math.isfinite(valid_ic):
        raise FloatingPointError('Nonfinite validation IC')
    return valid_ic > best


def train(config, output, smoke_days=0):
    import numpy as np
    import torch
    runner = make_runner(config)
    data, train_info = load_split(config, 'train')
    valid, valid_info = load_split(config, 'valid')
    write_json(output / 'data_summary.json', {'train': train_info, 'valid': valid_info})
    if smoke_days:
        # A separately named software check; never a candidate for selection.
        counts = valid.get_index().get_level_values('datetime').value_counts().sort_index()
        valid.end_idx = valid.start_idx + int(counts.iloc[:smoke_days].sum())
    loader = runner._init_data_loader(data, shuffle=True, drop_last=True, seed=config['seed'])
    history, best, stale, best_epoch, start = [], -float('inf'), 0, None, 0
    latest = output / 'latest.pt'
    if latest.exists():
        ck = torch.load(latest, map_location=runner.device)
        if ck['config'] != config:
            raise ValueError('Checkpoint config changed')
        runner.model.load_state_dict(ck['model'])
        runner.train_optimizer.load_state_dict(ck['optimizer'])
        history, best, stale, best_epoch, start = [ck[k] for k in
            ['history', 'best', 'stale', 'best_epoch', 'epoch']]
        restore_rng(ck['rng'])
    for epoch in range(start, 1 if smoke_days else config['epochs']):
        if stale > 20:
            break
        started = time.perf_counter()
        loss, train_metric = runner.train_epoch(
            ProgressLoader(loader, output, epoch + 1, smoke_days), None,
            mode=runner.mode, drop_extreme=False, loss_type=runner.loss_type)
        if not np.isfinite(loss) or any(not torch.isfinite(p).all() for p in runner.model.parameters()):
            raise FloatingPointError('Nonfinite training loss or parameters')
        write_json(output / 'progress.json', dict(phase='validation', epoch=epoch+1,
                   updated_utc=now(), pid=os.getpid()))
        valid_loss, metric = runner.predict(valid, None, mode=runner.mode, loss_type=runner.loss_type)
        score = float(metric['IC'])
        if choose_checkpoint(best, score):
            best, stale, best_epoch = score, 0, epoch + 1
            save_checkpoint(output / 'best.pt', runner.model.state_dict())
        else:
            stale += 1
        row = dict(epoch=epoch+1, train_loss=loss, train=train_metric,
                   validation_loss=valid_loss, validation=metric, best_epoch=best_epoch,
                   best_validation_ic=best, stale=stale, seconds=time.perf_counter()-started,
                   updated_utc=now())
        history.append(row)
        save_checkpoint(latest, dict(config=config, model=runner.model.state_dict(),
            optimizer=runner.train_optimizer.state_dict(), history=history, best=best,
            stale=stale, best_epoch=best_epoch, epoch=epoch+1, rng=rng_state()))
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', dict(phase='training', **row))
        print(json.dumps(row), flush=True)
        gc.collect()
    write_json(output / 'status.json', dict(
        phase='smoke_complete' if smoke_days else 'trained', epochs=len(history),
        best_epoch=best_epoch, best_validation_ic=best, test_evaluated=False,
        stop_reason='early_stopping' if stale > 20 else 'epoch_limit', updated_utc=now()))


def evaluate_selected(config, output):
    import numpy as np
    import pandas as pd
    import torch
    status = json.loads((output / 'status.json').read_text())
    if status['phase'] not in ('trained', 'evaluated'):
        raise ValueError('Only a completed full training run may be evaluated')
    runner = make_runner(config)
    runner.model.load_state_dict(torch.load(output / 'best.pt', map_location=runner.device))
    result = {}
    for split in ('valid', 'test'):
        data, summary = load_split(config, split)
        preds, metric = runner.predict(data, None, mode=runner.mode,
                                       loss_type=runner.loss_type, return_type='predictions')
        frame = pd.DataFrame(index=data.get_index())
        for name, batches in preds.items():
            frame['pred_' + name] = np.concatenate([x.detach().cpu().numpy() for x in batches])
        for j, name in enumerate(('return', 'vol_hit', 'volatility')):
            frame['label_' + name] = data.data.loc[frame.index].iloc[:, -3+j].to_numpy()
        assert np.isfinite(frame.to_numpy()).all() and frame.index.is_unique
        daily = frame.groupby(level='datetime').apply(lambda f:
            f.pred_return.corr(f.label_return))
        independent = dict(IC=float(daily.mean()), ICIR=float(daily.mean()/daily.std(ddof=0)))
        for key in independent:
            np.testing.assert_allclose(independent[key], metric[key], atol=1e-8, rtol=1e-6)
        frame.to_parquet(output / f'{split}_predictions.parquet')
        result[split] = dict(metrics=metric, independent=independent, data=summary)
        del data, preds, frame
        gc.collect()
    write_json(output / 'metrics.json', result)
    write_json(output / 'status.json', dict(status, phase='evaluated', test_evaluated=True,
                                          updated_utc=now()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--smoke-days', type=int, default=0)
    parser.add_argument('--evaluate-selected', action='store_true')
    args = parser.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    import pandas as pd
    import numpy as np
    if torch.__version__.split('+')[0] != '1.11.0' or pd.__version__ != '1.5.3':
        raise RuntimeError('Use the separate author environment: torch 1.11.0 / pandas 1.5.3')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable in the author environment')
    config = json.loads(args.config.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.smoke_days and 'smoke' not in output.parts:
        raise ValueError('Smoke output must live under a separate smoke/ directory')
    fingerprint = dict(config=config, device=args.device, smoke_days=args.smoke_days,
        torch=torch.__version__, pandas=pd.__version__, numpy=np.__version__,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
            [Path(__file__), ROOT/'upstream/base_model.py', ROOT/'upstream/utils.py', ROOT/'upstream/limt.py']})
    record = output / 'provenance.json'
    if record.exists() and json.loads(record.read_text()) != fingerprint:
        raise ValueError('Refusing to mix changed source/runtime/configuration with an existing run')
    write_json(record, fingerprint)
    write_json(output / 'config.json', config)
    try:
        if args.evaluate_selected:
            evaluate_selected(config, output)
        else:
            train(config, output, args.smoke_days)
    except Exception as exc:
        write_json(output / 'error.json', dict(type=type(exc).__name__, message=str(exc), updated_utc=now()))
        raise


if __name__ == '__main__':
    main()
