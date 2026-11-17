#!/usr/bin/env python3
"""Five-seed US LSTM replication of Appendix E.3, Table 6.

Test metrics are only computed after all baseline, horizon-selection and
selected-horizon fits finish. Incomplete downloads cannot train this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import torch

from core import (Forecaster, bilevel_step, daily_metrics, paired_block_bootstrap,
                  summarize_daily, target_loss, supervised_step)
from data import DailyStore, save_json, sha

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Native CUDA reduction ordering can still differ across GPU/tool versions.
    torch.use_deterministic_algorithms(True, warn_only=True)


def chunks(dates, size, *, full=False):
    return [dates[i:i + size] for i in range(0, len(dates), size)
            if not full or len(dates[i:i + size]) == size]


def new_model(config, seed, device):
    seed_all(seed)
    from backbones import make_forecaster
    return make_forecaster(config).to(device=device, dtype=torch.float64)


def atomic_checkpoint(path, content):
    temp = path.with_suffix('.pt.tmp')
    torch.save(content, temp)
    temp.replace(path)
    save_json(path.with_suffix('.receipt.json'), {'sha256': sha(path), 'bytes': path.stat().st_size,
                                                'signature': content['signature']})


def existing_checkpoint(path, signature, horizon=None):
    receipt = path.with_suffix('.receipt.json')
    if not path.exists() or not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    if saved['sha256'] != sha(path) or saved['bytes'] != path.stat().st_size:
        raise RuntimeError('Checkpoint checksum mismatch: ' + str(path))
    result = torch.load(path, map_location='cpu')
    if saved['signature'] != signature or result['signature'] != signature:
        raise RuntimeError('Checkpoint does not match the frozen run: ' + str(path))
    if horizon is not None and result['horizon'] != horizon:
        raise RuntimeError('Checkpoint training horizon does not match')
    return result


@torch.no_grad()
def validation_loss(model, store, horizon=390):
    model.eval()
    values = [float(target_loss(model, [store.day(d)], horizon - 1))
              for d in store.splits['validation']]
    if not values or not np.isfinite(values).all():
        raise ValueError('Validation loss is empty or nonfinite')
    return float(np.mean(values))


def model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def resume_path(path):
    return path.with_name(path.stem + '.resume.pt')


def fit_epochs(model, optimizer, rng, config, store, path, signature, progress,
               epoch_step, *, seed, horizon=390, logits=None, warmup_history=None):
    """Shared patience-five loop with exact epoch-boundary recovery."""
    resumed = existing_checkpoint(resume_path(path), signature)
    history, best, wait, best_state, best_logits = [], float('inf'), 0, None, None
    if resumed:
        model.load_state_dict(resumed['current_model'])
        if logits is not None:
            with torch.no_grad(): logits.copy_(resumed['current_logits'].to(logits))
        optimizer.load_state_dict(resumed['optimizer'])
        rng.bit_generator.state = resumed['numpy_rng']
        torch.set_rng_state(resumed['torch_rng'])
        if resumed.get('cuda_rng') is not None and store.device == 'cuda':
            torch.cuda.set_rng_state_all(resumed['cuda_rng'])
        history, best, wait = resumed['history'], resumed['best'], resumed['wait']
        best_state, best_logits = resumed['best_state'], resumed['best_logits']
        warmup_history = resumed['warmup_history']
    started = time.monotonic()
    limit = config.get('maximum_epochs')  # None in the primary protocol.
    while wait < config['patience'] and (limit is None or len(history) < limit):
        model.train()
        diagnostic = epoch_step()
        val = validation_loss(model, store, horizon)
        row = {'epoch': len(history) + 1, 'validation_loss': val, **diagnostic}
        if logits is not None:
            weights = logits.detach().softmax(0)
            row.update(selected_horizon=int(logits.argmax()) + 1,
                       weights=weights.cpu().tolist(),
                       maximum_weight_over_uniform=float(weights.max() * len(weights)))
        history.append(row)
        if val < best:
            best, wait, best_state = val, 0, model_state(model)
            best_logits = logits.detach().cpu().clone() if logits is not None else None
        else:
            wait += 1
        atomic_checkpoint(resume_path(path), {
            'signature': signature, 'current_model': model_state(model),
            'current_logits': logits.detach().cpu() if logits is not None else None,
            'optimizer': optimizer.state_dict(), 'numpy_rng': rng.bit_generator.state,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if store.device == 'cuda' else None,
            'history': history, 'best': best, 'wait': wait, 'best_state': best_state,
            'best_logits': best_logits, 'warmup_history': warmup_history})
        visible = {k: v for k, v in row.items() if k != 'weights'}
        progress({'phase': 'bilevel' if logits is not None else 'supervised',
                  'seed': seed, 'horizon': horizon, **visible})
        print(json.dumps(visible | {'seed': seed, 'horizon': horizon}), flush=True)
    result = {'signature': signature, 'seed': seed, 'state_dict': best_state,
              'best_validation_loss': best, 'validation_horizon': horizon,
              'history': history, 'elapsed_seconds_this_process': time.monotonic() - started,
              'stopping_reason': 'patience' if wait >= config['patience'] else 'explicit_epoch_cap'}
    if logits is None:
        result['horizon'] = horizon
    else:
        result.update(selected_horizon=int(best_logits.argmax()) + 1,
                      weights=best_logits.softmax(0).tolist(), warmup_history=warmup_history)
    atomic_checkpoint(path, result)
    return result


def adamw(model, config):
    return torch.optim.AdamW(model.parameters(), lr=config['adamw_learning_rate'],
                             weight_decay=config['weight_decay'])


def train_supervised(store, config, seed, horizon, path, signature, progress):
    completed = existing_checkpoint(path, signature, horizon)
    if completed is not None:
        return completed
    model = new_model(config, seed, store.device)
    optimizer = adamw(model, config)
    rng = np.random.default_rng(seed + 1000)
    windows = chunks(store.splits['train'], config['batch_days'], full=True)
    if not windows:
        raise ValueError('Training requires a full twenty-date batch')
    def epoch():
        losses = [supervised_step(model, store.days(windows[i]), optimizer, horizon - 1)
                  for i in rng.permutation(len(windows))]
        return {'train_loss': float(np.mean(losses))}
    return fit_epochs(model, optimizer, rng, config, store, path, signature, progress,
                      epoch, seed=seed, horizon=horizon)


def select_horizon(store, config, seed, path, signature, progress):
    completed = existing_checkpoint(path, signature)
    if completed is not None:
        return completed
    model = new_model(config, seed, store.device)
    rng = np.random.default_rng(seed + 2000)
    windows = chunks(store.splits['train'], config['batch_days'], full=True)
    if not windows:
        raise ValueError('BLO requires a full twenty-date batch')
    warm_history = []
    if existing_checkpoint(resume_path(path), signature) is None:
        optimizer = adamw(model, config)
        for epoch in range(config['warmup_epochs']):
            model.train()
            losses = [supervised_step(model, store.days(windows[i]), optimizer, 389, warmup=True)
                      for i in rng.permutation(len(windows))]
            warm_history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses))})
            progress({'phase': 'warmup', 'seed': seed, 'epoch': epoch + 1})
    logits = torch.nn.Parameter(torch.zeros(config['horizons'], device=store.device, dtype=torch.float64))
    optimizer = torch.optim.SGD([logits], lr=config['outer_learning_rate'])
    def epoch():
        diagnostics = []
        for i in rng.permutation(len(windows)):
            dates = windows[i]
            order = rng.permutation(len(dates))
            support = [dates[j] for j in order[:10]]
            query = [dates[j] for j in order[10:]]
            diagnostics.append(bilevel_step(model, store.days(support), store.days(query), logits,
                optimizer, config['inner_learning_rate'], config['entropy_weight'], 389))
        return {'hypergradient_norm': float(np.mean([d['hypergradient_norm'] for d in diagnostics]))}
    return fit_epochs(model, optimizer, rng, config, store, path, signature, progress,
                      epoch, seed=seed, logits=logits, warmup_history=warm_history)


@torch.no_grad()
def evaluate(model, store, method, seed, folder):
    model.eval()
    daily, predictions = [], []
    for date in store.splits['test']:
        x, _ = store.day(date)
        pred = model(x).cpu().numpy()
        raw = store.raw(date)
        target = raw['Y'][:, -1]
        daily.append({'date': date, 'method': method, 'seed': seed} | daily_metrics(pred, target))
        predictions.append(pd.DataFrame({'date': date, 'label_date': str(raw['label_date']),
                                         'permno': raw['permno'], 'score': pred, 'target_return': target}))
    pd.concat(predictions, ignore_index=True).to_parquet(folder / f'{method}-seed{seed}.parquet', index=False)
    return daily


def run(args):
    root = args.root.resolve()
    name = getattr(args, 'backbone', None) or 'LSTM'
    output = root / 'reproduction/results'
    if name != 'LSTM':
        output = output / 'backbones' / name.lower()
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_locked(args, root, output)


def run_locked(args, root, output):
    audit = json.loads((root / 'validation.json').read_text())
    audited = audit.get('summary', {})
    if (not audit.get('complete') or audited.get('integrity_error_count') != 0
            or audited.get('expected_sessions') != 1655 or audited.get('verified_sessions') != 1655
            or audited.get('pending_sessions') != 0):
        raise RuntimeError('Full 1,655-session independent raw-data audit must pass before training')
    config = json.loads((root / 'reproduction/protocol.json').read_text())
    name = getattr(args, 'backbone', None) or config['model']['name']
    config['model']['name'] = name
    config['model']['input_features'] = config['input_features']
    if name == 'TCN':
        config['model'].update(tcn_layers=3, tcn_kernel_size=3)
    from backbones import effective_model_config
    config['resolved_model'] = effective_model_config(config)
    if config['seeds'] != [0, 1, 2, 3, 4]:
        raise ValueError('The primary experiment requires all five prespecified seeds')
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(args.threads)
    store = DailyStore(root, device, max_cached_days=getattr(args, 'cached_days', 1400))
    if any(not dates for dates in store.splits.values()):
        raise RuntimeError('All three chronological splits must be present')
    source_files = ['core.py', 'data.py', 'train.py', 'build_tensors.py', 'protocol.json', 'backbones.py', 'paper_reference.json']
    source_files.extend(str(p.relative_to(root / 'reproduction'))
                        for p in sorted((root / 'reproduction/vendor/qlib_backbones').glob('*')) if p.is_file())
    source = {f: sha(root / 'reproduction' / f) for f in source_files}
    manifest = {'source': source, 'tensor_config_sha256': store.status['config_sha256'],
                'effective_model': config['resolved_model'],
                'tensor_receipts': {d: r['sha256'] for d, r in sorted(store.receipts.items())}}
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    frozen = output / 'manifest.json'
    if frozen.exists() and json.loads(frozen.read_text())['signature'] != signature:
        raise RuntimeError('Run sources/data changed. Archive results before starting a new protocol.')
    save_json(frozen, {'signature': signature, 'inputs': manifest, 'config': config,
                      'split_days': {k: len(v) for k, v in store.splits.items()},
                      'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
                      'numpy': np.__version__, 'cuda': torch.version.cuda,
                      'raw_audit_sha256': sha(root / 'validation.json'),
                      'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                      'cached_days': store.max_cached_days, 'dtype': 'float64', 'cudnn_enabled': False,
                      'pid': os.getpid()})
    def progress(value):
        save_json(output / 'status.json', value | {'updated_utc': now(), 'signature': signature})
    progress({'phase': 'preprocessing'})
    store.fit_scaler(output / 'feature_scaler.json')
    models = []
    for seed in config['seeds']:
        baseline = train_supervised(store, config, seed, 390,
            output / f'standard-seed{seed}.pt', signature, progress)
        selection = select_horizon(store, config, seed,
            output / f'selection-seed{seed}.pt', signature, progress)
        selected = train_supervised(store, config, seed, selection['selected_horizon'],
            output / f'selected-seed{seed}.pt', signature, progress)
        models.append((seed, baseline, selection, selected))
    # All training/selection is finished before the first held-out evaluation.
    progress({'phase': 'test_evaluation', 'fits_complete': True})
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    rows, summaries = [], []
    for seed, baseline, selection, selected in models:
        for method, fit in [('standard', baseline), ('selected', selected)]:
            model = new_model(config, seed, device)
            model.load_state_dict(fit['state_dict'])
            daily = evaluate(model, store, method, seed, prediction_dir)
            rows.extend(daily)
            summaries.append({'method': method, 'seed': seed, 'horizon': fit['horizon']} | summarize_daily(daily))
    daily_frame = pd.DataFrame(rows)
    daily_frame.to_csv(output / 'daily_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'per_seed_metrics.csv', index=False)
    metric_names = ['ic', 'icir', 'rank_ic', 'rank_icir', 'top_return', 'sharpe']
    averages = {}
    for method in ['standard', 'selected']:
        subset = [s for s in summaries if s['method'] == method]
        averages[method] = {k: {'mean': float(np.mean([s[k] for s in subset])),
                                'seed_std': float(np.std([s[k] for s in subset], ddof=1))}
                            for k in metric_names}
    paired = daily_frame.groupby(['date', 'method'])['ic'].mean().unstack('method')
    bootstrap = paired_block_bootstrap(paired['selected'] - paired['standard'])
    references = json.loads((root / 'reproduction/paper_reference.json').read_text())
    result = {'complete': True, 'signature': signature, 'completed_utc': now(), 'backbone': name,
              'scope': f'Independent S&P 500 Scenario 1 {name} replication; substitute disclosed features',
              'selected_horizons': {str(s): fit['selected_horizon'] for s, _, fit, _ in models},
              'five_seed_metrics': averages, 'paired_ic_bootstrap': bootstrap,
              'paper_table6_reference': references[name],
              'metric_conventions': 'sample daily std (ddof=1); raw IC units; decimal returns; gross price returns; Sharpe sqrt(252)',
              'limitations': ['Original complete feature formulas and architecture hyperparameters unavailable',
                             'Only this backbone / US Scenario 1; remaining backbones and A-share scenarios require separate results',
                             'Signal-quality top-decile returns omit execution costs and are not an executable strategy',
                             'Bootstrap conditions on fitted seeds; overlapping market exposures may persist beyond 10 days']}
    save_json(output / 'summary.json', result)
    progress({'phase': 'complete', 'complete': True})
    from report import write_report
    write_report(root)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--backbone', choices=['LSTM', 'GRU', 'TCN'], default='LSTM')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--cached-days', type=int, default=1400,
                        help='CPU tensor cache; 1400 keeps train/validation in about 6 GiB RAM')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        output = args.root / 'reproduction/results'
        if args.backbone != 'LSTM':
            output = output / 'backbones' / args.backbone.lower()
        save_json(output / 'status.json',
                  {'phase': 'failed', 'error': str(error), 'error_type': type(error).__name__,
                   'updated_utc': now()})
        raise
