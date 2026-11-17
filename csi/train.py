#!/usr/bin/env python3
"""Five-seed CSI daily adaptation: ten horizons, BLO and selected-horizon retrain."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.func import functional_call

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / 'reproduction'))
from core import corr, daily_metrics, entropy, paired_block_bootstrap


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class MLP(nn.Module):
    def __init__(self, in_dim=158, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout)])
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def grouped(values, day_ids, n_dates):
    out = values.new_zeros((n_dates, values.shape[1]))
    return out.scatter_add(0, day_ids[:, None].expand_as(values), values)


def label_standardization(labels, day_ids):
    if labels.ndim == 1:
        labels = labels[:, None]
    dates = int(day_ids.max()) + 1
    good = torch.isfinite(labels)
    safe = torch.where(good, labels, 0.0)
    count = grouped(good.to(labels.dtype), day_ids, dates)
    mean = grouped(safe, day_ids, dates) / count.clamp_min(1)
    dy = torch.where(good, safe - mean[day_ids], 0.0)
    variance = grouped(dy.square(), day_ids, dates) / count.clamp_min(1)
    usable = (count >= 3) & (variance > 1e-12)
    z = dy / variance.clamp_min(1e-12).sqrt()[day_ids]
    return z, good, count, usable


def daily_vector_loss(pred, labels, day_ids):
    """Equal-date standardized MSE vector; equivalent to the checked day loop."""
    if labels.ndim == 1:
        labels = labels[:, None]
    # Keep nonfinite predictions visible as an error rather than silently
    # deleting difficult stocks from a fitted model's loss.
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError('Nonfinite model predictions')
    zy, good, count, usable = label_standardization(labels, day_ids)
    dates = count.shape[0]
    safe_p = torch.where(good, pred[:, None], 0.0)
    pm = grouped(safe_p, day_ids, dates) / count.clamp_min(1)
    dp = torch.where(good, pred[:, None] - pm[day_ids], 0.0)
    variance = grouped(dp.square(), day_ids, dates) / count.clamp_min(1)
    zp = dp / variance.clamp_min(1e-12).sqrt()[day_ids]
    losses = grouped((zp - zy).square() * good, day_ids, dates) / count.clamp_min(1)
    n = usable.sum(0)
    loss = torch.where(usable, losses, 0.0).sum(0) / n.clamp_min(1)
    return loss, n > 0


def warmup_loss(pred, labels, day_ids):
    z, good, _, usable = label_standardization(labels, day_ids)
    mask = good & usable[day_ids]
    average = torch.where(mask, z, 0.0).sum(1) / mask.sum(1).clamp_min(1)
    average = torch.where(mask.any(1), average, torch.full_like(average, float('nan')))
    loss, valid = daily_vector_loss(pred, average, day_ids)
    if not bool(valid[0]):
        raise ValueError('No usable warmup labels')
    return loss[0]


def objective(model, batch, horizon=None, weights=None, params=None, warmup=False):
    x, y, ids = batch
    pred = model(x) if params is None else functional_call(model, params, (x,))
    if warmup:
        return warmup_loss(pred, y, ids)
    if horizon is not None:
        y = y[:, horizon - 1:horizon]
    losses, usable = daily_vector_loss(pred, y, ids)
    if not bool(usable.any()):
        raise ValueError('No usable date labels')
    if weights is None:
        return losses[0]
    effective = weights * usable
    return (losses * effective).sum() / effective.sum().clamp_min(1e-12)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def new_model(config, seed, device):
    seed_all(seed)
    return MLP(hidden=tuple(config['model']['hidden']), dropout=config['model']['dropout']).to(device)


def batches(dates, size, full=False):
    return [dates[i:i + size] for i in range(0, len(dates), size)
            if not full or len(dates[i:i + size]) == size]


def checkpoint_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.pt.tmp')
    torch.save(value, temp)
    temp.replace(path)
    save_json(path.with_suffix('.receipt.json'), {'sha256': sha(path), 'bytes': path.stat().st_size,
                                                'signature': value['signature']})


def checkpoint_load(path, signature):
    path = Path(path)
    receipt = path.with_suffix('.receipt.json')
    if not path.exists() or not receipt.exists():
        return None
    saved = json.loads(receipt.read_text())
    if saved['sha256'] != sha(path) or saved['bytes'] != path.stat().st_size:
        raise RuntimeError('Checkpoint checksum mismatch')
    if saved['signature'] != signature:
        raise RuntimeError('Checkpoint signature mismatch')
    value = torch.load(path, map_location='cpu')
    if value['signature'] != signature:
        raise RuntimeError('Checkpoint content signature mismatch')
    return value


@torch.no_grad()
def validation_loss(model, dataset, config):
    model.eval()
    total, count = 0.0, 0
    for dates in batches(dataset.splits['validation'], config['batch_days']):
        loss = float(objective(model, dataset.days(dates), horizon=config['target_horizon']))
        if not np.isfinite(loss):
            raise FloatingPointError('Nonfinite validation loss')
        total += len(dates) * loss
        count += len(dates)
    if count == 0:
        raise ValueError('Empty validation split')
    return total / count


def supervised_fit(dataset, config, seed, horizon, path, signature, progress):
    previous = checkpoint_load(path, signature)
    if previous is not None:
        if previous['horizon'] != horizon or previous['seed'] != seed:
            raise RuntimeError('Checkpoint horizon/seed mismatch')
        return previous
    model = new_model(config, seed, dataset.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['adamw_learning_rate'],
                                  weight_decay=config['weight_decay'])
    windows = batches(dataset.splits['train'], config['batch_days'])
    rng = np.random.default_rng(seed + 1000)
    history, best, wait, best_state = [], float('inf'), 0, None
    started = time.monotonic()
    for epoch in range(config['maximum_epochs']):
        model.train()
        losses = []
        for index in rng.permutation(len(windows)):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, dataset.days(windows[index]), horizon=horizon)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val = validation_loss(model, dataset, config)
        record = {'epoch': epoch + 1, 'train_loss': float(np.mean(losses)), 'validation_final_loss': val}
        history.append(record)
        progress({'phase': 'supervised', 'seed': seed, 'horizon': horizon, **record})
        if val < best:
            best, wait = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config['patience']:
                break
    value = {'signature': signature, 'seed': seed, 'horizon': horizon, 'state_dict': best_state,
             'best_validation_final_loss': best, 'history': history, 'seconds': time.monotonic() - started}
    checkpoint_save(path, value)
    return value


def select_horizon(dataset, config, seed, path, signature, progress):
    previous = checkpoint_load(path, signature)
    if previous is not None:
        if previous['seed'] != seed:
            raise RuntimeError('Checkpoint selection seed mismatch')
        return previous
    model = new_model(config, seed, dataset.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['adamw_learning_rate'],
                                  weight_decay=config['weight_decay'])
    rng = np.random.default_rng(seed + 2000)
    windows = batches(dataset.splits['train'], config['batch_days'])
    for epoch in range(config['warmup_epochs']):
        model.train()
        for index in rng.permutation(len(windows)):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model, dataset.days(windows[index]), warmup=True)
            loss.backward()
            optimizer.step()
        progress({'phase': 'warmup', 'seed': seed, 'epoch': epoch + 1})
    logits = nn.Parameter(torch.zeros(len(config['candidate_horizons']), device=dataset.device))
    outer_optimizer = torch.optim.Adam([logits], lr=config['outer_learning_rate'])
    windows = batches(dataset.splits['train'], config['batch_days'], full=True)
    if not windows:
        raise ValueError('BLO needs at least20 training dates')
    history, best, wait, best_logits, best_state = [], float('inf'), 0, None, None
    for epoch in range(config['maximum_epochs']):
        model.train()
        norms = []
        for index in rng.permutation(len(windows)):
            dates = windows[index]
            order = rng.permutation(len(dates))
            middle = len(dates) // 2
            support = dataset.days([dates[j] for j in order[:middle]])
            query = dataset.days([dates[j] for j in order[middle:]])
            params = dict(model.named_parameters())
            inner = objective(model, support, weights=logits.softmax(0), params=params)
            grads = torch.autograd.grad(inner, tuple(params.values()), create_graph=True)
            ahead = {name: value - config['inner_learning_rate'] * grad
                     for (name, value), grad in zip(params.items(), grads)}
            outer = objective(model, query, horizon=config['target_horizon'], params=ahead)
            outer = outer - config['entropy_weight'] * entropy(logits)
            gradient, = torch.autograd.grad(outer, logits)
            if not bool(torch.isfinite(gradient).all()):
                raise FloatingPointError('Nonfinite horizon hypergradient')
            outer_optimizer.zero_grad(set_to_none=True)
            logits.grad = gradient
            outer_optimizer.step()
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(ahead[name].detach())
            norms.append(float(gradient.norm()))
        val = validation_loss(model, dataset, config)
        weights = logits.detach().softmax(0).cpu()
        history.append({'epoch': epoch + 1, 'validation_final_loss': val,
                        'selected_horizon': int(weights.argmax() + 1),
                        'weights': weights.tolist(), 'hypergradient_norm': float(np.mean(norms))})
        progress({'phase': 'bilevel', 'seed': seed, **{k: v for k, v in history[-1].items() if k != 'weights'}})
        if val < best:
            best, wait = val, 0
            best_logits = logits.detach().cpu().clone()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config['patience']:
                break
    value = {'signature': signature, 'seed': seed, 'selected_horizon': int(best_logits.argmax() + 1),
             'weights': best_logits.softmax(0).tolist(), 'state_dict': best_state,
             'best_validation_final_loss': best, 'history': history}
    checkpoint_save(path, value)
    return value


class Dataset:
    def __init__(self, folder, device, config):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        if not self.manifest.get('complete'):
            raise RuntimeError('Full CSI data manifest must complete before financial training')
        self.records = {r['date']: r for r in self.manifest['records']}
        self.splits = {s: sorted(d for d, r in self.records.items() if r['split'] == s)
                       for s in ['train', 'validation', 'test']}
        if any(not dates for dates in self.splits.values()):
            raise RuntimeError('Missing a chronological CSI split')
        self.device, self.config = device, config
        self.items = {}
        self.mean = self.scale = None

    def load(self, date):
        if date in self.items:
            return self.items[date]
        record = self.records[date]
        path = self.folder / record['file']
        if sha(path) != record['sha256'] or path.stat().st_size != record['bytes']:
            raise RuntimeError('CSI tensor checksum mismatch: ' + date)
        with np.load(path, allow_pickle=False) as z:
            item = {key: z[key] for key in z.files}
        if item['X'].shape != (len(item['Y']), 158) or item['Y'].shape[1] != 10:
            raise ValueError('Unexpected CSI tensor shape')
        if not np.isfinite(item['Y'][:, -1]).all():
            raise ValueError('Final label must define the common observable-stock set')
        if self.mean is not None:
            item['X'] = self.transform(item['X'])
        self.items[date] = item
        return item

    def transform(self, x):
        return np.clip((np.where(np.isfinite(x), x, self.mean) - self.mean) / self.scale,
                       -10, 10).astype(np.float32)

    def fit_scaler(self, path):
        n, mean, m2 = np.zeros(158, np.int64), np.zeros(158), np.zeros(158)
        for date in self.splits['train']:
            x = self.load(date)['X'].astype(np.float64)
            good = np.isfinite(x)
            count = good.sum(0)
            local = np.where(good, x, 0).sum(0) / np.maximum(count, 1)
            delta = np.where(good, x - local, 0)
            local_m2 = (delta * delta).sum(0)
            total = n + count
            shift = local - mean
            m2 += local_m2 + shift * shift * n * count / np.maximum(total, 1)
            mean += shift * count / np.maximum(total, 1)
            n = total
        self.mean = mean.astype(np.float32)
        scale = np.sqrt(m2 / np.maximum(n, 1))
        self.scale = np.where(scale > 1e-8, scale, 1).astype(np.float32)
        for item in self.items.values():
            item['X'] = self.transform(item['X'])
        save_json(path, {'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                         'finite_count': n.tolist(), 'fit_dates': self.splits['train']})

    def days(self, dates):
        if self.mean is None:
            raise RuntimeError('Training-only scaler must be fitted first')
        items = [self.load(d) for d in dates]
        x = np.concatenate([item['X'] for item in items])
        y = np.concatenate([item['Y'] for item in items])
        ids = np.concatenate([np.full(len(item['Y']), i, np.int64) for i, item in enumerate(items)])
        return (torch.as_tensor(x, device=self.device), torch.as_tensor(y, device=self.device),
                torch.as_tensor(ids, device=self.device))


def summarize(rows):
    result = {}
    for field in ['ic', 'rank_ic', 'top_return_10sessions']:
        values = np.asarray([r[field] for r in rows], float)
        result[field] = float(np.nanmean(values))
        if field != 'top_return_10sessions':
            result['icir' if field == 'ic' else 'rank_icir'] = float(np.nanmean(values) / np.nanstd(values, ddof=1))
    return result


@torch.no_grad()
def evaluate(dataset, config, fit, method, seed, prediction_dir):
    model = new_model(config, seed, dataset.device)
    model.load_state_dict(fit['state_dict'])
    model.eval()
    rows, predictions = [], []
    for date in dataset.splits['test']:
        x, y, _ = dataset.days([date])
        pred = model(x).cpu().numpy()
        raw = dataset.load(date)
        metric = daily_metrics(pred, y[:, -1].cpu().numpy())
        rows.append({'date': date, 'method': method, 'seed': seed,
                     'ic': metric['ic'], 'rank_ic': metric['rank_ic'],
                     'top_return_10sessions': metric['top_return'], 'stocks': metric['stocks']})
        instrument = raw.get('instrument', raw.get('instruments'))
        if instrument is None:
            raise ValueError('CSI tensors must preserve security identifiers')
        predictions.append(pd.DataFrame({'date': date, 'instrument': instrument,
                                         'score': pred, 'target_return_10sessions': raw['Y'][:, -1]}))
    pd.concat(predictions, ignore_index=True).to_parquet(prediction_dir / f'{method}-seed{seed}.parquet', index=False)
    return rows


def fit_all(dataset, config, folder, signature, progress):
    """No test reads or evaluation in this stage, including the full sweep."""
    fitted, selections, selected = {}, {}, {}
    # Fit the primary target first, then BLO and selected-horizon retraining.
    # The remaining sweep is still completed before the first test evaluation.
    for seed in config['seeds']:
        path = folder / f'h10-seed{seed}.pt'
        fitted[(10, seed)] = supervised_fit(dataset, config, seed, 10, path, signature, progress)
        selections[seed] = select_horizon(dataset, config, seed, folder / f'blo-seed{seed}.pt', signature, progress)
        horizon = selections[seed]['selected_horizon']
        selected[seed] = supervised_fit(dataset, config, seed, horizon,
            folder / f'selected-seed{seed}.pt', signature, progress)
    for horizon in config['candidate_horizons']:
        for seed in config['seeds']:
            if (horizon, seed) not in fitted:
                # Same seed/recipe yields the same fit as selected retraining.
                # Reuse it when possible, avoiding duplicate optimization.
                if selected[seed]['horizon'] == horizon:
                    fitted[(horizon, seed)] = selected[seed]
                else:
                    fitted[(horizon, seed)] = supervised_fit(dataset, config, seed, horizon,
                        folder / f'h{horizon}-seed{seed}.pt', signature, progress)
    return fitted, selections, selected


def run(args):
    config = json.loads((HERE / 'protocol.json').read_text())
    if config['seeds'] != [0, 1, 2, 3, 4]:
        raise ValueError('Full CSI run requires all five seeds')
    folder = HERE / 'results' / args.universe
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.train.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = args.device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.set_num_threads(args.threads)
        dataset = Dataset(HERE / 'cache' / args.universe, device, config)
        input_manifest = {'source': {f: sha(HERE / f) for f in ['train.py', 'data.py', 'protocol.json']},
                          'core_sha256': sha(HERE.parent / 'reproduction/core.py'),
                          'tensor_receipts': {d: r['sha256'] for d, r in sorted(dataset.records.items())}}
        signature = hashlib.sha256(json.dumps(input_manifest, sort_keys=True).encode()).hexdigest()
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text())['signature'] != signature:
            raise RuntimeError('Code/data/protocol changed; archive prior results before a new run')
        save_json(manifest_path, {'signature': signature, 'inputs': input_manifest, 'config': config,
                                 'universe': args.universe, 'python': sys.version, 'torch': torch.__version__,
                                 'device': device, 'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None})
        def progress(value):
            save_json(folder / 'status.json', value | {'signature': signature, 'updated_utc': now()})
            print(json.dumps(value | {'universe': args.universe}), flush=True)
        progress({'phase': 'preprocessing'})
        dataset.fit_scaler(folder / 'scaler.json')
        fitted, selections, selected = fit_all(dataset, config, folder, signature, progress)
        progress({'phase': 'test_evaluation', 'all_fits_complete': True})
        prediction_dir = folder / 'predictions'
        prediction_dir.mkdir(exist_ok=True)
        daily, per_seed = [], []
        for (horizon, seed), fit in sorted(fitted.items()):
            method = f'h{horizon}'
            rows = evaluate(dataset, config, fit, method, seed, prediction_dir)
            daily.extend(rows)
            per_seed.append({'method': method, 'horizon': horizon, 'seed': seed, **summarize(rows)})
        for seed, fit in selected.items():
            rows = evaluate(dataset, config, fit, 'selected', seed, prediction_dir)
            daily.extend(rows)
            per_seed.append({'method': 'selected', 'horizon': fit['horizon'], 'seed': seed, **summarize(rows)})
        frame, seeds = pd.DataFrame(daily), pd.DataFrame(per_seed)
        frame.to_csv(folder / 'daily_metrics.csv', index=False)
        seeds.to_csv(folder / 'per_seed_metrics.csv', index=False)
        metrics = ['ic', 'icir', 'rank_ic', 'rank_icir', 'top_return_10sessions']
        def aggregate(method):
            subset = seeds[seeds.method == method]
            return {key: {'mean': float(subset[key].mean()), 'seed_std': float(subset[key].std(ddof=1))}
                    for key in metrics}
        paired = frame[frame.method.isin(['h10', 'selected'])].groupby(['date', 'method'])['ic'].mean().unstack('method')
        differences = paired['selected'] - paired['h10']
        boot = config['bootstrap']
        bootstrap = paired_block_bootstrap(differences, seed=boot['seed'], draws=boot['replicates'], block=boot['block_sessions'])
        sweep = [{'horizon': h, **aggregate(f'h{h}')} for h in config['candidate_horizons']]
        from scipy.stats import rankdata
        correlations = []
        for seed in config['seeds']:
            ic = seeds[(seeds.seed == seed) & (seeds.method != 'selected')].sort_values('horizon')['ic'].to_numpy()
            value = corr(rankdata(selections[seed]['weights']), rankdata(ic))
            correlations.append(value if np.isfinite(value) else None)
        finite_correlations = [value for value in correlations if value is not None]
        result = {'complete': True, 'scope': config['scope'], 'universe': args.universe, 'signature': signature,
                  'seeds': config['seeds'], 'target_horizon_sessions': 10,
                  'split_days': {key: len(dates) for key, dates in dataset.splits.items()},
                  'primary': {'standard': aggregate('h10'), 'selected': aggregate('selected')},
                  'selected_horizons': {str(s): fit['selected_horizon'] for s, fit in selections.items()},
                  'paired_ic_bootstrap': bootstrap, 'horizon_sweep': sweep,
                  'descriptive_best_test_horizon': max(sweep, key=lambda h: h['ic']['mean'])['horizon'],
                  'lambda_test_ic_rank_correlation': {'per_seed': correlations,
                      'mean': float(np.mean(finite_correlations)) if finite_correlations else None},
                  'completed_utc': now(), 'limitations': config['limitations']}
        save_json(folder / 'summary.json', result)
        progress({'phase': 'complete', 'complete': True})
        from report import write_report
        write_report(HERE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universe', choices=['csi300', 'csi500', 'csi1000'], required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save_json(HERE / 'results' / args.universe / 'status.json',
                  {'phase': 'failed', 'error': str(error), 'updated_utc': now()})
        raise
