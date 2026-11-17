"""Train the released PRISM-VQ architecture locally, with resumable checkpoints."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from runtime import ROOT, Panel, config, spatial_model, temporal_model, encode, from_codes, rankic, ranking_metrics
from build_data import save_json, sha


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # PyTorch's fused eval encoder assumes LayerNorm, but authors use RMSNorm.
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision('high')


def checkpoint(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj, tmp)
    tmp.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def log(folder, **kwargs):
    kwargs['updated_utc'] = datetime.now(timezone.utc).isoformat()
    save_json(folder / 'status.json', kwargs)
    print(json.dumps(kwargs), flush=True)


def build_code_cache(model, panel, folder, device, plans):
    # Cache exact STE outputs, so the optimization leaves native forward values unchanged.
    shape = (*panel.member.shape, model.vq_embed_dim)
    path = folder / 'codes.npy'
    receipt = folder / 'codes.json'
    key = dict(stage1_sha256=sha(Path(model.saved_model)),
               data_manifest_sha256=sha(panel.folder / 'manifest.json'),
               plan={s: list(map(int, ds)) for s, ds in plans.items()}, dtype='float32')
    if receipt.exists() and json.loads(receipt.read_text()) == key and path.exists():
        return np.load(path, mmap_mode='r')
    codes = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=shape)
    model.eval()
    for split, days in plans.items():
        for j, t in enumerate(days):
            x, p, y, ids = panel.batch(t, split, device)
            z, _ = encode(model, x)
            codes[ids, t] = z.cpu().numpy()
            if j % 250 == 0:
                log(folder, phase='cache_frozen_codes', split=split, completed=j, days=len(days))
    codes.flush()
    save_json(receipt, key)
    return np.load(path, mmap_mode='r')


def evaluate(model, panel, days, stage, device, codes=None, predictions=False):
    model.eval()
    losses, weights, ric, frames = [], [], [], []
    split = 'test' if predictions else 'valid'
    with torch.no_grad():
        for t in days:
            x, prior, y, ids = panel.batch(t, split, device)
            if stage == 1:
                loss = model(x, prior, y)[3]
            else:
                if codes is None:
                    pred, _, _, _, aux = model(x, prior)
                else:
                    z = torch.from_numpy(np.array(codes[ids, t], copy=True)).to(device)
                    pred, aux = from_codes(model, x, prior, z)
                good = torch.isfinite(y[:, 4])
                loss = ((pred[good]-y[good, 4])**2).mean() + model.aux_weight * aux
                ric.append(rankic(pred.cpu().numpy(), y[:, 4].cpu().numpy()))
                if predictions:
                    frames.append(pd.DataFrame(dict(date=str(panel.dates[t]), symbol=panel.symbols[ids],
                        prediction=pred.cpu().numpy(), label=y[:, 4].cpu().numpy())))
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite validation/test loss at {panel.dates[t]}')
            losses.append(float(loss))
            weights.append(len(ids))
    return float(np.average(losses, weights=weights)), ric, frames


def run(args):
    torch.set_num_threads(args.threads)
    seed = 42 if args.stage == 1 else args.seed
    seed_all(seed)
    stage1 = ROOT / f'results/{args.universe}/stage1/best.pt'
    suffix = 'smoke' if args.smoke_days else f'{args.universe}/stage{args.stage}'
    if args.stage == 2:
        suffix += f'/seed{seed}'
    folder = ROOT / 'results' / suffix
    folder.mkdir(parents=True, exist_ok=True)
    if args.stage == 2 and (folder / 'metrics.json').exists():
        print((folder / 'metrics.json').read_text())
        return
    panel = Panel(args.universe)
    plans = panel.plans.copy()
    if args.smoke_days:
        for s in plans:
            plans[s] = plans[s][:args.smoke_days]
        if args.stage == 2:
            stage1 = ROOT / 'results/smoke/best.pt'
    cfg = config(args.universe, seed, args.epochs, stage1)
    save_json(folder / 'config.json', cfg)
    save_json(folder / 'data_summary.json', panel.summary())
    total_steps = len(plans['train']) * args.epochs
    model = (spatial_model(cfg) if args.stage == 1 else temporal_model(cfg, total_steps)).to(args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4,
                              weight_decay=0.01 if args.stage == 1 else 1e-5)
    if args.stage == 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps)
    else:
        warmup = int(total_steps * .05)
        def lr_lambda(step):
            if step < warmup:
                return step/max(1, warmup)
            return .5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    best, early_best, stale, first = float('inf'), float('inf'), 0, 0
    history = []
    last = folder / 'last.pt'
    if last.exists():
        state = torch.load(last, map_location='cpu', weights_only=False)
        assert state['config'] == cfg, 'Config differs from resumable checkpoint'
        assert state['data_manifest_sha256'] == sha(panel.folder / 'manifest.json')
        sd = state['state_dict']
        model.load_state_dict({k.removeprefix('vqvae.'): v for k, v in sd.items()} if args.stage == 1 else sd)
        optim.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        best, early_best, stale = state['best'], state['early_best'], state['stale']
        first, history = state['epoch']+1, state['history']
        restore_rng(state['rng'])
    codes = None
    if args.stage == 2 and not args.smoke_days:
        codes = build_code_cache(model, panel, folder.parent, args.device, plans)
    log(folder, phase='training', stage=args.stage, seed=seed, start_epoch=first, epochs=args.epochs,
        train_days=len(plans['train']), valid_days=len(plans['valid']))
    for epoch in range(first, args.epochs):
        if stale >= args.patience:
            break
        model.train()
        order = plans['train'][np.random.permutation(len(plans['train']))]
        total, count = 0., 0
        start = time.monotonic()
        for step, t in enumerate(order):
            x, prior, y, ids = panel.batch(t, 'train', args.device)
            optim.zero_grad(set_to_none=True)
            if args.stage == 1:
                loss = model(x, prior, y)[3]
            else:
                if codes is None:
                    pred, _, _, _, aux = model(x, prior)
                else:
                    z = torch.from_numpy(np.array(codes[ids, t], copy=True)).to(args.device)
                    pred, aux = from_codes(model, x, prior, z)
                loss = torch.nn.functional.mse_loss(pred, y[:, 4]) + model.aux_weight*aux
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at epoch {epoch}, date {panel.dates[t]}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise ValueError('Nonfinite gradient norm')
            optim.step()
            scheduler.step()
            total += float(loss.detach())*len(ids)
            count += len(ids)
            if step and step % 250 == 0:
                log(folder, phase='training', stage=args.stage, seed=seed, epoch=epoch,
                    step=step, steps=len(order), mean_loss=total/count,
                    elapsed_seconds=time.monotonic()-start)
        valid, ric, _ = evaluate(model, panel, plans['valid'], args.stage, args.device, codes)
        record = dict(epoch=epoch, train_loss=total/count, valid_loss=valid,
                      seconds=time.monotonic()-start)
        if ric:
            record.update(valid_RankIC=ranking_metrics(ric)['RankIC'])
        history.append(record)
        if valid < early_best - 1e-5:
            early_best, stale = valid, 0
        else:
            stale += 1
        sd = ({'vqvae.'+k: v for k, v in model.state_dict().items()} if args.stage == 1 else model.state_dict())
        if valid < best:
            best = valid
            checkpoint(folder / 'best.pt', dict(state_dict=sd, epoch=epoch, valid_loss=valid, config=cfg))
        state = dict(state_dict=sd, optimizer=optim.state_dict(), scheduler=scheduler.state_dict(),
            rng=rng_state(), epoch=epoch, history=history, best=best, early_best=early_best,
            stale=stale, config=cfg, data_manifest_sha256=sha(panel.folder / 'manifest.json'))
        checkpoint(last, state)
        save_json(folder / 'history.json', history)
        log(folder, phase='epoch_complete', stage=args.stage, seed=seed, **record, best_valid_loss=best, patience_used=stale)
    best_state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    sd = best_state['state_dict']
    model.load_state_dict({k.removeprefix('vqvae.'): v for k, v in sd.items()} if args.stage == 1 else sd)
    if args.stage == 2:
        _, daily, frames = evaluate(model, panel, plans['test'], 2, args.device, codes, predictions=True)
        pred = pd.concat(frames, ignore_index=True)
        pred.to_parquet(folder / 'predictions.parquet', index=False)
        pd.DataFrame(dict(date=panel.dates[plans['test']], RankIC=daily)).to_csv(folder / 'daily_rankic.csv', index=False)
        metrics = dict(**ranking_metrics(daily), universe=args.universe, seed=seed,
            best_epoch=best_state['epoch'], trained_epochs=len(history), best_valid_loss=best,
            observations=len(pred), engineering_smoke=bool(args.smoke_days), exact_paper_reproduction=False,
            checkpoint_sha256=sha(folder / 'best.pt'), stage1_sha256=sha(stage1),
            predictions_sha256=sha(folder / 'predictions.parquet'))
        save_json(folder / 'metrics.json', metrics)
    log(folder, phase='complete', stage=args.stage, seed=seed, epochs_completed=len(history),
        best_epoch=best_state['epoch'], best_valid_loss=best, engineering_smoke=bool(args.smoke_days))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--universe', choices=['csi300', 'sp500'], required=True)
    p.add_argument('--stage', type=int, choices=[1, 2], required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--smoke-days', type=int, default=0)
    run(p.parse_args())
