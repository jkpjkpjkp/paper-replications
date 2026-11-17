"""Run reproducible, paired portfolio experiments without external services."""
import argparse
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from core import (ROOT, MLP, GaussianMLP, PortfolioPolicyNet, Diffsion, PortfolioLayer,
                  solve_portfolio, score_weights, ddim, gaussian_samples, diffusion_score_loss)

METHODS = ['deterministic_ts', 'gaussian_ts', 'diffusion_ts', 'policy',
           'deterministic_dfl', 'gaussian_reparam', 'gaussian_score',
           'diffusion_reparam', 'diffusion_score']


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def family(method):
    return method.split('_')[0]


def load_data(seed, n, device):
    panel = np.load(ROOT / 'data/crsp_panel.npz')
    choice = np.random.RandomState(seed).choice(len(panel['permnos']), n, replace=False)
    x, y = panel['x'][:, choice].copy(), panel['y'][:, choice].copy()
    t = len(y)
    a, b = int(.7*t), int(.7*t)+int(.1*t)
    # Embargo the last feature day before each boundary so no future labels cross it.
    splits = [np.arange(a-1), np.arange(a, b-1), np.arange(b, t)]
    mean = x[splits[0]].reshape(-1, 28).mean(0)
    std = x[splits[0]].reshape(-1, 28).std(0, ddof=1)+1e-5
    x = ((x-mean)/std).reshape(t, -1)
    meta = dict(permnos=panel['permnos'][choice], symbols=panel['symbols'][choice],
                dates=panel['dates'], target_dates=panel['target_dates'], mean=mean, std=std)
    return torch.tensor(x, device=device), torch.tensor(y, device=device), splits, meta


def make_model(kind, dx, n, device):
    if kind == 'diffusion':
        d = Diffsion(dx, n, device=device, timesteps=1000)
        return d.model_net, d
    if kind == 'gaussian':
        return GaussianMLP(dx, [1024, 1024], n).to(device), None
    if kind == 'policy':
        return PortfolioPolicyNet(dx, n).to(device), None
    return MLP(dx, 1024, n).to(device), None


def batches(indices, size, shuffle=False):
    idx = np.random.permutation(indices) if shuffle else indices
    for start in range(0, len(idx), size):
        yield idx[start:start+size]


def prediction_loss(kind, model, diffusion, x, y):
    if kind == 'diffusion':
        t = torch.randint(1000, (len(x),), device=x.device)
        return diffusion.diffusion_loss(y, t, x)
    if kind == 'gaussian':
        mu, lv = model(x)
        return F.gaussian_nll_loss(mu, y, lv.exp(), full=True)
    return F.mse_loss(model(x), y)


def pretrain(kind, model, diffusion, x, y, splits, args, cache):
    if cache.exists():
        model.load_state_dict(torch.load(cache, map_location=x.device, weights_only=True))
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best, state = float('inf'), None
    for epoch in range(args.pretrain):
        model.train()
        for idx in batches(splits[0], 256, True):
            loss = prediction_loss(kind, model, diffusion, x[idx], y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite pretraining loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else []):
            torch.manual_seed(9871)
            val = prediction_loss(kind, model, diffusion, x[splits[1]], y[splits[1]]).item()
        if val < best:
            best = val
            state = copy.deepcopy(model.state_dict())
        if epoch % 10 == 0 or epoch == args.pretrain-1:
            print(json.dumps({'stage':'pretrain', 'family':kind, 'epoch':epoch, 'val_loss':val}), flush=True)
    if state is not None:
        model.load_state_dict(state)
    torch.save(model.state_dict(), cache)


def predictions(kind, model, diffusion, x, args, grad=False):
    if kind == 'diffusion':
        return ddim(diffusion, x, args.mc_samples, args.ddim_steps, grad=grad, checkpoint=grad)
    if kind == 'gaussian':
        return gaussian_samples(model, x, args.mc_samples)[0]
    return model(x)[:, None]


@torch.no_grad()
def evaluate(kind, model, diffusion, x, y, indices, args, seed):
    model.eval()
    zs, ys = [], []
    with torch.random.fork_rng(devices=[x.device.index] if x.is_cuda else []):
        torch.manual_seed(seed)
        for idx in batches(indices, args.batch_size):
            if kind == 'policy':
                z = model(x[idx]); pred = None
            else:
                pred = predictions(kind, model, diffusion, x[idx], args)
                z, _ = solve_portfolio(pred, args.alpha, kkt=False)
                ys.append(pred.cpu().numpy())
            zs.append(z.cpu().numpy())
    weights = np.concatenate(zs)
    truth = y[indices].cpu().numpy()
    returns = (weights*truth).sum(1)
    oracle = truth.max(1)
    if not (np.isfinite(weights).all() and np.allclose(weights.sum(1), 1, atol=1e-6)
            and weights.min() >= -1e-6 and np.all(returns <= oracle+1e-6)):
        raise RuntimeError('Portfolio feasibility/oracle audit failed')
    metric = {'mean_daily_return_pct':float(100*returns.mean()),
              'daily_return_std_pct':float(100*returns.std(ddof=1)),
              'oracle_mean_daily_return_pct':float(100*oracle.mean()),
              'equal_weight_mean_daily_return_pct':float(100*truth.mean()),
              'regret_pct':float(100*(oracle-returns).mean()),
              'task_loss':float((.5*args.alpha*returns**2-returns).mean()),
              'observations':len(returns), 'max_constraint_violation':float(max(
                  np.max(np.abs(weights.sum(1)-1)), max(0., -weights.min()), max(0., weights.max()-1)))}
    if ys:
        sampled = np.concatenate(ys)
        metric['rmse_sample'] = float(np.sqrt(np.mean((sampled-truth[:, None])**2)))
        metric['rmse_predictive_mean'] = float(np.sqrt(np.mean((sampled.mean(1)-truth)**2)))
    return metric, weights, returns


def train_batch(method, model, diffusion, x, y, optimizer, args):
    kind = family(method)
    pars = list(model.parameters())
    if kind == 'policy':
        z = model(x)
        r = (z*y).sum(-1)
        loss = (.5*args.alpha*r*r-r).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return float(loss.detach())
    if method.endswith('score'):
        with torch.no_grad():
            sampled = predictions(kind, model, diffusion, x, args)
            z, mat = solve_portfolio(sampled, args.alpha, args.kkt_ridge)
            weights = score_weights(sampled, z, mat, y, args.alpha)
        if kind == 'diffusion':
            # Bound peak memory. A full 1000-time score is accumulated in chunks;
            # no graph is retained between them. Reduced-time runs use uniform MC.
            task_grads = [torch.zeros_like(p) for p in pars]
            for start in range(0, args.score_times, 8):
                count = min(8, args.score_times-start)
                task_surrogate = diffusion_score_loss(diffusion, sampled, x, weights, count,
                    time_start=start if args.score_times==1000 else None)
                chunk_grads = torch.autograd.grad(task_surrogate, pars)
                for dest, grad in zip(task_grads, chunk_grads):
                    dest.add_(grad, alpha=count/args.score_times)
        else:
            mu, lv = model(x)
            logp = -.5*(lv[:, None]+(sampled-mu[:, None]).square()*(-lv[:, None]).exp()).sum(-1)
            task_surrogate = (weights*logp).sum()
            task_grads = torch.autograd.grad(task_surrogate, pars)
    else:
        sampled = predictions(kind, model, diffusion, x, args, grad=True)
        z = PortfolioLayer.apply(sampled, args.alpha, args.kkt_ridge)
        r = (z*y).sum(-1)
        task_surrogate = (.5*args.alpha*r*r-r).mean()
        task_grads = torch.autograd.grad(task_surrogate, pars)
    # Gradient-norm balancing follows release (gamma .5 diffusion, .9 others).
    if kind == 'diffusion':
        regularizer = sum(prediction_loss(kind, model, diffusion, x, y) for _ in range(10))/10
    else:
        regularizer = F.mse_loss(predictions(kind, model, diffusion, x, args, grad=True).mean(1), y)
    reg_grads = torch.autograd.grad(regularizer, pars)
    tn = torch.sqrt(sum(g.square().sum() for g in task_grads)+1e-12)
    rn = torch.sqrt(sum(g.square().sum() for g in reg_grads)+1e-12)
    gamma = .5 if kind == 'diffusion' else .9
    optimizer.zero_grad()
    for p, tg, rg in zip(pars, task_grads, reg_grads):
        p.grad = gamma*(rn/tn+1e-8)*tg+(1-gamma)*rg
        if not torch.isfinite(p.grad).all():
            raise RuntimeError('Nonfinite DFL gradient')
    optimizer.step()
    return float(regularizer.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='local')
    ap.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    ap.add_argument('--seeds', nargs='+', type=int, default=[0,1,2])
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--pretrain', type=int, default=100)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--mc-samples', type=int, default=10)
    ap.add_argument('--ddim-steps', type=int, default=50)
    ap.add_argument('--score-times', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--alpha', type=float, default=1e-6)
    ap.add_argument('--kkt-ridge', type=float, default=1e-6)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    if not 1 <= args.ddim_steps <= 1000 or not 1 <= args.score_times <= 1000:
        ap.error('diffusion/score steps must be between 1 and 1000')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    base = ROOT / 'results' / args.output
    base.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(torch_version=torch.__version__, panel_sha256=json.loads((ROOT/'data/receipt.json').read_text())['panel_sha256'])
    existing = base/'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise RuntimeError('Output already contains a different config; choose a new --output')
    existing.write_text(json.dumps(config, indent=2))
    for seed in args.seeds:
        seed_all(seed)
        x, y, splits, meta = load_data(seed, args.n, args.device)
        sd = base / f'seed{seed}'
        sd.mkdir(exist_ok=True)
        np.savez_compressed(sd/'split_and_scaler.npz', **meta, train=splits[0], validation=splits[1], test=splits[2])
        for method in args.methods:
            dest = sd / method
            dest.mkdir(exist_ok=True)
            if (dest/'metrics.json').exists():
                continue
            started = time.time()
            seed_all(seed)
            kind = family(method)
            model, diffusion = make_model(kind, x.shape[1], args.n, x.device)
            pretrain(kind, model, diffusion, x, y, splits, args, sd/f'{kind}_pretrained.pt')
            history = []
            best, best_epoch, state = -np.inf, None, None
            if not method.endswith('_ts'):
                lr = 1e-2 if kind=='policy' else 1e-4 if kind=='deterministic' else 8e-6 if method.endswith('score') else 1e-5
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                for epoch in range(args.epochs):
                    model.train()
                    for idx in batches(splits[0], args.batch_size, True):
                        train_batch(method, model, diffusion, x[idx], y[idx], optimizer, args)
                    metric, _, _ = evaluate(kind, model, diffusion, x, y, splits[1], args, seed+10000)
                    metric.update(epoch=epoch, elapsed_seconds=time.time()-started)
                    history.append(metric)
                    (dest/'history.json').write_text(json.dumps(history, indent=2))
                    print(json.dumps({'method':method,'seed':seed,**metric}), flush=True)
                    if metric['mean_daily_return_pct'] > best:
                        best = metric['mean_daily_return_pct']; best_epoch = epoch
                        state = copy.deepcopy(model.state_dict())
                if state is not None:
                    model.load_state_dict(state)
            metric, weights, returns = evaluate(kind, model, diffusion, x, y, splits[2], args, seed+20000)
            metric.update(method=method, seed=seed, selected_epoch=best_epoch,
                          elapsed_seconds=time.time()-started, status='completed_alternative_data_reduced_compute')
            torch.save(model.state_dict(), dest/'model.pt')
            np.savez_compressed(dest/'portfolios.npz', weights=weights, returns=returns,
                                dates=meta['dates'][splits[2]], target_dates=meta['target_dates'][splits[2]],
                                permnos=meta['permnos'], actual_returns=y[splits[2]].cpu().numpy())
            pd.DataFrame({'date':meta['target_dates'][splits[2]], 'portfolio_return':returns,
                          'equal_weight_return':y[splits[2]].mean(1).cpu().numpy(),
                          'oracle_return':y[splits[2]].max(1).values.cpu().numpy()}).to_csv(dest/'daily.csv', index=False)
            (dest/'metrics.json').write_text(json.dumps(metric, indent=2))
            print(json.dumps(metric), flush=True)
            del model, diffusion
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
