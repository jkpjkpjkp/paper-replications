"""Train on past dates, select by validation loss, evaluate test once."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.model import Hermes, LinearBaseline, ranking_loss
from hermes.metrics import evaluate, PORTFOLIO_PROTOCOL
from hermes.evaluation import evaluated_artifacts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=ROOT / 'data/sp500.npz')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lookback', type=int, default=16)
    p.add_argument('--hidden', type=int, default=8)
    p.add_argument('--scales', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--windows', type=int, nargs='+', default=[3, 2, 1])
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--rank-reduction', choices=['sum', 'mean'], default='sum')
    p.add_argument('--ablation', choices=['full', 'no_fusion', 'single_scale', 'no_lead_lag', 'no_skip', 'initial_only', 'linear'], default='full')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--train-end', type=int, default=1006)
    p.add_argument('--val-end', type=int, default=1259)
    p.add_argument('--threads', type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'output must be empty: {args.output}')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    data = np.load(args.data)
    x = torch.as_tensor(data['features'], device=device)
    target = torch.as_tensor(data['returns'], device=device)
    valid = torch.as_tensor(data['valid'], device=device)
    pnl_source = data['pnl_returns'] if 'pnl_returns' in data.files else data['returns']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(torch_version=str(torch.__version__), numpy_version=str(np.__version__),
                  device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
                  evaluation_protocol=PORTFOLIO_PROTOCOL)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    if args.ablation == 'linear':
        model = LinearBaseline(args.lookback, x.shape[-1]).to(device)
    else:
        model = Hermes(data['incidence'], args.lookback, x.shape[-1], args.hidden,
                       args.scales, args.windows, args.ablation).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_days = np.arange(args.lookback, args.train_end)
    val_days = np.arange(args.train_end, args.val_end)
    test_days = np.arange(args.val_end, len(x))
    offsets = torch.arange(-args.lookback, 0, device=device)

    def batch(days):
        days = torch.as_tensor(days, device=device)
        histories = days[:, None] + offsets[None, :]
        inputs = x[histories].permute(0, 2, 1, 3)
        mask = valid[days]
        return inputs, target[days], mask

    @torch.no_grad()
    def predict(days):
        model.eval()
        preds, ys, masks, losses = [], [], [], []
        for start in range(0, len(days), args.batch_size):
            inputs, y, mask = batch(days[start:start + args.batch_size])
            prediction = model(inputs)
            loss, _, _ = ranking_loss(prediction, y, mask, args.alpha, args.rank_reduction)
            losses.append(float(loss) * len(inputs))
            preds.append(prediction.cpu().numpy())
            ys.append(y.cpu().numpy())
            masks.append(mask.cpu().numpy())
        p, y, m = map(np.concatenate, [preds, ys, masks])
        metrics, ic, portfolio = evaluate(p, y, m, args.top_k, pnl_source[days])
        return sum(losses) / len(days), metrics, (p, y, m, ic, portfolio)

    best_loss, best_epoch = float('inf'), 0
    started = time.monotonic()
    log = (args.output / 'history.jsonl').open('w', buffering=1)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        total = np.zeros(3)
        order = np.random.permutation(train_days)
        for start in range(0, len(order), args.batch_size):
            inputs, y, mask = batch(order[start:start + args.batch_size])
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss, mse, rank = ranking_loss(prediction, y, mask, args.alpha, args.rank_reduction)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite training loss at epoch {epoch}')
            loss.backward()
            optimizer.step()
            total += np.array([float(loss.detach()), float(mse), float(rank)]) * len(inputs)
        val_loss, val_metrics, _ = predict(val_days)
        record = {'epoch': epoch, 'train_loss': total[0] / len(train_days),
                  'train_mse': total[1] / len(train_days), 'train_rank': total[2] / len(train_days),
                  'val_loss': val_loss, 'val_IC': val_metrics['IC'],
                  'seconds': time.monotonic() - epoch_start}
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'config': config}, args.output / 'best.pt')
        log.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
        if epoch - best_epoch >= args.patience:
            break
    log.close()
    model.load_state_dict(torch.load(args.output / 'best.pt', map_location=device, weights_only=True)['model'])
    _, val_metrics, val_arrays = predict(val_days)
    vp, vy, vm, _, _ = val_arrays
    _, val_artifacts = evaluated_artifacts(vp, vy, vm, data['dates'][val_days], data['tickers'], args.top_k, pnl_source[val_days])
    np.savez_compressed(args.output / 'validation_predictions.npz', **val_artifacts)
    test_loss, test_metrics, arrays = predict(test_days)
    p, y, m, ic, portfolio = arrays
    _, test_artifacts = evaluated_artifacts(p, y, m, data['dates'][test_days], data['tickers'], args.top_k, pnl_source[test_days])
    np.savez_compressed(args.output / 'test_predictions.npz', **test_artifacts)
    # The untuned previous-day-return control shares exactly the same test dates and masks.
    naive_metrics, _, _ = evaluate(data['returns'][test_days - 1], y, m, args.top_k, pnl_source[test_days])
    result = {'dataset': args.data.stem, 'seed': args.seed, 'ablation': args.ablation,
              'evaluation_protocol': PORTFOLIO_PROTOCOL,
              'best_epoch': best_epoch, 'epochs_run': epoch,
              'parameters': sum(p.numel() for p in model.parameters()),
              'training_seconds': time.monotonic() - started,
              'test_loss': test_loss, 'validation': val_metrics, 'test': test_metrics,
              'previous_day_return_control': naive_metrics,
              'test_dates': [str(data['dates'][test_days[0]]), str(data['dates'][test_days[-1]])],
              'max_gpu_memory_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0}
    (args.output / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print('RESULT ' + json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
