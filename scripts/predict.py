"""Re-evaluate a saved checkpoint on a date interval, with no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.evaluation import predict_checkpoint, evaluated_artifacts


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--data', type=Path, default=ROOT / 'data/sp500.npz')
    p.add_argument('--start', type=int, default=1259)
    p.add_argument('--end', type=int, default=1611)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    data = np.load(args.data)
    prediction, saved = predict_checkpoint(args.checkpoint, data, args.start, args.end, args.device)
    c = saved['config']
    target, valid = data['returns'][args.start:args.end], data['valid'][args.start:args.end]
    pnl_source = data['pnl_returns'] if 'pnl_returns' in data.files else data['returns']
    metrics, artifacts = evaluated_artifacts(prediction, target, valid, data['dates'][args.start:args.end], data['tickers'], c['top_k'], pnl_source[args.start:args.end])
    if args.output:
        np.savez_compressed(args.output, **artifacts)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
