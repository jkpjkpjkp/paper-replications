"""Run a fixed experiment suite; each child writes its complete command and logs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--device', default='cuda')
    p.add_argument('--only', choices=['main', 'ablations', 'sensitivity', 'all'], default='all')
    args = p.parse_args()
    runs = []
    if args.only in ('all', 'main'):
        runs += [('full', s, 'sum') for s in [0, 1, 2]]
    if args.only in ('all', 'ablations'):
        runs += [(a, 0, 'sum') for a in ['no_fusion', 'single_scale', 'no_lead_lag', 'no_skip', 'initial_only', 'linear']]
    if args.only in ('all', 'sensitivity'):
        runs += [('full', s, 'mean') for s in [0, 1, 2]]
    for ablation, seed, reduction in runs:
        label = f'sp500-{ablation}-seed{seed}' + ('-meanrank' if reduction == 'mean' else '')
        dest = ROOT / 'results' / label
        if (dest / 'metrics.json').exists():
            print('completed, skip: ' + label, flush=True)
            continue
        command = [sys.executable, str(ROOT / 'scripts/train.py'), '--output', str(dest),
                   '--seed', str(seed), '--ablation', ablation, '--rank-reduction', reduction,
                   '--epochs', str(args.epochs), '--patience', str(args.patience), '--device', args.device]
        (ROOT / 'results' / f'{label}.command.json').write_text(json.dumps(command, indent=2) + '\n')
        print('running: ' + label, flush=True)
        with (ROOT / 'results' / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
