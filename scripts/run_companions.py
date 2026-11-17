"""Run clearly labeled NASDAQ/NYSE reconstructions using the available release."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    for market in ['nasdaq','nyse']:
        label = f'companion-{market}-full-seed0'
        dest = ROOT / 'results' / label
        if (dest / 'metrics.json').exists():
            continue
        command = [sys.executable, str(ROOT / 'scripts/train.py'), '--data', str(ROOT / 'data' / f'{market}.npz'),
                   '--output', str(dest), '--train-end', '756', '--val-end', '1008',
                   '--batch-size', '8', '--device', args.device, '--seed', '0', '--epochs', '100', '--patience', '15']
        (ROOT / 'results' / f'{label}.command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('running: '+label, flush=True)
        with (ROOT / 'results' / f'{label}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
