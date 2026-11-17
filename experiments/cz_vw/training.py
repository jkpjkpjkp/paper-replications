"""Reuse the established optimizer/checkpoint loop with isolated CZ paths."""
import argparse
from types import SimpleNamespace
from common import HERE, settings, model_config, check_lock
from panel import Panel
import train as original


def run(stage, smoke=False):
    p = settings()['training']
    if not smoke:
        check_lock()
    original.ROOT = HERE
    original.Panel = Panel
    original.config = model_config
    args = SimpleNamespace(universe='sp500', stage=stage,
        seed=p['stage1_seed'] if stage == 1 else p['stage2_seeds'][0],
        epochs=1 if smoke else p['epochs'], patience=p['patience'], device=p['device'],
        threads=p['threads'], smoke_days=3 if smoke else 0)
    original.run(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.stage, args.smoke)
