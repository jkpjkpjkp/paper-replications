#!/usr/bin/env python3
"""Resumable table reproduction. One JSON per independently seeded training run."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import time
import numpy as np
from threadpoolctl import threadpool_limits
from krrdp.market import Market
from krrdp.pricing import Training, train
from krrdp.reference import geometric_european, geometric_tree, max_call_european


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output = Path(output)
    manifest = output/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['config_sha256'] != fingerprint:
        raise RuntimeError('Output belongs to another config; use a different directory')
    atomic_json(manifest, dict(config=config, config_sha256=fingerprint,
                              python=platform.python_version(), platform=platform.platform(),
                              started_unix=time(), threads=config['threads']))
    with threadpool_limits(config['threads']):
        references = {}
        for dim in map(int, config['dimensions']):
            market = Market(dimension=dim, **config['market'])
            lower = geometric_tree(market, 400)
            higher = geometric_tree(market, 800)
            call, error = max_call_european(market)
            references[str(dim)] = dict(geometric_put=higher, geometric_tree_coarse=lower,
                geometric_tree_difference=abs(lower-higher),
                geometric_european=float(geometric_european(market)), max_call=call,
                max_call_quadrature_error=error)
        atomic_json(output/'references.json', references)
        for dim, sizes in config['dimensions'].items():
            for kind_i, kind in enumerate(['geometric_put', 'max_call']):
                for repetition in range(config['repetitions']):
                    path = output/'runs'/f'{kind}_d{dim}_r{repetition:02d}.json'
                    seed = config['base_seed']+int(dim)*1000+kind_i*100000+repetition
                    if path.exists():
                        old = json.loads(path.read_text())
                        if old.get('config_sha256') != fingerprint:
                            raise RuntimeError(f'Stale result {path}')
                        continue
                    result = train(Market(dimension=int(dim), **config['market']),
                                   Training(**sizes, **config['training']), kind, seed).asdict()
                    result.update(repetition=repetition, config_sha256=fingerprint)
                    atomic_json(path, result)
                    print(f'{kind:14s} d={dim:>2s} r={repetition:02d} price={result["price"]:.6f} '
                          f'seconds={result["seconds"]:.2f}', flush=True)
    atomic_json(output/'COMPLETE.json', dict(config_sha256=fingerprint, finished_unix=time(),
                   runs=len(config['dimensions'])*2*config['repetitions']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/main.json')
    parser.add_argument('--output', default='results/main')
    args = parser.parse_args()
    run(args.config, args.output)
