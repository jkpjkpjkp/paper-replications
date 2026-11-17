#!/usr/bin/env python3
"""Declared sensitivity experiments and an approximate Figure 2 reconstruction."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from threadpoolctl import threadpool_limits
from krrdp.market import Market
from krrdp.pricing import Training, train
from krrdp.reference import geometric_tree
from scripts.run_suite import atomic_json, run


def sensitivities():
    base = json.loads(Path('configs/main.json').read_text())
    base['dimensions'] = {k:v for k,v in base['dimensions'].items() if k in ['2','20']}
    configs = []
    for name in ['ten_steps', 'independent_mc']:
        config = json.loads(json.dumps(base))
        config['description'] = name+' sensitivity; not paper-reported protocol'
        if name == 'ten_steps':
            config['market']['steps'] = 10
        else:
            config['training']['mc_mode'] = 'independent'
        path = Path('configs')/(name+'.json')
        atomic_json(path, config)
        configs.append((name,path))
    for name,path in configs:
        run(path, Path('results')/name)


def convergence(repetitions=5):
    base = json.loads(Path('configs/main.json').read_text())
    tasks = {20:[200,400,800,1200,2000], 40:[400,800,1600,2400,4000]}
    for dimension, sizes in tasks.items():
        market = Market(dimension=dimension, **base['market'])
        reference = geometric_tree(market, 800)
        for n in sizes:
            m = round(50*(n/200)**.5)
            training = Training(n=n, m=m, max_centers=256, validation_n=128, batch_size=4096)
            for repetition in range(repetitions):
                path = Path('results/convergence')/f'd{dimension}_n{n}_r{repetition:02d}.json'
                if path.exists():
                    continue
                # Same base seed across n defines paired, not independent, n scenarios.
                seed = 741000+dimension*100+repetition
                result = train(market, training, 'geometric_put', seed).asdict()
                result.update(repetition=repetition, reference=reference,
                              squared_error=(result['price']-reference)**2,
                              description='Figure 2 reconstruction; direct Nystrom with at most 256 centers')
                atomic_json(path, result)
                print(f'convergence d={dimension} n={n} M={m} r={repetition} '
                      f'price={result["price"]:.5f} error2={result["squared_error"]:.6f} '
                      f'seconds={result["seconds"]:.2f}', flush=True)
    atomic_json(Path('results/convergence/COMPLETE.json'), dict(repetitions=repetitions, runs=10*repetitions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', choices=['sensitivities','convergence'])
    args = parser.parse_args()
    with threadpool_limits(2):
        globals()[args.experiment]()
