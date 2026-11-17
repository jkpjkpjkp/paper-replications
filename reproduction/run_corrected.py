"""Run fixed-budget corrected experiments in a fresh workspace; preserve archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--jobs', type=int, choices=[1, 2], default=2)
    parser.add_argument('--train-end', choices=['2020-11-30', '2020-12-31'], default='2020-11-30')
    args = parser.parse_args()
    if not args.seeds or args.seeds[0] != 0 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Unique seeds starting with 0 are required for the baseline report')
    results = ROOT/'results'
    results.mkdir(exist_ok=True)
    outputs = [f'csi300-seed{s}' for s in args.seeds] + ['gp-csi300-seed0', 'xgboost-csi300-seed0']
    for name in [*outputs, 'corrected-protocol.json', 'corrected-pipeline.json']:
        if (results/name).exists():
            raise ValueError(f'Existing experiment preserved: {results/name}; use a fresh workspace')
    sources = sorted((ROOT/'reproduction').glob('*.py')) + sorted((ROOT/'upstream').rglob('*.py'))
    protocol = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        seeds=args.seeds, train_end=args.train_end, alphasage_episodes=10000,
        pool_capacity=50, gp_population=1000, gp_generations=40, xgboost_rounds=[100,10],
        cpu_threads_per_job=2, jobs=args.jobs, selection='Fixed budgets; no test-based selection',
        data='Corrected community snapshot; original author data identity remains unresolved',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    save(results/'corrected-protocol.json', protocol)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MLFLOW_DISABLE_AGENT_HINT='1', PYTHONUNBUFFERED='1')
    started = time.monotonic()
    def run(script, arguments, log):
        with (results/log).open('x') as stream:
            subprocess.run([sys.executable, str(ROOT/'reproduction'/script), *arguments],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    def alpha(seed):
        name = f'csi300-seed{seed}'
        run('train.py', ['--seed',str(seed),'--episodes','10000','--capacity','50',
                         '--train-end',args.train_end,'--output',f'results/{name}'], name+'.log')
        run('evaluate.py', ['--pool',f'results/{name}/pool.json','--output',f'results/{name}/adaptive'], name+'-adaptive.log')
        return name

    def baseline(name):
        parameters = ['--population','1000','--generations','40'] if name == 'gp' else ['--rounds','100']
        directory = f'{name}-csi300-seed0'
        run(f'train_{name}.py', ['--seed','0','--train-end',args.train_end,*parameters,
                               '--output',f'results/{directory}'], directory+'.log')
        return directory

    completed, failed = [], []
    state = dict(status='running', pid=os.getpid(), started_at_utc=protocol['created_at_utc'],
                 outputs=outputs, completed=completed, failed=failed, protocol='corrected-protocol.json')
    save(results/'corrected-pipeline.json', state)
    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            tasks = {executor.submit(alpha, 0):'csi300-seed0',
                     executor.submit(baseline, 'xgboost'):'xgboost-csi300-seed0',
                     executor.submit(baseline, 'gp'):'gp-csi300-seed0'}
            tasks.update({executor.submit(alpha, seed):f'csi300-seed{seed}' for seed in args.seeds[1:]})
            for future in as_completed(tasks):
                try:
                    completed.append(future.result())
                except Exception as error:
                    failed.append(dict(experiment=tasks[future], error=str(error)))
                state['wall_seconds'] = time.monotonic()-started
                save(results/'corrected-pipeline.json', state)
                print(json.dumps(state), flush=True)
        if failed:
            raise RuntimeError('One or more experiments failed; see preserved logs')
        run('report.py', [], 'report.log')
        summaries = [json.loads((results/f'csi300-seed{s}/adaptive/summary.json').read_text()) for s in args.seeds]
        keys = ['ic','icir','rank_ic','rank_icir']
        aggregation = {}
        for key in keys:
            values = np.array([s['test'][key] for s in summaries])
            aggregation[key] = dict(values=values.tolist(), mean=float(values.mean()),
                sample_seed_sd=float(values.std(ddof=1)) if len(values)>1 else None)
        save(results/'seed-summary.json', dict(seeds=args.seeds, metrics=aggregation,
            interpretation='Sample seed SD is not a confidence interval; one seed has no empirical seed uncertainty',
            paper_table12=dict(ic_mean=.073, ic_seed_sd=.004, rank_ic_mean=.088, rank_ic_seed_sd=.008),
            comparability='Corrected community-data adaptation, not the authors exact snapshot'))
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state['wall_seconds'] = time.monotonic()-started
        save(results/'corrected-pipeline.json', state)


if __name__ == '__main__':
    main()
