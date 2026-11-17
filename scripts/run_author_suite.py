"""Durable, serial validation search for LiMT on the released original data."""
from __future__ import annotations
import argparse
import fcntl
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_author_data import now, sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def weight_grid():
    grid = list(itertools.product([.63, .65, .68, .70], [.30, .33, .35], [.03, .05, .08]))
    # Merely a scheduling order, fixed before any validation or test evaluation.
    first = (.65, .30, .05)
    return [first] + [w for w in grid if w != first]


def config_for(universe, seed, weights, inventory, threads):
    return dict(universe=universe, seed=seed, weights=list(weights),
                epochs=100, lr=1e-5, threads=threads, inventory=str(inventory),
                model='LiMT_v2_MMoE_RVol2Vola2', hidden=256,
                spatial_heads=4, temporal_heads=2, dropout=.5, experts=8,
                lookback=21, patience_nonimproving_epochs=21,
                label_processing='released train_epoch daily sample-std zscore; no additional dropping',
                data_processing='released TSDataSampler and serialized ffill+zero setting',
                selection='largest three-seed mean best validation IC across all 36 weight combinations')


def prepare(output, inventory, threads):
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for weights in weight_grid():
        tag = 'w' + '_'.join(f'{w:.2f}' for w in weights)
        for seed in [0, 1, 2]:
            for universe in ['csi300', 'csi500']:
                name = f'{universe}/{tag}/seed{seed}'
                config = config_for(universe, seed, weights, inventory, threads)
                config_path = output / 'configs' / (name.replace('/', '__') + '.json')
                if config_path.exists() and json.loads(config_path.read_text()) != config:
                    raise ValueError(f'Existing configuration changed: {config_path}')
                write_json(config_path, config)
                jobs.append(dict(name=name, config=str(config_path), output=str(output/'runs'/name),
                                 universe=universe, weights=list(weights), seed=seed))
    plan = dict(scope='LiMT main predictive model, CSI300 and CSI500, released 2008-start data',
        jobs=jobs, trials=len(jobs), seeds=[0, 1, 2], weight_combinations=len(weight_grid()),
        exact_published_configuration_verified=False,
        specified_by_paper=dict(hidden=256, lr=1e-5, spatial_heads=4, temporal_heads=2,
                               lookback=21, max_epochs=100, repetitions=3),
        release_defaults=dict(experts=8, dropout=.5, optimizer='Adam', gradient_value_clip=3,
                              patience=21, shuffle='same seed-permuted daily order each epoch'),
        unresolved=['Actual published seed IDs and winning weights are not disclosed.',
                    'Paper equations and released attention/fusion classes differ; this run uses released classes unchanged.',
                    'The released files contain raw-scale labels and ffill+zero, unlike parts of the README description.',
                    'Hardware differs from the reported Xeon/RTX3090 server.',
                    'Original APO/exchange implementation and full portfolio protocol are not released.'],
        selection='Each candidate has three runs; select greatest mean validation IC independently per universe; test only the selected three checkpoints.',
        tie_break='fixed weight_grid order',
        scope_limits='No substitute-price APO or fourteen-baseline results are presented as exact paper results.',
        inventory_sha256=sha(inventory),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT/'scripts/train_author_data.py', Path(__file__),
            ROOT/'upstream/base_model.py', ROOT/'upstream/limt.py', ROOT/'upstream/utils.py']})
    path = output/'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Existing suite plan/source has changed; use a new output directory')
    write_json(path, plan)
    return plan


def selected_jobs(jobs):
    import statistics
    selections = {}
    for universe in ['csi300', 'csi500']:
        scored = []
        for weights in weight_grid():
            group = [j for j in jobs if j['universe'] == universe and j['weights'] == list(weights)]
            if len(group) != 3:
                raise ValueError('Incomplete seed group')
            statuses = [json.loads((Path(j['output'])/'status.json').read_text()) for j in group]
            if any(s['phase'] not in ['trained', 'evaluated'] for s in statuses):
                raise ValueError('Selection requires every full trial')
            vals = [s['best_validation_ic'] for s in statuses]
            scored.append(dict(weights=list(weights), mean_validation_ic=statistics.mean(vals),
                               validation_seed_sd=statistics.stdev(vals), jobs=group))
        winner = max(scored, key=lambda s: s['mean_validation_ic'])
        selections[universe] = dict(winner=winner, candidates=scored)
    return selections


def device_for(job, requested, cuda_python=None):
    previous = Path(job['output'])/'provenance.json'
    if previous.exists():
        return json.loads(previous.read_text())['device']
    if requested != 'auto':
        return requested
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True)
        if int(result.stdout.splitlines()[0]) >= 2500 and cuda_python is not None:
            probe = subprocess.run([str(cuda_python), '-c',
                "import torch; assert torch.__version__.split('+')[0]=='1.11.0'; "
                "assert torch.cuda.is_available(); "
                "assert (torch.ones(2,device='cuda')+1).sum().item()==4"],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                capture_output=True, timeout=60)
            if probe.returncode == 0:
                return 'cuda'
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 'cpu'


def run_job(job, requested, evaluate=False, cuda_python=None):
    output = Path(job['output'])
    output.mkdir(parents=True, exist_ok=True)
    device = device_for(job, requested, cuda_python)
    interpreter = str(cuda_python) if device == 'cuda' and cuda_python else sys.executable
    command = [interpreter, '-u', str(ROOT/'scripts/train_author_data.py'),
               '--config', job['config'], '--output', str(output), '--device', device]
    if evaluate:
        command.append('--evaluate-selected')
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PYTHONUNBUFFERED='1')
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    else:
        env['CUDA_VISIBLE_DEVICES'] = '0'
    with (output/('evaluate.log' if evaluate else 'train.log')).open('a') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def run(output, plan, requested, cuda_python=None):
    jobs = plan['jobs']
    for i, job in enumerate(jobs):
        status_path = Path(job['output'])/'status.json'
        if status_path.exists() and json.loads(status_path.read_text())['phase'] in ['trained', 'evaluated']:
            continue
        write_json(output/'suite_status.json', dict(phase='training', job=job['name'],
            completed_trials=i, total_trials=len(jobs), pid=os.getpid(), updated_utc=now()))
        print(f'{now()} starting {job["name"]}', flush=True)
        run_job(job, requested, cuda_python=cuda_python)
    selection = selected_jobs(jobs)
    write_json(output/'selection.json', selection)
    for universe, result in selection.items():
        for job in result['winner']['jobs']:
            status = json.loads((Path(job['output'])/'status.json').read_text())
            if status['phase'] == 'evaluated':
                continue
            write_json(output/'suite_status.json', dict(phase='evaluating_selected',
                job=job['name'], completed_trials=len(jobs), total_trials=len(jobs),
                pid=os.getpid(), updated_utc=now()))
            run_job(job, requested, evaluate=True, cuda_python=cuda_python)
    import statistics
    summary = {}
    for universe, result in selection.items():
        records = [json.loads((Path(j['output'])/'metrics.json').read_text())['test']['metrics']
                   for j in result['winner']['jobs']]
        metrics = {k: dict(mean=statistics.mean(r[k] for r in records),
                          seed_sd=statistics.stdev(r[k] for r in records)) for k in records[0]}
        summary[universe] = dict(weights=result['winner']['weights'], test=metrics,
            paper_IC=.0575 if universe == 'csi300' else .0496,
            paper_ICIR=.4462 if universe == 'csi300' else .452)
    write_json(output/'comparison.json', summary)
    write_json(output/'suite_status.json', dict(phase='complete', completed_trials=len(jobs),
        total_trials=len(jobs), comparison=summary, updated_utc=now(),
        exact_published_configuration_verified=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, default=ROOT/'.agent/AUTHOR_DATA_INVENTORY.json')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--cuda-python', type=Path,
                        help='Separate torch 1.11 CUDA runtime; auto probes it before assigning CUDA')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'suite.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(output, args.inventory.resolve(), args.threads)
        if args.prepare_only:
            print(json.dumps(dict(trials=plan['trials'], output=str(output))))
            return
        write_json(output/'runtime_policy.json', dict(default_python=sys.executable,
            cuda_python=str(args.cuda_python) if args.cuda_python else None,
            requested_device=args.device, min_free_gpu_mb=2500, updated_utc=now()))
        try:
            run(output, plan, args.device, args.cuda_python)
        except Exception as exc:
            write_json(output/'suite_status.json', dict(phase='failed', error=str(exc), updated_utc=now()))
            raise


if __name__ == '__main__':
    main()
