"""Run the complete two-market, five-seed suite serially on the shared GPU."""
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from build_data import ROOT, save_json


def update(phase, **kw):
    save_json(ROOT / 'STATUS.json', dict(status=phase, pid=os.getpid(),
        updated_utc=datetime.now(timezone.utc).isoformat(),
        exact_published_results_reproduced=False, **kw))
    print(phase, kw, flush=True)


def external_train(universe, stage, seed):
    found = []
    for p in psutil.process_iter(['pid', 'cmdline', 'cwd']):
        try:
            cmd = p.info['cmdline'] or []
            if p.info['cwd'] != str(ROOT) or 'scripts/train.py' not in cmd:
                continue
            if cmd[cmd.index('--universe')+1] != universe or cmd[cmd.index('--stage')+1] != str(stage):
                continue
            if '--smoke-days' in cmd:
                continue
            if stage == 2 and ('--seed' not in cmd or cmd[cmd.index('--seed')+1] != str(seed)):
                continue
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                found.append(p.pid)
        except (psutil.Error, ValueError, IndexError):
            pass
    return found


def call(script, args, name):
    update('running', current_job=name)
    env = os.environ | dict(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2',
        MLFLOW_DISABLE_AGENT_HINT='1', MPLBACKEND='Agg', WANDB_MODE='disabled',
        CUBLAS_WORKSPACE_CONFIG=':4096:8')
    with (ROOT / 'results' / f'{name}.log').open('a') as f:
        result = subprocess.run([sys.executable, '-u', f'scripts/{script}', *args],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        update('failed', current_job=name, returncode=result.returncode, log=f'results/{name}.log')
        raise RuntimeError(f'{name} failed: see results/{name}.log')


def main():
    lock = (ROOT / '.suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    update('running', current_job='initialization')
    for universe in ['csi300', 'sp500']:
        for stage, seeds in [(1, [42]), (2, list(range(5)))]:
            for seed in seeds:
                while pids := external_train(universe, stage, seed):
                    update('running', current_job=f'{universe}-stage{stage}', adopted_training_pids=pids)
                    time.sleep(10)
                args = ['--universe', universe, '--stage', str(stage), '--epochs', '50', '--seed', str(seed)]
                call('train.py', args, f'{universe}-stage{stage}-seed{seed}')
                call('report.py', [], 'report')
        call('evaluate.py', ['--universe', universe, '--ridge'], f'ridge-{universe}-supervised')
        call('evaluate.py', ['--universe', universe], f'ensemble-{universe}')
        call('report.py', [], 'report')
    call('verify.py', [], 'verification')
    call('report.py', [], 'report')
    report = json.loads((ROOT / 'results/report.json').read_text())
    assert report['status'] == 'adapted_empirical_run_complete'
    update('adapted_empirical_run_complete', report='RESULTS.txt', verification='results/verification.json')
    save_json(ROOT / '.suite-complete', dict(verified=True,
        completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    main()
