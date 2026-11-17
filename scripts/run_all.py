"""Finite six-seed reproduction; no recurring jobs, tuning, or test-seed selection."""
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1000, 1, 2000, 3000, 4000, 5000]
DEVICE = os.environ.get('TRA_DEVICE', 'cuda')
RUNTIME_DIR = 'paper-runtime' if DEVICE == 'cuda' else 'paper-runtime-cpu'


def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2)+'\n')
    temp.replace(path)


def main():
    with open(ROOT/'results'/('run-'+DEVICE+'.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for seed in SEEDS:
            for stage in ['initialization', 'tra']:
                out = ROOT/'results'/RUNTIME_DIR/str(seed)/stage
                receipt = out/'complete.json'
                if receipt.exists():
                    info = json.loads(receipt.read_text())
                    assert info['kind'] == 'full_stage'
                    for name, expected in info['files'].items():
                        actual = hashlib.sha256((out/name).read_bytes()).hexdigest()
                        assert actual == expected, (seed, stage, name)
                    continue
                save(ROOT/'results/status.json', dict(
                    state='running', device=DEVICE, runtime_dir=RUNTIME_DIR,
                    seed=seed, stage=stage, pid=os.getpid(),
                    started_utc=datetime.now(timezone.utc).isoformat(),
                    seeds=SEEDS, method='Unmodified Qlib 0.7.1 ALSTM+TRA'))
                log = ROOT/'logs'/('{}-{}-{}.log'.format(DEVICE, seed, stage))
                command = [str(ROOT/'.venv/bin/python'), '-u', str(ROOT/'scripts/run_stage.py'),
                           '--seed', str(seed), '--stage', stage]
                with log.open('a') as stream:
                    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                if result.returncode:
                    save(ROOT/'results/status.json', dict(state='failed', device=DEVICE, seed=seed,
                         stage=stage, exit_code=result.returncode, log=str(log)))
                    return result.returncode
                subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/summarize.py')],
                               cwd=ROOT, check=True)
        subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/summarize.py')],
                       cwd=ROOT, check=True)
        save(ROOT/'results/status.json', dict(state='six_seed_runs_complete', device=DEVICE, seeds=SEEDS,
             completed_utc=datetime.now(timezone.utc).isoformat(),
             exact_paper_reproduction_established=False,
             reason='Paper says five seeds; released script lists six without identifying the table subset. See report.txt.'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
