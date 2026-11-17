"""Finish the authorized AlphaSAGE, GP and XGBoost runs in sequence."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MLFLOW_DISABLE_AGENT_HINT='1')
status_path = ROOT/'STATUS.json'


def state(stage, status='running', **kwargs):
    payload = dict(status=status, stage=stage, **kwargs)
    status_path.write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload),flush=True)


def run(stage, script, args, completed, log):
    if (ROOT/completed).exists():
        return
    state(stage)
    with (ROOT/log).open('w') as stream:
        subprocess.run([sys.executable,str(ROOT/'reproduction'/script),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    state('waiting_for_alphasage_training', training_progress='results/csi300-seed0/progress.json')
    while not (ROOT/'results/csi300-seed0/summary.json').exists():
        time.sleep(10)
    run('alphasage_adaptive_combination','evaluate.py',
        ['--pool','results/csi300-seed0/pool.json','--output','results/csi300-seed0/adaptive'],
        'results/csi300-seed0/adaptive/summary.json','results/csi300-seed0-adaptive.log')
    run('genetic_programming','train_gp.py',
        ['--seed','0','--population','1000','--generations','40','--output','results/gp-csi300-seed0'],
        'results/gp-csi300-seed0/summary.json','results/gp-csi300-seed0.log')
    run('xgboost','train_xgboost.py',
        ['--seed','0','--rounds','100','--output','results/xgboost-csi300-seed0'],
        'results/xgboost-csi300-seed0/summary.json','results/xgboost-csi300-seed0.log')
    subprocess.run([sys.executable,str(ROOT/'reproduction/report.py')],cwd=ROOT,env=env,check=True)
    state('finished',status='complete',report='results/RUN_REPORT.txt',comparison='results/comparison.csv')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        state('failed',status='failed',error=str(error))
        raise
