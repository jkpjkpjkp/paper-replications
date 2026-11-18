#!/usr/bin/env python3
"""Keep building caches, then run the guarded five-seed experiment automatically."""
import argparse
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def state(name,**fields):
    value={'state':name,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**fields}
    temp=HERE/'run_status.json.tmp';temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(HERE/'run_status.json')
    print(json.dumps(value),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-python',type=Path,default=Path('/home/jkp/data/alphapareto-reproduction/.venv/bin/python'))
    args=parser.parse_args()
    lock=(HERE/'.coordinator.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state('building_available_tensors')
        subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'build_tensors.py')],check=True,cwd=ROOT)
        tensors=json.loads((HERE/'tensor_status.json').read_text())
        audit=json.loads((ROOT/'validation.json').read_text())
        download=json.loads((ROOT/'run_status.json').read_text())
        if tensors['complete'] and audit['complete']:break
        if download['state'] in ['failed','validation_failed']:
            state('data_recovery_needed',download_status=download,tensors=tensors['summary']);return 1
        state('waiting_for_training_history',tensors=tensors['summary'],download_state=download['state'])
        # Only the supervising background process waits here. Every cycle writes
        # an explicit state; it never substitutes smoke output for primary results.
        for _ in range(3):time.sleep(60)
    state('checking_engineering_invariants')
    subprocess.run([str(ROOT/'.venv/bin/python'),str(HERE/'verify_engineering.py'),'--features','--download'],check=True)
    subprocess.run([str(args.model_python),str(HERE/'verify_engineering.py'),'--model'],check=True)
    state('training_five_seed_lstm_comparison')
    subprocess.run([str(args.model_python),str(HERE/'train_lstm.py')],check=True,cwd=ROOT)
    state('rendering_results')
    subprocess.run([str(args.model_python),str(HERE/'plot_results.py')],check=True,cwd=ROOT)
    metrics=json.loads((HERE/'results/metrics.json').read_text())
    state('complete',metrics_file=str(HERE/'results/metrics.json'),comparison=metrics['five_seed_mean_by_method'])
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        state('failed',error=str(exc));raise
