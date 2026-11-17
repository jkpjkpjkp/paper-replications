"""Resume data, at most two training processes, then baseline and analysis."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parent
MODEL_PYTHON=Path(os.environ.get('LIMT_MODEL_PYTHON', sys.executable))
ANALYSIS_PYTHON=Path(os.environ.get('LIMT_ANALYSIS_PYTHON', sys.executable))


def status(stage,**kwargs):
    temp=ROOT/'run_status.tmp'
    temp.write_text(json.dumps(dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),**kwargs),indent=2)+'\n')
    temp.replace(ROOT/'run_status.json')


def main():
    lock=(ROOT/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    status('building_data')
    subprocess.run([str(MODEL_PYTHON),str(ROOT/'usdata.py')],cwd=ROOT,env=env,check=True)
    status('pretraining_checks')
    with (ROOT/'tests.log').open('w') as log:
        subprocess.run([str(MODEL_PYTHON),'-m','pytest','tests','-q'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    pending=[0,1,2];running={};handles=[]
    while pending or running:
        while pending and len(running)<2:
            seed=pending.pop(0)
            output=ROOT/'results'/f'seed{seed}'
            output.mkdir(parents=True,exist_ok=True)
            log=(output/'train.log').open('a');handles.append(log)
            child=subprocess.Popen([str(MODEL_PYTHON),str(ROOT/'train.py'),'--seed',str(seed)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            running[seed]=child
        status('training',running_seeds=list(running),pending_seeds=pending)
        for seed,child in list(running.items()):
            result=child.poll()
            if result is not None:
                if result:
                    status('failed',seed=seed,returncode=result)
                    raise RuntimeError(f'Seed {seed} failed; other already-running seeds retain their checkpoints')
                del running[seed]
        time.sleep(5)
    for log in handles:log.close()
    status('baseline')
    subprocess.run([str(MODEL_PYTHON),str(ROOT/'baseline.py')],cwd=ROOT,env=env,check=True)
    status('analysis')
    subprocess.run([str(ANALYSIS_PYTHON),str(ROOT/'analyze.py')],cwd=ROOT,env=env,check=True)
    status('source_audit')
    subprocess.run([str(ANALYSIS_PYTHON),str(ROOT/'audit_results.py')],cwd=ROOT,env=env,check=True)
    status('complete')


if __name__=='__main__':main()
