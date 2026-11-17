"""Run the preregistered three seeds on both universes, with resumable jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import os
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--run-name',default='main')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=21)
    a=p.parse_args()
    folder=ROOT/'results'/a.run_name
    folder.mkdir(parents=True,exist_ok=True)
    def job(item):
        universe,seed=item
        status=folder/universe/f'seed{seed}'/'status.json'
        if status.exists() and json.loads(status.read_text()).get('phase')=='complete':
            return dict(universe=universe,seed=seed,exit_code=0,skipped_completed=True)
        cmd=[sys.executable,'-u','scripts/train.py','--universe',universe,'--seed',str(seed),
             '--run-name',a.run_name,'--epochs',str(a.epochs),'--patience',str(a.patience)]
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (folder/f'{universe}-seed{seed}.log').open('a') as log:
            result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
        row=dict(universe=universe,seed=seed,exit_code=result.returncode)
        print(json.dumps(row),flush=True)
        return row
    tasks=[(u,s) for s in range(3) for u in ['csi300','csi500']]
    (folder/'suite_status.json').write_text(json.dumps(dict(phase='running',tasks=tasks,pid=os.getpid()),indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        rows=list(pool.map(job,tasks))
    (folder/'suite_status.json').write_text(json.dumps(dict(phase='complete' if all(r['exit_code']==0 for r in rows) else 'failed',
        results=rows,updated_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    if any(r['exit_code'] for r in rows):raise SystemExit(1)


if __name__=='__main__':main()
