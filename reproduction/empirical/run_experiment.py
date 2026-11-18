"""Sequential, restartable full Table-2 forecast runs, with per-stage receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from build_panel import DEFAULT, save_json

HERE=Path(__file__).resolve().parent


def main(out):
    out.mkdir(parents=True,exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    if (out/'run_status.json').exists():
        previous=json.loads((out/'run_status.json').read_text())
        process=Path('/proc')/str(previous.get('pid',-1))/'cmdline'
        if previous.get('state')=='running' and process.exists() and b'run_experiment.py' in process.read_bytes():
            raise RuntimeError(f"Coordinator {previous['pid']} is already running; inspect its status/logs")
        with (out/'attempts.jsonl').open('a') as receipt:
            receipt.write(json.dumps(previous)+'\n')
    state={'pid':os.getpid(),'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
           'state':'running','stages_completed':[],
           'planned_neural_fits':40,'epochs_per_fit_max':150,'ensemble_seeds':list(range(521,531)),
           'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    def run(name,args):
        state.update(current_stage=name,state='running');save_json(out/'run_status.json',state)
        print('Starting',name,flush=True)
        with (out/'logs'/f'{name}.log').open('a') as log:
            env=os.environ.copy();env['LFM_EMPIRICAL_OUTPUT']=str(out)
            subprocess.run([sys.executable,*args],cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['stages_completed'].append(name);save_json(out/'run_status.json',state)
    try:
        if not (out/'panel_audit.json').exists():run('build_panel',['build_panel.py','--out',str(out)])
        run('tests',['-m','pytest','-q','test_empirical.py'])
        run('baselines',['train.py','--kind','baselines','--out',str(out)])
        run('evaluate_baselines',['evaluate.py','--out',str(out)])
        for seed in range(521,531):
            for kind in ['mlp','uq_lstm','lstm','uq_mlp']:
                run(f'{kind}_{seed}',['train.py','--out',str(out),'--kind',kind,'--seed',str(seed)])
                run(f'evaluate_{kind}_{seed}',['evaluate.py','--out',str(out)])
        state.update(state='complete',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    except Exception as exc:
        state.update(state='failed',error=str(exc));raise
    finally:save_json(out/'run_status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    main(parser.parse_args().out)
