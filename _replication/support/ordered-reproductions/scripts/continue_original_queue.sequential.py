"""Run the next ready paper after DoubleAdapt finishes, retaining strict compute order."""
import fcntl
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = open(ROOT/'queue.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)


def update(paper, **fields):
    p=ROOT/'queue.json'
    data=json.loads(p.read_text())
    for entry in data['papers']:
        if entry['paper']==paper:entry.update(fields)
    data['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(p)


while subprocess.run(['systemctl','--user','is-active','--quiet','ordered-factor-doubleadapt.service']).returncode==0:
    time.sleep(20)
summary=ROOT/'results/doubleadapt-original/summary.json'
if not summary.exists():
    update('DoubleAdapt',stage='original_experiment_failed',notes='Inspect logs/doubleadapt-original.log; next compute job was not started.')
    raise SystemExit('DoubleAdapt did not finish. Queue stopped for diagnosis.')
update('DoubleAdapt',stage='original_experiment_complete_pending_fidelity_review',summary_file=str(summary.relative_to(ROOT)))
update('TRA',stage='original_experiment_running',service='ordered-factor-queue.service',
       status_file='results/tra-original/status.json',log='logs/tra-original.log')
with open(ROOT/'logs/tra-original.log','a') as log:
    result=subprocess.run([str(ROOT/'.venv-tra/bin/python'),'-u',str(ROOT/'scripts/run_tra_original.py')],
                          cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT)
if result.returncode:
    update('TRA',stage='original_experiment_failed',returncode=result.returncode)
    raise SystemExit(result.returncode)
update('TRA',stage='original_experiment_complete_pending_fidelity_review',summary_file='results/tra-original/summary.json')
print('Ready original experiments finished. CZ training remains gated on independent fidelity review.',flush=True)
