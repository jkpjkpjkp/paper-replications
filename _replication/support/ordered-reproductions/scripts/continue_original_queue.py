"""Track CPU DoubleAdapt and GPU TRA after the user's idle-GPU request."""
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


jobs={'DoubleAdapt':('ordered-factor-doubleadapt.service','doubleadapt-original'),
      'TRA':('ordered-factor-tra.service','tra-original')}
while jobs:
    finished=[]
    for paper,(service,folder) in jobs.items():
        if subprocess.run(['systemctl','--user','is-active','--quiet',service]).returncode==0:
            continue
        summary=ROOT/'results'/folder/'summary.json'
        if summary.exists():
            update(paper,stage='original_experiment_complete_pending_fidelity_review',summary_file=str(summary.relative_to(ROOT)))
        else:
            update(paper,stage='original_experiment_failed')
        finished.append(paper)
        print(f'{paper}: {"complete, review pending" if summary.exists() else "failed; inspect log"}',flush=True)
    for paper in finished:del jobs[paper]
    if jobs:time.sleep(20)
print('Original jobs stopped. No automatic CZ promotion.',flush=True)
