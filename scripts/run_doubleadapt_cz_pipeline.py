"""Full gated CZ training, followed by the pre-specified portfolio/HAC evaluation."""
import fcntl
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/doubleadapt-cz'
lock=open(ROOT/'queue.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def update(stage):
    path=ROOT/'queue.json';q=json.loads(path.read_text())
    for row in q['papers']:
        if row['paper']=='DoubleAdapt':
            row.update(full_cz_stage=stage,cz_service='ordered-factor-doubleadapt-cz.service',
                       cz_status_file='results/doubleadapt-cz/status.json',cz_log='logs/doubleadapt-cz.log')
    q['cz_plan']['full_training_started']=True
    q['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=path.with_suffix('.json.pipeline.tmp');tmp.write_text(json.dumps(q,indent=2,ensure_ascii=False));tmp.replace(path)
update('full_training_running')
try:
    subprocess.run([str(ROOT/'.venv-tra/bin/python'),'-u',str(ROOT/'scripts/run_doubleadapt_cz.py'),'--device','cuda'],check=True)
    update('evaluation_running')
    subprocess.run([str(ROOT/'.venv-tra/bin/python'),'-u',str(ROOT/'scripts/evaluate_cz.py'),
        '--scores',str(OUT/'scores.csv'),'--output',str(OUT/'evaluation'),'--cost-bps','10','--hac-lags','6'],check=True)
    update('full_training_and_evaluation_complete')
    p=OUT/'status.json';s=json.loads(p.read_text());s['stage']='full_training_and_evaluation_complete'
    s['evaluation']='evaluation/report.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(p)
except Exception as exc:
    update('failed_inspect_cz_log')
    (OUT/'pipeline-error.json').write_text(json.dumps(dict(error=repr(exc),time_utc=datetime.now(timezone.utc).isoformat()),indent=2))
    raise
