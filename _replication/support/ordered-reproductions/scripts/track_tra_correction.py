"""Record corrected TRA completion without touching original results or gates."""
import fcntl,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
root=Path(__file__).resolve().parents[1]
while subprocess.run(['systemctl','--user','is-active','--quiet','ordered-factor-tra-corrected.service']).returncode==0:
    time.sleep(20)
with open(root/'queue.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    p=root/'queue.json';q=json.loads(p.read_text())
    summary=root/'results/tra-boundary-corrected/summary.json'
    for row in q['papers']:
        if row['paper']=='TRA':
            row['stage']='boundary_corrected_complete_pending_review' if summary.exists() else 'boundary_corrected_run_failed'
            if summary.exists():row['corrected_summary_file']=str(summary.relative_to(root))
    q['updated_utc']=datetime.now(timezone.utc).isoformat()
    tmp=p.with_suffix('.json.correction.tmp');tmp.write_text(json.dumps(q,indent=2,ensure_ascii=False));tmp.replace(p)
