"""Wake isolated read-only Codex checks, serially, with report/snapshot continuity."""
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'monitor';OUT.mkdir(exist_ok=True)
if (OUT/'cancelled.json').exists():
    raise SystemExit('Scheduled Codex checks were cancelled by the user; no check started.')
network=Path('/home/jkp/.config/factor-status/network.json')
if network.exists():
    allowed={'HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','NO_PROXY','http_proxy','https_proxy','all_proxy','no_proxy'}
    os.environ.update({k:v for k,v in json.loads(network.read_text()).items() if k in allowed})
lock=open(OUT/'monitor.lock','w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Existing check still running; skipped duplicate')
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run=OUT/'runs'/stamp;run.mkdir(parents=True)

def command(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=30)
        return dict(returncode=p.returncode,output=(p.stdout+p.stderr)[-18000:])
    except Exception as exc:return dict(error=repr(exc))

snapshot=dict(time_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
    blocker_refresh=command(['/usr/bin/python',str(ROOT/'scripts/update_blocker_report.py')]),
    status=command(['/usr/bin/python',str(ROOT/'scripts/status.py')]),
    gpu=command(['/usr/bin/nvidia-smi']),
    memory=command(['/usr/bin/free','-h']),disk=command(['/usr/bin/df','-h',str(ROOT)]))
for name in ['doubleadapt','tra','queue','doubleadapt-cz','tra-corrected','doubleadapt-diagnostic']:
    unit=f'ordered-factor-{name}.service'
    snapshot[unit]=command(['/usr/bin/systemctl','--user','show',unit,'-p','ActiveState','-p','SubState',
        '-p','MainPID','-p','CPUUsageNSec','-p','MemoryCurrent','-p','Result','-p','ExecMainStatus'])
for name in ['doubleadapt-original','tra-original','doubleadapt-cz','tra-boundary-corrected','doubleadapt-online-lr-diagnostic']:
    log=ROOT/'logs'/f'{name}.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-30000))
            snapshot[name+'_log_tail']='\n'.join(f.read().decode(errors='replace').replace('\r','\n').splitlines()[-45:])
(run/'snapshot.json').write_text(json.dumps(snapshot,indent=2))
prompt=(OUT/'prompt.txt').read_text()+'\n\nFRESH SNAPSHOT:\n'+json.dumps(snapshot,indent=2)
if (OUT/'latest.txt').exists():prompt+='\n\nPREVIOUS REPORT:\n'+(OUT/'latest.txt').read_text()[-18000:]
if (OUT/'latest-snapshot.json').exists():prompt+='\n\nPREVIOUS SNAPSHOT:\n'+(OUT/'latest-snapshot.json').read_text()[-30000:]
args=['/home/jkp/.npm-global/bin/codex','-a','never','-s','read-only','exec']
threadfile=OUT/'thread-id.txt'
args+=['--skip-git-repo-check','--json','--output-last-message',str(run/'report.txt')]
args+=['-']
started=time.monotonic()
with (run/'events.jsonl').open('w') as events,(run/'stderr.log').open('w') as error:
    child=subprocess.run(args,input=prompt,text=True,cwd=ROOT.parent,stdout=events,stderr=error)
for line in (run/'events.jsonl').read_text().splitlines():
    try:event=json.loads(line)
    except json.JSONDecodeError:continue
    if event.get('type')=='thread.started':
        threadfile.write_text(event['thread_id']+'\n')
receipt=dict(time_shanghai=snapshot['time_shanghai'],exit_code=child.returncode,
             elapsed_seconds=time.monotonic()-started,run=str(run),
             thread_id=threadfile.read_text().strip() if threadfile.exists() else None)
(run/'receipt.json').write_text(json.dumps(receipt,indent=2))
(OUT/'last-run.json').write_text(json.dumps(receipt,indent=2))
if child.returncode==0 and (run/'report.txt').exists():
    tmp=OUT/'latest.txt.tmp';tmp.write_text((run/'report.txt').read_text());tmp.replace(OUT/'latest.txt')
    (OUT/'latest-snapshot.json').write_text(json.dumps(snapshot,indent=2))
    print((run/'report.txt').read_text(),flush=True)
else:
    print(f'Codex check failed; inspect {run}',flush=True)
raise SystemExit(child.returncode)
