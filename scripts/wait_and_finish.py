"""Finalize the current suite automatically after every training job succeeds."""
from pathlib import Path
import json
import subprocess
import sys
import time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
status=ROOT/'results/main/suite_status.json'
while True:
    j=json.loads(status.read_text())
    if j['phase']=='failed':raise SystemExit('Training suite failed; inspect its per-seed logs before resuming.')
    if j['phase']=='complete':break
    time.sleep(15)
subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'scripts/finish.py'],cwd=ROOT,check=True)
(ROOT/'results/main/completion.json').write_text(json.dumps(dict(phase='complete',
    report=str(ROOT/'REPORT.md'),finished_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
