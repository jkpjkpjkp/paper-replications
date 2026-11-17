"""Finite completion chain: verify finished runs and refresh six-paper report."""
import json,subprocess,time,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
names=['TRA','AlphaSAGE','AlphaPROBE']
while True:
    pending=[];failed=[]
    for name in names:
        out=HERE/'results'/name
        if (out/'evaluation/report.json').exists():
            if name in ['AlphaSAGE','AlphaPROBE'] and not (out/'ols-verification.json').exists():
                python=HERE.parent/(name.lower()+'-reproduction')/'.venv/bin/python'
                subprocess.run([str(python),str(HERE/'verify_pool.py'),name],check=True)
            if not (out/'evaluation/verification.json').exists():
                subprocess.run([sys.executable,str(HERE/'verify_monthly.py'),str(out)],check=True)
        else:
            pending.append(name)
            unit='protocol-six-'+name.lower()+'-20261116.service'
            result=subprocess.run(['systemctl','--user','is-active',unit],capture_output=True,text=True)
            if result.stdout.strip()!='active':failed.append(name)
    subprocess.run([sys.executable,str(HERE/'summarize.py')],check=True)
    (HERE/'completion.json').write_text(json.dumps(dict(stage='complete' if not pending else 'failed' if failed else 'running',pending=pending,failed=failed,updated=time.time()),indent=2))
    if not pending or failed:break
    time.sleep(30)
