#!/usr/bin/env python3
"""Run the resumable download and then audit every completed file."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
def status(value):
    value['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
    temporary=ROOT/'run_status.json.tmp'
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(ROOT/'run_status.json')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=range(1,5),default=3,
                        help='Initial parallel workers; reduce by one after each failed attempt')
    args=parser.parse_args(argv)
    for attempt,workers in enumerate(range(args.workers,0,-1),1):
        status({'state':'downloading','attempt':attempt,'workers':workers})
        completed=subprocess.run([sys.executable,str(ROOT/'download_minutes.py'),'--workers',str(workers)],cwd=ROOT)
        if completed.returncode==0:break
        status({'state':'retry_pending' if workers>1 else 'failed','attempt':attempt,
                'workers':workers,'exit_code':completed.returncode})
        if workers==1:return completed.returncode
        time.sleep(30)
    status({'state':'validating'})
    checked=subprocess.run([sys.executable,str(ROOT/'validate_download.py')],cwd=ROOT)
    report=json.loads((ROOT/'validation.json').read_text()) if (ROOT/'validation.json').exists() else {}
    complete=checked.returncode==0 and report.get('complete',False)
    status({'state':'complete' if complete else 'validation_failed',
            'exit_code':checked.returncode,'complete':report.get('complete',False)})
    return 0 if complete else (checked.returncode or 1)


if __name__=='__main__':
    sys.exit(main())
