#!/usr/bin/env python3
"""Build/adopt verified daily inputs and run CSI300/500/1000 sequentially."""
import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from report import write_report

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, data):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def builder_alive():
    try:
        pid = int((HERE / 'data_build.pid').read_text().strip())
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return str(HERE / 'data.py').encode() in command
    except (ValueError, FileNotFoundError):
        return False


def main(args):
    universes = args.universes.split(',')
    if set(universes) - {'csi300', 'csi500', 'csi1000'}:
        raise ValueError('Unsupported CSI universe')
    with (HERE / '.run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (HERE / 'run.pid').write_text(str(os.getpid()) + '\n')
        def status(phase, **extra):
            value = {'phase': phase, 'pid': os.getpid(), 'universes': universes,
                     'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
            save(HERE / 'run_status.json', value)
            write_report(HERE)
            print(json.dumps(value), flush=True)
        status('waiting_for_data_build')
        while builder_alive():
            time.sleep(15)
            write_report(HERE)
        complete = all(read(HERE / 'cache' / u / 'manifest.json').get('complete') for u in universes)
        if not complete:
            status('building_data')
            result = subprocess.run([sys.executable, str(HERE / 'data.py'), '--universes', ','.join(universes),
                                     '--workers', '4'], cwd=HERE)
            if result.returncode:
                status('failed', error='CSI data build failed', exit_code=result.returncode)
                return result.returncode
        for universe in universes:
            if read(HERE / 'results' / universe / 'summary.json').get('complete'):
                continue
            status('training', universe=universe)
            result = subprocess.run([sys.executable, str(HERE / 'train.py'), '--universe', universe,
                                     '--device', args.device, '--threads', str(args.threads)], cwd=HERE)
            if result.returncode:
                status('failed', universe=universe, error='CSI training failed', exit_code=result.returncode)
                return result.returncode
            if not read(HERE / 'results' / universe / 'summary.json').get('complete'):
                status('failed', universe=universe, error='Trainer exited without completed summary')
                return 1
            status('universe_complete', universe=universe)
        status('complete', complete=True)
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--universes', default='csi300,csi500,csi1000')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    try:
        sys.exit(main(args))
    except Exception as error:
        save(HERE / 'run_status.json', {'phase': 'failed', 'error': str(error),
                                      'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        write_report(HERE)
        raise
