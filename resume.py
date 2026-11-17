#!/usr/bin/env python3
"""Resume the verified US download and the configured paper experiment.

The public daily demo and the paper's minute experiment are different protocols.
This entry point controls the latter. It never starts the older, duplicate US
model coordinator and never changes another job's GPU allocation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / 'bars').resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def process(pid_file, script):
    """Reject stale PID files and PID reuse, including zombie processes."""
    try:
        pid = int(pid_file.read_text().strip())
        proc = Path('/proc') / str(pid)
        if proc.joinpath('stat').read_text().split(') ', 1)[1].startswith('Z '):
            return None
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if str(script.resolve()).encode() not in argv:
            return None
        return {'pid': pid, 'argv': [arg.decode() for arg in argv if arg]}
    except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError):
        return None


def snapshot():
    return {
        'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'download_process': process(SOURCE / 'download.pid', SOURCE / 'run_full_download.py'),
        'pipeline_process': process(ROOT / 'reproduction/pipeline.pid', ROOT / 'reproduction/supervise.py'),
        'download': read(SOURCE / 'status.json'),
        'audit': {key: read(SOURCE / 'validation.json').get(key) for key in ['complete', 'summary']},
        'pipeline': read(ROOT / 'reproduction/pipeline_status.json'),
        'training': read(ROOT / 'reproduction/results/status.json'),
        'financial_complete': read(ROOT / 'reproduction/results/summary.json').get('complete', False),
    }


def start(python, script, pid_file, log_file, *args, device='cpu'):
    environment = {**os.environ,
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                   'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'}
    if device == 'cpu':
        environment['CUDA_VISIBLE_DEVICES'] = ''
    elif environment.get('CUDA_VISIBLE_DEVICES') == '':
        environment.pop('CUDA_VISIBLE_DEVICES')
    with log_file.open('a') as log:
        child = subprocess.Popen([str(python), '-u', str(script), *args],
                                 cwd=script.parent, env=environment,
                                 stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    # A new script can publish its own richer status after it acquires its lock.
    pid_file.write_text(str(child.pid) + '\n')
    return child.pid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true', help='Read status without starting jobs')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'],
                        default=read(ROOT / 'request_scope.json').get('execution_device', 'cpu'),
                        help='Training device; defaults to the saved request scope')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1,
                        help='Missing-data workers; one preserves capacity for other WRDS jobs')
    parser.add_argument('--backbones', default='LSTM,GRU,TCN',
                        help='Sequential independent US backbone runs; default reuses the existing public models')
    args = parser.parse_args()
    with (ROOT / '.resume.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = snapshot()
        started = {}
        names = args.backbones.split(',')
        if set(names) - {'LSTM', 'GRU', 'TCN'} or len(names) != len(set(names)):
            raise ValueError('Backbones must be unique LSTM,GRU,TCN names')
        def finished(name):
            folder = ROOT / 'reproduction/results'
            if name != 'LSTM':
                folder = folder / 'backbones' / name.lower()
            return read(folder / 'summary.json').get('complete', False)
        if not args.status and not all(finished(name) for name in names):
            if not state['audit']['complete'] and state['download_process'] is None:
                started['download'] = start(SOURCE / '.venv/bin/python', SOURCE / 'run_full_download.py',
                                            SOURCE / 'download.pid', SOURCE / 'download.log',
                                            '--workers', str(args.workers))
            if state['pipeline_process'] is None:
                started['pipeline'] = start(ROOT / 'reproduction/.venv/bin/python',
                                            ROOT / 'reproduction/supervise.py',
                                            ROOT / 'reproduction/pipeline.pid', ROOT / 'reproduction/pipeline.log',
                                            '--device', args.device, '--threads', '2',
                                            '--backbones', args.backbones, device=args.device)
        print(json.dumps({'started': started, **snapshot()}, indent=2))


if __name__ == '__main__':
    main()
