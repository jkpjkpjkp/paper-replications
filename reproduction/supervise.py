#!/usr/bin/env python3
"""Queue the replication behind the already-running licensed-data downloader.

This process does not start another downloader. It observes its audit, builds
verified tensors, trains all five seeds, then evaluates and refreshes REPORT.
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
import time

from data import save_json
from report import write_report

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def other_replication_running(root):
    # The raw source project has its own concurrently launched coordinator.
    # Give it priority instead of competing for this machine's small GPU.
    source = (root / 'bars').resolve().parent / 'reproduction'
    if source.resolve() == (root / 'reproduction').resolve():
        return False
    pid_file = source / 'reproduction.pid'
    if not pid_file.exists():
        return False
    try:
        pid = int(pid_file.read_text().strip())
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'run_reproduction.py' in command
    except (ValueError, FileNotFoundError, ProcessLookupError):
        return False


def cuda_free_mib():
    """Inspect capacity without creating a CUDA context or changing other jobs."""
    checked = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10)
    if checked.returncode:
        raise RuntimeError('Cannot inspect GPU capacity: ' + checked.stderr.strip())
    return int(checked.stdout.strip().splitlines()[0])


def main(args):
    here = args.root.resolve() / 'reproduction'
    with (here / '.pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return main_locked(args, here)


def main_locked(args, here):
    names = getattr(args, 'backbones', 'LSTM').split(',')
    if not names or set(names) - {'LSTM', 'GRU', 'TCN'} or len(names) != len(set(names)):
        raise ValueError('Backbones must be a unique list drawn from LSTM,GRU,TCN')
    (here / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
    def status(phase, **extra):
        value = {'phase': phase, 'pid': os.getpid(), 'device': args.device, 'backbones': names,
                 'threads': args.threads, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
        save_json(here / 'pipeline_status.json', value)
        write_report(args.root)
        print(json.dumps(value), flush=True)
    last_progress = None
    while True:
        audit = load(args.root / 'validation.json')
        if audit.get('complete') and audit.get('summary', {}).get('integrity_error_count') == 0:
            break
        download = load(args.root / 'status.json')
        supervisor = load(args.root / 'run_status.json')
        if supervisor.get('state') in ['failed', 'validation_failed']:
            status('failed', error='Data supervisor failed; inspect ../download.log and ../validation.json')
            return 1
        current = download.get('verified_sessions', 0)
        if current != last_progress:
            status('waiting_for_complete_data_audit', download_sessions=current,
                   planned_sessions=download.get('planned_sessions', 1655))
            last_progress = current
        time.sleep(args.poll_seconds)
    status('building_tensors')
    built = subprocess.run([sys.executable, str(here / 'build_tensors.py'), '--root', str(args.root),
                            '--require-complete'], cwd=here)
    if built.returncode:
        status('failed', error='Tensor build did not complete', exit_code=built.returncode)
        return built.returncode
    waiting_logged = False
    while other_replication_running(args.root):
        if not waiting_logged:
            status('waiting_for_other_replication', reason='Existing source-project runner has GPU priority')
            waiting_logged = True
        time.sleep(args.poll_seconds)
    completed = []
    for name in names:
        output = here / 'results'
        if name != 'LSTM':
            output = output / 'backbones' / name.lower()
        while True:
            if args.device == 'cuda':
                last_memory = None
                while True:
                    free = cuda_free_mib()
                    if free >= args.minimum_cuda_free_mib:
                        break
                    if last_memory is None or abs(free - last_memory) >= 256:
                        status('waiting_for_gpu_memory', backbone=name, free_cuda_mib=free,
                               required_free_cuda_mib=args.minimum_cuda_free_mib)
                        last_memory = free
                    time.sleep(args.poll_seconds)
            status('training', backbone=name, completed_backbones=completed)
            trained = subprocess.run([sys.executable, str(here / 'train.py'), '--root', str(args.root),
                                      '--device', args.device, '--threads', str(args.threads),
                                      '--backbone', name], cwd=here)
            if trained.returncode == 0:
                break
            failure = load(output / 'status.json')
            if args.device == 'cuda' and failure.get('error_type') == 'OutOfMemoryError':
                status('waiting_to_retry_gpu_memory', backbone=name,
                       reason='GPU capacity changed during the fit; completed checkpoints are retained')
                time.sleep(args.poll_seconds)
                continue
            status('failed', backbone=name, error='Training failed; inspect pipeline.log and backbone status', exit_code=trained.returncode)
            return trained.returncode
        summary = load(output / 'summary.json')
        if not summary.get('complete'):
            status('failed', backbone=name, error='Training exited without a complete financial summary')
            return 1
        completed.append(name)
    status('complete', complete=True, completed_backbones=completed,
           scope='Independent US backbone runs; remaining original-paper experiments are incomplete')
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--poll-seconds', type=int, default=30)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='cpu',
                   help='CPU by default to avoid the existing GPU training job')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--backbones', default='LSTM,GRU,TCN', help='Sequential independent US backbone runs')
    p.add_argument('--minimum-cuda-free-mib', type=int, default=2800,
                   help='Free GPU memory required before launching the measured 390-horizon workload')
    args = p.parse_args()
    try:
        sys.exit(main(args))
    except Exception as error:
        save_json(args.root / 'reproduction/pipeline_status.json', {
            'phase': 'failed', 'error': str(error), 'pid': os.getpid(),
            'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        write_report(args.root)
        raise
