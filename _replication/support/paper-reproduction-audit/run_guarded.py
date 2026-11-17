#!/usr/bin/env python3
"""Run one foreground job inside a hard-bounded systemd user cgroup.

Exit 75 means deferred. Existing user jobs are never stopped or modified.
The fixed service name and ExitType=cgroup protect the whole descendant lifetime,
even if this admission process is killed. This is not an automatic scheduler.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
UNIT = 'paper-reproduction-compute.service'
RESERVE_GIB = 5.0
DISK_RESERVE_GIB = 20.0
TASK_LIMIT = 128


def memory_available_gib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) / 1024 ** 2


def defer(reason):
    print(json.dumps({'status': 'waiting_resources', 'reason': reason}), flush=True)
    return 75


def atomic_json(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def unit_state():
    result = subprocess.run(
        ['systemctl', '--user', 'show', UNIT, '--property=LoadState,ActiveState,SubState,ControlGroup'],
        text=True, capture_output=True, timeout=15)
    fields = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    if fields.get('LoadState') == 'not-found':
        return fields
    if result.returncode or not fields.get('ActiveState'):
        raise RuntimeError('Cannot verify user-service state: ' + result.stderr.strip())
    return fields


def occupied(state):
    # Activating, deactivating, failed or unknown units are unsafe admission windows.
    return state.get('LoadState') != 'not-found' and state.get('ActiveState') != 'inactive'


def verify_limits_and_exec():
    """Executed inside the service; refuse work unless kernel limits are effective."""
    memory_bytes, threads = int(sys.argv[2]), int(sys.argv[3])
    command = sys.argv[5:]
    if sys.argv[4] != '--' or not command:
        return defer('Invalid internal command')
    try:
        entry = next(line for line in Path('/proc/self/cgroup').read_text().splitlines()
                     if line.startswith('0::'))
        cgroup = Path('/sys/fs/cgroup') / entry.split(':', 2)[2].lstrip('/')
        memory_max = (cgroup / 'memory.max').read_text().strip()
        swap_max = (cgroup / 'memory.swap.max').read_text().strip()
        quota, period = (cgroup / 'cpu.max').read_text().split()
        pids_max = (cgroup / 'pids.max').read_text().strip()
        if memory_max == 'max' or int(memory_max) > memory_bytes:
            raise RuntimeError('MemoryMax is not effective')
        if swap_max != '0':
            raise RuntimeError('MemorySwapMax=0 is not effective')
        if quota == 'max' or int(quota) / int(period) > threads + 1e-9:
            raise RuntimeError('CPUQuota is not effective')
        if pids_max == 'max' or int(pids_max) > TASK_LIMIT:
            raise RuntimeError('TasksMax is not effective')
    except (OSError, ValueError, StopIteration, RuntimeError) as exc:
        return defer(f'Hard resource limits cannot be verified: {exc}')
    if memory_available_gib() < memory_bytes / 1024 ** 3 + RESERVE_GIB:
        return defer('Available RAM changed before execution; 5 GiB reserve would be breached')
    print(json.dumps({'status': 'starting_bounded_job', 'unit': UNIT,
                      'memory_max_bytes': int(memory_max), 'swap_max_bytes': 0,
                      'cpu_quota_cores': int(quota) / int(period),
                      'tasks_max': int(pids_max)}), flush=True)
    state = ROOT / 'active-compute.json'
    info = json.loads(state.read_text())
    info['limits_verified_at_unix'] = time.time()
    info['main_pid'] = os.getpid()
    info['control_group'] = str(cgroup)
    atomic_json(state, info)
    os.execvpe(command[0], command, os.environ)


def valid_env_name(name):
    return bool(name) and (name[0].isalpha() or name[0] == '_') and all(
        character.isascii() and (character.isalnum() or character == '_') for character in name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-id', type=int, required=True)
    parser.add_argument('--memory-gib', type=float, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--gpu-mib', type=int, default=0)
    parser.add_argument('--disk-gib', type=float, default=0,
                        help='Expected peak additional disk use; 20 GiB must remain free')
    parser.add_argument('--cwd', default=os.getcwd())
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or not math.isfinite(args.memory_gib) or args.memory_gib <= 0
            or not math.isfinite(args.disk_gib) or args.disk_gib < 0
            or not 1 <= args.paper_id <= 437 or not 1 <= args.threads <= 2 or args.gpu_mib < 0):
        parser.error('paper ID 1-437, finite positive memory, finite nonnegative disk, 1-2 threads, nonnegative GPU memory and a command are required')
    if not shutil.which('systemd-run') or not shutil.which('systemctl'):
        return defer('systemd user cgroups are required for hard limits; no unbounded fallback')
    cwd = Path(args.cwd).resolve()
    if not cwd.is_dir():
        parser.error('--cwd must be an existing directory')
    if args.memory_gib > 1024:
        return defer('Requested memory exceeds this workstation capacity')
    memory_bytes = max(1, math.ceil(args.memory_gib * 1024 ** 3))
    with (ROOT / 'compute.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return defer('Another new reproduction job owns the shared computation slot')
        try:
            service = unit_state()
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return defer(str(exc))
        if occupied(service):
            return defer(f'{UNIT} still owns a computation cgroup ({service})')
        available = memory_available_gib()
        if available < args.memory_gib + RESERVE_GIB:
            return defer(f'{available:.2f} GiB available; need {args.memory_gib:.2f} GiB plus {RESERVE_GIB:.1f} GiB reserve')
        if shutil.disk_usage(cwd).free / 1024 ** 3 < DISK_RESERVE_GIB + args.disk_gib:
            return defer(f'Need {args.disk_gib:.2f} GiB additional disk plus 20 GiB reserve')
        if args.gpu_mib:
            try:
                rows = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                    text=True, timeout=15).strip().splitlines()
                gpu_index, free_mib = (int(value.strip()) for value in rows[0].split(','))
            except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
                return defer(f'GPU capacity cannot be verified: {exc}')
            if free_mib < args.gpu_mib + 768:
                return defer(f'{free_mib} MiB GPU free; need {args.gpu_mib} MiB plus 768 MiB reserve')
        env = os.environ.copy()
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[key] = str(args.threads)
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_index) if args.gpu_mib else ''
        info = {'paper_id': args.paper_id, 'started_at_unix': time.time(),
                'guard_pid': os.getpid(), 'unit': UNIT,
                'memory_estimate_gib': args.memory_gib, 'memory_max_bytes': memory_bytes,
                'memory_swap_max_bytes': 0, 'threads': args.threads,
                'cpu_quota_percent': args.threads * 100, 'tasks_max': TASK_LIMIT,
                'gpu_mib': args.gpu_mib, 'disk_estimate_gib': args.disk_gib,
                'cwd': str(cwd), 'command': command}
        state = ROOT / 'active-compute.json'
        atomic_json(state, info)
        bounded = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--quiet',
                   '--no-ask-password', '--expand-environment=no', '--service-type=exec',
                   '--unit=' + UNIT, '--working-directory=' + str(cwd),
                   '--property=MemoryMax=' + str(memory_bytes),
                   '--property=MemoryHigh=' + str(max(1, int(memory_bytes * .9))),
                   '--property=MemorySwapMax=0', '--property=CPUQuota=' + str(args.threads * 100) + '%',
                   '--property=TasksMax=' + str(TASK_LIMIT), '--property=KillMode=control-group',
                   '--property=ExitType=cgroup', '--property=OOMPolicy=kill',
                   '--property=TimeoutStopSec=15s', '--property=SendSIGKILL=yes', '--property=Nice=10']
        # Names only: copy environment without exposing secret values in argv.
        bounded += ['--setenv=' + key for key in sorted(env) if valid_env_name(key)]
        bounded += ['--', str(Path(sys.executable).resolve()), str(Path(__file__).resolve()),
                    '--verify-limits-and-exec', str(memory_bytes), str(args.threads), '--'] + command
        child = None
        result_code = None
        try:
            child = subprocess.Popen(bounded, env=env, start_new_session=True)
            def forward(_signum, _frame):
                # Stop only this programme's unit; systemd includes every descendant.
                subprocess.run(['systemctl', '--user', 'stop', '--no-block', UNIT],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
            signal.signal(signal.SIGTERM, forward)
            signal.signal(signal.SIGINT, forward)
            result_code = child.wait()
            latest = json.loads(state.read_text()) if state.exists() else {}
            if latest.get('guard_pid') == os.getpid():
                info.update(latest)
            if result_code and not info.get('limits_verified_at_unix'):
                return defer(f'Bounded service did not admit the command (client exit {result_code}); inspect its error, no unbounded fallback')
            return result_code
        finally:
            try:
                still_occupied = occupied(unit_state())
            except (OSError, RuntimeError, subprocess.SubprocessError):
                still_occupied = True
            if still_occupied:
                info['guard_wait_ended'] = True
                info['client_returncode'] = result_code
                atomic_json(state, info)
            else:
                state.unlink(missing_ok=True)
            info['ended_at_unix'] = time.time()
            info['client_returncode'] = result_code
            info['cgroup_still_occupied'] = still_occupied
            atomic_json(ROOT / 'last-compute-result.json', info)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--verify-limits-and-exec':
        sys.exit(verify_limits_and_exec())
    sys.exit(main())
