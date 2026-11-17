#!/usr/bin/env python3
"""Administrative dispatcher: exact registered threads, max two, no new threads.

This process can run in a transient user service, without autostart. It never executes paper
computation itself; workers must use the fixed run_guarded.py service. Original
and no-cost experiment completeness remain evidence-based worker/supervisor
decisions, not a consequence of a finished turn.
"""
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
from local_control import LocalControl

ROOT = Path(__file__).resolve().parent
LOCK = (ROOT / 'bulk-dispatcher.lock').open('a+')
fcntl.flock(LOCK, fcntl.LOCK_EX | fcntl.LOCK_NB)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(name, default=None):
    p = ROOT / name
    return json.loads(p.read_text()) if p.exists() else default


def save(name, data):
    p = ROOT / name
    t = p.with_name(p.name + f'.{os.getpid()}.tmp')
    t.write_text(json.dumps(data, indent=2) + '\n')
    t.replace(p)


def inventory(event=None):
    cmd = ['python', str(ROOT / 'supervisor_inventory.py')]
    if event:
        cmd.append(json.dumps(event))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if r.returncode:
        raise RuntimeError('Inventory update failed: ' + r.stderr[-1000:])


def unit():
    r = subprocess.run(['systemctl', '--user', 'show', 'paper-reproduction-compute.service',
                        '--property=LoadState,ActiveState,ControlGroup'],
                       capture_output=True, text=True, timeout=15)
    fields = dict(v.split('=', 1) for v in r.stdout.splitlines() if '=' in v)
    if not fields.get('ActiveState'):
        raise RuntimeError('Compute unit state unverifiable')
    return fields, read('active-compute.json', {})


state = read('bulk-dispatcher-state.json', {})
state.update(pid=os.getpid(), status='starting', started_at=now(),
             automatic_restart=False, worker_limit=2,
             transport='existing managed local Unix WebSocket',
             scope='Eligible queued non-heavy papers plus saved resource waits whose gates have cleared; partial and external-blocker follow-ups remain explicit.')
for stale_key in ('stopped_at', 'error', 'note'):
    state.pop(stale_key, None)
state['administrative_service'] = os.environ.get('PAPER_SUPERVISOR_UNIT')
state.setdefault('activation_failures', {})
state.setdefault('resource_retries', {})
state.setdefault('settled', [])
state.setdefault('activations', [])
stop = False


def request_stop(_signum, _frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, request_stop)
signal.signal(signal.SIGINT, request_stop)


def call(c, method, params):
    response = c.call(method, params)
    if response.get('error'):
        raise RuntimeError(json.dumps({'method': method, 'error': response['error']}))
    return response.get('result', {})


def prompt(task):
    return (f"ACTIVATE paper{task['paper_id']}. Read updated WORKER_INSTRUCTIONS.txt, LLM_PREFILTER_PROTOCOL.txt and RETURN_EVALUATION_PROTOCOL.txt in {ROOT}. "
            "Resume bulk reproduction of the exact paper. Pre-filter essential heavy LLM work BEFORE downloads/training; title-only holds are provisional. "
            "Verify full text, observed financial data, official code/data, all original datasets/baselines/seeds/ablations/metrics; reuse existing work and preserve active owners. "
            "PRIMARY user extension: NO-COST market-neutral50%long/50%short,gross1/net0; genuine chronological daily ledger, HAC95%CI/p,>=1000block bootstrap where feasible, all-variant selection/multiplicity. "
            "Do not invent trading signals for nontrading papers; record N/A or exact blocker. "
            "All substantial computation MUST run_guarded.py with accurate RAM/CPU/GPU/disk, synchronous foreground;75=>save exact command and end. "
            f"Atomically reviews/{task['paper_id']:03d}.json with llm_prefilter, market_neutral_nocost, full coverage/evidence and remaining feasible work. "
            "No new threads, duplicate jobs, paid/remote LLMs or test tuning.")


def retry_gate(review, service, last_guard, prior):
    """Admit a saved resource wait only after its actual resource gate clears.

    Never execute its command here. A new guard-start marker or changed command
    is required after a previous retry, avoiding repeated unchanged admissions.
    """
    if service.get('ActiveState') != 'inactive':
        return None
    commands = review.get('next_commands') or []
    for command in commands:
        if not isinstance(command, str):
            continue
        try:
            args = shlex.split(command)
            if str(ROOT / 'run_guarded.py') not in args:
                continue
            def value(flag, default=None):
                return float(args[args.index(flag) + 1]) if flag in args else default
            memory, disk, gpu = value('--memory-gib'), value('--disk-gib'), value('--gpu-mib', 0)
            if memory is None or disk is None or memory <= 0 or disk < 0 or gpu < 0:
                continue
            if value('--paper-id') != review.get('paper_id'):
                continue
        except (ValueError, IndexError):
            continue
        fingerprint = {'command': command, 'previous_guard_start': last_guard.get('started_at_unix')}
        if prior and all(prior.get(k) == v for k, v in fingerprint.items()):
            continue
        meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(meminfo['MemAvailable'].split()[0]) / 1024 ** 2
        if available < memory + 5 or shutil.disk_usage(ROOT).free / 1024 ** 3 < disk + 20:
            continue
        if gpu:
            probe = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=15)
            try:
                if probe.returncode or int(probe.stdout.strip().splitlines()[0]) < gpu + 768:
                    continue
            except (ValueError, IndexError):
                continue
        return {**fingerprint, 'checked_at': now(), 'memory_available_gib': available,
                'reason': 'Fixed compute service now inactive; saved RAM/disk/GPU request passes reserves. Worker must recheck via guard.'}
    return None


c = None
try:
    c = LocalControl(timeout=25)
    registered = read('dispatch.json')['tasks']
    tasks = {t['paper_id']: t for t in registered}
    existing = {v['paper_id'] for v in read('existing-projects.json')['projects'] if v.get('paper_id')}
    # Adopt existing reservations by querying actual state; never start them again.
    for worker in read('supervisor-state.json', {}).get('active_threads', []):
        meta = call(c, 'thread/read', {'threadId': worker['thread_id']})['thread']
        state.setdefault('adopted', []).append({'paper_id': worker['paper_id'], 'thread_id': worker['thread_id'],
                                              'observed_status': meta.get('status'), 'at': now()})
    state['status'] = 'running'
    save('bulk-dispatcher-state.json', state)
    print(json.dumps({'event': 'dispatcher_ready', 'pid': os.getpid(), 'adopted': state.get('adopted', [])[-2:]}), flush=True)
    refreshed = 0
    while not stop:
        reservations = read('supervisor-state.json', {}).get('active_threads', [])
        live = []
        service, owned = unit()
        for worker in reservations:
            tid, pid = worker['thread_id'], worker['paper_id']
            meta = call(c, 'thread/read', {'threadId': tid})['thread']
            runtime = meta.get('status', {}).get('type')
            if runtime == 'active':
                live.append({'paper_id': pid, 'thread_id': tid, 'status': 'active'})
                continue
            turns = call(c, 'thread/turns/list', {'threadId': tid, 'limit': 1, 'itemsView': 'summary'}).get('data', [])
            turn = turns[0] if turns else {}
            if turn.get('status') == 'inProgress':
                live.append({'paper_id': pid, 'thread_id': tid, 'status': 'inProgress'})
                continue
            if service['ActiveState'] != 'inactive' and owned.get('paper_id') == pid:
                live.append({'paper_id': pid, 'thread_id': tid, 'status': 'unit_still_occupied'})
                continue
            review = read(f'reviews/{pid:03d}.json', {})
            outcome = review.get('status', 'missing_review')
            inventory({'event': 'settle', 'paper_id': pid, 'thread_id': tid, 'outcome': outcome,
                       'reason': 'Actual turn finished; review outcome recorded, not inferred as reproduction completeness. All partial/blocked/pending extension work remains outstanding.',
                       'feasible_available_work_remaining': review.get('remaining_feasible_work')})
            item = {'paper_id': pid, 'thread_id': tid, 'review_status': outcome,
                    'turn_status': turn.get('status'), 'at': now(),
                    'no_cost_status': review.get('market_neutral_nocost', {}).get('status')}
            state['settled'].append(item)
            print(json.dumps({'event': 'settled', **item}), flush=True)
            if outcome in {'queued', 'reviewing', 'reproducing', 'missing_review'}:
                state['activation_failures'][str(pid)] = 'Turn finished without final review; explicit follow-up required, no automatic repeat.'
            if pid not in {23, 64}:
                try:
                    call(c, 'thread/archive', {'threadId': tid})
                except RuntimeError as exc:
                    state.setdefault('archive_warnings', []).append({'paper_id': pid, 'error': str(exc), 'at': now()})
        reservations = read('supervisor-state.json', {}).get('active_threads', [])
        available = 2 - len(reservations)
        if available > 0:
            candidates = []
            triage = {v['id']: v for f in ROOT.glob('screen-*.json') for v in json.loads(f.read_text())}
            for pid, task in tasks.items():
                review = read(f'reviews/{pid:03d}.json', {})
                review_status = review.get('status')
                if review_status not in {'queued', 'waiting_resources'} or str(pid) in state['activation_failures']:
                    continue
                if review.get('llm_prefilter', {}).get('bulk_dispatch_eligible') is not True:
                    continue
                if any(v['paper_id'] == pid for v in reservations):
                    continue
                resource_retry = None
                if review_status == 'waiting_resources':
                    resource_retry = retry_gate(review, service, read('last-compute-result.json', {}), state['resource_retries'].get(str(pid)))
                    if resource_retry is None:
                        continue
                text = task['title'].lower()
                observed = bool(re.search(r'stock|portfolio|credit|fraud|launder|asset pric|financial time series|ethereum|anomaly|volatility|prediction market', text))
                rank = (0 if triage.get(pid, {}).get('triage') == 'candidate' and observed else
                        1 if triage.get(pid, {}).get('triage') == 'uncertain' and observed else
                        2 if triage.get(pid, {}).get('triage') == 'candidate' else 3,
                        0 if pid in existing else 1, pid)
                candidates.append(((0 if resource_retry else 1, *rank), task, resource_retry))
            candidates.sort(key=lambda x: x[0])
            for _, task, resource_retry in candidates[:available]:
                pid, tid = task['paper_id'], task['thread_id']
                try:
                    meta = call(c, 'thread/read', {'threadId': tid})['thread']
                    if meta.get('status', {}).get('type') == 'active':
                        inventory({'event': 'activate', 'paper_id': pid, 'thread_id': tid,
                                   'reason': 'Found already-active exact thread; adopted without starting another turn.'})
                        continue
                    if '/archived_sessions/' in str(meta.get('path', '')):
                        call(c, 'thread/unarchive', {'threadId': tid})
                    call(c, 'thread/resume', {'threadId': tid})
                    message = prompt(task)
                    if resource_retry:
                        message += ' RESOURCE RESUME: the previously occupied fixed compute service is now inactive and your saved request passes current reserves. Inspect completed artifacts, then resume the exact saved guarded stage; do not rerun finished work. If the guard returns75 again, save and end.'
                        state['resource_retries'][str(pid)] = resource_retry
                        save('bulk-dispatcher-state.json', state)
                    request = {'threadId': tid, 'input': [{'type': 'text', 'text': message}]}
                    result = call(c, 'turn/start', request)
                    turn = result['turn']
                    inventory({'event': 'activate', 'paper_id': pid, 'thread_id': tid,
                               'reason': 'Exact registered thread turn/start accepted via native local control. Required no-cost and LLM-prefilter protocols explicit; no model override/new thread.'})
                    item = {'paper_id': pid, 'thread_id': tid, 'turn_id': turn['id'], 'at': now()}
                    state['activations'].append(item)
                    save(f'control-receipts/bulk-{turn["id"]}.json', {'request': request, 'response': {'turn_id': turn['id'], 'status': turn.get('status')}})
                    print(json.dumps({'event': 'activated', **item}), flush=True)
                except Exception as exc:
                    # A failed response may follow admission. Reconcile, never replay.
                    state['activation_failures'][str(pid)] = str(exc)
                    try:
                        current = call(c, 'thread/read', {'threadId': tid})['thread']
                        if current.get('status', {}).get('type') == 'active':
                            inventory({'event': 'activate', 'paper_id': pid, 'thread_id': tid,
                                       'reason': 'Activation response uncertain; actual runtime active, conservatively reserved, no retry.'})
                    except Exception:
                        raise RuntimeError('Unverifiable activation; stop admissions: ' + str(exc))
                    print(json.dumps({'event': 'activation_failure_no_replay', 'paper_id': pid, 'error': str(exc)}), flush=True)
        if time.monotonic() - refreshed > 45:
            inventory()
            refreshed = time.monotonic()
        state.update(status='running', heartbeat_at=now(), active_threads=read('supervisor-state.json', {}).get('active_threads', []),
                     remaining_eligible_queued=sum(read(f'reviews/{pid:03d}.json', {}).get('status') == 'queued' and read(f'reviews/{pid:03d}.json', {}).get('llm_prefilter', {}).get('bulk_dispatch_eligible') is True for pid in tasks))
        save('bulk-dispatcher-state.json', state)
        c.events.clear()
        time.sleep(15)
except Exception as exc:
    state.update(status='stopped_error', error=str(exc), stopped_at=now())
    save('bulk-dispatcher-state.json', state)
    try:
        inventory({'event': 'control_unavailable', 'error': 'Bulk dispatcher stopped safely: ' + str(exc)})
    except Exception:
        pass
    print(json.dumps({'event': 'dispatcher_stopped_error', 'error': str(exc)}), flush=True)
    raise
finally:
    if c:
        c.close()
    if stop:
        state.update(status='stopped_requested', stopped_at=now(), note='Existing worker jobs preserved; no further admissions.')
        save('bulk-dispatcher-state.json', state)
