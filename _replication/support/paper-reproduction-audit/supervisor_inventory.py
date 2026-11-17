#!/usr/bin/env python3
"""Read-only queue inventory plus atomic supervisor-owned status files.

This never dispatches work or changes a worker review, dispatch.json, or projects.
Thread activation and monitoring are performed by the supervisor via Codex tools.
"""
import collections
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import hashlib
import fcntl

ROOT = Path(__file__).resolve().parent
TERMINAL = {'complete', 'excluded', 'blocked_access', 'blocked_data', 'blocked_method'}
RESUME = 'Resume supervision from SUPERVISOR_INSTRUCTIONS.txt and supervisor-state.json; reconcile actual thread, review and systemd-unit state before activation.'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def write(path, obj):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def return_outcome(item):
    """Classify explicit worker outcomes; never infer significance from a point estimate."""
    status = item.get('status', 'missing')
    if status in {'not_applicable', 'blocked', 'pending', 'running'}:
        return status.replace('_', '-')
    outcome = item.get('outcome') or item.get('result_category')
    if isinstance(outcome, str):
        outcome = outcome.lower().replace('_', '-')
        if outcome in {'positive-significant', 'nominal-only', 'positive-inconclusive', 'nonpositive', 'blocked', 'not-applicable'}:
            return outcome
    return 'complete-unclassified' if status == 'complete' else 'pending'


def variant_summary(rows, categories):
    """Keep evaluated variants visible when other variants block a paper.

    These are explicit worker reports, not new significance adjudications.
    Shared market dates and related variants are not independent experiments.
    """
    totals, papers = collections.Counter(), []
    for row in rows:
        item = row.get('market_neutral_nocost', {})
        table = item.get('full_result_table')
        if not isinstance(table, list) or not table:
            continue
        counts = collections.Counter()
        for variant in table:
            if not isinstance(variant, dict):
                continue
            if variant.get('cost_convention', 'nocost') != 'nocost':
                continue
            metrics = variant.get('metrics', {})
            counts[return_outcome({
                'status': variant.get('status', 'missing'),
                'outcome': variant.get('outcome') or metrics.get('outcome')
            })] += 1
        if not counts:
            continue
        totals.update(counts)
        papers.append({'paper_id': row['paper_id'],
                       'paper_extension_status': item.get('status'),
                       'reported_variant_count': sum(counts.values()),
                       'reported_outcome_counts': dict(counts),
                       'evaluation_level': item.get('evaluation_level'),
                       'review_path': row['review_path'],
                       'results_paths': item.get('results_paths', [])})
    return {'unit': 'reported no-cost strategy/universe/seed/horizon rows; not independent trading days or papers',
            'reported_outcome_counts': {key: totals.get(key, 0) for key in categories},
            'reported_variant_count': sum(totals.values()),
            'evidence_policy': 'Counts retain explicit worker outcomes. Partial paper coverage and programme-wide selection remain unresolved; these counts do not certify profitability.',
            'papers': papers}


def refresh():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    papers = read(ROOT / 'papers.json', [])
    dispatch = read(ROOT / 'dispatch.json', {})
    tasks = {t['paper_id']: t for t in dispatch.get('tasks', [])}
    triage = {p['id']: p for f in sorted(ROOT.glob('screen-*.json')) for p in read(f, [])}
    state = read(ROOT / 'supervisor-state.json', {})
    # Inspect the fixed service even if its former guard process has ended.
    try:
        probe = subprocess.run(['systemctl', '--user', 'show', 'paper-reproduction-compute.service',
                                '--property=LoadState,ActiveState,SubState,ControlGroup'],
                               capture_output=True, text=True, timeout=15)
        unit = dict(line.split('=', 1) for line in probe.stdout.splitlines() if '=' in line)
        if not unit.get('ActiveState'):
            unit['error'] = probe.stderr.strip() or f'exit {probe.returncode}'
    except (OSError, subprocess.SubprocessError) as exc:
        unit = {'error': str(exc)}
    state['compute_service'] = {'checked_at': now, **unit}
    state['active_compute_record'] = read(ROOT / 'active-compute.json')
    if time.time() - state.get('resource_probe_unix', 0) >= 45:
        memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        resource = dict(checked_at=now,
                        memory_available_gib=int(memory['MemAvailable'].split()[0]) / 1024 ** 2,
                        free_disk_gib=shutil.disk_usage(ROOT).free / 1024 ** 3)
        try:
            gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                                 capture_output=True, text=True, timeout=15)
            resource['gpu_free_mib'] = int(gpu.stdout.strip().splitlines()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
            resource['gpu_probe_error'] = str(exc)
        state['resource_snapshot'] = resource
        state['resource_probe_unix'] = time.time()
    active = {t['paper_id']: t for t in state.get('active_threads', [])}
    existing = read(ROOT / 'existing-projects.json', {})
    projects = collections.defaultdict(list)
    for p in existing.get('projects', []):
        if p.get('paper_id') is not None:
            projects[p['paper_id']].append(p)
    owners = {p['paper_id']: p['thread_id'] for p in read(ROOT / 'existing-task-owners.json', {}).get('mappings', [])}
    reviews, errors = {}, []
    for f in sorted((ROOT / 'reviews').glob('*.json')):
        try:
            r = read(f)
            pid = r.get('paper_id')
            if not isinstance(pid, int):
                errors.append(f'{f.name}: missing integer paper_id')
                continue
            reviews[pid] = r
        except (ValueError, OSError) as e:
            errors.append(f'{f.name}: {e}')
    outcomes, flags, rows = {}, [], []
    for p in papers:
        pid = p['id']
        r = reviews.get(pid, {})
        status = r.get('status', 'unregistered' if pid not in tasks else 'registration_pending')
        if pid in active:
            if state.get('control_status') == 'unavailable':
                owned = state.get('active_compute_record') or {}
                effective = ('guarded_job_running_thread_unverified'
                             if unit.get('ActiveState') == 'active' and owned.get('paper_id') == pid
                             else 'thread_state_unverified')
            else:
                effective = 'active_worker'
        else:
            effective = status
        row = dict(paper_id=pid, title=p['title'], status=status, effective_status=effective,
                   thread_id=tasks.get(pid, {}).get('thread_id'), review_path=str(ROOT / 'reviews' / f'{pid:03d}.json') if r else None,
                   triage=triage.get(pid, {}).get('triage'), uses_real_finance_data=r.get('uses_real_finance_data'),
                   feasible_on_machine=r.get('feasible_on_machine'), reproduction_level=r.get('reproduction_level'),
                   existing_owner_thread_id=owners.get(pid), updated_at=r.get('updated_at'))
        for key in ('eligibility_reason', 'missing_requirements', 'next_commands', 'results_paths', 'sources', 'validation', 'paper_comparison', 'experiment_coverage', 'completed_experiments', 'resume_condition', 'market_neutral_5bp', 'market_neutral_nocost', 'llm_prefilter'):
            if key in r:
                row[key] = r[key]
        if status not in {'queued', 'registration_pending', 'unregistered'}:
            outcomes[str(pid)] = row
        if status == 'complete':
            if r.get('uses_real_finance_data') is not True:
                flags.append({'paper_id': pid, 'issue': 'complete requires observed real-finance evidence'})
            for key in ('completed_experiments', 'validation', 'results_paths', 'paper_comparison', 'reproduction_level', 'experiment_coverage'):
                if not r.get(key):
                    flags.append({'paper_id': pid, 'issue': f'complete missing {key}'})
        if status == 'excluded' and not r.get('sources'):
            flags.append({'paper_id': pid, 'issue': 'exclusion needs source evidence'})
        rows.append(row)
    counts = dict(collections.Counter(r['status'] for r in rows))
    effective_counts = dict(collections.Counter(r['effective_status'] for r in rows))
    sentinel = (ROOT / 'DISPATCH_IN_PROGRESS').exists()
    extension_statuses = dict(collections.Counter(r.get('market_neutral_5bp', {}).get('status', 'missing') for r in rows))
    reported_categories = collections.Counter(return_outcome(r.get('market_neutral_5bp', {})) for r in rows)
    categories = ['positive-significant', 'nominal-only', 'positive-inconclusive', 'nonpositive', 'blocked', 'not-applicable', 'pending', 'running', 'complete-unclassified']
    pending_returns = [r['paper_id'] for r in rows if r.get('market_neutral_5bp', {}).get('status') not in {'complete', 'not_applicable', 'blocked'}]
    adjudications = read(ROOT / 'return-evaluation-adjudications.json', {})
    verified_categories = collections.Counter(v.get('outcome') for v in adjudications.get('papers', {}).values() if v.get('verified') is True)
    extension = {
        'required': False, 'role': 'secondary_historical_costed_evidence',
        'protocol_path': str(ROOT / 'RETURN_EVALUATION_PROTOCOL_5BP_2026-11-06.txt'),
        'protocol_sha256': hashlib.sha256((ROOT / 'RETURN_EVALUATION_PROTOCOL_5BP_2026-11-06.txt').read_bytes()).hexdigest(),
        'statuses': extension_statuses,
        'reported_outcome_counts': {key: reported_categories.get(key, 0) for key in categories},
        'supervisor_verified_outcome_counts': {key: verified_categories.get(key, 0) for key in categories},
        'pending_or_running_paper_ids': pending_returns,
        'completed_paper_ids': [r['paper_id'] for r in rows if r.get('market_neutral_5bp', {}).get('status') == 'complete'],
        'propagation': read(ROOT / 'return-evaluation-propagation.json', {}),
        'evidence_policy': 'Worker-reported categories and supervisor adjudications are separate. No significance is inferred from accuracy, positive returns alone, independent seeds on common dates, or incomplete source evidence.',
        'programme_wide_multiplicity': {'status': 'pending_all_reported_variants_and_selection_history',
                                        'scope': 'Collect every tested strategy/universe/seed/horizon and raw/adjusted p-values; no programme-wide positive-significance certification yet.'},
        'papers': [{'paper_id': r['paper_id'], 'thread_id': r['thread_id'],
                    'status': r.get('market_neutral_5bp', {}).get('status', 'missing'),
                    'reported_outcome': return_outcome(r.get('market_neutral_5bp', {})),
                    'details': r.get('market_neutral_5bp', {})} for r in rows]}
    queue = {'updated_at': now, 'protocol_path': extension['protocol_path'],
             'protocol_sha256': extension['protocol_sha256'],
             'worker_limit': 2, 'no_mass_restart': True,
             'already_reviewed_pending_paper_ids': [r['paper_id'] for r in rows if r['status'] != 'queued' and r['paper_id'] in pending_returns],
             'items': [{'paper_id': r['paper_id'], 'thread_id': r['thread_id'],
                        'existing_owner_thread_id': r['existing_owner_thread_id'],
                        'paper_status': r['status'],
                        'extension_status': r.get('market_neutral_5bp', {}).get('status', 'missing'),
                        'applicable': r.get('market_neutral_5bp', {}).get('applicable'),
                        'review_path': r['review_path']} for r in rows]}
    write(ROOT / 'market-neutral-5bp-queue.json', queue)
    primary_statuses = dict(collections.Counter(r.get('market_neutral_nocost', {}).get('status', 'missing') for r in rows))
    primary_categories = collections.Counter(return_outcome(r.get('market_neutral_nocost', {})) for r in rows)
    primary_pending = [r['paper_id'] for r in rows if r.get('market_neutral_nocost', {}).get('status') not in {'complete', 'not_applicable', 'blocked'}]
    primary_variants = variant_summary(rows, categories)
    primary = {
        'required': True, 'primary': True, 'cost_assumptions': 'Idealized zero commissions/fees/spread/slippage/impact/borrow/financing; consistent corporate-action/dividend P&L retained.',
        'protocol_path': str(ROOT / 'RETURN_EVALUATION_PROTOCOL.txt'),
        'protocol_sha256': hashlib.sha256((ROOT / 'RETURN_EVALUATION_PROTOCOL.txt').read_bytes()).hexdigest(),
        'statuses': primary_statuses,
        'reported_outcome_counts': {key: primary_categories.get(key, 0) for key in categories},
        'reported_variant_summary': primary_variants,
        'supervisor_verified_outcome_counts': {key: 0 for key in categories},
        'programme_wide_multiplicity': extension['programme_wide_multiplicity'],
        'evidence_policy': extension['evidence_policy'],
        'pending_or_running_paper_ids': primary_pending,
        'papers': [{'paper_id': r['paper_id'], 'thread_id': r['thread_id'],
                    'status': r.get('market_neutral_nocost', {}).get('status', 'missing'),
                    'reported_outcome': return_outcome(r.get('market_neutral_nocost', {})),
                    'details': r.get('market_neutral_nocost', {})} for r in rows]}
    write(ROOT / 'market-neutral-nocost-queue.json', {'updated_at': now, 'primary': True,
          'protocol_path': primary['protocol_path'], 'worker_limit': 2,
          'items': [{'paper_id': r['paper_id'], 'thread_id': r['thread_id'],
                     'existing_owner_thread_id': r['existing_owner_thread_id'],
                     'paper_status': r['status'],
                     'extension_status': r.get('market_neutral_nocost', {}).get('status', 'missing'),
                     'llm_prefilter': r.get('llm_prefilter', {}),
                     'review_path': r['review_path']} for r in rows]})
    prefilter_counts = dict(collections.Counter(r.get('llm_prefilter', {}).get('status', 'missing') for r in rows))
    held_ids = [r['paper_id'] for r in rows if r.get('llm_prefilter', {}).get('bulk_dispatch_eligible') is False]
    pending_returns = primary_pending
    outstanding = [r['paper_id'] for r in rows if r['status'] not in TERMINAL or r['paper_id'] in active or r['paper_id'] in pending_returns]
    state.update(updated_at=now, heartbeat_at=now, supervisor_thread_id='01a1b0f8-3143-7af0-819a-e40158506629',
                 resume_instruction=RESUME, automatic_restart_scheduler=False, per_paper_outcomes=outcomes,
                 registered_tasks=len(tasks), total_papers=len(papers), dispatch_in_progress=sentinel)
    state['primary_return_field'] = 'market_neutral_nocost'
    dispatcher = read(ROOT / 'bulk-dispatcher-state.json', {})
    dispatcher_pid = dispatcher.get('pid')
    process_present = False
    if isinstance(dispatcher_pid, int):
        try:
            process_present = b'bulk_dispatcher.py' in Path(f'/proc/{dispatcher_pid}/cmdline').read_bytes()
        except OSError:
            pass
    heartbeat_age = None
    try:
        heartbeat_age = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(dispatcher['heartbeat_at'])).total_seconds()
    except (KeyError, TypeError, ValueError):
        pass
    state['live_bulk_dispatcher'] = {
        'pid': dispatcher_pid, 'reported_status': dispatcher.get('status'),
        'administrative_service': dispatcher.get('administrative_service'),
        'process_present': process_present, 'heartbeat_age_seconds': heartbeat_age,
        'live_at_check': process_present and dispatcher.get('status') == 'running' and heartbeat_age is not None and 0 <= heartbeat_age < 120,
        'checked_at': now, 'state_path': str(ROOT / 'bulk-dispatcher-state.json'),
        'automatic_restart': False}
    state['llm_prefilter'] = {'protocol_path': str(ROOT / 'LLM_PREFILTER_PROTOCOL.txt'),
                             'counts': prefilter_counts, 'held_paper_ids': held_ids}
    state['market_neutral_nocost'] = {'protocol_path': primary['protocol_path'],
                                  'protocol_sha256': primary['protocol_sha256'],
                                  'statuses': primary_statuses,
                                  'queue_path': str(ROOT / 'market-neutral-nocost-queue.json'),
                                  'propagation_path': str(ROOT / 'return-evaluation-propagation.json'),
                                  'mandatory_activation_suffix': 'Read RETURN_EVALUATION_PROTOCOL.txt and LLM_PREFILTER_PROTOCOL.txt. PRIMARY market_neutral_nocost:50/50 long-short,gross1,net0,zero costs, genuine chronological daily P&L,HAC/bootstrap and selection/multiplicity. Pre-filter essential heavy LLM before any model work; N/A or exact blockers for unsupported trading tests.'}
    write(ROOT / 'supervisor-state.json', state)
    status = dict(schema_version=1, updated_at=now,
                  status='in_progress' if state.get('control_status') != 'unavailable' else 'incomplete_control_transport_unavailable',
                  supervision_status=state.get('supervision_status', 'supervising'),
                  control_status=state.get('control_status', 'available'), control_error=state.get('control_error'),
                  total_papers=len(papers), registered_tasks=len(tasks), review_files=len(reviews),
                  dispatch_in_progress=sentinel, worker_limit=2,
                  active_threads=list(active.values()), counts=counts, effective_counts=effective_counts,
                  live_bulk_dispatcher=state['live_bulk_dispatcher'],
                  followup_queue=state.get('followup_queue', {}), compute_service=state['compute_service'],
                  active_compute_record=state['active_compute_record'], resource_snapshot=state.get('resource_snapshot'),
                  supervisor_observations=read(ROOT / 'supervisor-observations.json', {}),
                  market_neutral_5bp=extension,
                  market_neutral_nocost=primary, primary_return_field='market_neutral_nocost',
                  llm_prefilter={'counts': prefilter_counts, 'held_paper_ids': held_ids,
                                'provisional_screen_not_final_exclusion': True,
                                'protocol_path': str(ROOT / 'LLM_PREFILTER_PROTOCOL.txt')},
                  outstanding_paper_ids=outstanding, review_errors=errors, evidence_flags=flags,
                  supervisor_thread_id=state['supervisor_thread_id'], heartbeat_at=now,
                  automatic_restart_scheduler=False, resume_instruction=RESUME,
                  coverage_counts={
                      'full_faithful_execution': sum(r['status'] == 'complete' and not any(f['paper_id'] == r['paper_id'] for f in flags) for r in rows),
                      'partial_or_reduced_with_executed_components': sum(r['status'] != 'complete' and bool(r.get('completed_experiments')) for r in rows),
                      'partial_review_status': counts.get('partial', 0),
                      'existing_running': counts.get('existing_running', 0),
                      'excluded': counts.get('excluded', 0),
                      'access_data_method_blocked': sum(counts.get(k, 0) for k in ('blocked_access', 'blocked_data', 'blocked_method')),
                      'waiting_resources': counts.get('waiting_resources', 0),
                      'numerical_agreement_verified': sum(
                          isinstance(r.get('paper_comparison'), dict)
                          and r['paper_comparison'].get('numerical_agreement_verified') is True for r in rows),
                      'numerical_agreement_explicitly_unverified_or_unspecified': sum(
                          not isinstance(r.get('paper_comparison'), dict)
                          or r['paper_comparison'].get('numerical_agreement_verified') is not True for r in rows)},
                  existing_inventory_as_of=existing.get('as_of_utc'),
                  notes=['Triage is provisional, not an eligibility decision.',
                         'Completed worker turns may be blocked or queued; only reviews with experiment evidence count as complete.',
                         'Existing processes and owners are preserved. Active worker slots include research and guarded execution.'],
                  papers=rows)
    if not outstanding and not sentinel and not flags and not errors:
        status['status'] = 'feasible_work_complete_with_documented_outcomes'
    write(ROOT / 'programme-status.json', status)
    text = [f'Paper reproduction programme — {now}',
            f"Programme: {status['status']}; {len(papers)} workbook papers; {len(tasks)} registered tasks; {len(reviews)} reviews.",
            f'Dispatch still arriving: {sentinel}. Reserved worker slots: {len(active)}/2; control: {state.get("control_status", "available")}.',
            'Review outcomes: ' + ', '.join(f'{k}={v}' for k, v in sorted(counts.items())),
            'PRIMARY no-cost equal-long/short extension: ' + ', '.join(f'{k}={v}' for k, v in sorted(primary_statuses.items())),
            'NO-COST outcomes (worker-reported): ' + ', '.join(f'{k}={primary_categories.get(k, 0)}' for k in categories),
            'NO-COST variant outcomes (worker-reported, related rows): ' + ', '.join(f'{k}={v}' for k, v in primary_variants['reported_outcome_counts'].items() if v) + f"; total={primary_variants['reported_variant_count']}. Paper-level blocked status can include evaluated variants.",
            f"Bulk dispatcher live at check: {state['live_bulk_dispatcher']['live_at_check']}; PID {dispatcher_pid}; heartbeat age {heartbeat_age}s; no automatic restart.",
            f"Administrative service: {state['live_bulk_dispatcher']['administrative_service'] or 'foreground tool process'}; paper computation still requires the fixed guarded compute service.",
            'LLM prefilter: ' + ', '.join(f'{k}={v}' for k, v in sorted(prefilter_counts.items())) + '. Provisional holds are skipped by bulk dispatch, not final paper exclusions.',
            'Historical5bp secondary evidence retained; no-cost significance is now primary.',
            'Return inference: HAC95% CI/p-value + block bootstrap + all-variant selection/multiplicity evidence required. No programme-wide significance certification.',
            ('Unverified thread reservations: ' if state.get('control_status') == 'unavailable' else 'Active: ') + (', '.join(f"#{pid} {a['thread_id']}" for pid, a in active.items()) or 'none'),
            'Partial-paper follow-ups: ' + (', '.join(f"#{pid} {work.get('status')}" for pid, work in state.get('followup_queue', {}).items()) or 'none'),
            f'Outstanding: {len(outstanding)}. Evidence flags: {len(flags)}. Review read errors: {len(errors)}.',
            'Existing jobs remain under their existing owners. New substantial runs require run_guarded.py.',
            f"Supervisor: {state['supervisor_thread_id']}; heartbeat {now}. No automatic restart scheduler.",
            f'Resume this exact thread: {RESUME}',
            'Details and exact evidence: programme-status.json; reviews/NNN.json; supervisor-state.json.']
    if state.get('control_status') == 'unavailable':
        mode = 'STOPPED' if state.get('supervision_status', '').startswith('stopped') else 'DEGRADED'
        text.insert(2, 'SUPERVISION ' + mode + ': ' + state.get('control_error', 'Thread-control transport unavailable.'))
        text.insert(3, 'No new activations. Reserved thread states are unverified; local cgroup and review snapshots are recorded separately.')
        if (ROOT / 'supervisor-observations.json').exists():
            text.insert(4, 'Local execution checkpoint: supervisor-observations.json; paper23 CPU pilot ended exit0 at37/100000steps, all175 original full runs remain unexecuted. Canonical review may still say reproducing; no live programme compute unit was present at the checkpoint.')
    for r in rows:
        if r['status'] in TERMINAL | {'waiting_resources', 'existing_running', 'partial'}:
            reason = r.get('eligibility_reason') or r.get('missing_requirements') or ''
            text.append(f"#{r['paper_id']:03d} {r['status']} ({r.get('reproduction_level') or 'level pending'}): {str(reason)[:450]}")
    if errors:
        text.extend(errors)
    write(ROOT / 'STATUS.txt', '\n'.join(text) + '\n')
    candidates = [r for r in rows if r['paper_id'] in tasks and r['status'] in {'queued', 'registration_pending'} and r['paper_id'] not in active and r['paper_id'] not in held_ids]
    def rank(r):
        title = r['title'].lower()
        market = bool(re.search(r'stock|portfolio|credit|fraud|launder|asset pric|financial time series|ethereum|anomaly|volatility', title))
        # Reuse projects while memory is occupied, then bounded market methods.
        return (0 if r['paper_id'] in projects else 1,
                0 if r['triage'] == 'candidate' and market else 1 if r['triage'] == 'uncertain' and market else 2 if r['triage'] == 'candidate' else 3,
                r['paper_id'])
    candidates.sort(key=rank)
    print(json.dumps({k: status[k] for k in ('updated_at', 'registered_tasks', 'review_files', 'dispatch_in_progress', 'counts', 'effective_counts', 'resource_snapshot', 'compute_service', 'evidence_flags', 'review_errors')}, ensure_ascii=False))
    print('RETURN_EXTENSION', json.dumps({'statuses': extension_statuses, 'reported_outcomes': extension['reported_outcome_counts'], 'supervisor_verified': extension['supervisor_verified_outcome_counts']}))
    print('PRIMARY_NOCOST', json.dumps({'statuses': primary_statuses, 'reported_outcomes': primary['reported_outcome_counts'], 'llm_prefilter': prefilter_counts}))
    print('NEXT', [(r['paper_id'], r['triage'], r['title'][:70]) for r in candidates[:6]])


def main():
    if len(sys.argv) > 1:
        event = json.loads(sys.argv[1])
        path = ROOT / 'supervisor-state.json'
        state = read(path, {'active_threads': [], 'events': []})
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        kind = event['event']
        if kind == 'activate':
            if state.get('control_status') == 'unavailable':
                raise SystemExit('Thread-control transport unavailable: reconcile actual state before activation')
            if len(state.get('active_threads', [])) >= 2:
                raise SystemExit('Two-worker limit reached')
            if any(a['paper_id'] == event['paper_id'] or a['thread_id'] == event['thread_id'] for a in state.get('active_threads', [])):
                raise SystemExit('Already active')
            task = next(t for t in read(ROOT / 'dispatch.json')['tasks'] if t['paper_id'] == event['paper_id'])
            if task['thread_id'] != event['thread_id']:
                raise SystemExit('Thread identity differs from dispatch.json')
            state.setdefault('active_threads', []).append(dict(paper_id=event['paper_id'], thread_id=event['thread_id'], activated_at=now))
        elif kind == 'settle':
            state['active_threads'] = [a for a in state.get('active_threads', []) if a['thread_id'] != event['thread_id']]
            if event.get('outcome') == 'partial':
                review = read(ROOT / 'reviews' / f"{event['paper_id']:03d}.json", {})
                state.setdefault('followup_queue', {})[str(event['paper_id'])] = {
                    'paper_id': event['paper_id'], 'thread_id': event['thread_id'],
                    'status': event.get('followup_status', 'queued_followup_not_running'), 'queued_at': now,
                    'feasible_available_work_remaining': event.get('feasible_available_work_remaining'),
                    'reason': event.get('reason'),
                    'missing_requirements': review.get('missing_requirements'),
                    'next_commands': review.get('next_commands', []),
                    'review_path': str(ROOT / 'reviews' / f"{event['paper_id']:03d}.json")}
        elif kind == 'snapshot':
            state.setdefault('thread_snapshots', {}).update(event['snapshots'])
        elif kind == 'control_unavailable':
            state['status'] = 'degraded_control_transport_unavailable'
            state['control_status'] = 'unavailable'
            state['control_error'] = event['error']
            state['control_failure_observed_at'] = now
            state['supervision_status'] = 'local_monitor_only'
            for worker in state.get('active_threads', []):
                worker['current_thread_state'] = 'unverified'
                worker['thread_state_reason'] = 'Control transport unavailable; do not infer a live thread from the saved reservation.'
        elif kind == 'supervision_stopped':
            state['status'] = 'stopped_control_transport_unavailable'
            state['supervision_status'] = 'stopped_control_transport_unavailable'
            state['supervision_stopped_at'] = now
            if event.get('final_control_error'):
                state['control_error'] = event['final_control_error']
        elif kind == 'control_recovered':
            if not event.get('reconciliation_evidence'):
                raise SystemExit('Live thread/review/systemd reconciliation evidence is required')
            state['control_status'] = 'available'
            state['control_error'] = None
            state['status'] = 'supervising'
            state['supervision_status'] = 'supervising'
            state['control_recovered_at'] = now
        else:
            raise SystemExit('Unknown event')
        event['at'] = now
        state.setdefault('events', []).append(event)
        state['updated_at'] = now
        write(path, state)
    refresh()


if __name__ == '__main__':
    with (ROOT / 'supervisor-state.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
