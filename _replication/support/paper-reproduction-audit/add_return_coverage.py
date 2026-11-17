#!/usr/bin/env python3
"""Seed required extension metadata in parked review files; never run paper jobs."""
import datetime
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
protocol = ROOT / 'RETURN_EVALUATION_PROTOCOL.txt'
sha = hashlib.sha256(protocol.read_bytes()).hexdigest()
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
# Root delivered metadata-only turns to these exact tasks. Their writers own them.
skip_writers = {23, 64}
changed, existing, skipped, races = [], [], [], []
for f in sorted((ROOT / 'reviews').glob('*.json')):
    before = f.stat()
    review = json.loads(f.read_text())
    pid = review['paper_id']
    if pid in skip_writers:
        skipped.append(pid)
        continue
    if isinstance(review.get('market_neutral_5bp'), dict):
        existing.append(pid)
        continue
    item = {
        'status': 'pending', 'applicable': None, 'outcome': None,
        'reason': 'Required user extension registered; exact applicability, existing evidence and executable timing await the dedicated worker. No return/significance result is inferred from title, forecast accuracy or paper status.',
        'required_user_extension': True, 'protocol_path': str(protocol),
        'protocol_sha256': sha, 'registered_at': now,
        'evaluation_level': None,
        'exposure_convention': {'long_nav_fraction': .5, 'short_nav_fraction': -.5,
                                'gross_nav_fraction': 1., 'net_nav_fraction': 0.},
        'execution_convention': {'buy_commission_rate': .0005,
                                 'buys_include': ['long purchases', 'short covers'],
                                 'sell_commission_rate': 0.,
                                 'requires_actual_trade_cash_holdings_ledger': True,
                                 'requires_chronological_daily_net_pnl': True},
        'inference_required': ['two-sided 95% HAC mean-return interval and p-value',
                               'time-block bootstrap, at least 1000 resamples where feasible',
                               'frozen strategy construction and all tested variants',
                               'selection/multiplicity evidence and programme-wide context'],
        'gross_metrics': None, 'net_metrics': None, 'results_paths': []}
    review['market_neutral_5bp'] = item
    coverage = review.get('experiment_coverage')
    if coverage is None:
        coverage = {}
        review['experiment_coverage'] = coverage
    if not isinstance(coverage, dict):
        raise ValueError(f'Unexpected coverage type: {f}')
    extensions = coverage.setdefault('required_user_extensions', {})
    extensions['market_neutral_5bp'] = {
        'status': 'pending', 'required': True, 'separate_from_original_paper': True,
        'protocol_path': str(protocol), 'protocol_sha256': sha,
        'details_field': 'market_neutral_5bp'}
    temporary = f.with_name(f.name + f'.return-seed-{os.getpid()}.tmp')
    temporary.write_text(json.dumps(review, indent=2, ensure_ascii=False) + '\n')
    current = f.stat()
    if (current.st_ino, current.st_mtime_ns, current.st_size) != (before.st_ino, before.st_mtime_ns, before.st_size):
        temporary.unlink()
        races.append(pid)
        continue
    temporary.replace(f)
    changed.append(pid)
result = dict(at=now, protocol_sha256=sha, changed=changed, already_present=existing,
              skipped_active_metadata_writers=skipped, skipped_concurrent_changes=races,
              scope='Pending requirement only, not a worker delivery or completed experiment. Original reproduction statuses and existing project files preserved.')
f = ROOT / 'return-coverage-registration.json'
t = f.with_name(f.name + f'.{os.getpid()}.tmp')
t.write_text(json.dumps(result, indent=2) + '\n')
t.replace(f)
print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in result.items()}))
