#!/usr/bin/env python3
"""Verified daily CSI Alpha158 tensors, an adaptation of Song et al. (2026).

This data builder does not reconstruct the paper's private minute features.
It uses public Alpha158 definitions and daily next-entry VWAP returns.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ.setdefault('MLFLOW_DISABLE_AGENT_HINT', '1')

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path('/home/jkp/data/csi/qlib/cn_data')
PRICE_FIELDS = ('open', 'high', 'low', 'close', 'vwap')
RAW_FIELDS = PRICE_FIELDS + ('volume', 'factor')
SPLITS = {
    'train': ('2019-01-01', '2023-07-01'),
    'validation': ('2023-07-01', '2024-07-01'),
    'test': ('2024-07-01', '2025-08-01'),
}
_WORKER_CALENDAR = None
_WORKER_FIELDS = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def alpha158_config():
    from qlib.contrib.data.loader import Alpha158DL
    expressions, names = Alpha158DL.get_feature_config()
    expressions = [re.sub(r'\$(open|high|low|close|vwap)\b', r'($\1*$factor)', expr)
                   for expr in expressions]
    if len(names) != 158 or len(set(names)) != 158:
        raise ValueError('Unexpected installed Qlib Alpha158 feature definition')
    return expressions, names


def parse_membership(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        symbol, start, end = line.split()
        result.setdefault(symbol.upper(), []).append((start, end))
    return result


def active_members(membership, date):
    """Use distributed inclusive historical intervals on the decision date."""
    return [symbol for symbol, intervals in membership.items()
            if any(start <= date <= end for start, end in intervals)]


def split_for(date):
    return next((name for name, (start, stop) in SPLITS.items() if start <= date < stop), None)


def label_plan(calendar, index, horizons=10):
    """Entry at t+1, exits t+2,...,t+11; purge all horizons together."""
    date = str(calendar[index])
    split = split_for(date)
    if split is None:
        return None, 'outside_decision_period'
    if index + horizons + 1 >= len(calendar):
        return None, 'missing_label_calendar'
    entry = str(calendar[index + 1])
    exits = [str(calendar[index + 1 + h]) for h in range(1, horizons + 1)]
    if split != 'test' and exits[-1] >= SPLITS[split][1]:
        return None, 'split_boundary_purge'
    return {'date': date, 'split': split, 'entry_date': entry, 'label_exit_dates': exits}, None


def vwap_labels(vwap, factor, index, horizons=10):
    """Input arrays are stocks x dates; return stocks x candidate horizons."""
    entry = index + 1
    end = index + 1 + np.arange(1, horizons + 1)
    base = np.asarray(vwap[:, entry], dtype=np.float64) * np.asarray(factor[:, entry], dtype=np.float64)
    future = np.asarray(vwap[:, end], dtype=np.float64) * np.asarray(factor[:, end], dtype=np.float64)
    good_base = np.isfinite(base) & (base > 0) & (vwap[:, entry] > 0) & (factor[:, entry] > 0)
    good_future = np.isfinite(future) & (future > 0) & (vwap[:, end] > 0) & (factor[:, end] > 0)
    good = good_base[:, None] & good_future
    result = np.full(future.shape, np.nan, dtype=np.float64)
    np.divide(future, base[:, None], out=result, where=good)
    result -= 1
    return result.astype(np.float32)


def verified_source(source, universes, symbols):
    """Re-hash every used raw field, calendar and membership file locally."""
    project = source.parents[1]
    transfer_path = project / 'download_manifest.json'
    transfer = json.loads(transfer_path.read_text())
    paths = ['calendars/day.txt'] + [f'instruments/{u}.txt' for u in universes]
    paths += [f'features/{s.lower()}/{f}.day.bin' for s in sorted(symbols) for f in RAW_FIELDS]
    verified = {}
    for relative in paths:
        path = source / relative
        expected = transfer['files'].get(relative)
        if expected is None or not path.exists():
            raise ValueError('Required source file absent from transfer receipt: ' + relative)
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError('Source file checksum mismatch: ' + relative)
        verified[relative] = expected['sha256']
    return {'revision': transfer['revision'], 'archive_sha256': transfer['archive_sha256'],
            'transfer_manifest_sha256': sha(transfer_path), 'verified_files': verified,
            'source_page': transfer['source_page'],
            'quality_scope': 'Local bytes match pinned publisher transfer; vendor values and constituent accuracy unverified'}


def init_worker(source, dates):
    global _WORKER_CALENDAR, _WORKER_FIELDS
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=source, region=REG_CN, kernels=1, logging_level=40)
    _WORKER_CALENDAR = pd.DatetimeIndex(dates)
    expressions, _ = alpha158_config()
    _WORKER_FIELDS = expressions + ['$' + f for f in RAW_FIELDS]


def extract_instrument(symbol):
    from qlib.data import D
    from qlib.data.cache import H
    dates = _WORKER_CALENDAR
    frame = D.features([symbol], _WORKER_FIELDS, start_time=str(dates[0].date()),
                       end_time=str(dates[-1].date()), freq='day')
    if len(frame):
        frame = frame.xs(symbol, level='instrument').reindex(dates)
        values = frame.to_numpy(dtype=np.float32)
    else:
        values = np.full((len(dates), len(_WORKER_FIELDS)), np.nan, dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    # A positive as-provided factor is required for each adjusted feature date.
    factor = values[:, 158 + RAW_FIELDS.index('factor')]
    values[~(np.isfinite(factor) & (factor > 0)), :158] = np.nan
    H.clear()
    return values[:, :158], values[:, 158:]


def record_reusable(folder, record, signature):
    return (record.get('content_signature') == signature and
            (folder / record['file']).exists() and
            (folder / record['file']).stat().st_size == record['bytes'] and
            sha(folder / record['file']) == record['sha256'])


def build(args):
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.build.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def status(phase, **extra):
        value = {'phase': phase, 'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
        save_json(output / 'build_status.json', value)
        print(json.dumps(value), flush=True)
    universes = args.universes.split(',')
    unknown = set(universes) - {'csi300', 'csi500', 'csi1000'}
    if unknown:
        raise ValueError('Unsupported universes: ' + str(unknown))
    calendar = np.array((source / 'calendars/day.txt').read_text().splitlines(), dtype='U10')
    memberships = {u: parse_membership(source / f'instruments/{u}.txt') for u in universes}
    decisions = [i for i, d in enumerate(calendar) if split_for(str(d)) is not None]
    if args.dates:
        wanted = set(args.dates.split(','))
        decisions = [i for i in decisions if str(calendar[i]) in wanted]
        if len(decisions) != len(wanted):
            raise ValueError('Requested engineering date outside decision calendar')
    symbols = sorted({s for membership in memberships.values() for s, intervals in membership.items()
                      if any(a <= str(calendar[decisions[-1]]) and b >= str(calendar[decisions[0]]) for a, b in intervals)})
    status('verifying_source', instruments=len(symbols))
    provenance = verified_source(source, universes, symbols)
    expressions, feature_names = alpha158_config()
    import qlib
    specification = {'builder_sha256': sha(__file__), 'qlib_version': qlib.__version__, 'source': provenance,
                     'universes': universes, 'features': feature_names,
                     'feature_expressions': expressions, 'splits': SPLITS,
                     'minimum_stocks': args.minimum_stocks, 'candidate_horizons': list(range(1, 11)),
                     'adjustment': 'Every price reference uses positive source price * positive cumulative factor; raw volume retained',
                     'adjustment_caveat': 'Factor convention inferred from provider source and corporate-action invariants; vendor accuracy unverified',
                     'entry_exit': 'entry t+1; exit t+h+1; exchange-session offsets, h=1..10',
                     'membership': 'historical inclusive intervals on decision date, never future membership',
                     'stock_filter': 'positive current close,vwap,volume,factor and finite final label; missing intermediate labels masked',
                     'engineering_dates': args.dates}
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    reports = {}
    for universe in universes:
        folder = output / universe
        folder.mkdir(exist_ok=True)
        reports[universe] = {'complete': False, 'scope': 'Daily CSI adaptation, not original minute paper experiment',
                            'content_signature': signature, 'configuration': specification,
                            'features': feature_names, 'records': [], 'excluded': [],
                            'expected_decision_dates': len(decisions)}
    # Query only needed decision history and the final eleven-session label tail.
    first, last = decisions[0], min(decisions[-1] + 11, len(calendar) - 1)
    query_calendar = calendar[first:last + 1]
    day_offset = {str(d): i for i, d in enumerate(query_calendar)}
    symbol_offset = {s: i for i, s in enumerate(symbols)}
    with tempfile.TemporaryDirectory(prefix='.feature-stage-', dir=output) as stage:
        feature_map = np.lib.format.open_memmap(Path(stage) / 'features.npy', mode='w+', dtype=np.float32,
                                                shape=(len(symbols), len(query_calendar), 158))
        raw_map = np.lib.format.open_memmap(Path(stage) / 'raw.npy', mode='w+', dtype=np.float32,
                                            shape=(len(symbols), len(query_calendar), len(RAW_FIELDS)))
        status('extracting_alpha158', completed_instruments=0, instruments=len(symbols))
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                                 initargs=(str(source), query_calendar.tolist())) as executor:
            for i, (features, raw) in enumerate(executor.map(extract_instrument, symbols, chunksize=1)):
                feature_map[i] = features
                raw_map[i] = raw
                if (i + 1) % 50 == 0 or i + 1 == len(symbols):
                    status('extracting_alpha158', completed_instruments=i + 1, instruments=len(symbols))
        for date_count, calendar_index in enumerate(decisions):
            date = str(calendar[calendar_index])
            plan, reason = label_plan(calendar, calendar_index)
            index = day_offset[date]
            for universe in universes:
                report = reports[universe]
                folder = output / universe
                if reason:
                    report['excluded'].append({'date': date, 'reason': reason})
                    continue
                ids = [s for s in active_members(memberships[universe], date) if s in symbol_offset]
                positions = np.array([symbol_offset[s] for s in ids], dtype=np.int64)
                raw = raw_map[positions, index:index + 12, :]
                vwap = raw[:, :, RAW_FIELDS.index('vwap')]
                factor = raw[:, :, RAW_FIELDS.index('factor')]
                y = vwap_labels(vwap, factor, 0)
                current = raw[:, 0, :]
                positive = np.isfinite(current) & (current > 0)
                keep = positive[:, [RAW_FIELDS.index(f) for f in ['close', 'vwap', 'volume', 'factor']]].all(1)
                keep &= np.isfinite(y[:, -1])
                if keep.sum() < args.minimum_stocks:
                    report['excluded'].append({'date': date, 'reason': 'insufficient_evaluable_stocks',
                                               'stocks': int(keep.sum())})
                    continue
                path = folder / (date + '.npz')
                receipt_path = path.with_suffix('.receipt.json')
                existing = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
                if existing and record_reusable(folder, existing, signature):
                    report['records'].append(existing)
                    continue
                x = np.asarray(feature_map[positions[keep], index, :])
                y = y[keep]
                instruments = np.array(ids)[keep]
                temp = path.with_suffix('.npz.tmp')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream, X=x, Y=y, instrument=instruments,
                                        date=np.datetime64(date, 'D'), entry_date=np.datetime64(plan['entry_date'], 'D'),
                                        label_exit_dates=np.array(plan['label_exit_dates'], dtype='datetime64[D]'))
                temp.replace(path)
                record = {**plan, 'file': path.name, 'content_signature': signature, 'sha256': sha(path),
                          'bytes': path.stat().st_size, 'stocks': int(keep.sum()),
                          'excluded_stocks': int(len(ids) - keep.sum()),
                          'missing_feature_values': int(np.isnan(x).sum()),
                          'missing_label_values': int(np.isnan(y).sum())}
                save_json(receipt_path, record)
                report['records'].append(record)
            if (date_count + 1) % 50 == 0 or date_count + 1 == len(decisions):
                status('writing_daily_tensors', completed_dates=date_count + 1, dates=len(decisions),
                       records={u: len(r['records']) for u, r in reports.items()})
    for universe, report in reports.items():
        report['complete'] = not bool(args.dates)
        report['engineering_subset'] = bool(args.dates)
        report['created_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        report['summary'] = {'days': len(report['records']),
                             'by_split': {s: sum(r['split'] == s for r in report['records']) for s in SPLITS},
                             'stock_days': sum(r['stocks'] for r in report['records']),
                             'excluded_days': len(report['excluded'])}
        if len(report['records']) + len(report['excluded']) != report['expected_decision_dates']:
            raise RuntimeError('Decision-date accounting mismatch')
        save_json(output / universe / 'manifest.json', report)
    status('complete' if not args.dates else 'engineering_subset_complete', complete=not bool(args.dates),
           summaries={u: r['summary'] for u, r in reports.items()}, content_signature=signature)
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=HERE / 'cache')
    parser.add_argument('--universes', default='csi300,csi500,csi1000')
    parser.add_argument('--minimum-stocks', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dates', help='Comma-separated engineering subset; complete will remain false')
    arguments = parser.parse_args()
    try:
        build(arguments)
    except Exception as error:
        save_json(arguments.output / 'build_status.json', {'phase': 'failed', 'error': str(error),
                  'updated_utc': dt.datetime.now(dt.timezone.utc).isoformat()})
        raise
