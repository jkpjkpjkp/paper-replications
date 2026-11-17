#!/usr/bin/env python3
"""Export month–security–return tables from existing replication inputs.

No raw input is copied. Calendar-month returns are never inferred by relabeling
5/20-day forward predictions. See table_specs.json for per-source timing/units.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPECS = Path(__file__).with_name('table_specs.json')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class Inputs:
    def __init__(self, root):
        self.root = root.resolve()
        self.used = {}

    def file(self, relative):
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError(f'Input specifications must be relative to the collection: {relative}')
        candidates = [self.root / rel, self.root / '.replication-local/artifacts' / rel]
        for candidate in candidates:
            if candidate.is_file():
                actual = candidate.resolve()
                stat = actual.stat()
                self.used[str(actual)] = (relative, stat.st_size, stat.st_mtime_ns)
                return actual
        raise FileNotFoundError(f'Missing shared input: {relative}')

    def provenance(self):
        records = []
        for actual, (relative, size, modified) in self.used.items():
            path = Path(actual)
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != (size, modified):
                raise RuntimeError(f'Input changed while exporting: {relative}')
            digest = sha256(path)
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != (size, modified):
                raise RuntimeError(f'Input changed while hashing: {relative}')
            records.append({'source': relative, 'resolved_path': actual,
                            'bytes': size, 'sha256': digest})
        return records


def months(values, encoding):
    values = pd.Series(values).reset_index(drop=True)
    if values.isna().any():
        raise ValueError('Null source month')
    if encoding == 'yyyymm':
        result = pd.to_datetime(values.astype(str), format='%Y%m', errors='raise').dt.strftime('%Y-%m')
    elif encoding in {'period_ordinal', 'year_month_index'}:
        numeric = pd.to_numeric(values, errors='raise')
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError('Noninteger monthly index')
        offset = 1970 * 12 if encoding == 'year_month_index' else 0
        result = pd.Series([str(pd.Period(ordinal=int(v) - offset, freq='M')) for v in numeric])
    elif encoding == 'date':
        result = pd.to_datetime(values, errors='raise').dt.strftime('%Y-%m')
    else:
        raise ValueError(f'Unknown month encoding {encoding!r}')
    if not result.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError('Invalid calendar month')
    return result


def identifiers(values):
    values = pd.Series(values).reset_index(drop=True)
    if values.isna().any():
        raise ValueError('Null security identifier')
    result = values.astype(str)
    if result.str.strip().eq('').any():
        raise ValueError('Empty security identifier')
    return result


def validate(frame):
    if len(frame) == 0:
        raise ValueError('Empty table')
    if frame[['month', 'firm']].isna().any().any() or frame.duplicated(['month', 'firm']).any():
        raise ValueError('Month/security keys must be nonnull and unique')
    if not frame.month.str.fullmatch(r'\d{4}-(0[1-9]|1[0-2])').all():
        raise ValueError('Invalid calendar-month output key')
    returns = pd.to_numeric(frame['return'], errors='raise')
    if np.isinf(returns).any() or returns.dropna().lt(-1.000001).any():
        raise ValueError('Return is not a finite-or-missing simple decimal return')
    if not frame.return_observed.equals(returns.notna()):
        raise ValueError('return_observed is inconsistent with return availability')
    if 'prediction_score' in frame and np.isinf(frame.prediction_score).any():
        raise ValueError('Infinite prediction score')
    return frame.sort_values(['month', 'firm']).reset_index(drop=True)


def read_frame(path, columns):
    columns = list(dict.fromkeys(columns))
    if path.suffix == '.parquet':
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def monthly(spec, inputs):
    columns = [spec[key] for key in ['month', 'firm', 'return']]
    columns += [spec[key] for key in ['observed', 'formation_month'] if key in spec]
    source = read_frame(inputs.file(spec['source']), columns).reset_index(drop=True)
    values = pd.to_numeric(source[spec['return']], errors='raise').astype(float)
    if 'observed' in spec:
        observed = source[spec['observed']]
        if not observed.dropna().isin([True, False]).all():
            raise ValueError('Observation mask must be Boolean')
        values = values.where(observed.fillna(False).astype(bool))
    out = pd.DataFrame({'month': months(source[spec['month']], spec['month_encoding']),
                        'firm': identifiers(source[spec['firm']]), 'return': values})
    if 'formation_month' in spec:
        out['formation_month'] = months(source[spec['formation_month']], spec['month_encoding'])
    out['return_observed'] = out['return'].notna()
    out = validate(out)
    if 'scores' in spec:
        s = spec['scores']
        cols = [s['month'], s['firm'], s['value']]
        if 'return' in s:
            cols.append(s['return'])
        source_scores = read_frame(inputs.file(s['source']), cols).reset_index(drop=True)
        scores = pd.DataFrame({'month': months(source_scores[s['month']], s.get('month_encoding', spec['month_encoding'])),
                               'firm': identifiers(source_scores[s['firm']]),
                               'prediction_score': pd.to_numeric(source_scores[s['value']], errors='raise')})
        if scores.duplicated(['month', 'firm']).any():
            raise ValueError('Duplicate prediction month/security keys')
        if 'return' in s:
            scores['_score_return'] = source_scores[s['return']].astype(float)
        how = s.get('sample', 'predicted')
        if how == 'predicted':
            out = scores.merge(out, on=['month', 'firm'], how='left', validate='one_to_one', indicator=True)
            if not out.pop('_merge').eq('both').all():
                raise ValueError('Prediction key lacks a realized-return source row')
        elif how == 'all':
            out = out.merge(scores, on=['month', 'firm'], how='left', validate='one_to_one')
        else:
            raise ValueError('Score sample must be predicted or all')
        if '_score_return' in out:
            a, b = out.pop('_score_return'), out['return']
            if not np.allclose(a, b, rtol=1e-7, atol=1e-9, equal_nan=True):
                raise ValueError('Prediction-file realized returns disagree with the source panel')
    return validate(out), {'method': 'declared native calendar-month outcome', 'independent_formula_checks': 0}


def aggregate(dates, firms, values, kind, valid=None, membership=None, allowed_invalid_dates=None):
    dates = np.asarray(dates).astype(str)
    firms = identifiers(firms).to_numpy()
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(dates), len(firms)):
        raise ValueError(f'Expected date-by-security array, got {values.shape}')
    if pd.Index(firms).has_duplicates:
        raise ValueError('Duplicate source security axis')
    parsed = pd.DatetimeIndex(pd.to_datetime(dates, errors='coerce'))
    bad = parsed.isna()
    removed_dates = sorted(set(dates[bad]))
    permitted = set(allowed_invalid_dates or [])
    if bad.any():
        if not set(dates[bad]).issubset(permitted):
            raise ValueError('Unexpected invalid source date')
        # A sentinel may represent an absent terminal target only.
        if np.isfinite(values[bad]).any():
            raise ValueError('Invalid/sentinel date has observed returns')
    keep = ~bad
    dates = parsed[keep]
    values = values[keep]
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None:
        raise ValueError('Source dates must be unique, ordered and timezone-naive')
    if not (dates == dates.normalize()).all():
        raise ValueError('Expected daily dates, not intraday timestamps')
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != (len(parsed), len(firms)):
            raise ValueError('Invalid observation-mask shape')
        values = np.where(valid[keep], values, np.nan)
    values[~np.isfinite(values)] = np.nan
    if kind == 'price':
        values[values <= 0] = np.nan
    elif kind == 'daily_return':
        if np.nanmin(values) < -1.000001:
            raise ValueError('Daily simple return below -100%')
    else:
        raise ValueError('Only prices or one-session realized returns may be aggregated')
    wide = pd.DataFrame(values, index=dates, columns=firms)
    period = dates.to_period('M')
    calendar = pd.period_range(period.min(), period.max(), freq='M')
    counts = wide.groupby(period).count().reindex(calendar, fill_value=0)
    expected = pd.Series(1, index=dates).groupby(period).sum().reindex(calendar, fill_value=0)
    if kind == 'daily_return':
        computed = (1 + wide).groupby(period).prod(min_count=1).reindex(calendar) - 1
        computed = computed.where(counts.eq(expected, axis=0) & counts.gt(0))
    else:
        endpoints = wide.groupby(period).tail(1)
        endpoints.index = endpoints.index.to_period('M')
        endpoints = endpoints.reindex(calendar)
        computed = endpoints / endpoints.shift(1) - 1
        computed = computed.where(endpoints.gt(0) & endpoints.shift(1).gt(0))
    # Neither boundary month is certified complete from a bounded input panel.
    computed = computed.iloc[1:-1]
    computed.index.name = 'month'
    computed.columns.name = 'firm'
    out = computed.stack(future_stack=True).rename('return').reset_index()
    out['month'] = out.month.astype(str)
    out['return_observed'] = out['return'].notna()
    counts.index.name = 'month'
    counts.columns.name = 'firm'
    n = counts.stack(future_stack=True).rename('observations').reset_index()
    n['month'] = n.month.astype(str)
    out = out.merge(n, on=['month', 'firm'], validate='one_to_one')
    out['expected_observations'] = out.month.map({str(k): int(v) for k, v in expected.items()})
    if membership is not None:
        membership = np.asarray(membership, dtype=bool)[keep]
        if membership.shape != values.shape:
            raise ValueError('Invalid membership shape')
        mem = pd.DataFrame(membership, index=dates, columns=firms).groupby(period).tail(1)
        mem.index = mem.index.to_period('M')
        mem = mem.reindex(calendar).shift(1)
        mem.index.name, mem.columns.name = 'month', 'firm'
        m = mem.stack(future_stack=True).rename('eligible_at_formation').reset_index()
        m['month'] = m.month.astype(str)
        out = out.merge(m, on=['month', 'firm'], validate='one_to_one')
        out['eligible_at_formation'] = out.eligible_at_formation.astype('boolean')
    out = validate(out)
    # Independent scalar reference calculation, separate from pandas aggregation.
    checks = []
    for month, firm, actual in out[['month', 'firm', 'return']].iloc[np.linspace(0, len(out)-1, min(12, len(out)), dtype=int)].itertuples(index=False, name=None):
        j = int(np.flatnonzero(firms == firm)[0])
        current = pd.Period(month, freq='M')
        ix = np.flatnonzero(period == current)
        if kind == 'daily_return':
            observed = [float(values[i, j]) for i in ix]
            reference = math.prod(1 + x for x in observed) - 1 if observed and all(math.isfinite(x) for x in observed) else math.nan
        else:
            prev = np.flatnonzero(period == current-1)
            a = float(values[ix[-1], j]) if len(ix) else math.nan
            b = float(values[prev[-1], j]) if len(prev) else math.nan
            reference = a/b-1 if math.isfinite(a) and math.isfinite(b) and b > 0 else math.nan
        if not (math.isnan(reference) and math.isnan(actual)) and not math.isclose(reference, actual, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError(f'Independent scalar monthly check failed: {month}, {firm}')
        checks.append({'month': month, 'firm': firm})
    return out, {'method': 'product(1 + daily_return) - 1' if kind == 'daily_return' else 'exact adjacent month-end price ratio - 1',
                 'excluded_boundary_months': [str(calendar[0]), str(calendar[-1])],
                 'invalid_terminal_dates_removed': removed_dates,
                 'calendar': 'All sessions present in the source date axis; not independently certified as an exchange calendar',
                 'independent_formula_checks': len(checks), 'checked_keys': checks}


def arrays(spec, inputs):
    def read_npy(name):
        return np.load(inputs.file(spec['source'] + '/' + name), mmap_mode='r', allow_pickle=False)
    if spec['kind'] == 'npz':
        with np.load(inputs.file(spec['source']), allow_pickle=False) as source:
            dates = source[spec['dates']].copy()
            firms = source[spec['firms']].copy()
            values = source[spec['values']].copy()
            valid = source[spec['mask']].copy() if 'mask' in spec else None
            membership = source[spec['membership']].copy() if 'membership' in spec else None
    else:
        dates = read_npy(spec.get('dates', 'dates.npy'))
        firms = read_npy(spec.get('firms', 'symbols.npy'))
        values = read_npy(spec['values'])
        valid = read_npy(spec['mask']) if 'mask' in spec else None
        membership = read_npy(spec['membership']) if 'membership' in spec else None
    if 'channel' in spec:
        values = values[..., spec['channel']]
    if spec.get('transpose'):
        values = values.T
        valid = valid.T if valid is not None else None
        membership = membership.T if membership is not None else None
    return aggregate(dates, firms, values, spec['value_kind'], valid, membership, spec.get('allowed_invalid_dates'))


def daily_partitions(spec, inputs):
    """Aggregate observed one-session returns, never forward multi-session labels."""
    calendar = pd.DatetimeIndex(pd.read_parquet(inputs.file(spec['calendar']), columns=[spec['date']])[spec['date']])
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('Invalid source session calendar')
    expected = pd.Series(calendar.to_period('M').astype(str)).value_counts()
    frames, checks, seen_dates = [], [], []
    for relative in spec['sources']:
        raw = read_frame(inputs.file(relative), [spec['date'], spec['firm'], spec['return']]).reset_index(drop=True)
        dates = pd.to_datetime(raw[spec['date']], errors='raise')
        if dates.isna().any() or not dates.isin(calendar).all():
            raise ValueError('Daily source date missing from calendar')
        raw = pd.DataFrame({'date': dates, 'month': dates.dt.strftime('%Y-%m'),
                            'firm': identifiers(raw[spec['firm']]),
                            'value': pd.to_numeric(raw[spec['return']], errors='raise')})
        if raw.duplicated(['date', 'firm']).any() or np.isinf(raw.value).any() or raw.value.dropna().lt(-1.000001).any():
            raise ValueError('Duplicate daily key or invalid one-session return')
        seen_dates.extend([raw.date.min(), raw.date.max()])
        raw['gross'] = 1 + raw.value.astype(float)
        groups = raw.groupby(['month', 'firm'], sort=True)
        out = groups.agg(observed_sessions=('value', 'count'), source_sessions=('date', 'size'))
        out['expected_sessions'] = out.index.get_level_values('month').map(expected)
        if out.expected_sessions.isna().any() or out.source_sessions.gt(out.expected_sessions).any():
            raise ValueError('Daily session coverage exceeds the declared calendar')
        out['return'] = (groups.gross.prod(min_count=1)-1).where(out.observed_sessions.eq(out.expected_sessions))
        out['return_observed'] = out['return'].notna()
        out = out.reset_index()
        # An independent scalar calculation for one observed key in every partition.
        sample = out.iloc[len(out)//2]
        values = raw.loc[raw.month.eq(sample.month) & raw.firm.eq(sample.firm), 'value'].tolist()
        reference = math.prod(1 + float(v) for v in values)-1 if len(values) == sample.expected_sessions and all(math.isfinite(v) for v in values) else math.nan
        actual = sample['return']
        if not (math.isnan(reference) and math.isnan(actual)) and not math.isclose(reference, actual, rel_tol=1e-10, abs_tol=1e-12):
            raise AssertionError('Independent partitioned daily-return check failed')
        checks.append({'month': sample.month, 'firm': sample.firm})
        frames.append(out)
    out = pd.concat(frames, ignore_index=True)
    boundaries = [str(pd.Timestamp(min(seen_dates)).to_period('M')), str(pd.Timestamp(max(seen_dates)).to_period('M'))]
    out = out.loc[~out.month.isin(boundaries)]
    return validate(out), {'method': 'product(1 + observed one-session return) - 1; missing source sessions make month unavailable',
                           'excluded_boundary_months': boundaries, 'independent_formula_checks': len(checks), 'checked_keys': checks,
                           'calendar': 'All sessions in the supplied source calendar; not independently certified as an exchange calendar'}


def qlib_prices(spec, inputs):
    """Read the existing Qlib float32 close bins without building another dataset."""
    prefix = spec['source']
    calendar = pd.DatetimeIndex(pd.read_csv(inputs.file(prefix + '/calendars/day.txt'), header=None)[0])
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError('Invalid shared Qlib calendar')
    intervals = pd.read_csv(inputs.file(prefix + '/instruments/' + spec['universe'] + '.txt'),
                            sep='\t', header=None, names=['firm', 'start', 'end'])
    intervals['start'] = pd.to_datetime(intervals.start, errors='raise')
    intervals['end'] = pd.to_datetime(intervals.end, errors='raise')
    start, end = pd.Timestamp(spec['sample_start']), pd.Timestamp(spec['sample_end'])
    intervals = intervals.loc[intervals.start.le(end) & intervals.end.ge(start)]
    firms = sorted(intervals.firm.unique())
    # Keep a trailing calendar month solely to certify the sample's final month.
    selected = (calendar >= start) & (calendar <= end + pd.offsets.MonthEnd(1))
    dates = calendar[selected]
    rows = np.flatnonzero(selected)
    values = np.full((len(rows), len(firms)), np.nan)
    membership = np.zeros(values.shape, dtype=bool)
    for j, firm in enumerate(firms):
        path = inputs.file(prefix + '/features/' + firm.lower() + '/close.day.bin')
        series = np.fromfile(path, dtype='<f4')
        if len(series) < 2 or not np.isfinite(series[0]) or int(series[0]) != series[0]:
            raise ValueError(f'Invalid Qlib start-index header for {firm}')
        offset = int(series[0])
        if offset < 0 or offset + len(series) - 1 > len(calendar):
            raise ValueError(f'Qlib close series exceeds calendar for {firm}')
        ix = rows - offset
        valid = (ix >= 0) & (ix < len(series)-1)
        values[valid, j] = series[1:][ix[valid]]
        for interval in intervals.loc[intervals.firm.eq(firm)].itertuples():
            membership[:, j] |= (dates >= interval.start) & (dates <= interval.end)
    out, audit = aggregate(dates.astype(str), firms, values, 'price', membership=membership)
    audit['qlib_format'] = 'little-endian float32: first value is calendar start index, remaining values are native $close'
    audit['universe_selection'] = 'security has at least one CSI membership interval intersecting declared sample'
    return out, audit


def export(spec, data_root, output_root):
    inputs = Inputs(data_root)
    if spec['kind'] == 'monthly':
        frame, audit = monthly(spec, inputs)
    elif spec['kind'] == 'daily_partitions':
        frame, audit = daily_partitions(spec, inputs)
    elif spec['kind'] == 'qlib_prices':
        frame, audit = qlib_prices(spec, inputs)
    else:
        frame, audit = arrays(spec, inputs)
    provenance = inputs.provenance()
    dest = output_root / spec['project'] / spec['table']
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / 'month_firm_return.parquet'
    with tempfile.NamedTemporaryFile(dir=dest, suffix='.parquet', delete=False) as temp:
        tmp = Path(temp.name)
    try:
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    metadata = {'schema_version': 1, 'project': spec['project'], 'table': spec['table'],
                'specification': spec, 'rows': len(frame), 'firms': int(frame.firm.nunique()),
                'months': int(frame.month.nunique()), 'first_month': frame.month.min(), 'last_month': frame.month.max(),
                'missing_returns': int(frame['return'].isna().sum()),
                'predictions_available': int(frame.prediction_score.notna().sum()) if 'prediction_score' in frame else 0,
                'return_units': 'simple decimal return: 0.01 = 1%',
                'firm_definition': spec['firm_identifier'],
                'prediction_definition': spec.get('scores', {}).get('meaning', 'not exported: no matching calendar-month prediction selected'),
                'inputs': provenance, 'audit': audit, 'output_sha256': sha256(path),
                'exporter_sha256': sha256(Path(__file__))}
    (dest / 'provenance.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return {key: metadata[key] for key in ['project', 'table', 'rows', 'firms', 'months', 'first_month', 'last_month', 'missing_returns', 'predictions_available']}


def self_test():
    assert months([202012, 202101], 'yyyymm').tolist() == ['2020-12', '2021-01']
    assert months([0, 12], 'period_ordinal').tolist() == ['1970-01', '1971-01']
    assert months([24000, 24012], 'year_month_index').tolist() == ['2000-01', '2001-01']
    dates = ['2020-12-31', '2021-01-04', '2021-01-29', '2021-02-01', '2021-02-26', '2021-03-01']
    daily = np.array([[0, 0], [.1, .1], [-.1, np.nan], [.02, 0], [.03, 0], [0, 0]])
    result, _ = aggregate(dates, ['A', 'B'], daily, 'daily_return')
    r = result.set_index(['month', 'firm'])['return']
    assert math.isclose(r.loc['2021-01', 'A'], -.01, abs_tol=1e-12)
    assert math.isnan(r.loc['2021-01', 'B'])
    assert math.isclose(r.loc['2021-02', 'A'], .0506, abs_tol=1e-12)
    prices = np.array([[100], [110], [np.nan], [120], [125], [130]])
    result, _ = aggregate(dates, ['A'], prices, 'price')
    assert result['return'].isna().all()  # A missing endpoint must not be forward-filled.
    try:
        aggregate(['2020-12-31', '2021-01-04', '2021-01-04'], ['A'], [[0], [0], [0]], 'daily_return')
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate dates accepted')
    try:
        validate(pd.DataFrame({'month': ['2021-01']*2, 'firm': ['A']*2,
                               'return': [0.0]*2, 'return_observed': [True]*2}))
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate month/security keys accepted')
    masked, _ = aggregate(dates, ['A', 'B'], np.nan_to_num(daily), 'daily_return', valid=np.isfinite(daily))
    assert pd.isna(masked.set_index(['month', 'firm']).loc[('2021-01', 'B'), 'return'])
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        pd.DataFrame({'month': [600], 'id': [1], 'ret': [.2], 'observed': [False]}).to_parquet(root/'panel.parquet')
        result, _ = monthly({'source': 'panel.parquet', 'month': 'month', 'month_encoding': 'period_ordinal',
                              'firm': 'id', 'return': 'ret', 'observed': 'observed'}, Inputs(root))
        assert result['return'].isna().all()  # Do not reuse a noncontiguous later outcome.
    print('Passed month encoding, compounding, missingness, endpoint, duplicate-date and observation-mask checks.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', help='Project name, or all')
    parser.add_argument('--data-root', type=Path, default=ROOT)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    catalogue = json.loads(SPECS.read_text())
    specs = catalogue['tables']
    references = catalogue.get('references', [])
    if args.list:
        for spec in specs:
            print(f"{spec['project']}/{spec['table']}: {spec['definition']}")
        for ref in references:
            print(f"{ref['project']}/{ref['table']}: reference to {ref['references']} (no duplicated table)")
        return
    if not args.project:
        parser.error('--project is required unless using --list or --self-test')
    chosen = [s for s in specs if args.project == 'all' or s['project'] == args.project]
    chosen_references = [s for s in references if args.project == 'all' or s['project'] == args.project]
    if not chosen and not chosen_references:
        parser.error(f'No table specification for {args.project!r}; use --list')
    output = (args.output_root or args.data_root / '.replication-local/tables').resolve()
    records, failed = [], []
    for spec in chosen:
        try:
            record = export(spec, args.data_root, output)
            records.append(record)
            print(json.dumps(record), flush=True)
        except Exception as exc:
            failed.append({'project': spec['project'], 'table': spec['table'], 'error': str(exc)})
            print(json.dumps(failed[-1]), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    linked = []
    for ref in chosen_references:
        try:
            target = output / ref['references']
            if not (target / 'provenance.json').exists():
                target_spec = next(s for s in specs if s['project'] + '/' + s['table'] == ref['references'])
                export(target_spec, args.data_root, output)
            provenance = json.loads((target / 'provenance.json').read_text())
            if sha256(target / 'month_firm_return.parquet') != provenance['output_sha256']:
                raise ValueError('Referenced table no longer matches its provenance hash')
            dest = output / ref['project'] / ref['table']
            dest.mkdir(parents=True, exist_ok=True)
            metadata = {'specification': ref, 'reference': '../../' + ref['references'] + '/month_firm_return.parquet',
                        'provenance_reference': '../../' + ref['references'] + '/provenance.json',
                        'referenced_output_sha256': provenance['output_sha256']}
            (dest / 'reference.json').write_text(json.dumps(metadata, indent=2) + '\n')
            linked.append({'project': ref['project'], 'table': ref['table'], 'references': ref['references']})
        except Exception as exc:
            failed.append({'project': ref['project'], 'table': ref['table'], 'error': str(exc)})
    report = {'exported': records, 'references': linked, 'failed': failed}
    (output / ('verification-' + args.project + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
