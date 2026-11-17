#!/usr/bin/env python3
"""Read live processes and existing evidence without launching or rerunning jobs."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
from pathlib import Path
from reproduction.progress import collect_progress, describe

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def inspect(root=ROOT):
    root = Path(root)
    spec = importlib.util.spec_from_file_location('song_resume_status', root / 'resume.py')
    resume = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resume)
    source = (root / 'bars').resolve().parent
    plan = read(source / 'download-plan.json')
    dates = [d['date'] if isinstance(d, dict) else str(d) for d in plan.get('dates', [])]
    present = {d for d in dates if (source / 'bars' / d[:4] / (d + '.parquet')).exists()
               and (source / 'bars' / d[:4] / (d + '.receipt.json')).exists()}
    raw_audit = read(source / 'validation.json')
    jobs = {}
    for name, pid_file, script, status_file in [
        ('download', source / 'download.pid', source / 'run_full_download.py', source / 'run_status.json'),
        ('us_pipeline', root / 'reproduction/pipeline.pid', root / 'reproduction/supervise.py', root / 'reproduction/pipeline_status.json'),
    ]:
        live = resume.process(pid_file, script)
        saved = read(status_file)
        jobs[name] = {'live': live is not None, 'pid': live['pid'] if live else None,
                      'phase': saved.get('phase', saved.get('state', 'unknown')) if live else 'stopped',
                      'last_saved_phase': saved.get('phase', saved.get('state', 'unknown')),
                      'last_saved_utc': saved.get('updated_utc')}
    csi = {}
    for universe in ['csi300', 'csi500', 'csi1000']:
        summary = read(root / 'csi/results' / universe / 'summary.json')
        if summary.get('complete'):
            csi[universe] = {'complete': True, 'scope': summary['scope'],
                            'standard_ic': summary['primary']['standard']['ic']['mean'],
                            'selected_ic': summary['primary']['selected']['ic']['mean'],
                            'paired_ic_bootstrap': summary['paired_ic_bootstrap']}
        else:
            csi[universe] = {'complete': False}
    us_summary = read(root / 'reproduction/results/summary.json')
    verified_progress = collect_progress(root)
    independent_backbones = {}
    for name in ['LSTM', 'GRU', 'TCN']:
        folder = root / 'reproduction/results'
        if name != 'LSTM':
            folder = folder / 'backbones' / name.lower()
        summary = read(folder / 'summary.json')
        independent_backbones[name] = {'complete': verified_progress[name]['financial_complete'],
                                       'status': read(folder / 'status.json'),
                                       'result_path': str(folder / 'summary.json')}
        if verified_progress[name]['financial_complete']:
            independent_backbones[name]['five_seed_metrics'] = summary['five_seed_metrics']
    pipeline = read(root / 'reproduction/pipeline_status.json')
    current_backbone = pipeline.get('backbone', 'LSTM')
    coverage = read(root / 'paper_coverage.json')
    return {
        'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'full_published_results_replicated': False,
        'reason': 'Daily CSI tasks differ from paper minute tasks; full original features/settings unavailable; original experiment matrix incomplete',
        'processes': jobs,
        'us_data': {'planned_sessions': len(dates), 'receipted_sessions_present': len(present),
                    'missing_sessions': [d for d in dates if d not in present],
                    'independent_audit_complete': raw_audit.get('complete', False),
                    'independent_audit_summary': raw_audit.get('summary', {}),
                    'independent_audit_utc': raw_audit.get('created_utc')},
        'us_independent_lstm_complete': verified_progress['LSTM']['financial_complete'],
        'us_training': independent_backbones.get(current_backbone, {}).get('status', {}),
        'independent_us_backbones': independent_backbones,
        'configured_us_backbones': pipeline.get('backbones', []),
        'verified_fit_progress': verified_progress,
        'queued_us_backbones': [name for name in pipeline.get('backbones', [])
                                if jobs['us_pipeline']['live']
                                and not independent_backbones.get(name, {}).get('complete')],
        'daily_csi_adaptations': csi,
        'required_inputs': ['Authorized CSI one-minute OHLC, amount, volume and actual trade count for2018–2025',
                            'Original complete feature definitions, architecture and optimizer configurations',
                            'Missing backtest execution-cost parameters and MTL details'],
        'coverage_artifact': 'paper_coverage.json' if coverage else None,
        'not_counted_as_replication': ['Synthetic mechanism checks', 'Passed code tests', 'Queued or running jobs', 'Published reference values copied into reports'],
    }


def write_markdown(root, status):
    raw = status['us_data']
    def backbone_state(name):
        return describe(status['verified_fit_progress'][name])
    lines = ['# Published-result replication status', '',
             '**The paper’s published results have not been fully replicated.**', '',
             'Checked: ' + status['checked_utc'], '',
             '| Experiment | Current evidence |', '|---|---|',
             '| Daily CSI300/500/1000 adaptations | Completed; different inputs, horizons and architecture from original minute experiments |',
             f"| Independent US minute LSTM | {backbone_state('LSTM')} |",
             f"| Independent US minute GRU | {backbone_state('GRU')} |",
             f"| Independent US minute TCN | {backbone_state('TCN')} |",
             '| Original CSI minute tables, full backbone matrix, stress, ablations, portfolio and efficiency results | Incomplete; see published-result audit |', '',
             f"US files present with receipts: **{raw['receipted_sessions_present']}/{raw['planned_sessions']}**; "
             f"missing: **{len(raw['missing_sessions'])}**. Independent raw audit complete: **{raw['independent_audit_complete']}**.", '',
             '| Process | Live now | PID | Current phase |', '|---|---|---:|---|']
    for name, job in status['processes'].items():
        lines.append(f"| {name} | {job['live']} | {job['pid'] or '—'} | {job['phase']} |")
    active = status['us_training']
    if 'maximum_weight_over_uniform' in active:
        lines += ['', f"Last saved BLO weights: maximum/uniform = {active['maximum_weight_over_uniform']:.12f}; "
                  f"provisional argmax = {active['selected_horizon']} minutes. "
                  'This is a training diagnostic, not a validated horizon or financial result.']
    lines += ['', 'Saved JSON labels are treated as stale when their matching process is absent. '
              'The process check rejects PID reuse and zombies.', '',
              '| Daily adaptation | Standard IC | Selected IC | Paired change95% block interval |',
              '|---|---:|---:|---|']
    for universe, result in status['daily_csi_adaptations'].items():
        if result.get('complete'):
            ci = result['paired_ic_bootstrap']['ci_95']
            lines.append(f"| {universe.upper()} | {result['standard_ic']:.5f} | {result['selected_ic']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] |")
    lines += ['', 'All three daily CSI paired intervals contain zero. The existing daily runs do not '
              'establish a reliable improvement from BLO and do not reproduce the original minute tables.', '',
              '[Full published-result audit](PUBLISHED_RESULT_AUDIT.md) · [Minute-data source audit](csi/minute_source_audit_20261110.json) · '
              '[US report](reproduction/REPORT.md) · [Daily CSI report](csi/REPORT.md) · '
              '[November 11 methodology audit](methodology_audit_20261111/REPORT.md) · '
              '[Published-protocol checks](reproduction/PROTOCOL_CHECK.md)', '',
              'Refresh this live check with `python status.py`. Resume the existing download/pipeline with '
              '`python resume.py --workers 1`.', '']
    (root / 'STATUS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='Print full machine-readable status')
    args = parser.parse_args()
    status = inspect()
    save(ROOT / 'STATUS.json', status)
    write_markdown(ROOT, status)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(json.dumps({'full_published_results_replicated': False,
                          'us_receipted_sessions': status['us_data']['receipted_sessions_present'],
                          'missing_us_sessions': len(status['us_data']['missing_sessions']),
                          'processes': status['processes'],
                          'daily_csi_complete': all(v.get('complete') for v in status['daily_csi_adaptations'].values())}, indent=2))
