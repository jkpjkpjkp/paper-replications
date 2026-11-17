"""Finalize the six-run released-code experiment; refuses partial results."""
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / 'reproduction'
FRESH = REPRO / 'fresh_local'
for pool in (40, 50):
    paths = list((FRESH / 'results').glob(f'dp{pool}_*test_metrics.csv'))
    assert len(paths) == 3, f'Pool {pool}: expected 3 completed result files, found {len(paths)}'
    assert (FRESH / f'pool{pool}_finished.txt').exists(), f'Pool {pool} process is not finished'

subprocess.run([sys.executable, str(REPRO / 'audit_results.py'), '--results', str(FRESH / 'results'),
                '--label', 'fresh_local_comparison'], check=True, cwd=ROOT)
subprocess.run([sys.executable, str(REPRO / 'audit_date_alignment.py'), '--sit-results', str(FRESH / 'results'),
                '--sit-label', 'SIT_fresh', '--output', str(REPRO / 'fresh_local_common_dates')], check=True, cwd=ROOT)
table = pd.read_csv(REPRO / 'fresh_local_comparison/table1_comparison.csv')
assert len(table) == 16 and set(table.runs) == {3}
audit = pd.read_csv(REPRO / 'fresh_local_comparison/daily_replay_audit.csv')
assert len(audit) == 6
assert audit.max_daily_return_error.max() < 1e-6
matches = int(table.printed_match.sum())
rows = []
for path in sorted((FRESH / 'results').glob('*test_metrics.csv')):
    row = pd.read_csv(path).iloc[0].to_dict()
    row.update(file=path.name, pool=int(re.search(r'^dp(\d+)', path.name)[1]),
               run=int(re.search(r'_(\d+)_test_metrics.csv$', path.name)[1]))
    rows.append(row)
pd.DataFrame(rows).to_csv(FRESH / 'per_run_metrics.csv', index=False)

durations = {}
epochs = {}
for pool in (40, 50):
    started = dt.datetime.fromisoformat((FRESH / f'pool{pool}_started.txt').read_text().strip())
    finished = dt.datetime.fromisoformat((FRESH / f'pool{pool}_finished.txt').read_text().strip())
    durations[pool] = int((finished - started).total_seconds())
    log = (REPRO / f'logs/fresh_local_dp{pool}.log').read_text()
    epochs[pool] = [len(re.findall(r'\[Epoch \d+\]', part)) for part in log.split('>>>>>>>start training :')[1:]]

text = ['CURRENT RESULT',
        'All six fresh local training runs completed: three per asset pool.',
        f'Printed-precision agreement: {matches} of 16 Table 1 mean/SD cells.',
        'This is a fresh reproduction of the released configuration with disclosed',
        'environment differences. It is NOT an exact whole-paper reproduction.', '']
if matches < 16:
    text.append('The published headline results were not completely matched.')
for pool in (40, 50):
    text += ['', f'Asset pool {pool}: three runs, mean +/- sample standard deviation',
             'Metric           Paper                    Fresh local']
    for metric in ('Sharpe', 'Sortino', 'MaxDrawdown', 'FinalWealthFactor'):
        subset = table[(table.pool == pool) & (table.metric == metric)].set_index('statistic')
        mean, sd = subset.loc['mean'], subset.loc['sample_sd']
        text.append(f'{metric:<20}{mean.paper:.4f} +/- {sd.paper:.4f}       {mean.replayed:.6f} +/- {sd.replayed:.6f}')
    text.append(f'Execution: {durations[pool]} seconds, epochs per run {epochs[pool]}.')
text += ['', 'Lower maximum drawdown is better. A mismatch does not necessarily mean',
         'worse performance on that metric. These small sets of training runs do not',
         'establish statistical equivalence or identify the cause of discrepancies.',
         'Evidence: reproduction/fresh_local_comparison/table1_comparison.csv,',
         'reproduction/fresh_local/per_run_metrics.csv and the fresh checkpoints/logs.', '']
report = ROOT / 'REPRODUCTION_REPORT.txt'
body = report.read_text().replace('Working reproduction report', 'Reproduction report')
body = re.sub(r'CURRENT RESULT\n.*?\nWHAT WAS EXECUTED\n', '\n'.join(text) + '\nWHAT WAS EXECUTED\n', body, flags=re.S)
report.write_text(body)

now = dt.datetime.now().astimezone().isoformat()
summary = dict(status='completed_with_disclosed_environment_differences', completed_at=now,
               completed_runs=6, printed_matches=matches, comparison_cells=16,
               all_primary_printed_match=matches == 16, whole_paper_exact_reproduction=False,
               elapsed_seconds=durations, epochs_per_run=epochs,
               max_daily_return_replay_error=float(audit.max_daily_return_error.max()),
               result_scope='SIT Table 1 40/50-asset rows; other baselines, universes and ablations not all freshly rerun.')
(FRESH / 'COMPLETED.json').write_text(json.dumps(summary, indent=2) + '\n')
status_path = ROOT.parent / 'results/comparison_status.json'
status = json.loads(status_path.read_text())
status['SIT'] = dict(summary, fresh_training_completed=True, report='SIT/REPRODUCTION_REPORT.txt',
                     official_EW_fresh_matches=8, official_EW_cells=8,
                     shipped_artifact_printed_matches=15, shipped_artifact_cells=16)
status_path.write_text(json.dumps(status, indent=2) + '\n')
protocol_path = ROOT.parent / 'protocol.json'
protocol = json.loads(protocol_path.read_text())
protocol['current_status']['SIT'] = f'Fresh local 40/50-asset training completed: six runs, {matches}/16 primary cells match; whole-paper exact replication not established. See SIT/REPRODUCTION_REPORT.txt.'
protocol['compute']['intended_wall_time_hours'] = f'Actual GPU execution: pool40 {durations[40]}s; pool50 {durations[50]}s, excluding preparation. RTX 4060; pool50 used CUDA managed memory.'
protocol_path.write_text(json.dumps(protocol, indent=2) + '\n')
paper = ROOT / 'PAPER.txt'
paper.write_text(re.sub(r'Status:.*', f'Status: Sole selected paper; six fresh local runs completed. {matches}/16 Table 1 mean/SD cells matched. Environment/protocol differences prevent an exact whole-paper reproduction claim. See REPRODUCTION_REPORT.txt.', paper.read_text()))
bias = ROOT / 'LOOKAHEAD_BIAS_AUDIT.txt'
bias.write_text(bias.read_text().replace('fresh SIT training is pending.', 'six fresh SIT training runs are complete; see REPRODUCTION_REPORT.txt.'))
(REPRO / 'RUNNING_STATUS.json').write_text(json.dumps(summary, indent=2) + '\n')

manifest = []
for path in sorted(FRESH.rglob('*')):
    if path.is_file() and path.name != 'output_manifest.json':
        manifest.append(dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(FRESH / 'output_manifest.json').write_text(json.dumps({'created_at':now, 'files':manifest}, indent=2) + '\n')
print(json.dumps(summary, indent=2))
