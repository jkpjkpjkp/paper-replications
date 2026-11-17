"""Write a conservative, machine-readable completion statement."""
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from metrics import evaluate

ROOT = Path(__file__).resolve().parent
diagnostics = []
for market in ['NASDAQ', 'NYSE']:
    name = f'{market.lower()}-full-universe-diagnostic-memory-efficient'
    folder = ROOT / 'results' / name
    if not (folder / 'metrics.json').exists():
        continue
    config = json.loads((folder / 'config.json').read_text())
    result = json.loads((folder / 'metrics.json').read_text())
    z = np.load(folder / 'predictions.npz')
    replay = evaluate(z['prediction'], z['truth'], z['mask'])
    for key in ['sharpe5','mrr','ndcg5_percentile','downside_deviation_daily','downside_deviation_annual','simple_return','compound_return']:
        np.testing.assert_allclose(replay[key], result[key], rtol=1e-12, atol=1e-12)
    diagnostics.append({'market':market,'stocks':config['stocks'],'seed':config['seed'],
        'optimizer_updates':config['train_steps'],'validation_days':252,'test_days':result['days'],
        'saved_prediction_replay_passed':True,'results':result})

tests = (ROOT/'logs/tests.log').read_text()
status = {'generated_utc':datetime.now(timezone.utc).isoformat(),
    'overall_status':'partial_reconstruction; exact_paper_results_not_reproduced',
    'exact_table2_cells_reproduced':0,'exact_table2_pdu_cells_total':16,
    'tests':{'passed': 'Ran 7 tests' in tests and tests.rstrip().endswith('OK'), 'count':7},
    'datasets_reconstructed':['NASDAQ','NYSE'],
    'datasets_not_recovered':['PDU-author-prepared NASDAQ','PDU-author-prepared NYSE','PDU-author-prepared SP500','PDU-author-prepared SZSE'],
    'diagnostics': diagnostics,
    'full_training_completed':False,'three_seed_results_completed':False,
    'baselines_completed':False,'ablations_completed':False,'online_case_study_completed':False,
    'blocking_inputs':['Author-prepared panels, exact normalization and feature definitions, frozen graph/ticker/date mapping',
        'Resolution of paper/code hyperparameters and trading horizon discrepancies',
        'Valid NDCG/DD definitions and model-selection/seed protocol'],
    'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(list(ROOT.glob('*.py'))+list((ROOT/'repaired').glob('*.py')))}}
(ROOT/'results/STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
lines=['PDU reproduction result','=======================','',
    'Exact paper reproduction: NOT ACHIEVED (0 of 16 PDU Table 2 cells certified).',
    'Runnable, explicitly repaired reconstruction: implemented and tested.',
    f"Automated tests: {status['tests']['count']} passed = {status['tests']['passed']}.",
    'NASDAQ/NYSE reference-source panels recovered: 1026/1737 stocks, 1224 dates.',
    'Source interval: 2013-01-02 to 2017-11-08; test interval: 216 days in 2017.',
    '', 'Pipeline diagnostics (ONE optimizer update; not converged experiments):']
for d in diagnostics:
    r=d['results']
    lines.extend([f"  {d['market']}: full {d['stocks']}-stock universe, seed {d['seed']};",
        f"    252 validation days + {d['test_days']} test days; saved predictions independently reloaded and checked.",
        f"    SR={r['sharpe5']:.6f}, MRR={r['mrr']:.6f}, corrected percentile NDCG@5={r['ndcg5_percentile']:.6f}.",
        f"    Daily DD={r['downside_deviation_daily']:.6f}; annualized DD={r['downside_deviation_annual']:.6f}.",
        '    These figures are diagnostic outputs and are NOT matches to the paper.'])
lines.extend(['','Do not interpret proximity of any diagnostic number to a published number as',
    'replication. One update, reconstructed preprocessing, code repairs, metric',
    'differences and source-data defects invalidate that comparison.',
    '', 'Material data defect: most NYSE stocks have zero OHLC fields on 2017-07-31.',
    'Only 31 contemporaneous observations pass the explicit validity rule; 24-26',
    'stocks have valid four-day input windows and targets over 2017-07-31 through 2017-08-04.',
    'No forward/back filling or stock-universe substitution was applied.',
    '', 'Full 300-epoch runs, three-seed statistics, baselines, ablations and online',
    'case study have NOT been completed. Exact replication is blocked by missing',
    'author-prepared inputs and unresolved release/paper discrepancies.',
    '', 'See README.txt for commands, provenance, defects, repair choices and the',
    'precise additional inputs needed. See results/STATUS.json for structured status.'])
(ROOT/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
