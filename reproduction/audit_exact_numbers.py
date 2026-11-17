"""Audit saved predictions against Table 1 without retraining or tuning to test data.

Run: OMP_NUM_THREADS=2 .venv/bin/python reproduction/audit_exact_numbers.py
The released metric functions are extracted unchanged from the pinned source.
Chunking Spearman by eight days bounds memory; each day's computation is unchanged.
This tests metric conventions, not equivalence of data, training, or portfolios.
"""
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream/src'))
import numpy as np
import torch
from scipy.stats import rankdata
from alphagen.utils import correlation


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def paper_rows():
    rows, market = [], None
    keys = ['ic', 'icir', 'rank_ic', 'rank_icir', 'ar', 'mdd', 'sr']
    for line in (ROOT / 'sources/paper-results.txt').read_text().splitlines():
        cells = line.split('\t')
        if cells[0] in ('CSI300', 'CSI500', 'S&P500'):
            market, cells = cells[0], cells[1:]
        if market and len(cells) == 8:
            values = {key: float(value.rstrip('%')) / (100 if '%' in value else 1)
                      for key, value in zip(keys, cells[1:])}
            rows.append(dict(market=market, method=cells[0].replace('(ours)', ''), **values))
    assert len(rows) == 24
    return rows


def independent_metrics(x, y):
    """Daily paired-mask Pearson/Spearman; independent of existing summary files."""
    daily = []
    for a, b in zip(x.astype(np.float64), y.astype(np.float64)):
        ok = np.isfinite(a) & np.isfinite(b)
        a, b = a[ok], b[ok]
        values = []
        for u, v in ((a, b), (rankdata(a), rankdata(b))):
            u, v = u - u.mean(), v - v.mean()
            denom = np.sqrt((u @ u) * (v @ v))
            values.append(float(u @ v / denom) if denom else 0.)
        daily.append(values)
    ic, ric = np.asarray(daily).T
    return dict(ic=float(ic.mean()), icir=float(ic.mean()/ic.std(ddof=1)),
                rank_ic=float(ric.mean()), rank_icir=float(ric.mean()/ric.std(ddof=1)))


def main():
    torch.set_num_threads(2)
    destination = ROOT / 'results/exact-number-audit'
    destination.mkdir(exist_ok=True)
    source = ROOT / 'upstream/run_adaptive_combination.py'
    tree = ast.parse(source.read_text())
    names = {'get_tensor_metrics', 'chunk_batch_spearmanr'}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == 2
    namespace = dict(vars(correlation), torch=torch, args=SimpleNamespace(chunk_size=8, label_days=20))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), 'exec'), namespace)

    manifest = json.loads((ROOT / 'sources/artifact-manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.exists() or digest(path) != entry['sha256']:
            failures.append(entry['path'])
    print(json.dumps(dict(manifest_checked=len(manifest['files']), failures=failures)), flush=True)

    references = paper_rows()
    cases = [
        ('AlphaSAGE', 'adaptive_50_pool', 'results/csi300-seed0/adaptive/predictions.npz'),
        ('GP', 'best_training_ic_single_expression', 'results/gp-csi300-seed0/test_predictions.npz'),
        ('XGBoost', 'intended_100_rounds', 'results/xgboost-csi300-seed0/intended_100_rounds_test_predictions.npz'),
        ('XGBoost', 'literal_notebook_10_rounds', 'results/xgboost-csi300-seed0/literal_notebook_10_rounds_test_predictions.npz'),
    ]
    original_report = json.loads((ROOT / 'results/report.json').read_text())
    results = []
    for method, variant, relative in cases:
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as archive:
            dates = archive['dates']
            mask = (dates >= np.datetime64('2022-01-01')) & (dates <= np.datetime64('2024-12-31'))
            x, y = archive['predictions'][mask], archive['targets'][mask]
        assert len(x) == 726 and not np.isinf(x).any() and not np.isinf(y).any()
        independent = independent_metrics(x, y)
        saved = next(r for r in original_report['comparison']
                     if r['method'] == method and r['variant'] == variant and r['split'] == 'test')
        saved_error = max(abs(value-saved[key]) for key, value in independent.items())
        assert saved_error < 1e-10, (method, saved_error)
        with torch.no_grad():
            released, returns = namespace['get_tensor_metrics'](torch.from_numpy(x), torch.from_numpy(y))
        mapping = {'ic':'ic', 'icir':'icir', 'rank_ic':'ric', 'rank_icir':'ricir'}
        paper = next(r for r in references if r['market'] == 'CSI300' and r['method'] == method)
        comparison = {}
        for key, released_key in mapping.items():
            raw = released[released_key]
            value = abs(raw) if key in ('ic', 'rank_ic') else raw
            comparison[key] = dict(published=paper[key], recomputed_standard=independent[key],
                                   released_evaluator=raw, paper_convention_value=value,
                                   difference=value-paper[key],
                                   matches_published_three_decimals=f'{value:.3f}' == f'{paper[key]:.3f}')
        result = dict(method=method, variant=variant, predictions=relative,
                      predictions_sha256=digest(path), test_days=len(x),
                      original_summary_max_error=saved_error, correlation_comparison=comparison,
                      released_helper_raw_output=released,
                      portfolio_comparable_to_paper=False)
        results.append(result)
        print(json.dumps(result), flush=True)

    report = dict(
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        exact_reproduction=False,
        conclusion='Saved runs do not reproduce the published correlation numbers, including with the released metric functions.',
        scope='Fresh evaluation of existing CSI300 seed-0 predictions; no new training, other markets, ablations, or executable portfolio backtest.',
        paper_url='https://arxiv.org/html/2509.25055v2',
        upstream_commit='517467a34909512a92d6e3139df54891957560de',
        upstream_evaluator_sha256=digest(source),
        audit_script_sha256=digest(Path(__file__)),
        torch_version=torch.__version__,
        manifest=dict(checked=len(manifest['files']), mismatches=failures),
        table_1_targets=references,
        results=results,
        blockers=[
            'The local community Qlib snapshot is not identified as the authors exact snapshot.',
            'Pinned released training uses 2010-2016 even for CSI300; paper specifies 2010-2020.',
            'Existing local training ends 2020-11-30 with a label purge, differing from the paper.',
            'Local adaptive OLS removes neither rows nor columns using the released rank-deficiency heuristics.',
            'Table 1 is not linked here to original prediction files, final pools, exact experiment commands, and run aggregation.',
            'CPU execution differs from the released CUDA environment.',
            'Released portfolio helper uses overlapping forward labels/20, days/3 annualization, and positive arithmetic drawdown; not a daily trade ledger.',
        ],
        required_for_exact_attempt=[
            'Authors Qlib snapshot/checksum including adjustments, calendars, and point-in-time universe files.',
            'Table-specific commit, commands, seeds, checkpoint selection/aggregation, and original pools or predictions.',
            'Exact adaptive regression settings and original portfolio return series or executable backtest.',
        ])
    (destination / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Written {destination / "audit.json"}', flush=True)


if __name__ == '__main__':
    main()
