"""Summarize completed runs without opening prediction files or evaluating models.

Run after the three full runs finish. The optional root argument permits bounded
schema checks with synthetic summaries, without changing real experiment files.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    'alphasage_training': 'results/csi300-seed0/summary.json',
    'alphasage_adaptive': 'results/csi300-seed0/adaptive/summary.json',
    'gp': 'results/gp-csi300-seed0/summary.json',
    'xgboost': 'results/xgboost-csi300-seed0/summary.json',
}
PROTOCOLS = {
    'alphasage': 'results/csi300-seed0/protocol.json',
    'gp': 'results/gp-csi300-seed0/protocol.json',
    'xgboost': 'results/xgboost-csi300-seed0/protocol.json',
}
REFERENCES = {
    'AlphaSAGE': {'ic': 0.079, 'rank_ic': 0.094},
    'GP': {'ic': 0.026, 'rank_ic': 0.028},
    'XGBoost': {'ic': 0.031, 'rank_ic': 0.033},
}
SOURCES = {
    'repository': 'https://github.com/BerkinChen/AlphaSAGE',
    'paper': 'https://arxiv.org/html/2509.25055v2',
    'paper_table': 'Table 1, CSI300 rows; reference IC and RIC use absolute-value reporting',
    'data_repository': 'https://huggingface.co/datasets/joshuaxql/qlib_data',
    'data_revision': 'ee0fa6b2299cb5e927d2df491888c84f6c9c027a',
    'data_archive': 'https://huggingface.co/datasets/joshuaxql/qlib_data/resolve/ee0fa6b2299cb5e927d2df491888c84f6c9c027a/cn_data.zip',
    'data_archive_sha256': '50d6fbb35634eb858f0e665e4811cbcbdcaccfba1d1e0084e8b7227deae17757',
}


class PendingArtifacts(ValueError):
    pass


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PendingArtifacts(f'Cannot read completed artifact {path}: {error}') from error


def numeric(value, label: str):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise PendingArtifacts(f'{label} must be a finite number; got {value!r}')
    return value


def collect(root: Path):
    missing = [relative for relative in [*FILES.values(), *PROTOCOLS.values()] if not (root/relative).is_file()]
    if missing:
        raise PendingArtifacts('Runs are not all complete yet. Missing:\n  ' + '\n  '.join(missing))
    summaries = {key: read_json(root/relative) for key, relative in FILES.items()}
    protocols = {key: read_json(root/relative) for key, relative in PROTOCOLS.items()}
    for key, summary in summaries.items():
        if summary.get('status') != 'complete':
            raise PendingArtifacts(f'{FILES[key]} is not marked complete')
    train, gp, xgb = summaries['alphasage_training'], summaries['gp'], summaries['xgboost']
    expected = [
        (train.get('completed_episodes'), 10000, 'AlphaSAGE completed episodes'),
        (train.get('capacity'), 50, 'AlphaSAGE pool capacity'),
        (gp.get('generations_completed'), 40, 'GP completed generations'),
        (gp.get('population'), 1000, 'GP population'),
        (xgb.get('rounds'), 100, 'XGBoost completed rounds'),
    ]
    for actual, required, description in expected:
        if actual != required:
            raise PendingArtifacts(f'{description}: expected {required}, got {actual!r}; refusing to label a reduced run as full budget')
    for name, protocol in protocols.items():
        training = protocol.get('training', {})
        seed = protocol.get('seed', training.get('seed') if isinstance(training, dict) else None)
        if seed != 0:
            raise PendingArtifacts(f'{name} protocol seed must be 0, got {seed!r}')
    train_dates = protocols['alphasage'].get('actual_train')
    if train_dates != protocols['gp'].get('training') or train_dates != protocols['xgboost'].get('ranges', {}).get('train'):
        raise PendingArtifacts('Training dates differ across methods; refusing a mixed-protocol comparison')
    variants = xgb.get('variants', {})
    for variant in ['intended_100_rounds', 'literal_notebook_10_rounds']:
        if variant not in variants:
            raise PendingArtifacts(f'XGBoost summary lacks {variant}')
    configurations = [
        ('AlphaSAGE', 'adaptive_50_pool', summaries['alphasage_adaptive'], FILES['alphasage_adaptive']),
        ('GP', 'best_training_ic_single_expression', gp, FILES['gp']),
        ('XGBoost', 'intended_100_rounds', variants['intended_100_rounds'], FILES['xgboost']),
        ('XGBoost', 'literal_notebook_10_rounds', variants['literal_notebook_10_rounds'], FILES['xgboost']),
    ]
    rows = []
    for method, variant, result, source in configurations:
        for split in ['validation', 'test']:
            metric = result.get(split, {})
            proxy = result.get(f'{split}_return_proxy') or {}
            row = dict(method=method, variant=variant, split=split, seed=0,
                       period='2021' if split == 'validation' else '2022-2024',
                       **{key: numeric(metric.get(key), f'{method}/{variant}/{split}/{key}')
                          for key in ['ic', 'icir', 'rank_ic', 'rank_icir', 'days']},
                       ic_released_evaluator=metric.get('ic_released_evaluator'),
                       icir_released_evaluator=metric.get('icir_released_evaluator'),
                       paper_reference_ic=REFERENCES[method]['ic'],
                       paper_reference_rank_ic=REFERENCES[method]['rank_ic'],
                       paper_reference_period='2022-2024',
                       paper_reference_comparability='separate absolute-value reference; data and integration differ',
                       proxy_annualized_arithmetic_return=proxy.get('annualized_arithmetic_return'),
                       proxy_sharpe=proxy.get('sharpe'), proxy_maximum_drawdown=proxy.get('maximum_drawdown'),
                       summary_file=source)
            for key in ['proxy_annualized_arithmetic_return', 'proxy_sharpe', 'proxy_maximum_drawdown']:
                if row[key] is not None:
                    numeric(row[key], f'{method}/{variant}/{split}/{key}')
            rows.append(row)
    return summaries, protocols, rows


def optional_json(path: Path):
    return read_json(path) if path.is_file() else None


def verification(root: Path):
    records = {}
    for filename in ['determinism.json', 'model-health.json',
                     'gp-csi300-seed0/artifact-verification.json',
                     'xgboost-csi300-seed0/artifact-verification.json']:
        if (root/'results'/filename).is_file():
            records[filename] = read_json(root/'results'/filename)
    for filename in ['verification.log', 'baseline-verification.log', 'data-verification.log']:
        path = root/'results'/filename
        if not path.is_file():
            continue
        text = path.read_text()
        # Logs may begin with library diagnostics before their JSON result.
        decoder = json.JSONDecoder()
        for position, character in enumerate(text):
            if character != '{':
                continue
            try:
                value, _ = decoder.raw_decode(text[position:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get('status') == 'passed':
                records[filename] = value
    return records


def package_versions():
    names = ['torch', 'numpy', 'pandas', 'pyqlib', 'torchgfn', 'torch-geometric', 'xgboost-cpu', 'xgboost', 'scikit-learn']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def source_state(root: Path, protocols):
    commit = protocols['alphasage'].get('upstream_commit')
    dirty = None
    if (root/'upstream/.git').exists():
        result = subprocess.run(['git', '-C', str(root/'upstream'), 'status', '--porcelain'], capture_output=True, text=True)
        if result.returncode == 0:
            dirty = bool(result.stdout.strip())
    return dict(**SOURCES, upstream_commit=commit, upstream_worktree_modified=dirty,
                local_data_directory=str(root.parent/'csi/qlib/cn_data'),
                original_data_verification=optional_json(root.parent/'csi/status.json'),
                local_data_preflight=optional_json(root/'sources/data-preflight.json'))


def write_comparison(root: Path, rows):
    path = root/'results/comparison.csv'
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readme(root: Path, summaries, protocols, rows, sources, checks):
    tests = [row for row in rows if row['split'] == 'test']
    lines = [
        'AlphaSAGE, Genetic Programming and XGBoost reproduction',
        '',
        'Completed scope: CSI300, seed 0, community daily Qlib data.',
        'AlphaSAGE uses 10,000 episodes, a 50-alpha capacity, the GNN encoder, and adaptive combination.',
        'GP uses the released 1,000-individual, 40-generation search; its final factor has the best training IC.',
        'XGBoost uses the released 295 lagged OHLCV-return features and hyperparameters.',
        'Both the intended 100-round model and the notebook-equivalent first 10 rounds are reported.',
        'This is an execution reproduction on substituted data with explicit integration changes.',
        'It does not establish a numerical replication of the paper, a multi-seed result, or a complete backtest.',
        '',
        'Signed test correlation results (2022-2024)',
        'Method / variant                         IC       ICIR     RankIC   RankICIR  Days',
    ]
    names = {'adaptive_50_pool': 'AlphaSAGE / adaptive',
             'best_training_ic_single_expression': 'GP / single expression',
             'intended_100_rounds': 'XGBoost / intended 100 rounds',
             'literal_notebook_10_rounds': 'XGBoost / notebook 10 rounds'}
    for row in tests:
        lines.append(f"{names[row['variant']]:<38} {row['ic']:>8.5f} {row['icir']:>8.5f} {row['rank_ic']:>8.5f} {row['rank_icir']:>9.5f} {row['days']:>5}")
    lines += [
        '',
        'Paper Table 1 reference values (CSI300 IC / RankIC):',
        'AlphaSAGE 0.079 / 0.094; GP 0.026 / 0.028; XGBoost 0.031 / 0.033.',
        'The paper reports absolute IC and RankIC. Local primary metrics retain their signs;',
        'no direction is chosen or reversed using held-out test results. The CSV keeps paper',
        'references in separate columns because the source snapshot and integration differ.',
        '',
        'Data and information timing',
        'Nominal paper Chinese train period: 2010-01-01 through 2020-12-31.',
        'Local training period: ' + ' through '.join(protocols['alphasage']['actual_train']) + '.',
        'Label boundary: ' + protocols['alphasage']['purge'] + '. Validation is 2021;',
        'the fixed test period is 2022-01-01 through 2024-12-31.',
        'AlphaSAGE and XGBoost use fixed budgets. GP selects by training IC only.',
        'Validation diagnostics do not select the reported model. No test checkpoint selection is used.',
        'Adaptive regression updates throughout evaluation using labels available before each prediction:',
        'its exclusive fit cutoff is cur-21 for a 20-day forward label, matching the released lag.',
        'Therefore test-period past observations can enter later adaptive fits; the generator remains fixed.',
        'Prices use the stored cumulative adjustment factors; volume is converted from lots to adjusted shares.',
        'Membership applies at formation; full post-exit close histories remain available to construct labels.',
        'Operations and normalization exclude nonmembers; tied cross-sectional ranks are averaged.',
        'Missing training outcomes remain missing. Absent pre-2010 history',
        'is padded with NaNs; no price observations are fabricated.',
        'The community CSI300 membership/price coverage begins on 2010-01-29; 19 earlier',
        'empty calendar days remain in the tensors and retain the documented empty-day convention.',
        '',
        'Source repository: ' + SOURCES['repository'],
        'Pinned upstream commit: ' + str(sources['upstream_commit']),
        'Paper: ' + SOURCES['paper'],
        'Data source: ' + SOURCES['data_repository'],
        'Pinned data revision: ' + SOURCES['data_revision'],
        'Source archive SHA256: ' + SOURCES['data_archive_sha256'],
        'Data directory: ' + sources['local_data_directory'],
        'The dataset maintainer attributes the snapshot to Tushare. This is not the authors\' exact',
        'snapshot or adjustment anchor. Per-run data-provenance.json files hash calendar, membership',
        'and every consumed feature file. No current-vintage reconstruction establishes original data identity.',
        '',
        'Integration changes and practical limits',
        '- Canonicalize mixed src.alphagen/alphagen imports to avoid incompatible duplicate classes.',
        '- Replace hard-coded vendor paths and US training dates with the local CSI300 adapter and split.',
        '- Run on CPU with Torch 2.4.0; full installed versions are in sources/requirements.lock.txt.',
        '- Use average-tie rank correlation without the released quadratic rank matrix allocation.',
        '- Primary IC is standard paired-mask Pearson correlation. The released evaluator replaces',
        '  the denominator when either daily standard deviation is below 1e-3; this can materially',
        '  change low-variance prediction IC. Separate ic_released_evaluator and',
        '  icir_released_evaluator fields preserve that convention alongside the primary metrics.',
        '- Accumulate expanding OLS sufficient statistics in float64, with rcond=1e-12 and no ridge.',
        '  Retain training observations rather than applying the released rank-deficiency row/column',
        '  removal heuristics. This is an explicit evaluator change, especially for collinear factors.',
        '- GP generation callbacks record training diagnostics. The released repeated test evaluation and',
        '  notebook refitting over test data are excluded from model selection.',
        '- The XGBoost notebook puts num_boost_round=100 inside params but omits the xgb.train argument.',
        '  The call defaults to 10 rounds; both first-10 and intended-100 predictions are saved here.',
        '- Only CSI300 and seed 0 are run. CSI500, S&P500, ablations and the other baselines are outside scope.',
        '',
        'Return figures are disabled by default',
        'An explicit --legacy-return-proxy flag can retain historical diagnostics, which rank the top 20%',
        'by prediction and average their 20-day forward return / 20.',
        'These overlapping forward labels do not describe a self-financing daily portfolio.',
        'The proxy restricts ranking to names with finite future targets. Future index exits or missing',
        'observations can therefore affect eligibility, creating an availability bias.',
        'Local proxy annualization uses 252 trading days and compounded negative maximum drawdown;',
        'the released helper uses test-days/3 annualization and an arithmetic drawdown convention.',
        'They omit an execution ledger, position timing, tradeability and transaction costs.',
        'Proxy return, Sharpe and drawdown figures must not be compared directly with paper backtest claims.',
        '',
        'Verification',
    ]
    if checks:
        for name, result in checks.items():
            lines.append(f'- {name}: {result.get("status")} (details in results/report.json).')
        lines += ['The recorded integration checks cover rank ties/paired NaN masks, direct versus',
                  'sufficient-statistics OLS, unavailable future-label perturbations, missing early history,',
                  'calendar/tensor alignment and the 20-day target. See verification.log for numerical errors.']
        if checks.get('baseline-verification.log', {}).get('status') == 'passed':
            lines += ['Baseline checks confirm the 295 lagged columns against the notebook, causal feature timing,',
                      'training-only normalization and label scaling, bundled GP string-terminal evolution,',
                      'and equality of first-10-round predictions to independently trained 10-round XGBoost.']
        if checks.get('determinism.json', {}).get('status') == 'passed':
            lines += ['Two 64-episode seed-0 CPU checks produced bitwise-identical model parameters and',
                      'identical alpha-pool files. This checks short-run determinism, not multi-seed robustness.']
    else:
        lines += ['No passing verification JSON was found in the expected logs; inspect the verification logs.']
    lines += [
        f'AlphaSAGE finite/nonfinite losses: {summaries["alphasage_training"].get("finite_losses")} / {summaries["alphasage_training"].get("nonfinite_losses")}.',
        '',
        'Artifacts',
        'results/comparison.csv                  Validation and test comparison, signed metrics and paper references',
        'results/report.json                     Full summaries, protocols, provenance and recorded checks',
        'results/csi300-seed0/model.pt            AlphaSAGE policy state',
        'results/csi300-seed0/pool.json           Final AlphaSAGE expressions and weights',
        'results/csi300-seed0/adaptive/           Adaptive predictions, daily metrics and selections',
        'results/gp-csi300-seed0/                 GP expression, search cache, history and predictions',
        'results/xgboost-csi300-seed0/model.ubj    100-round XGBoost model',
        'results/xgboost-csi300-seed0/literal_notebook_10_rounds.ubj  Saved first-10-round model',
        'results/xgboost-csi300-seed0/             Both XGBoost prediction variants and normalization statistics',
        'sources/artifact-manifest.json           SHA256 hashes of code and completed artifacts',
        'sources/*integration.patch              Auditable upstream integration diffs',
        'upstream/                               Pinned source checkout',
        '',
        'Re-run commands (execute from this directory; use fresh output directories)',
        '  export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1',
        '  .venv/bin/python reproduction/verify.py',
        '  .venv/bin/python reproduction/train.py --episodes 10000 --seed 0 --capacity 50 --update-freq 64 --log-freq 500 --output results/new-alphasage',
        '  .venv/bin/python reproduction/evaluate.py --pool results/new-alphasage/pool.json --output results/new-alphasage/adaptive',
        '  .venv/bin/python reproduction/train_gp.py --seed 0 --population 1000 --generations 40 --output results/new-gp',
        '  .venv/bin/python reproduction/train_xgboost.py --seed 0 --rounds 100 --output results/new-xgboost',
        '  .venv/bin/python reproduction/report.py',
        'The report script summarizes the standard result directories used by run_remaining.py.',
        'It refuses absent, incomplete or reduced-budget summaries; it never evaluates prediction files.',
        '',
        'To recreate the Python 3.11 environment with uv:',
        '  uv venv --python 3.11 .venv',
        '  uv pip sync --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match sources/requirements.lock.txt',
        'The shared CSI dataset must remain available at the recorded path, or update the DATA setting',
        'in reproduction/common.py and rebuild the local derived cache.',
        '',
        'Source notes are preserved in sources/paper-*.txt and sources/xgboost-notebook-cell.py.',
    ]
    return '\n'.join(lines) + '\n'


def artifact_manifest(root: Path, generated_at: str):
    files = []
    candidates = [root/'README.txt']
    for folder in ['reproduction', 'runtime', 'sources', 'results']:
        if (root/folder).exists():
            candidates.extend((root/folder).rglob('*'))
    excluded = {'sources/artifact-manifest.json', 'results/pipeline.log'}
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        if not path.is_file() or path.is_symlink() or relative in excluded or '__pycache__' in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8*1024*1024), b''):
                digest.update(chunk)
        files.append(dict(path=relative, bytes=path.stat().st_size, sha256=digest.hexdigest()))
    return dict(generated_at_utc=generated_at, algorithm='SHA256', files=files,
                exclusions=['.venv/', 'cache/', 'upstream/', '**/__pycache__/', 'STATUS.json',
                            'results/pipeline.log (updated after report generation)',
                            'sources/artifact-manifest.json (self)'],
                note='Hashes establish local artifact identity; they do not validate experimental claims.')


def build_report(root: Path):
    summaries, protocols, rows = collect(root)
    generated_at = datetime.now(timezone.utc).isoformat()
    sources = source_state(root, protocols)
    checks = verification(root)
    report = dict(status='complete', generated_at_utc=generated_at,
                  scope=dict(universe='csi300', seed=0, horizon_trading_days=20,
                             train=protocols['alphasage']['actual_train'], validation=['2021-01-01','2021-12-31'],
                             test=['2022-01-01','2024-12-31'],
                             alphasage=dict(episodes=10000,pool_capacity=50),
                             gp=dict(generations=40,population=1000),xgboost_rounds=[100,10]),
                  conclusion='Completed single-market execution reproduction with substituted data and documented changes; paper numerical replication not established.',
                  metrics_convention='Signed local correlations; paper absolute-value IC/RIC references are separate. No test-based direction flip.',
                  comparison=rows,paper_references=REFERENCES,summaries=summaries,protocols=protocols,
                  sources=sources,verification=checks,packages=package_versions(),
                  limitations=['One seed, CSI300 only; not a multi-market or multi-seed reproduction',
                               'Community snapshot differs from the authors exact data',
                               protocols['alphasage']['purge'],
                               'Adaptive OLS accumulation/rank handling differs from released evaluator heuristics',
                               'No executable portfolio backtest; legacy return proxies are opt-in diagnostics only'],
                  artifact_manifest='sources/artifact-manifest.json')
    (root/'results').mkdir(exist_ok=True)
    (root/'sources').mkdir(exist_ok=True)
    write_comparison(root, rows)
    (root/'results/report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (root/'results/RUN_REPORT.txt').write_text(readme(root,summaries,protocols,rows,sources,checks))
    manifest = artifact_manifest(root, generated_at)
    (root/'sources/artifact-manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    return dict(status='complete',report='results/RUN_REPORT.txt',comparison='results/comparison.csv',
                machine_readable='results/report.json',hashed_artifacts=len(manifest['files']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT,help='Experiment root; defaults to this reproduction directory')
    args=parser.parse_args()
    try:
        result=build_report(args.root.resolve())
    except (PendingArtifacts,KeyError,TypeError) as error:
        print(f'Report not generated: {error}',file=sys.stderr)
        return 2
    print(json.dumps(result,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
