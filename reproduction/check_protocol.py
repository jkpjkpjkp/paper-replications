#!/usr/bin/env python3
"""Check disclosed settings, run regression tests, and publish bounded evidence."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys

from build_tensors import FIELDS
from data import save_json, sha
from progress import collect_progress

HERE = Path(__file__).resolve().parent


def main():
    config = json.loads((HERE / 'protocol.json').read_text())
    expected = {'patch_minutes': 15, 'patches': 26, 'horizons': 390,
                'batch_days': 20, 'warmup_epochs': 3, 'inner_learning_rate': 1e-6,
                'outer_learning_rate': 1e-3, 'entropy_weight': 1e-3, 'patience': 5}
    for key, value in expected.items():
        if config[key] != value:
            raise AssertionError(f'{key}: expected {value}, observed {config[key]}')
    if FIELDS != ['open', 'high', 'low', 'close', 'amount', 'volume', 'trade_count']:
        raise AssertionError('Raw fields differ from Appendix A.2')
    progress = collect_progress(HERE.parent)
    if any(p['source_mismatches'] or p['invalid_checkpoints'] for p in progress.values()):
        raise AssertionError('Run sources or completed checkpoint receipts changed')
    log_path = HERE / 'protocol-check-tests.log'
    with log_path.open('w') as log:
        checked = subprocess.run([sys.executable, '-m', 'unittest', 'test_reproduction',
                                  'test_backbones', 'test_supervisor', 'test_paper_protocol',
                                  'test_progress', '-v'], cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if checked.returncode:
        raise RuntimeError('Tests failed; inspect ' + str(log_path))
    tests_run = int(re.search(r'Ran (\d+) tests? in ', log_path.read_text()).group(1))
    diagnostic_path = HERE / 'diagnostics/hypergradient-20261111/result.json'
    diagnostic = json.loads(diagnostic_path.read_text())
    if diagnostic['status'] != 'passed':
        raise AssertionError('Real-data hypergradient check did not pass')
    tensors = json.loads((HERE / 'tensor_status.json').read_text())
    audit = json.loads((HERE.parent / 'validation.json').read_text())
    rows = [
        ['Raw inputs', 'A.2, p.12', 'Seven one-minute fields', 'Names/schema aligned; vendor trade eligibility is an assumption'],
        ['Feature aggregation', 'A.3–A.4, pp.12–13', '26 non-overlapping 15-minute patches (US)', 'Patch locality tested; exact 66 features and D are not established'],
        ['Ordinary training', 'A.5.1, p.13', 'AdamW; cross-sectionally standardized MSE', 'MSE–IC identity and streamed optimizer update tested'],
        ['Batch size', 'A.5.1, p.13', '20 trading days', 'Full 20-date windows; last 17 training dates omitted by fixed batching'],
        ['Warm-up', '4.2; A.5.2, pp.6,13', 'Three epochs; mean standardized candidate levels', 'Same AdamW settings as ordinary training; label calculation tested'],
        ['Support/query split', 'A.5.2, pp.13–14', 'Random 10/10 partition within each batch', 'Disjoint date indices in train.select_horizon; epoch RNG recovery tested'],
        ['Inner learning rate', 'A.5.2, p.14', '1e-6', 'Differentiable single SGD step, retained student writeback'],
        ['Outer learning rate', 'A.5.2, p.14', '1e-3', 'SGD on softmax logits: explicit equation interpretation, optimizer still ambiguous'],
        ['Entropy coefficient', 'A.5.2, p.14', '1e-3', 'Subtract entropy from minimized query objective'],
        ['Early stopping', 'A.5.1, p.13', 'Patience 5; restore best validation checkpoint', 'Checkpoint/recovery tests pass; no primary epoch cap'],
        ['Final model', '4.2, p.6', 'Argmax horizon then retrain', 'Fresh same-seed model; all 15 fits required before backbone test metrics'],
    ]
    now = datetime.now(timezone.utc).isoformat()
    source_files = ['protocol.json', 'core.py', 'train.py', 'data.py', 'build_tensors.py',
                    'backbones.py', 'check_hypergradient.py', 'check_protocol.py',
                    'progress.py', 'report.py', 'test_progress.py']
    result = {'checked_utc': now, 'status': 'passed_with_documented_unknowns',
              'full_published_replication_complete': False,
              'protocol_version': config['protocol_version'], 'checked_constants': expected,
              'raw_fields': FIELDS, 'checklist': rows,
              'regression_test_count': tests_run, 'regression_log_sha256': sha(log_path),
              'hypergradient_diagnostic_sha256': sha(diagnostic_path),
              'source_sha256': {name: sha(HERE / name) for name in source_files},
              'raw_audit_complete': audit.get('complete'),
              'tensor_dates': dict(Counter(r['split'] for r in tensors['cached'])),
              'frozen_model_source_unchanged': not any(p['source_mismatches'] for p in progress.values()),
              'progress_at_check': progress,
              'paper_pdf_sha256': sha(HERE.parent / 'archive/us-v1/reference/song26l.pdf'),
              'sources': ['https://proceedings.mlr.press/v306/song26l.html',
                          'https://github.com/EFundAI/label-horizon-paradox'],
              'unresolved': config['unspecified_choices']}
    save_json(HERE / 'protocol_check.json', result)
    numerical = diagnostic['current_weights']
    max_error = max(c['absolute_error'] for c in diagnostic['checks'])
    lines = ['# Published protocol: implementation check', '',
             f'Checked {now}. Protocol `{config["protocol_version"]}`.', '',
             '**Published numerical results remain unreplicated.** These checks establish implementation behavior, '
             'not equivalence to the authors’ private inputs or configuration.', '',
             '| Item | Published source | Specification | Observed implementation / limit |',
             '|---|---|---|---|']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    lines += ['', 'The baseline/warm-up learning rate (1e-3), weight decay (1e-4), architecture sizes, '
              'seed IDs 0–4, preprocessing, missing-bar handling, corporate-action policy, point-in-time '
              'membership convention, and boundary purging are implementation choices. The paper does '
              'not fully specify them. Appendix A.3 gives representative feature categories and leaves D unspecified. '
              'The 66 implemented statistics must not be represented as the original tensor.', '',
              'The [authors’ public repository](https://github.com/EFundAI/label-horizon-paradox), checked live '
              'on November 11, identifies itself as a separate daily Alpha158 demo and says the original '
              'experimental features are not public. No new original feature/configuration release was found there.', '',
              '## Independent numerical evidence', '',
              f'All **{tests_run} regression tests passed** (training/data/backbone checks and progress-integrity checks). '
              'The primary model source hashes and completed checkpoint receipts still match the frozen run.', '',
              f'A consistent snapshot of LSTM seed 0 at BLO epoch {diagnostic["checkpoint_epoch"]} was checked on '
              '20 real TRAINING dates, with a seeded random 10/10 split. A separate implementation of standardized '
              'MSE and first-order inner updates recomputes the query objective at perturbed logits. Two independent '
              f'directions and two step sizes agree with the analytic hypergradient to maximum absolute error {max_error:.3g}. '
              'No validation or test scores were read for this diagnostic.', '',
              f'At that snapshot, maximum/uniform weight = {numerical["maximum_weight_over_uniform"]:.12f}. '
              f'The provisional argmax is {numerical["argmax_horizon"]} minutes; its weight exceeds '
              f'the runner-up ({numerical["runner_up_horizon"]} minutes) by only {numerical["top_two_gap"]:.3g}. '
              'This is a nearly uniform distribution, not evidence of a stable selected optimum.', '',
              'The task hypergradient norm is '
              f'{diagnostic["task_hypergradient_norm"]:.3g}. At the same LR of 1e-3, a fresh SGD step changes '
              f'the logits by norm {diagnostic["optimizer_sensitivity"]["SGD"]["logit_delta_norm"]:.3g}; '
              'a fresh AdamW step (epsilon 1e-8, zero weight decay) changes them by '
              f'{diagnostic["optimizer_sensitivity"]["AdamW"]["logit_delta_norm"]:.3g}. '
              'This is a one-step numerical sensitivity check, not a trained alternative or evidence that AdamW improves '
              'financial performance. It confirms that choosing an optimizer can materially change the trajectory. '
              'The running primary protocol remains unchanged.', '',
              '## Work still outstanding', '',
              'The active coordinator is executing the five-seed LSTM comparison, followed by GRU and TCN. '
              'Each requires 15 completed fits before test evaluation. See [live status](../STATUS.md) for actual '
              'completed counts; the monitor refreshes it every 30 seconds while the coordinator runs. '
              'Early stopping has no reliable finish-time estimate while validation continues improving.', '',
              'Original CSI minute inputs, exact feature definitions/configurations, seven other architectures, '
              'and the remaining original-paper experiments are still unavailable or unfinished. '
              'The older daily CSI results are not minute-paper replications.', '',
              '[Published paper](https://proceedings.mlr.press/v306/song26l.html) · '
              '[Machine-readable check](protocol_check.json) · [Regression log](protocol-check-tests.log) · '
              '[Frozen numerical diagnostic](diagnostics/hypergradient-20261111/result.json)', '',
              'Re-run: `.venv/bin/python check_hypergradient.py` (reuses its frozen snapshot), then '
              '`.venv/bin/python check_protocol.py`. Neither changes the fitted primary models.', '']
    (HERE / 'PROTOCOL_CHECK.md').write_text('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'tests_passed': tests_run,
                      'full_published_replication_complete': False, 'report': str(HERE / 'PROTOCOL_CHECK.md')}, indent=2))


if __name__ == '__main__':
    main()
