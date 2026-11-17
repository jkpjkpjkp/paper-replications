"""One fixed-seed AlphaSAGE run on recovered inputs, with native combination.

Keeps the released GFN learning loop and the released SVD/QR row/column
handling. Its data vintage and release ambiguities preclude an automatic CZ
fidelity promotion. A smoke run is never counted as an original experiment.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[key] = '2'
import argparse
import difflib
import hashlib
import json
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT.parent / 'alphasage-reproduction'
UPSTREAM = OLD / 'upstream'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def module(name, text, path):
    value = types.ModuleType(name)
    value.__file__ = str(path)
    sys.modules[name] = value
    exec(compile(text, str(path), 'exec'), value.__dict__)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    panel, out = args.panel.resolve(), args.output.resolve()
    verification = json.loads((panel / 'verification.json').read_text())
    complete = json.loads((panel / 'complete.json').read_text())
    assert verification['passed']
    if not args.smoke:
        assert complete['state'] == 'panel_reconstructed' and complete['stocks'] == 780
    out.mkdir(parents=True, exist_ok=True)
    if any((out / p).exists() for p in ['protocol.json', 'training/pool.json', 'complete.json']):
        raise ValueError('Existing experiment preserved; choose a fresh output directory')
    integration = out / 'integration'
    for directory in [integration / 'sources', integration / 'runtime', out / 'evaluation']:
        directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    files = [UPSTREAM / 'train_gfn.py', UPSTREAM / 'run_adaptive_combination.py',
        OLD / 'reproduction/common.py', OLD / 'reproduction/train.py']
    spec = dict(method='AlphaSAGE CSI300 GNN, mining and released adaptive combination',
        seed=0, episodes=64 if args.smoke else 10000, pool_capacity=50, n_factors=20,
        train=['2010-01-01', '2020-12-31'], validation=['2021-01-01', '2021-12-31'],
        test=['2022-01-01', '2024-12-31'], smoke=args.smoke, device='cpu',
        selection='Fixed final episode; no test-based selection',
        source_sha256={str(p): sha(p) for p in files}, panel_protocol_sha256=sha(panel / 'protocol.json'),
        panel_verification_sha256=sha(panel / 'verification.json'), runner_sha256=sha(Path(__file__)),
        original_fidelity_passed=False,
        declared_choices=['Paper Chinese dates replace hard-coded US dates in train_gfn.py',
            'README n_factors=20 chosen before training; eval.sh says 50',
            'Native SVD/QR row/column handling and native least-squares solver retained',
            'CPU execution; canonical import aliases; calendar-preserving data adapter',
            'Nominal paper train endpoint retained; last 20-day labels reach early validation',
            'Released return metrics remain overlapping-label proxies, not a verified trading ledger'],
        data_limitations=json.loads((panel / 'protocol.json').read_text())['limitations'])
    save(out / 'protocol.json', spec)
    try:
        # The earlier tested calendar adapter is reused in a fresh module/cache.
        common_source = (OLD / 'reproduction/common.py').read_text()
        common_source = common_source.replace("'ee0fa6b2299cb5e927d2df491888c84f6c9c027a'", repr(sha(panel / 'protocol.json')))
        # Resolve imports against the pinned upstream, regardless of this wrapper's location.
        sys.path[:0] = [str(UPSTREAM / 'src'), str(UPSTREAM)]
        common = module('common', common_source, OLD / 'reproduction/common.py')
        common.ROOT, common.DATA = integration, panel / 'qlib'
        import torch
        import numpy as np
        torch.set_num_threads(2)
        original = (OLD / 'reproduction/train.py').read_text()
        patched = original.replace('2020-11-30', '2020-12-31').replace(
            'Last month of training excluded so 20-day training labels precede validation',
            'Nominal paper endpoint retained; final training labels reach early validation')
        (integration / 'sources/runner.patch').write_text(''.join(difflib.unified_diff(
            original.splitlines(True), patched.splitlines(True), fromfile='prior/train.py', tofile='recovered/train.py')))
        train = module('recovered_train', patched, OLD / 'reproduction/train.py')
        save(out / 'status.json', dict(state='training', episodes=spec['episodes']))
        sys.argv = [str(Path(__file__)), '--episodes', str(spec['episodes']), '--seed', '0',
            '--capacity', '50', '--update-freq', '64', '--log-freq', '64' if args.smoke else '500',
            '--output', str(out / 'training')]
        os.chdir(integration)
        train.main()
        save(out / 'status.json', dict(state='native_adaptive_combination', seconds=time.monotonic() - started))
        source = (UPSTREAM / 'run_adaptive_combination.py').read_text()
        adapted = source.replace('.cuda()', ".to('cpu')")
        needle = '        all_pred = torch.stack(pred_list, dim=0)'
        assert adapted.count(needle) == 1
        adapted = adapted.replace(needle, needle + '\n        capture(all_pred, tgt_tensor, start_day, data_all, data_valid.n_days)')
        needle = "        test_results, ret_s = get_tensor_metrics(pred_test.to('cpu'), tgt_test.to('cpu'))"
        assert adapted.count(needle) == 1
        adapted = adapted.replace(needle, needle + '\n        record_metrics(valid_results, test_results)')
        (integration / 'sources/adaptive.patch').write_text(''.join(difflib.unified_diff(
            source.splitlines(True), adapted.splitlines(True), fromfile='upstream/adaptive.py', tofile='cpu/adaptive.py')))
        native = module('recovered_adaptive', adapted, UPSTREAM / 'run_adaptive_combination.py')
        native.StockData = common.LocalStockData
        native.args = argparse.Namespace(expressions_file=str(out / 'training/pool.json'),
            instruments='csi300', train_end_year=2020, threshold_ric=.015, threshold_ricir=.15,
            seed=0, cuda=-1, n_factors=20, chunk_size=32, window='inf', label_days=20,
            use_weights=False, corr_threshold=.95, ridge_alpha=1e-6, use_vif=False, linear_dep_tol=1e-10)
        # Chunking bounds the temporary rank-comparison tensor; it does not alter ranks.
        captured = {}
        def capture(pred, target, start, data, valid_days):
            p, y = pred.detach().cpu().numpy(), target[start:, :, 0].detach().cpu().numpy()
            dates = data._dates[data.max_backtrack_days:-data.max_future_days][start:]
            np.savez_compressed(out / 'evaluation/predictions.npz', predictions=p, targets=y,
                dates=dates.to_numpy(), stocks=np.asarray(data._stock_ids, dtype=str), validation_days=valid_days)
            captured.update(pred=p, target=y, valid_days=valid_days)
        native.capture = capture
        native.record_metrics = lambda valid, test: save(out / 'evaluation/native-metrics.json', dict(validation=valid, test=test))
        torch.manual_seed(0)
        np.random.seed(0)
        native.run(native.args)
        scores = json.loads((out / 'evaluation/native-metrics.json').read_text())
        n = captured['valid_days']
        independent = common.metrics(captured['pred'][n:], captured['target'][n:])
        np.testing.assert_allclose(scores['test']['ic'], independent['ic_released_evaluator'], atol=3e-6, rtol=1e-5)
        np.testing.assert_allclose(scores['test']['ric'], independent['rank_ic'], atol=3e-6, rtol=1e-5)
        save(out / 'complete.json', dict(state='smoke_complete' if args.smoke else 'original_method_run_complete_fidelity_unresolved',
            wall_seconds=time.monotonic() - started, independent_metrics=independent,
            paper_comparison=dict(paper_ic=.079, paper_rank_ic=.094,
                local_native_ic=scores['test']['ic'], local_native_rank_ic=scores['test']['ric']),
            original_fidelity_passed=False, cz_training_started=False,
            limitations=spec['declared_choices'] + spec['data_limitations']))
        save(out / 'status.json', json.loads((out / 'complete.json').read_text()))
    except BaseException as error:
        save(out / 'status.json', dict(state='failed', error=repr(error), seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
