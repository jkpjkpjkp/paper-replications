"""Paper Appendix B: select online learning rates on 2015-2016, then evaluate once."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'sources/DoubleAdapt-qlib'
BASE = ROOT / 'results/gap-resolution-20261113'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean', action='store_true')
    p.add_argument('--step', type=int, choices=[5, 20], default=20)
    p.add_argument('--coherent', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    args = p.parse_args()
    assert not args.coherent or args.clean
    assert args.step == 20 or args.clean
    if args.step == 5:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    name = 'available-inputs' if args.clean else 'script-matched'
    clean_dir = 'available-inputs-step5' if args.step == 5 else 'available-inputs'
    if args.step == 5: name += '-step5'
    if args.coherent:
        name += '-coherent'
    out = BASE / ('validation-rates-' + name)
    out.mkdir(exist_ok=True)
    assert not (out / 'selection.json').exists(), 'Do not overwrite a completed selection'
    (out / 'runner-at-launch.py').write_bytes(Path(__file__).read_bytes())
    checkpoint_path = (BASE / clean_dir if args.clean else ROOT / 'results/doubleadapt-script-matched') / ('best-coherent-state.pt' if args.coherent else 'offline-model.pt')
    assert checkpoint_path.exists()
    start = time.monotonic()

    def save(filename, data):
        dest = out / filename
        temp = dest.with_suffix(dest.suffix + '.tmp')
        temp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
        temp.replace(dest)

    # Native rates first: ties retain the released configuration. No held-out metric is consulted.
    grid = [dict(lr_ma=ma, lr_da=da) for ma in [.001, .0005] for da in [.01, .001, 0.]]
    save('protocol.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        basis='DoubleAdapt paper Appendix B explicitly suggests validation selection using one pretrained model; discusses model-adapter rates 0.0005/0.001 and freezing the data adapter.',
        candidates=grid, forecast_inner_lr=.001, selection='Maximum mean daily Pearson IC strictly inside 2015-2016, stable grid-order ties. Native blocks include 2014-12-31; excluded only from selection score, preserved in adaptation sequence.',
        validation_dates=['2015-01-01', '2016-12-31'], test_dates=['2017-01-01', '2020-07-31'],
        checkpoint=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        input_availability_filter=args.clean, coherent_optimizer=args.coherent, step=args.step,
        common_scoring_dates=['2015-01-05','2016-12-19'],
        test_policy='One evaluation of the validation winner only; no selection by test IC or return.',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    import numpy as np
    import pandas as pd
    import torch
    import qlib
    sys.path.insert(0, str(REPO))
    torch.set_num_threads(2)
    if args.step == 5:
        assert torch.cuda.is_available()
        torch.cuda.set_per_process_memory_fraction(0.30)
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    original_init = qlib.init

    def initialize(*a, **kw):
        kw.update(kernels=2, joblib_backend='threading', expression_cache=None, dataset_cache=None)
        return original_init(*a, **kw)

    qlib.init = initialize
    spec = importlib.util.spec_from_file_location('author_main', REPO / 'examples/benchmarks_dynamic/incremental/main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if args.clean:
        native_task = module.DoubleAdaptManager.run_task

        def available_task(self, original, phase):
            task = original.copy()
            support = task['X_train'].reshape(len(task['X_train']), -1).ne(0).any(dim=1)
            assert support.any()
            for key in ['X_train', 'y_train']:
                task[key] = task[key][support]
            if 'train_idx' in task:
                task['train_idx'] = task['train_idx'][support.numpy()]
            query = task['X_test'].reshape(len(task['X_test']), -1).ne(0).any(dim=1)
            assert query.any() and 'X_extra' not in task
            assert phase != 'train', 'This runner only validates/replays'
            task['mask_y'] = query
            return native_task(self, task, phase)

        module.DoubleAdaptManager.run_task = available_task
    os.chdir(out)
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(43)
    cfg = dict(data_dir='crowd_data', root_path=str(ROOT / 'data') + '/', market='csi300',
               forecast_model='GRU', alpha=360, step=args.step, rank_label=False, num_head=8, tau=10,
               lr=.001, lr_da=.01, online_lr=None, begin_valid_epoch=30, preprocess_tensor=True,
               save=False, tag=name + '-validation-rates', h_path=str(ROOT / 'results/doubleadapt-original/handler.pkl'))
    exp = module.Incremental(**cfg)
    module.R.set_uri((out / 'mlruns').as_uri())
    exp.meta_dataset_offline, exp.meta_dataset_online = exp.dump_data()
    tasks = exp.meta_dataset_offline.prepare_tasks('test')
    boundaries = []
    for task in tasks:
        data = task.get_meta_input()
        dates = data['test_idx'].get_level_values('datetime')
        support = data['train_idx'].get_level_values('datetime')
        boundaries.append(dict(support_start=str(support.min()), support_end=str(support.max()), query_start=str(dates.min()), query_end=str(dates.max())))
    save('native-validation-boundaries.json', boundaries)
    assert all(b['query_end'] <= '2016-12-31' for b in boundaries), boundaries
    # Released rolling blocks straddle the split by one query day (2014-12-31).
    # Preserve the task sequence; score selection strictly inside 2015-2016.
    assert all(b['query_start'] >= '2014-12-31' for b in boundaries), boundaries
    manager = module.DoubleAdaptManager(exp.basic_task, x_dim=exp.x_dim, lr_model=exp.lr,
        weight_decay=exp.weight_decay, first_order=exp.first_order, alpha=exp.alpha,
        begin_valid_epoch=exp.begin_valid_epoch, factor_num=exp.factor_num,
        lr_da=exp.lr_da, lr_ma=exp.lr_ma, online_lr=exp.online_lr,
        lr_x=exp.lr_x, lr_y=exp.lr_y, adapt_x=exp.adapt_x, adapt_y=exp.adapt_y,
        reg=exp.reg, num_head=exp.num_head, temperature=exp.temperature)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    results = []
    for i, rates in enumerate(grid):
        manager.load_state_dict(copy.deepcopy(checkpoint))
        manager.online_lr = rates.copy()
        manager.framework.train()
        manager.lr_model = .001
        tick = time.monotonic()
        predictions, native_ic = manager.run_epoch('test', tasks)
        selected = predictions.loc['2015-01-05':'2016-12-19']
        daily = selected.groupby(level='datetime').apply(lambda g: g.pred.corr(g.label))
        ic = float(daily.mean())
        assert np.isfinite(ic)
        assert len(daily) == 479, 'All candidates must use identical validation dates'
        assert predictions.index.get_level_values('datetime').max() <= pd.Timestamp('2016-12-31')
        predictions.to_pickle(out / f'validation-{i}.pkl')
        results.append(dict(candidate=i, rates=rates, validation_ic=float(ic), native_block_validation_ic=float(native_ic), scored_days=len(daily), scored_dates=[str(daily.index.min()), str(daily.index.max())], rows=len(predictions), seconds=time.monotonic()-tick))
        save('validation-progress.json', dict(results=results, elapsed_seconds=time.monotonic()-start))
        print(json.dumps(results[-1]), flush=True)
        # Native validation promises to restore the entire model and optimizer checkpoint.
        for key, value in checkpoint['framework'].items():
            # Check values on the same device; CUDA validation restores CUDA weights.
            assert torch.equal(value.cpu(), manager.framework.state_dict()[key].cpu())
    winner = max(results, key=lambda r: r['validation_ic'])
    save('selection.json', dict(winner=winner, all_candidates=results, selection_complete_before_test=True))
    if not args.evaluate:
        return
    if not args.clean and winner['candidate'] == 0:
        baseline = json.loads((ROOT / 'results/doubleadapt-script-matched/summary.json').read_text())
        save('summary.json', dict(status='selected_native_configuration_existing_evaluation_reused', selection=winner,
                                  metrics=baseline['metrics'], evaluation_source='doubleadapt-script-matched',
                                  no_new_test_evaluation=True, seconds=time.monotonic()-start))
        return
    manager.load_state_dict(copy.deepcopy(checkpoint))
    manager.online_lr = winner['rates'].copy()
    manager.lr_model = .001
    manager.fitted = True
    rec = exp.online_training(exp.meta_dataset_online, manager)
    pred, label = rec.load_object('pred.pkl'), rec.load_object('label.pkl')
    joined = pd.concat([pred.iloc[:, 0].rename('prediction'), label.iloc[:, 0].rename('raw_label')], axis=1).dropna()
    baseline = pd.read_pickle(ROOT / 'results/doubleadapt-script-matched/predictions-raw-labels.pkl')
    assert joined.index.equals(baseline.index), 'Evaluation rows must remain identical'
    assert np.array_equal(joined.raw_label.values, baseline.raw_label.values)
    joined.to_pickle(out / 'predictions-raw-labels.pkl')
    daily = joined.groupby(level='datetime').apply(lambda x: pd.Series(dict(ic=x.prediction.corr(x.raw_label), rank_ic=x.prediction.corr(x.raw_label, method='spearman'))))
    daily.to_csv(out / 'daily-correlations.csv')
    metrics = rec.list_metrics()
    for key, val in [('IC', daily.ic.mean()), ('Rank IC', daily.rank_ic.mean()),
                     ('ICIR', daily.ic.mean()/daily.ic.std()), ('Rank ICIR', daily.rank_ic.mean()/daily.rank_ic.std())]:
        assert np.isclose(metrics[key], val, rtol=1e-7, atol=1e-9)
    torch.save(manager.state_dict(), out / 'online-model.pt')
    save('summary.json', dict(status='validation_selected_evaluation_complete', selection=winner, metrics=metrics,
                              observations=len(joined), trading_days=len(daily), seconds=time.monotonic()-start,
                              comparison='Single seed, paper-described validation tuning; not the published 10-run mean.',
                              full_cz_authorized_by_gate=False))


if __name__ == '__main__':
    main()
