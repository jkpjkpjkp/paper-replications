"""Paper-grounded five-session task candidate with verified input cleaning and native full budget."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
# Native CUDA candidate, conditional on the standalone full-task capacity check.
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT/'sources/DoubleAdapt-qlib'
OUT = ROOT/'results/gap-resolution-20261113/available-inputs-step5'


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/name
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False))
    temp.replace(p)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if (OUT/'summary.json').exists() and not args.preflight:
        raise RuntimeError('Complete experiment already exists; refusing overwrite')
    import numpy as np
    import pandas as pd
    import torch
    import qlib
    # Import the built fork before adding examples: the source tree has no Cython .so files.
    sys.path.insert(0, str(REPO))
    torch.set_num_threads(2)
    capacity=json.loads((ROOT/'results/gap-resolution-20261113/cuda-step5-probe.json').read_text())
    assert capacity['passed'], 'Native CUDA full-task capacity must pass first'
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(0.30)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    init_original = qlib.init
    def init_local(*a, **kw):
        kw.update(kernels=2, joblib_backend='threading', expression_cache=None, dataset_cache=None)
        return init_original(*a, **kw)
    qlib.init = init_local
    mainpath = REPO/'examples/benchmarks_dynamic/incremental/main.py'
    spec = importlib.util.spec_from_file_location('author_doubleadapt_main', mainpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    contract = json.loads((ROOT/'results/gap-resolution-20261113/doubleadapt-data-contract.json').read_text())
    assert contract['passed'], 'Independent data contract must pass before training'
    native_task = module.DoubleAdaptManager.run_task
    def available_task(self, original, phase):
        task = original.copy()
        # Availability is determined from current inputs only, not future labels.
        support = task['X_train'].reshape(len(task['X_train']), -1).ne(0).any(dim=1)
        assert support.any()
        for key in ['X_train','y_train']:
            task[key] = task[key][support]
        if 'train_idx' in task: task['train_idx'] = task['train_idx'][support.numpy()]
        query = task['X_test'].reshape(len(task['X_test']), -1).ne(0).any(dim=1)
        assert query.any()
        assert 'X_extra' not in task, 'This controlled experiment retains use_extra=False'
        if phase == 'train':
            for key in ['X_test','y_test']:
                task[key] = task[key][query]
            task['test_idx'] = task['test_idx'][query.numpy()]
        else:
            # Keep all predictions and all evaluation labels. Mask ONLY the update loss.
            task['mask_y'] = query
        return native_task(self, task, phase)
    module.DoubleAdaptManager.run_task = available_task
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)
    started = time.monotonic()
    state = dict(stage='preparing_data', started_utc=datetime.now(timezone.utc).isoformat(), epochs_trained=0)
    save('status.json', state)
    config = dict(data_dir='crowd_data', root_path=str(ROOT/'data')+'/',
                  market='csi300', forecast_model='GRU', alpha=360, step=5,
                  rank_label=False, num_head=8, tau=10, lr=0.001, lr_da=0.01,
                  online_lr=None,
                  begin_valid_epoch=30, preprocess_tensor=True, save=False,
                  tag='available-inputs-step5-seed43', h_path=str(ROOT/'results/doubleadapt-original/handler.pkl'))
    provenance = dict(config=config, numpy_seed=0, torch_seed=43, device='cuda',torch_version=torch.__version__,
                      source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
                      data_receipt=json.loads((ROOT/'data/doubleadapt-data-receipt.json').read_text()),
                      runner_sha256=sha(__file__),
                      paper_target={'IC':0.0687,'ICIR':0.5497,'Rank IC':0.0621,'Rank ICIR':0.5110,
                                    '1day.excess_return_with_cost.annualized_return':0.1296,
                                    '1day.excess_return_with_cost.information_ratio':1.5123},
                      comparison='Paper Appendix B 5-session task candidate with verified unavailable-input loss mask. Same architecture/source/seed/native budget; modern CUDA runtime. Validation selection only, not an exact paper/runtime attribution.',
                      choices=['Offline checkpoint selection uses only the 479 common validation dates 2015-01-05 through 2016-12-19. The release extra 2014-12-31 score precedes final training-label availability and is excluded.',
                               'Exclude proven all-zero/missing-current-price inputs from support and query update losses; retain full evaluation universe, released preprocessing and all output rows.',
                               'Paper Appendix B specifies 5-session tasks as an improvement candidate; remaining run.sh settings including begin_valid_epoch=30 and no online LR override retained. Candidate declared before evaluation.',
                               'PyTorch 2.4.1 CUDA, FP32 with TF32 disabled; native full 5-session tasks. Explicit runtime difference from the preserved CPU 20-session baseline.',
                               'Qlib loading uses two threads; no model or preprocessing math changed.',
                               'One complete run is the first seed of native run_all (NumPy=0, Torch=43), as in the preserved original baseline. This does not reproduce the published multi-seed average.'])
    save('protocol.json', provenance)
    (OUT/'runner-at-launch.py').write_bytes(Path(__file__).read_bytes())
    np.random.seed(0)
    random.seed(0)
    exp = module.Incremental(**config)
    module.R.set_uri((OUT/'mlruns').as_uri())
    exp.meta_dataset_offline, exp.meta_dataset_online = exp.dump_data()
    save('data-preflight.json', dict(x_dim=exp.x_dim, factor_num=exp.factor_num,
                                   offline_tasks=len(exp.meta_dataset_offline.meta_task_l),
                                   online_tasks=len(exp.meta_dataset_online.meta_task_l),
                                   segments=exp.basic_task['dataset']['kwargs']['segments']))
    assert exp.x_dim == 360 and exp.factor_num == 6
    if args.preflight:
        save('status.json', dict(stage='data_preflight_complete', wall_seconds=time.monotonic()-started))
        return
    best_full = [None, -float('inf'), None]
    native_epoch = module.DoubleAdaptManager.run_epoch
    def logged_epoch(self, phase, task_list, *a, **kw):
        state.update(stage='offline_training' if phase in ['train','test'] else 'online_training', phase=phase,
                     tasks=len(task_list), wall_seconds=time.monotonic()-started)
        save('status.json', state)
        result = native_epoch(self, phase, task_list, *a, **kw)
        if phase == 'test':
            # The paper's validation years exclude the release's extra 2014-12-31 day.
            # Score on the same 479 dates as the prespecified 20-session comparison.
            validation = result[0].loc['2015-01-05':'2016-12-19']
            daily_ic = validation.groupby(level='datetime').apply(lambda g: g.pred.corr(g.label))
            assert len(daily_ic) == 479
            state['native_block_validation_ic'] = float(result[1])
            result = (result[0], float(daily_ic.mean()))
        if phase == 'train':
            state['epochs_trained'] += 1
        if phase == 'test':
            state['validation_ic'] = float(result[1])
            if float(result[1]) >= best_full[1]:
                best_full[:] = [copy.deepcopy(self.state_dict()),float(result[1]),state['epochs_trained']]
                torch.save(best_full[0], OUT/'best-coherent-state.pt')
                save('best-validation.json',dict(ic=best_full[1],epoch=best_full[2]))
        state['wall_seconds'] = time.monotonic()-started
        save('status.json', state)
        print(json.dumps(state), flush=True)
        return result
    module.DoubleAdaptManager.run_epoch = logged_epoch
    trained_start = time.monotonic()
    manager = exp.offline_training(seed=43)
    offline_seconds = time.monotonic()-trained_start
    torch.save(manager.state_dict(), OUT/'offline-model.pt')
    save('optimizer-state-review.json', dict(best_validation_epoch=best_full[2],final_epoch=state['epochs_trained'],
        best_validation_ic=best_full[1],note='Native fit restores best framework weights only. Saved full best state separately for a prespecified validation-only optimizer-state comparison; primary output retains native behavior.'))
    online_start = time.monotonic()
    rec = exp.online_training(exp.meta_dataset_online, manager)
    online_seconds = time.monotonic()-online_start
    prediction = rec.load_object('pred.pkl')
    labels = rec.load_object('label.pkl')
    joined = pd.concat([prediction.iloc[:,0].rename('prediction'), labels.iloc[:,0].rename('raw_label')], axis=1).dropna()
    joined.to_pickle(OUT/'predictions-raw-labels.pkl')
    daily = joined.groupby(level='datetime').apply(lambda x: pd.Series({
        'ic': x.prediction.corr(x.raw_label),
        'rank_ic': x.prediction.rank(method='average').corr(x.raw_label.rank(method='average'))}))
    daily.to_csv(OUT/'daily-correlations.csv')
    independent = {'IC':float(daily.ic.mean()),'ICIR':float(daily.ic.mean()/daily.ic.std()),
                   'Rank IC':float(daily.rank_ic.mean()),'Rank ICIR':float(daily.rank_ic.mean()/daily.rank_ic.std())}
    native_metrics = rec.list_metrics()
    for key, value in independent.items():
        if not np.isclose(value, native_metrics[key], rtol=1e-7, atol=1e-9):
            raise AssertionError((key,value,native_metrics[key]))
    torch.save(manager.state_dict(), OUT/'online-model.pt')
    comparison = {k:{'paper':v,'run':float(native_metrics[k]),'difference':float(native_metrics[k])-v}
                  for k,v in provenance['paper_target'].items()}
    save('summary.json', dict(status='experiment_complete_pending_fidelity_review',
                             metrics=native_metrics, independently_verified=independent,
                             comparison=comparison, dates=[str(daily.index.min()),str(daily.index.max())],
                             trading_days=len(daily), observations=len(joined),
                             offline_seconds=offline_seconds,online_seconds=online_seconds,
                             total_seconds=time.monotonic()-started,
                             full_cz_authorized_by_gate=False))
    save('status.json', dict(stage='experiment_complete_pending_fidelity_review',wall_seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save('status.json',dict(stage='failed',error=repr(e),updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
