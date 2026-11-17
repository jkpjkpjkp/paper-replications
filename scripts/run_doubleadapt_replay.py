"""Predeclared fixed-checkpoint replay: control, then change only online lr_da."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
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
OUT = ROOT/'results/doubleadapt-online-lr-diagnostic'


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
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    if args.control: OUT = ROOT/'results/doubleadapt-replay-control'
    if (OUT/'summary.json').exists() and not args.preflight:
        raise RuntimeError('Complete experiment already exists; refusing overwrite')
    import numpy as np
    import pandas as pd
    import torch
    import qlib
    # Import the built fork before adding examples: the source tree has no Cython .so files.
    sys.path.insert(0, str(REPO))
    torch.set_num_threads(2)
    init_original = qlib.init
    def init_local(*a, **kw):
        kw.update(kernels=2, joblib_backend='threading', expression_cache=None, dataset_cache=None)
        return init_original(*a, **kw)
    qlib.init = init_local
    mainpath = REPO/'examples/benchmarks_dynamic/incremental/main.py'
    spec = importlib.util.spec_from_file_location('author_doubleadapt_main', mainpath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)
    started = time.monotonic()
    state = dict(stage='preparing_data', started_utc=datetime.now(timezone.utc).isoformat(), epochs_trained=0)
    save('status.json', state)
    config = dict(data_dir='crowd_data', root_path=str(ROOT/'data')+'/',
                  market='csi300', forecast_model='GRU', alpha=360, step=20,
                  rank_label=False, num_head=8, tau=10, lr=0.001, lr_da=0.01,
                  online_lr={'lr':0.001, 'lr_da':0.0001, 'lr_ma':0.001} if args.control else None,
                  begin_valid_epoch=10, preprocess_tensor=True, save=False,
                  tag='replay-control' if args.control else 'replay-online-da001', h_path=str(ROOT/'results/doubleadapt-original/handler.pkl'))
    provenance = dict(config=config, numpy_seed=0, torch_seed=43, device='cpu',
                      source_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip(),
                      data_receipt=json.loads((ROOT/'data/doubleadapt-data-receipt.json').read_text()),
                      runner_sha256=sha(__file__),
                      paper_target={'IC':0.0687,'ICIR':0.5497,'Rank IC':0.0621,'Rank ICIR':0.5110,
                                    '1day.excess_return_with_cost.annualized_return':0.1296,
                                    '1day.excess_return_with_cost.information_ratio':1.5123},
                      comparison='Diagnostic only: identical saved offline checkpoint, changing online data-adapter LR from 0.0001 to 0.01; not test-set tuning or full run.sh replication.',
                      offline_checkpoint_sha256=sha(ROOT/'results/doubleadapt-original/offline-model.pt'),
                      choices=['Control replays original online settings; treatment removes online override, retaining offline lr_da=0.01.',
                               'No offline retraining. Uses the original saved best framework and optimizer states.',
                               'Same CPU PyTorch 1.9, source, data, task sequence and backtest; hold other changes fixed.',
                               'Report both outcomes; do not select whichever resembles the published table.'])
    save('protocol.json', provenance)
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
    native_epoch = module.DoubleAdaptManager.run_epoch
    def logged_epoch(self, phase, task_list, *a, **kw):
        state.update(stage='offline_training' if phase in ['train','test'] else 'online_training', phase=phase,
                     tasks=len(task_list), wall_seconds=time.monotonic()-started)
        save('status.json', state)
        result = native_epoch(self, phase, task_list, *a, **kw)
        if phase == 'train':
            state['epochs_trained'] += 1
        if phase == 'test':
            state['validation_ic'] = float(result[1])
        state['wall_seconds'] = time.monotonic()-started
        save('status.json', state)
        print(json.dumps(state), flush=True)
        return result
    module.DoubleAdaptManager.run_epoch = logged_epoch
    trained_start = time.monotonic()
    torch.manual_seed(43)
    manager = module.DoubleAdaptManager(exp.basic_task, x_dim=exp.x_dim, lr_model=exp.lr,
        weight_decay=exp.weight_decay, first_order=exp.first_order, alpha=exp.alpha,
        begin_valid_epoch=exp.begin_valid_epoch, factor_num=exp.factor_num,
        lr_da=exp.lr_da, lr_ma=exp.lr_ma, online_lr=exp.online_lr,
        lr_x=exp.lr_x, lr_y=exp.lr_y, adapt_x=exp.adapt_x, adapt_y=exp.adapt_y,
        reg=exp.reg, num_head=exp.num_head, temperature=exp.temperature)
    checkpoint = torch.load(ROOT/'results/doubleadapt-original/offline-model.pt', map_location='cpu')
    manager.load_state_dict(checkpoint)
    manager.fitted = True
    for key, value in checkpoint['framework'].items():
        assert torch.equal(value, manager.framework.state_dict()[key])
    offline_seconds = time.monotonic()-trained_start
    torch.save(manager.state_dict(), OUT/'offline-model.pt')
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
    save('summary.json', dict(status='diagnostic_replay_complete',
                             metrics=native_metrics, independently_verified=independent,
                             comparison=comparison, dates=[str(daily.index.min()),str(daily.index.max())],
                             trading_days=len(daily), observations=len(joined),
                             checkpoint_load_seconds=offline_seconds,online_seconds=online_seconds,
                             offline_training_performed=False,control_run=args.control,
                             total_seconds=time.monotonic()-started,
                             full_cz_authorized_by_gate=False))
    save('status.json', dict(stage='diagnostic_replay_complete',wall_seconds=time.monotonic()-started))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        save('status.json',dict(stage='failed',error=repr(e),updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
