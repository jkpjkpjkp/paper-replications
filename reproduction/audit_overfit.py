"""Post-hoc inspection of frozen predictors; no model/strategy selection.

Primary comparator: one per-asset mean using every row before the original
80% split. Test targets, preprocessing and saved weights are exactly those of
the release replay. Raw-unit errors invert its full-panel StandardScaler.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '2')
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'upstream/src'))
from data import prepare_time_series_data
from model import dense_nn_mlp, predict_all
from utils import set_seed
torch.set_num_threads(2)
OUT = ROOT/'results/overfit_audit'
MARKETS = ['euro50', 'ftse100', 'nikkei225', 'kospi200']
SEEDS = [16, 37, 47, 214, 1947]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def panel(market, raw=False):
    path = ROOT/f'data/daily/{market}_160101_251231.csv'
    frame = pd.read_csv(path, index_col=0)
    cut = int(.8*len(frame))
    scaler = StandardScaler().fit(frame)
    values = frame if raw else pd.DataFrame(scaler.transform(frame), index=frame.index, columns=frame.columns)
    splits = {}
    for name, start, end in [('train', 0, cut), ('test', cut, len(frame))]:
        # Literal-method tests keep the first 100 test dates; release tests do not.
        window_start = start-100 if raw and name == 'test' else start
        X, y = prepare_time_series_data(values.iloc[window_start:end], 100)
        first = window_start+100
        splits[name] = {'X': X.flatten(1), 'y': y, 'dates': frame.index[first:end].tolist(),
                        'raw_y': frame.iloc[first:end].to_numpy(np.float64)}
    return frame, scaler, splits, cut, path


def infer(path, n, X, literal=False):
    state = torch.load(path, map_location='cpu', weights_only=not literal)
    if literal:
        state = state['model']
    model = dense_nn_mlp(n*100, n, [512, 256], 'tanh', 0. if literal else .1)
    model.load_state_dict(state)
    model.eval()
    pred = predict_all(model, X).numpy()
    if not np.isfinite(pred).all():
        raise ValueError(f'Nonfinite predictions: {path}')
    return pred, sum(p.numel() for p in model.parameters())


def block_ci(model, base, block, seed=829):
    """Paired circular moving-block bootstrap, dates are the sampling unit."""
    rng = np.random.default_rng(seed)
    T = len(base)
    starts = rng.integers(0, T, size=(2000, (T+block-1)//block))
    idx = ((starts[..., None]+np.arange(block)) % T).reshape(2000, -1)[:, :T]
    ratios = model[idx].mean(1)/base[idx].mean(1)
    return np.quantile(ratios, [.025, .975]).tolist()


def predictions():
    OUT.mkdir(parents=True, exist_ok=True)
    daily_dir = OUT/'daily_errors'; daily_dir.mkdir(exist_ok=True)
    rows = []; receipts = []; checks = []
    for market in MARKETS:
        frame, scaler, splits, cut, data_path = panel(market)
        n = frame.shape[1]
        full_mean = scaler.transform(frame).astype(np.float64)[:cut].mean(0)
        target_mean = splits['train']['y'].numpy().astype(np.float64).mean(0)
        baselines = {'train_mean': full_mean, 'train_target_mean': target_mean,
                     'raw_zero': -scaler.mean_/scaler.scale_}
        candidates = []
        for name, value in baselines.items():
            candidates.append((name, name, None, None, value))
        candidates.append(('rolling100', 'rolling100', None, None, None))
        for seed in SEEDS:
            candidates.append((f'PFL_seed{seed}', 'PFL', seed, None,
                               ROOT/f'artifacts_released/{market}_PFL_MSE_seed{seed}/model.pt'))
            for rho in [10, 15, 20]:
                k = round(n*rho/100)
                candidates.append((f'DFL_rho{rho}_seed{seed}', 'DFL', seed, rho,
                                   ROOT/f'artifacts_released/{market}_k{rho}_actualk{k}_alpha0p5_seed{seed}/model.pt'))
        for run, method, seed, rho, item in candidates:
            daily = {}; row = {'setting': 'release', 'market': market, 'method': method,
                               'run': run, 'seed': seed, 'rho': rho, 'n_assets': n,
                               'train_dates': len(splits['train']['y']), 'test_dates': len(splits['test']['y'])}
            for split, data in splits.items():
                if isinstance(item, Path):
                    pred, parameters = infer(item, n, data['X'])
                    row['parameters'] = parameters
                elif method == 'rolling100':
                    pred = data['X'].reshape(-1,100,n).mean(1).numpy()
                else:
                    pred = np.broadcast_to(item, data['y'].shape)
                pred = pred.astype(np.float64)
                norm_errors = ((pred-data['y'].numpy().astype(np.float64))**2).mean(1)
                raw_prediction = pred*scaler.scale_+scaler.mean_
                raw_errors = ((raw_prediction-data['raw_y'])**2).mean(1)
                for unit, errors in [('normalized', norm_errors), ('raw', raw_errors)]:
                    row[f'{split}_{unit}_mse'] = float(errors.mean())
                    daily[f'{split}_{unit}'] = errors
                daily[f'{split}_dates'] = np.array(data['dates'])
            if isinstance(item, Path):
                meta = json.loads(item.with_name('meta.json').read_text())
                expected = meta['test_mse'] if method == 'PFL' else meta['test_avg_pfl_loss']
                error = abs(row['test_normalized_mse']-expected)
                if error > 2e-6:
                    raise AssertionError((item, error))
                checks.append({'model': str(item.relative_to(ROOT)), 'saved_test_mse_abs_error': error})
                receipts.append({'model': str(item.relative_to(ROOT)), 'sha256': sha(item)})
            np.savez_compressed(daily_dir/f'{market}_{run}.npz', **daily)
            rows.append(row)
        receipts.append({'data': str(data_path.relative_to(ROOT)), 'sha256': sha(data_path),
                         'split': str(frame.index[cut]), 'test_first': splits['test']['dates'][0],
                         'test_last': splits['test']['dates'][-1]})
        pd.DataFrame(rows).to_csv(OUT/'prediction_runs.csv', index=False)
        print(f'Prediction inspection complete: {market}', flush=True)
        del splits; gc.collect()
    df = pd.DataFrame(rows)
    groups = []
    for market in MARKETS:
        baseline = df[(df.market==market)&(df.method=='train_mean')].iloc[0]
        base_daily = np.load(daily_dir/f'{market}_train_mean.npz')
        for method in ['PFL', 'DFL']:
            for rho in ([None] if method=='PFL' else [10,15,20]):
                selected = df[(df.market==market)&(df.method==method)]
                if rho is not None:
                    selected = selected[selected.rho==rho]
                row = {'market':market, 'method':method, 'rho':rho, 'seeds':len(selected)}
                for unit in ['normalized', 'raw']:
                    for split in ['train', 'test']:
                        key = f'{split}_{unit}_mse'
                        row[key] = selected[key].mean()
                        row[f'{split}_{unit}_ratio'] = selected[key].mean()/baseline[key]
                    daily_model = np.mean([np.load(daily_dir/f'{market}_{run}.npz')[f'test_{unit}'] for run in selected.run], axis=0)
                    daily_base = base_daily[f'test_{unit}']
                    for block in [20,60]:
                        lo,hi = block_ci(daily_model,daily_base,block)
                        row[f'test_{unit}_ratio_ci{block}_low'] = lo
                        row[f'test_{unit}_ratio_ci{block}_high'] = hi
                    row[f'test_{unit}_seeds_worse'] = int((selected[f'test_{unit}_mse']>baseline[f'test_{unit}_mse']).sum())
                    row[f'test_{unit}_r2_vs_train_mean'] = 1-row[f'test_{unit}_ratio']
                groups.append(row)
    pd.DataFrame(groups).to_csv(OUT/'prediction_comparison.csv', index=False)
    save_json(OUT/'prediction_verification.json', {'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'80 saved release models, alpha=.5 DFL, all four markets, all five seeds; no training or test selection',
              'baseline':'per-asset fixed full-training-segment mean; target-only mean and raw zero also retained',
              'bootstrap':'2000 paired circular moving-block resamples of whole dates; seed/rho comparisons are dependent, not independent market observations',
              'receipts':receipts,'saved_mse_checks':checks, 'script_sha256':sha(__file__)})


def curves():
    """Re-run unchanged source trainer with read-only optimizer post-step hook."""
    import train as author_train
    from losses import get_loss_fn
    from types import SimpleNamespace
    frame, scaler, splits, cut, path = panel('euro50')
    train = [SimpleNamespace(x=x, y_norm=y) for x,y in zip(splits['train']['X'], splits['train']['y'])]
    problem = SimpleNamespace(get_train_data=lambda:train, device=torch.device('cpu'))
    n=frame.shape[1]; records=[]; checks=[]
    original_adam=torch.optim.Adam
    for seed in SEEDS:
        set_seed(seed)
        model=dense_nn_mlp(n*100,n,[512,256],'tanh',.1)
        update=[0]
        def snapshot():
            training=model.training
            rng=torch.get_rng_state().clone()
            row={'market':'euro50','seed':seed,'updates':update[0]}
            for split,data in splits.items():
                pred=predict_all(model,data['X']).double()
                row[f'{split}_mse']=float((pred-data['y'].double()).square().mean())
            records.append(row)
            model.train(training)
            assert torch.equal(rng,torch.get_rng_state()), 'Evaluation changed dropout RNG'
        snapshot()
        def hook(optimizer,args,kwargs):
            update[0]+=1
            if update[0] in [5,10,25,50,100,150,200,250,300]:
                snapshot()
        def factory(*args,**kwargs):
            optimizer=original_adam(*args,**kwargs)
            optimizer.register_step_post_hook(hook)
            return optimizer
        torch.optim.Adam=factory
        try:
            model,_=author_train.train_model(model,problem,get_loss_fn('mse',problem),300,64,.001,.0001,seed,log_prefix='overfit-inspection')
        finally:
            torch.optim.Adam=original_adam
        saved=torch.load(ROOT/f'artifacts_released/euro50_PFL_MSE_seed{seed}/model.pt',weights_only=True,map_location='cpu')
        error=max(float((v-saved[k]).abs().max()) for k,v in model.state_dict().items())
        if error!=0:
            raise AssertionError(f'Learning-curve trainer drifted from saved model: {error}')
        checks.append({'seed':seed,'final_model_max_abs_error':error})
        pd.DataFrame(records).to_csv(OUT/'learning_curves.csv',index=False)
        save_json(OUT/'learning_curve_verification.json',{'checks':checks,'purpose':'Post-hoc diagnostic, no checkpoint selection; author train_model unchanged; snapshots preserve RNG'})
        print(f'Curve inspection complete: seed {seed}; final parameter error {error}',flush=True)


def portfolios():
    from problem import SparseSharpeProblem
    from evaluate import evaluate_split, EvalConfig
    from artifacts import make_exp_name
    portfolio_dir=OUT/'portfolios';portfolio_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for market in MARKETS:
        problem=SparseSharpeProblem(market,ROOT,torch.device('cpu'))
        normalized=StandardScaler().fit_transform(problem.df_raw)
        cut=int(.8*len(normalized))
        mean=torch.tensor(normalized[:cut].mean(0),dtype=torch.float32)
        forecasts=mean[None,:].expand(len(problem.test),-1).contiguous()
        for rho in [10,15,20]:
            k=round(problem.n_assets*rho/100)
            for selector in ['oscar','oscar_hard']:
                name=f'{market}_rho{rho}_{selector}'
                trace_path=portfolio_dir/f'{name}.csv'
                if trace_path.exists():
                    base=pd.read_csv(trace_path,index_col=0)
                else:
                    metrics,trace,_=evaluate_split(problem,problem.test,forecasts,k,selector,EvalConfig(print_every=200))
                    base=pd.DataFrame.from_dict(trace,orient='index');base.index.name='date'
                    base.to_csv(trace_path)
                base_r=base.ret_true_raw.to_numpy()
                base_sr=base_r.mean()/base_r.std(ddof=1)
                for method in (['PFL','DFL'] if selector=='oscar' else ['PFL']):
                    learned=[];srs=[]
                    for seed in SEEDS:
                        exp=make_exp_name(method.lower(),market,rho,k,.5 if method=='DFL' else 0.,seed,selector)
                        path=ROOT/'artifacts_released'/exp/'test_trace.csv'
                        frame=pd.read_csv(path,index_col=0)
                        if not frame.index.equals(base.index):
                            raise AssertionError(f'Unmatched test dates: {path}')
                        r=frame.ret_true_raw.to_numpy();learned.append(r);srs.append(r.mean()/r.std(ddof=1))
                    learned=np.array(learned)
                    rng=np.random.default_rng(829);T=len(base_r);block=20
                    starts=rng.integers(0,T,size=(2000,(T+block-1)//block))
                    idx=((starts[...,None]+np.arange(block))%T).reshape(2000,-1)[:,:T]
                    boot_base=base_r[idx];boot_model=learned[:,idx]
                    boot_delta=(boot_model.mean(2)/boot_model.std(2,ddof=1)).mean(0)-boot_base.mean(1)/boot_base.std(1,ddof=1)
                    lo,hi=np.quantile(boot_delta,[.025,.975])
                    rows.append({'market':market,'rho':rho,'selector':selector,'method':method,
                                 'test_days':T,'fixed_train_mean_daily_sr':base_sr,'model_daily_sr_mean':np.mean(srs),
                                 'model_daily_sr_seed_sd':np.std(srs,ddof=1),'sr_delta':np.mean(srs)-base_sr,
                                 'sr_delta_ci20_low':lo,'sr_delta_ci20_high':hi,
                                 'model_seeds_worse':sum(sr<base_sr for sr in srs),
                                 'fixed_mean_daily_return':base_r.mean(),'model_daily_return_seed_mean':learned.mean(),
                                 'fixed_mean_max_gross':base.weight_abs_sum.max(),
                                 'fixed_mean_max_budget_error':float(abs(base.weight_sum-1).max()),
                                 'fixed_mean_holdings_max':base['weight_nnz_1e-8'].max()})
                pd.DataFrame(rows).to_csv(OUT/'portfolio_comparison.csv',index=False)
        print(f'Fixed-mean portfolio inspection complete: {market}',flush=True)
        del problem;gc.collect()


def literal_predictions():
    rows=[]
    for market in MARKETS:
        frame,scaler,splits,cut,path=panel(market,raw=True)
        n=frame.shape[1]
        means={'train_mean':frame.iloc[:cut].to_numpy().mean(0),
               'train_target_mean':frame.iloc[100:cut].to_numpy().mean(0),'raw_zero':np.zeros(n)}
        models=[]
        for result in sorted((ROOT/'results/paper_method/study').glob(f'{market}_*/result.json')):
            info=json.loads(result.read_text())
            if info['method'] not in ['dfl','pfl'] or not (result.parent/'checkpoint.pt').exists():
                continue
            models.append((result.parent.name,info['method'],info.get('rho_percent'),info.get('seed'),result.parent/'checkpoint.pt',info))
        for name,method,rho,seed,item,info in [*[(k,k,None,None,v,{}) for k,v in means.items()],*models]:
            row={'market':market,'method':method,'rho':rho,'seed':seed,'run':name,
                 'epochs':info.get('epochs'),'optimizer':info.get('optimizer'),'status':info.get('status')}
            initial=None
            if isinstance(item,Path):
                set_seed(seed)
                initial=dense_nn_mlp(n*100,n,[512,256],'tanh',0.)
            for split,data in splits.items():
                pred,_=infer(item,n,data['X'],literal=True) if isinstance(item,Path) else (item,None)
                row[f'{split}_raw_mse']=float(np.mean((pred-data['raw_y'])**2))
                if initial is not None:
                    init_pred=predict_all(initial,data['X']).numpy()
                    row[f'{split}_initial_raw_mse']=float(np.mean((init_pred-data['raw_y'])**2))
            rows.append(row)
        del splits;gc.collect()
    pd.DataFrame(rows).to_csv(OUT/'literal_prediction_runs.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['predictions','curves','portfolios','literal'])
    args=parser.parse_args()
    {'predictions':predictions,'curves':curves,'portfolios':portfolios,'literal':literal_predictions}[args.phase]()
