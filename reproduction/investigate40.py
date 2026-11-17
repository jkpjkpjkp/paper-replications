"""CPU-only forensic checks of the completed 40-asset result; no fitting/tuning."""
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import joblib
import numpy as np
import pandas as pd
import torch
from pandas.tseries.offsets import BDay

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_provider.data_loader import Dataset_Sig, PrecomputedSigDataset
from exp.exp_main import REBAL_DATE_SET
from model.SIT import Model
from audit_results import metrics

OUT = ROOT / 'reproduction/discrepancy40'
OUT.mkdir(exist_ok=True)
torch.set_num_threads(2)
started = time.perf_counter()
prices = pd.read_csv(ROOT / 'asset_data/full_dataset.csv', parse_dates=['Date']).set_index('Date').iloc[:, :40]
frame = prices.loc[(pd.Timestamp('2020-01-01') - BDay(61)).strftime('%Y-%m-%d'):'2024-12-31']
scaler = joblib.load(ROOT / 'signature_cache_6020/pool_40/signature_scaler.pkl')
original = Dataset_Sig(frame, frame, 60, 20, sig_scaler=scaler,
                       pred_start='2020-01-01', pred_end='2024-12-31')
cached = PrecomputedSigDataset(str(ROOT / 'signature_cache_6020/pool_40'), 'test')
cfg = SimpleNamespace(data_pool=40, d_model=8, dropout=.1, ff_dim=32,
                      n_heads=4, num_layers=2, sig_input_dim=2, cross_sig_dim=1,
                      hidden_c=8, max_position=5, time_feat_dim=3)
fresh_paths = sorted((ROOT / 'reproduction/fresh_local/results').glob('dp40_*test_positions.csv'))
assert len(fresh_paths) == 3
reference = pd.read_csv(fresh_paths[0], parse_dates=['Date'])
decision_indices = [i for i, d in enumerate(reference.Date) if i == 0 or str(d.date()) in REBAL_DATE_SET]
samples = {}
cache_max_error = {k: 0. for k in ['x_sigs', 'cross_sigs', 'date_feats', 'future_return_unscaled']}
for count, i in enumerate(decision_indices):
    sample_index = max(0, i - 19)
    a, b = original[sample_index], cached[sample_index]
    assert a['dates_horizon'] == b['dates_horizon']
    for k in cache_max_error:
        cache_max_error[k] = max(cache_max_error[k], float(np.max(np.abs(a[k] - b[k]))))
        assert np.array_equal(a[k], b[k]), (sample_index, k)
    assert a['dates_horizon'][i - sample_index] == str(reference.Date.iloc[i].date())
    samples[i] = (a, i - sample_index)
    if count % 15 == 0:
        print(f'Original-loader decisions checked: {count+1}/{len(decision_indices)}', flush=True)

inference_checks = []
for path in fresh_paths:
    run = int(re.search(r'_(\d+)_test_positions.csv$', path.name)[1])
    positions = pd.read_csv(path, parse_dates=['Date'])
    ckpt = ROOT / 'reproduction/fresh_local/checkpoints' / path.name.replace('_test_positions.csv', '') / 'checkpoint.pth'
    model = Model(cfg).eval()
    model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
    inferred_weights = np.empty((len(positions), 40), dtype=np.float32)
    for j, i in enumerate(decision_indices):
        sample, horizon_index = samples[i]
        with torch.no_grad():
            tensors = [torch.from_numpy(sample[k]).unsqueeze(0) for k in ['x_sigs', 'cross_sigs', 'date_feats']]
            w = torch.softmax(model(*tensors)[1] / 1.3, dim=-1)[0, horizon_index].numpy()
        stop = decision_indices[j+1] if j+1 < len(decision_indices) else len(positions)
        inferred_weights[i:stop] = w
    observed = positions.filter(like='Weight_').values
    max_error = float(np.max(np.abs(observed-inferred_weights)))
    indices = prices.index.get_indexer(positions.Date)
    raw = prices.values.astype(np.float32)
    rets = (raw[indices+1]-raw[indices])/(raw[indices]+1e-8)
    replay = np.array([float(np.dot(w,r)) for w,r in zip(inferred_weights, rets)])
    saved = pd.read_csv(str(path).replace('positions.csv','metrics.csv')).iloc[0]
    replay_metrics = metrics(replay)
    inference_checks.append(dict(run=run, decisions=len(decision_indices),
        original_vs_cached_max_error=cache_max_error,
        cpu_original_input_vs_saved_gpu_weight_max_error=max_error,
        original_loader_checkpoint_replayed_metrics=replay_metrics,
        metric_max_error=max(abs(replay_metrics[k]-saved[k]) for k in replay_metrics)))
    assert max_error < 2e-6, inference_checks[-1]
    print('Checkpoint checked:', run, 'max weight error', max_error, flush=True)

# Attribution is descriptive accounting of the original frozen runs.
# Assets receive log(1+portfolio_return) in proportion to their same-day
# simple-return contribution. Their sums exactly equal log wealth.
year_rows, asset_rows, weight_rows, run_rows = [], [], [], []
for label, folder in [('shipped', ROOT/'results'), ('fresh', ROOT/'reproduction/fresh_local/results')]:
    for path in sorted(folder.glob('dp40_*test_positions.csv')):
        pos = pd.read_csv(path, parse_dates=['Date'])
        run = int(re.search(r'_(\d+)_test_positions.csv$', path.name)[1])
        idx = prices.index.get_indexer(pos.Date)
        raw = prices.values.astype(np.float32)
        asset_ret = (raw[idx+1]-raw[idx])/(raw[idx]+1e-8)
        w = pos.filter(like='Weight_').values.astype(np.float32)
        contributions = w.astype(np.float64)*asset_ret.astype(np.float64)
        r = contributions.sum(axis=1)
        scale = np.ones(len(r))
        np.divide(np.log1p(r),r,out=scale,where=r!=0)
        allocated_log = contributions*scale[:,None]
        assert abs(allocated_log.sum()-np.log1p(r).sum())<1e-12
        economic_end = prices.index[idx+1]
        for year in sorted(set(economic_end.year)):
            mask = economic_end.year == year
            year_rows.append(dict(source=label,run=run,year=int(year),days=int(mask.sum()),
                wealth_factor=float(np.prod(1+r[mask])),log_growth=float(np.log1p(r[mask]).sum())))
        for j, asset in enumerate(prices.columns):
            asset_rows.append(dict(source=label,run=run,asset=asset,
                allocated_log_growth=float(allocated_log[:,j].sum()),mean_weight=float(w[:,j].mean())))
        weight_rows.append(dict(source=label,run=run,
            mean_HHI=float(np.square(w).sum(axis=1).mean()),
            mean_effective_assets=float((1/np.square(w).sum(axis=1)).mean()),
            mean_largest_weight=float(w.max(axis=1).mean())))
        run_rows.append(dict(source=label,run=run,**metrics(r)))
pd.DataFrame(year_rows).to_csv(OUT/'yearly_growth.csv',index=False)
pd.DataFrame(asset_rows).to_csv(OUT/'asset_log_growth_attribution.csv',index=False)
pd.DataFrame(weight_rows).to_csv(OUT/'portfolio_concentration.csv',index=False)
pd.DataFrame(run_rows).to_csv(OUT/'per_run_comparison.csv',index=False)
summary = dict(scope='40 assets only; no extra training or selection',
    inference_checks=inference_checks,seconds=time.perf_counter()-started,
    attribution_note='Post hoc descriptive accounting, not a causal explanation of training differences. Economic return end dates used for years; identical 1255-day spans.')
(OUT/'forensic_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
