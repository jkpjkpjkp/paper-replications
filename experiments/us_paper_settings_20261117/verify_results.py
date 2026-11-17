"""Audit both frozen follow-up runs against source, prices, weights and forecasts."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/best_effort'
P = json.loads((ROOT / 'protocol.json').read_text())


def ledger_from_prices(prices, table, cost):
    """Independent fixed-share valuation using price ratios within each block."""
    all_nav = []
    prev_nav = 1.
    before_trade = np.zeros(prices.shape[1])
    dates = pd.to_datetime(table.iloc[:, 0])
    for block, day in enumerate(dates):
        start = prices.index.get_loc(day)
        stop = prices.index.get_loc(dates.iloc[block+1]) if block+1 < len(dates) else len(prices)
        target = table.iloc[block, 1:].to_numpy(dtype=float)
        turnover = np.abs(target - before_trade).sum()
        initial_nav = prev_nav * (1 - cost * turnover)
        relatives = prices.iloc[start:stop].to_numpy() / prices.iloc[start-1].to_numpy()
        block_gross_nav = relatives @ target
        all_nav.extend(initial_nav * block_gross_nav)
        prev_nav = all_nav[-1]
        before_trade = relatives[-1] * target / block_gross_nav[-1]
    return np.array(all_nav)


def main():
    pre = json.loads((ROOT / 'PRE_RUN_MANIFEST.json').read_text())
    for file, expected in pre.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file
    prices = pd.read_parquet(ROOT / 'data/US_prices.parquet')
    rets = np.log(prices).diff()
    results = []
    checks = []
    for seed in P['seeds']:
        result = json.loads((OUT / f'US_{seed}_result.json').read_text())
        state = torch.load(ROOT / f'checkpoints/US_{seed}.pt', map_location='cpu', weights_only=False)
        fingerprint = hashlib.sha256(b''.join((ROOT / file).read_bytes() for file in
            ['protocol.json', 'diffusion_model.py', 'run_empirical.py', 'data/US_prices.parquet', 'data/global_macro.parquet'])).hexdigest()
        assert state['fingerprint'] == fingerprint
        assert state['training_complete'] and state['training']['epochs'] == 200
        assert state['training']['selected_epoch'] == 200
        assert state['training']['train_last_label'] == '2024-09-30'
        history = pd.read_csv(OUT / f'US_{seed}_training.csv')
        assert history.epoch.tolist() == list(range(1, 201))
        assert np.isfinite(history.train_noise_mse).all()
        for row in result['metrics']:
            method, cost = row['method'], row['cost_bps']
            ledger = pd.read_csv(OUT / f'US_{seed}_{method}_{cost}bps_daily.csv')
            weights = pd.read_csv(OUT / f'US_{seed}_{method}_weights.csv')
            nav = ledger_from_prices(prices, weights, cost/10000)
            np.testing.assert_allclose(nav, ledger.wealth, rtol=1e-10, atol=1e-12)
            daily = nav / np.r_[1., nav[:-1]] - 1
            np.testing.assert_allclose(daily, ledger['return'], rtol=1e-8, atol=1e-12)
            assert len(nav) == len(prices.loc[P['test_start']:P['test_end']]) == 124
            assert np.isclose(row['ASR'], np.sqrt(252)*daily.mean()/daily.std(ddof=1))
            assert np.isclose(row['total_return_percent'], 100*(nav[-1]-1))
            full_nav = np.r_[1.,nav]
            assert np.isclose(row['RMDD_percent'],100*np.max(1-full_nav/np.maximum.accumulate(full_nav)))
            w = weights.iloc[:,1:].to_numpy()
            np.testing.assert_allclose(w.sum(axis=1),1,atol=1e-10)
            assert (np.abs(w).sum(axis=1) <= 2+1e-8).all()
            checks.append({'seed':seed,'method':method,'cost_bps':cost,'passed':True,
                           'independent_price_ledger_max_error':float(np.max(np.abs(nav-ledger.wealth)))})
        saved = np.load(OUT / f'US_{seed}_generated_paths.npz')
        for date, paths in zip(saved['first_return_dates'], saved['paths']):
            assert paths.shape == (50,20,55) and np.isfinite(paths).all()
            j = prices.index.get_loc(date)
            if j+20 <= len(prices):
                mse = np.square(paths.mean(axis=0)-rets.iloc[j:j+20].to_numpy()).mean()
                row = next(x for x in result['forecast_metrics'] if x['method']=='STABLE' and x['first_return_date']==date)
                assert np.isclose(mse,row['MSE'],rtol=1e-12,atol=1e-15)
        results.append(result)
    perf = pd.DataFrame([row for result in results for row in result['metrics']])
    forecasts = pd.DataFrame([row for result in results for row in result['forecast_metrics']])
    perf.to_csv(OUT / 'portfolio_metrics.csv',index=False)
    forecasts.to_csv(OUT / 'forecast_metrics_by_origin.csv',index=False)
    fm = forecasts.groupby(['seed','method'],as_index=False)[['MSE','normalized_DTW']].mean()
    fm.to_csv(OUT / 'forecast_metrics.csv',index=False)
    primary = perf[(perf.method=='STABLE_repaired') & (perf.cost_bps==0)]
    summary = {'updated_at':datetime.now(timezone.utc).isoformat(),'seeds':P['seeds'],
               'setup':'revised US reconstruction after paper-settings audit',
               'exact_author_reproduction':False,'training_runs':len(results),
               'mean_sharpe':float(primary.ASR.mean()),'sample_seed_sd':float(primary.ASR.std(ddof=1)),
               'min_sharpe':float(primary.ASR.min()),'max_sharpe':float(primary.ASR.max()),
               'equal_weight_sharpe':float(perf[(perf.method=='CRP') & (perf.cost_bps==0)].ASR.mean()),
               'paper_sharpe':1.85,'paper_equal_weight_sharpe':.82,
               'all_checks_passed':True,'ledgers_independently_verified':len(checks),
               'primary_fallbacks':sum(x['evaluation']['fallback_count'] for x in results),
               'primary_gross_caps':sum(x['evaluation']['gross_cap_count'] for x in results),
               'primary_rebalances':sum(x['evaluation']['rebalances'] for x in results),
               'original_three_seed_results_pooled':False,'checks':checks}
    (ROOT / 'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    env = {name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','pyarrow','scipy']}
    env.update(device=results[0]['device'],gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    (ROOT / 'environment.json').write_text(json.dumps(env,indent=2)+'\n')
    lines = ['STABLE US: two additional seeds after paper-settings audit',summary['updated_at'],'',
             'This experiment changes setup; do not pool it with the original three seeds.',
             'All changes and unresolved author settings are in ../../PAPER_SETTINGS_AUDIT.txt.',
             'Both seeds completed 200 epochs. Final epoch selected; no test selection.',
             'Train labels end 2024-09-30. Test: 2024-10-01 to 2025-03-31 (124 sessions).','',
             'Main reconstructed STABLE results (no costs, zero risk-free rate):',
             'Seed          Sharpe     Total return %    Annual vol %    Max drawdown %']
    for row in primary.itertuples():
        lines.append(f'{row.seed}    {row.ASR:9.6f}     {row.total_return_percent:10.4f}       {row.AVol_percent:9.4f}       {row.RMDD_percent:9.4f}')
    lines += ['',f"Mean ASR {summary['mean_sharpe']:.6f}; sample seed SD {summary['sample_seed_sd']:.6f} (not a confidence interval).",
              f"Equal-weight ASR {summary['equal_weight_sharpe']:.6f}; paper equal weight 0.82; paper STABLE 1.85.",
              f"Equal-weight fallbacks {summary['primary_fallbacks']}/{summary['primary_rebalances']}; gross caps {summary['primary_gross_caps']}.",'',
              'Forecast mean squared error (x1e4):']
    for row in fm.itertuples():lines.append(f'{row.seed} {row.method:15} {row.MSE*1e4:.6f}')
    lines += ['',f"Verification: all {len(checks)} daily ledgers rebuilt independently from price ratios and fixed shares;",
              'source/data fingerprints unchanged, both checkpoints complete, metrics and weights verified.',
              'These seeds vary training and sampling RNG jointly, not training RNG alone.',
              'Limitations: substitute vendor/GICS/security identities; unverified network and optimizer;',
              'sampling/covariance/leverage repairs remain assumptions; end-of-test universe is retrospective.',
              'No author implementation or exact paper metric cell is certified.']
    (ROOT / 'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
