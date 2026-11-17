"""Verify completed runs and publish a report without selecting on test results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import shutil
import numpy as np
import pandas as pd
import torch
from run_empirical import ROOT,OUT,P,features,origins


def main():
    results=[];checks=[];training=[];allocations=[]
    gm=pd.read_parquet(ROOT/'data/global_macro.parquet')
    for market in P['markets']:
        prices=pd.read_parquet(ROOT/f'data/{market}_prices.parquet')
        f=features(prices,gm)
        for seed in P['seeds']:
            path=OUT/f'{market}_{seed}_result.json'
            result=json.loads(path.read_text())
            state=torch.load(ROOT/f'checkpoints/{market}_{seed}.pt',map_location='cpu',weights_only=False)
            fingerprint=hashlib.sha256((ROOT/'protocol.json').read_bytes()+
                (ROOT/'diffusion_model.py').read_bytes()+
                (ROOT/result.get('training_source','run_empirical.py')).read_bytes()+
                (ROOT/f'data/{market}_prices.parquet').read_bytes()+
                (ROOT/'data/global_macro.parquet').read_bytes()).hexdigest()
            assert state['fingerprint']==fingerprint
            assert state['training_complete'] and result['training']['epochs']==200
            history=pd.read_csv(OUT/f'{market}_{seed}_training.csv')
            assert len(history)==200
            assert result['training']['best_epoch']==int(history.loc[history.validation_noise_mse.idxmin(),'epoch'])
            assert np.isclose(result['training']['validation_noise_mse'],history.validation_noise_mse.min())
            for row in result['metrics']:
                ledger=pd.read_csv(OUT/f"{market}_{seed}_{row['method']}_{row['cost_bps']}bps_daily.csv")
                rets=ledger['return'].to_numpy()
                # Independently reconstruct wealth and reported metrics from the ledger.
                nav=np.cumprod(1+rets)
                np.testing.assert_allclose(nav,ledger.wealth,rtol=1e-11,atol=1e-12)
                assert np.isclose(row['total_return_percent'],100*(nav[-1]-1))
                assert np.isclose(row['ASR'],np.sqrt(252)*np.mean(rets)/np.std(rets,ddof=1))
                assert row['test_days']==len(prices.loc[P['test_start']:P['test_end']])
                weights=pd.read_csv(OUT/f"{market}_{seed}_{row['method']}_weights.csv").iloc[:,1:].to_numpy()
                np.testing.assert_allclose(weights.sum(axis=1),1,atol=1e-10)
                assert (abs(weights).sum(axis=1)<=2+1e-8).all()
            training.append(dict(market=market,**result['training'],**result['evaluation']))
            diag=pd.read_csv(OUT/f'{market}_{seed}_allocation_diagnostics.csv')
            diag.insert(0,'seed',seed);diag.insert(0,'market',market);allocations.append(diag)
            results.append(result)
            checks.append({'market':market,'seed':seed,'passed':True,
                           'checks':['200 epochs completed','validation checkpoint minimum',
                                     'self-financing ledger wealth','reported return and Sharpe',
                                     'full test coverage','budget and gross limits at rebalance']})
    perf=pd.DataFrame([row for r in results for row in r['metrics']])
    forecast=pd.DataFrame([row for r in results for row in r['forecast_metrics']])
    forecast['MSE_x1e4']=forecast.MSE*1e4
    forecast['DTW_x1e3']=forecast.normalized_DTW*1e3
    seedforecast=forecast.groupby(['market','seed','method'],as_index=False)[['MSE_x1e4','DTW_x1e3']].mean()
    perf.to_csv(OUT/'portfolio_metrics.csv',index=False)
    forecast.to_csv(OUT/'forecast_metrics_by_origin.csv',index=False)
    seedforecast.to_csv(OUT/'forecast_metrics.csv',index=False)
    pd.DataFrame(training).to_csv(OUT/'training_summary.csv',index=False)
    diag=pd.concat(allocations,ignore_index=True)
    diag.to_csv(OUT/'allocation_diagnostics.csv',index=False)
    mean=perf.groupby(['market','method','cost_bps'],as_index=False)[
        ['ASR','RMDD_percent','AVol_percent','total_return_percent','total_turnover']].mean()
    mean.to_csv(OUT/'portfolio_mean_across_seeds.csv',index=False)
    stamp=datetime.now(timezone.utc).isoformat()
    (OUT/'verification.json').write_text(json.dumps({'updated_utc':stamp,'runs_checked':len(checks),
        'all_passed':True,'checks':checks,'unit_checks':8,'prior_equation_checks':14},indent=2))
    if not (ROOT/'EQUATION_REPORT.txt').exists():shutil.copyfile(ROOT/'REPORT.txt',ROOT/'EQUATION_REPORT.txt')
    if not (ROOT/'STATUS_equation_only.json').exists():shutil.copyfile(ROOT/'STATUS.json',ROOT/'STATUS_equation_only.json')
    if not (ROOT/'MANIFEST_equation_only.json').exists():shutil.copyfile(ROOT/'MANIFEST.json',ROOT/'MANIFEST_equation_only.json')
    paper={'US':1.85,'China':-.41,'Europe':2.92,'Korea':1.61}
    primary=perf[(perf.method=='STABLE_repaired')&(perf.cost_bps==0)]
    lines=[
      'STABLE best-effort empirical reconstruction',stamp,'',
      'Completed: 12 learned-model training runs (four markets x three seeds),',
      '200 epochs each, with held-out backtests and forecast evaluation.',
      'This is an independent reconstruction using substitute data and explicit',
      'repairs. It does not reproduce the author implementation or certify any',
      'of the paper\'s 536 published metric cells. The main Q1/Q2 pipeline runs;',
      'the full appendix and all competitor implementations are outside this run.',
      '',
      'Test period: 2024-10-01 through 2025-03-31, local trading sessions.',
      'Fit: 2013-01-01 through 2023-12-31. Validation: 2024-01 through 2024-09.',
      'Each target is 20 sessions; targets crossing split boundaries are removed.',
      'Checkpoint selection uses validation noise MSE only. All three fixed seeds',
      'are reported; there is no test-based seed or parameter selection.',
      '',
      'Outcome: this reconstruction does not recover the claimed broad advantage.',
      'Mean Sharpe is below equal weight in three of four markets. Mean forecast',
      'MSE is worse than a zero-return forecast in all four. These observations',
      'describe this substitute setup; they do not establish that the unavailable',
      'author implementation or its measurements are false.',
      '',
      'Annualized Sharpe ratios (zero risk-free rate; no transaction costs)',
      'Market      STABLE mean [seed minimum, maximum]    Equal weight    Paper STABLE',
    ]
    for m in P['markets']:
        q=primary[primary.market==m]
        crp=perf[(perf.market==m)&(perf.method=='CRP')&(perf.cost_bps==0)].ASR.mean()
        lines.append(f'{m:<10}  {q.ASR.mean():7.3f} [{q.ASR.min():7.3f}, {q.ASR.max():7.3f}]       {crp:7.3f}        {paper[m]:6.2f}')
    lines += ['', 'The paper values above are context, not like-for-like benchmarks: the',
      'stock IDs, vendor data, architecture and portfolio repairs differ.',
      'Three seeds measure training/sampling variation on one fixed historical',
      'test period; they are not independent market samples or confidence intervals.',
      '', 'Main reconstructed STABLE metrics (mean across all three seeds)',
      'Market     Cost bps      ASR     Max drawdown %    Annual vol %    Total return %']
    for m in P['markets']:
        for cost in [0,10]:
            z=mean[(mean.market==m)&(mean.method=='STABLE_repaired')&(mean.cost_bps==cost)].iloc[0]
            lines.append(f'{m:<10} {cost:6} {z.ASR:10.3f} {z.RMDD_percent:15.3f} {z.AVol_percent:15.3f} {z.total_return_percent:16.3f}')
    lines += ['', 'Forecast error on complete 20-session test horizons',
       'Market        STABLE MSE x1e4    Zero forecast MSE x1e4    STABLE DTW x1e3']
    for m in P['markets']:
        z=seedforecast[(seedforecast.market==m)&(seedforecast.method=='STABLE')]
        zero=seedforecast[(seedforecast.market==m)&(seedforecast.method=='zero')].MSE_x1e4.mean()
        lines.append(f'{m:<10} {z.MSE_x1e4.mean():18.4f} {zero:25.4f} {z.DTW_x1e3.mean():21.4f}')
    lines += ['', 'MSE uses the mean of 50 generated daily log-return paths. STABLE DTW',
        'averages L1 dynamic-time-warping distance over the 50 paths and stocks,',
        'divided by the 20-day horizon. The L1 cost/normalizer is an explicit',
        'implementation choice because the paper does not fully specify it.',
        'Partial final holding blocks are included in portfolio returns but are',
        'excluded from full-horizon forecast scoring.',
        '', 'Allocation repairs and their observed use',
        'Market seed          Equal-weight fallbacks / rebalances     Gross caps']
    for r in training:
        lines.append(f"{r['market']:<7} {r['seed']} {r['fallback_count']:19} / {r['rebalances']:<12} {r['gross_cap_count']:8}")
    total_fallback=sum(r['fallback_count'] for r in training)
    total_rebalance=sum(r['rebalances'] for r in training)
    lines += ['',f'Fallbacks occurred in {total_fallback}/{total_rebalance} main-model allocation decisions.',
       'A fallback uses equal weights, so that observation is not evidence of a',
       'successful learned allocation. The raw covariance remains singular for',
       '55 assets with 50 samples; ridge makes the specified solve executable.',
       'A nonpositive normalization denominator triggers equal weights. Valid',
       'directions exceeding 2x gross exposure are shrunk toward equal weights.',
       'The gross limit is enforced at rebalance; holdings may drift beyond it.',
       '', 'Data and substitutions',
       'US: 55 S&P500 members at the training cutoff; CRSP daily total returns.',
       'China/Europe/Korea: 55/37/44 regional primary ordinary-share proxies from',
       'Compustat Global, selected by capitalization within GICS sectors. These',
       'are not verified CSI300/EUROSTOXX/KOSPI200 constituent sets. Selection',
       'uses 2024-09-30 capitalization and training-history coverage, whereas the',
       'paper selects at 2025-03-31. Company classifications and primary issue',
       'flags come from vendor snapshots and can contain retrospective changes.',
       'The selected training universe is retrospective; this is not a historical',
       'point-in-time universe simulation. No test return enters stock selection.',
       'Global adjusted price = prccd / qunit / ajexdi * trfd. Local currencies',
       'are retained, without FX conversion. Holidays use regional activity.',
       'No selected test prices needed missing-value filling; training suspensions',
       'and isolated gaps are forward-filled and counted in data/*_data_audit.json.',
       '',
       'Macro inputs: selected-universe equal-weight market proxy, FRED broad',
       'dollar index DTWEXBGS, FRED 10y-minus-3m spread T10Y3M, CBOE VIX, and',
       'CRSP GLD close as the gold proxy. Global observations are aligned to a',
       'strictly earlier calendar date. These are revised source snapshots, not',
       'vintage release feeds. Spread changes use differences, not logs of',
       'potentially negative values. All fitted scalers use the fit period only.',
       '', 'Model and execution',
       'Shared 1D convolutional U-Net, 16-dimensional macro/stock embeddings,',
       'learned gate in [0,2], causal Kalman coefficients, 200 forward steps,',
       '30 DDIM steps and 50 generated paths. A cosine schedule, width 32,',
       'Adam lr .001, L2 beta .001, batch 256 and normalized-clean clipping at',
       '[-8,8] are explicit choices. Clipping frequency is saved per rebalance.',
       'The compact architecture and nonoverlapping 20-session training origins',
       'use substantially fewer examples/operations than an unspecified larger',
       'author model; 200 epochs is not a claim of equivalent compute budget.',
       'Trailing covariance is scaled to a 20-day horizon; ridge is .01 times',
       'mean diagonal variance. Equilibrium prior mean follows Appendix A.3.',
       'Main sampling aligns independently generated per-stock paths. The first',
       'seed also tests correlated initial shocks and ridge values .0001/.1.',
       'These sensitivities are reported, never used to pick the main result.',
       '',
       'Features use close t; execution is close t+1; first earned return ends',
       't+2. Signed positions buy and hold for 20 sessions with daily weight',
       'drift. Returns include distributions. The 10-bps case charges .001 per',
       'unit of absolute weight turnover, including entry. Financing and borrow',
       'fees are omitted and the risk-free rate is zero. This is a retrospective',
       'research backtest, not evidence of executable net profitability.',
       '', 'Verification and artifacts',
       '14 earlier numerical checks and eight new model/chronology/accounting',
       'checks passed. Each completed checkpoint and every daily ledger were',
       'checked for epoch completion, validation selection, metric arithmetic,',
       'full test-date coverage, budget and rebalance exposure constraints.',
       'protocol.json fixes assumptions. data/ contains frozen panels, source',
       'file metadata, data hashes and exact substitute stock lists. checkpoints/',
       'contains all trained weights. results/best_effort/ contains every seed,',
       'daily ledger, weight vector, generated path, training history and metric.',
       'The original equation audit is preserved in EQUATION_REPORT.txt.',
       '', 'Sources',
       'Paper: https://datalab.snu.ac.kr/~ukang/papers/2026ICLR-stable.pdf',
       'Dollar: https://fred.stlouisfed.org/series/DTWEXBGS',
       'Spread: https://fred.stlouisfed.org/series/T10Y3M',
       'Prices/macros: local checksum-verified WRDS archives listed in data/source_files.json.',
       'The scientific-critical-thinking skill informed evidence separation.',
       'Methods attribution, verified earlier today and preserved with source',
       'metadata: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M.',
       '(2026). Scientific Agent Skills: A Library of Procedural Knowledge for',
       'Research Agents. https://doi.org/10.48550/arXiv.2609.00065',
    ]
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    status={'paper':'STABLE','updated_at':stamp,'status':'best_effort_empirical_reconstruction_completed',
       'empirical_status':'trained_and_backtested_on_substitute_data_with_explicit_repairs',
       'training_runs':len(results),'epochs_per_run':200,'markets':P['markets'],'seeds':P['seeds'],
       'primary_market_backtests':len(results),'portfolio_ledgers_including_baselines_costs_sensitivities':len(perf),
       'published_metric_cells_reproduced':0,'exact_author_reproduction':False,
       'numerical_checks_passed':14,'empirical_unit_checks_passed':8,
       'completed_runs_independently_checked':len(checks),'fallback_decisions':total_fallback,
       'total_primary_rebalances':total_rebalance,'report':str(ROOT/'REPORT.txt'),
       'metrics':str(OUT/'portfolio_metrics.csv'),'remaining_exact_reproduction_gaps':P['known_limits']}
    (ROOT/'STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    environment={k:importlib.metadata.version(k) for k in ['numpy','pandas','pyarrow','torch','scipy','requests']}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in str(p)
       and p.name!='MANIFEST.json' and 'logs' not in p.parts}
    (ROOT/'MANIFEST.json').write_text(json.dumps({'created_at':stamp,'sha256':manifest},indent=2))
    print(json.dumps(status,indent=2));print('\n'.join(lines[18:29]))


if __name__=='__main__':main()
