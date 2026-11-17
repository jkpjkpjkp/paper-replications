"""Verify saved portfolios and summarize seed-level results without selecting on test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PAPER={'deterministic_ts':(.04,.04),'gaussian_ts':(.10,.04),'diffusion_ts':(.13,.03),
       'policy':(1.74,.41),'deterministic_dfl':(.07,.00),'gaussian_reparam':(.08,.03),
       'gaussian_score':(.14,.05),'diffusion_reparam':(4.17,.24),'diffusion_score':(3.98,.31)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='local');args=ap.parse_args()
    base=ROOT/'results'/args.output
    config=json.loads((base/'config.json').read_text())
    rows=[]
    for seed in config['seeds']:
        sd=base/f'seed{seed}'
        for method in config['methods']:
            folder=sd/method
            metric=json.loads((folder/'metrics.json').read_text())
            p=np.load(folder/'portfolios.npz')
            w,y=p['weights'].astype(float),p['actual_returns'].astype(float)
            r=(w*y).sum(1)
            assert np.isfinite(w).all() and np.isfinite(r).all()
            assert np.allclose(w.sum(1),1,atol=1e-6) and w.min()>=-1e-6 and w.max()<=1+1e-6
            assert np.allclose(r,p['returns'],atol=1e-7)
            assert np.all(r<=y.max(1)+1e-6)
            assert abs(100*r.mean()-metric['mean_daily_return_pct'])<1e-5
            assert (p['dates']<p['target_dates']).all()
            if metric['selected_epoch'] is not None:
                history=json.loads((folder/'history.json').read_text())
                assert metric['selected_epoch']==max(history,key=lambda h:h['mean_daily_return_pct'])['epoch']
            rows.append(metric)
    frame=pd.DataFrame(rows)
    frame.to_csv(base/'all_runs.csv',index=False)
    summaries=[]
    for method in config['methods']:
        f=frame[frame.method==method]
        summaries.append({'method':method,'runs':len(f),
            'local_mean_daily_return_pct':f.mean_daily_return_pct.mean(),
            'local_seed_sem_pct':f.mean_daily_return_pct.std(ddof=1)/np.sqrt(len(f)),
            'local_rmse_sample':f.rmse_sample.mean(),
            'paper_mean_daily_return_pct':PAPER[method][0], 'paper_seed_sem_pct':PAPER[method][1],
            'mean_equal_weight_pct':f.equal_weight_mean_daily_return_pct.mean(),
            'mean_oracle_pct':f.oracle_mean_daily_return_pct.mean()})
    summary=pd.DataFrame(summaries)
    summary.to_csv(base/'comparison.csv',index=False)
    audit={'passed':True,'runs_verified':len(rows),'expected_runs':len(config['methods'])*len(config['seeds']),
           'daily_weights_replayed':True,'simplex_feasibility':True,'oracle_bound':True,
           'validation_checkpoint_selection':True,'test_observations_per_run':sorted(frame.observations.unique().tolist()),
           'exact_paper_reproduction':False}
    (base/'verification.json').write_text(json.dumps(audit,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.2,1]})
    names=summary.method.str.replace('_',' ').tolist();pos=np.arange(len(names))
    for ax,col,err,title in [(axes[0],'local_mean_daily_return_pct','local_seed_sem_pct','Local CRSP replication'),
                              (axes[1],'paper_mean_daily_return_pct','paper_seed_sem_pct','Published Table 1 (different protocol)')]:
        ax.barh(pos,summary[col],xerr=summary[err].fillna(0),color=['#147d92' if 'diffusion' in x else '#758497' for x in names],capsize=3)
        ax.set_yticks(pos,names if ax==axes[0] else []);ax.invert_yaxis();ax.axvline(0,color='black',lw=.6)
        ax.set_title(title);ax.set_xlabel('Mean daily return (%)');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Diffusion-DFL portfolio experiment\nError bars: standard error across seeds; results are not directly comparable',fontsize=11)
    fig.tight_layout();fig.savefig(base/'comparison.png',dpi=160);plt.close(fig)
    lines=['DIFFUSION-DFL: PORTFOLIO REPLICATION RESULTS','',
        'Status: completed alternative-data, reduced-compute comparison; the paper is NOT numerically reproduced.',
        'Original Lookahead repository was not modified. Work is in '+str(ROOT)+'.','',
        'Paper: https://openreview.net/forum?id=uhv3f80jmG',
        'Full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC13460018/',
        'Official code: https://github.com/GT-KOALA/Diffusion_DFL','',
        'Mean daily return (%) +/- standard error across seeds:']
    for item in summaries:
        lines.append(f"  {item['method']:24s} local {item['local_mean_daily_return_pct']:.4f} +/- {item['local_seed_sem_pct']:.4f}; paper {item['paper_mean_daily_return_pct']:.2f} +/- {item['paper_seed_sem_pct']:.2f}")
    lines += ['',f"Verified {len(rows)} completed runs; each has {frame.observations.iloc[0]} held-out days.",
        'Every saved portfolio was independently replayed, checked for simplex feasibility and bounded by the daily perfect-foresight oracle.',
        'Checkpoints were selected only by validation return. Seed error bars include random basket and training variability; they are not time-series confidence intervals.',
        'No transaction costs are deducted. These are arithmetic daily returns, not annualized returns, CAGR, or cumulative wealth.',
        '', 'CONFIGURATION',json.dumps(config,indent=2),'',
        'MATERIAL LIMITATIONS',
        '1. CRSP CIZ replaces Quandl WIKI. The exact author data, feature/PCA implementation, stock lists, checkpoints, and experiment configs were not provided.',
        '2. Fixed January 2004 S&P 500 membership is restricted to complete history through 2016. This survival conditioning is not an investable point-in-time universe.',
        '3. The released 28-features-per-stock implementation is used: 1400 inputs for 50 assets, not the paper appendix description of 28 total PCA/volume features.',
        '4. This local comparison uses fewer seeds, diffusion sampling steps, Monte Carlo draws, and fine-tuning epochs than the paper/release; see config above.',
        '5. Upstream models are reused, but data handling, mixed derivatives, Gaussian score gradients, and metric aggregation were corrected. See results/release_audit.json and REPRODUCTION.txt.',
        '6. KKT ridge regularization creates an approximate gradient at active simplex vertices; the forward optimization is certified against its original objective.',
        '7. The synthetic and power-scheduling experiments have not been run. This report covers the portfolio task only.',
        '8. Forecast calibration is poor: local diffusion sample RMSE is approximately 3 in decimal-return units, and deterministic DFL approximately 80, versus realized returns near a few percent. Finite training and feasible portfolios do not establish predictor convergence.',
        '', 'The local results cannot establish or refute the published numerical claim because the data and protocol differ.',
        'See comparison.csv, all_runs.csv, verification.json, model checkpoints, and per-method daily.csv/portfolios.npz beside this report.']
    diagnostic=base/'sampler_validation_check.json'
    if diagnostic.exists():
        lines += ['', 'VALIDATION-ONLY SAMPLER CHECK (SEED 0)',
                  'Saved checkpoints are unchanged; no test selection or retraining used this diagnostic.']
        for row in json.loads(diagnostic.read_text()):
            lines.append(f"  {row['method']:22s} steps={row['ddim_steps']:4d}: validation mean daily return {row['mean_daily_return_pct']:.4f}%, sample RMSE {row['rmse_sample']:.4f}.")
        lines.append('Returning to the full 1000-step sampler does not resolve poor forecast calibration in these checkpoints; this does not test full-budget training.')
    (base/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':main()
