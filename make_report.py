"""Produce a compact result table and diagnostic figure from a completed run."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    args=p.parse_args()
    run=Path(args.run)
    result=json.loads((run/'results.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    pool=json.loads((run/'selected_pool.json').read_text())
    frame=pd.DataFrame(result['metrics']).T
    frame.loc['test_final_pool']=result['final_pool_test_metrics']
    frame.to_csv(run/'metrics_summary.csv',index_label='split_or_checkpoint')
    lines=[
        'AlphaQCM: measured corrected-release replication',
        '',
        f"Status: {result['status']}",
        f"Universe/seed: {manifest['config']['universe']} / {manifest['config']['seed']}",
        f"Interactions: {result['steps']:,}; learning updates: {result['learning_updates']:,}",
        f"Validation-selected step: {result['selected_step']:,}",
        f"Selected during random warmup: {result['selected_during_random_warmup']}",
        f"Wall time: {result['elapsed_seconds']:.1f} seconds",
        '',
        frame.to_string(float_format=lambda x:f'{x:.6f}'),
        '',
        'IC and RankIC are signed daily cross-sectional correlations, in decimal units.',
        'ICIR is not annualized. Final-pool test metrics are a diagnostic, not a selection criterion.',
        'Paper reference: CSI300 IQN IC 0.0849, seed standard deviation 0.0103.',
        'The paper reference is NOT directly comparable to this single shortened run.',
        '',
        'Validation-selected formulas (weights fitted on training data):',
    ]
    lines += [f'{w:.10g} * {e}' for e,w in zip(pool['exprs'],pool['weights'])]
    lines += ['', 'Numerical replication of the paper is NOT established.',
        f"Scope: community snapshot, one seed, P={manifest['config']['pool']}, lambda={manifest['config']['lambda']}, {result['steps']:,} steps.",
        'Upstream fixes, purged labels, released total-IC reward/L1 pool fitting and',
        'remaining paper/code differences are documented in README.txt.',
        'No portfolio returns, causal claims, significance tests or SOTA claims are made.']
    (run/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history=pd.DataFrame(json.loads((run/'history.json').read_text()))
    daily=pd.read_csv(run/'test_daily_metrics.csv',parse_dates=['date'])
    fig,axes=plt.subplots(2,1,figsize=(10,7),layout='constrained')
    axes[0].plot(history.step,history.valid_ic,label='Validation IC: released evaluator',color='#2368a0')
    axes[0].axvline(10000,color='gray',ls='--',label='Replay warmup ends')
    axes[0].axvline(result['selected_step'],color='#d47c23',ls=':',label='Selected checkpoint')
    axes[0].set(xlabel='Environment steps',ylabel='Signed IC',title='Validation-only checkpoint selection')
    axes[0].legend(loc='best',fontsize=9)
    axes[1].plot(daily.date,daily.ic,alpha=.25,color='#2368a0',label='Daily test IC')
    axes[1].plot(daily.date,daily.ic.rolling(63,min_periods=20).mean(),color='#2368a0',label='63-session mean')
    axes[1].axhline(0,color='gray',lw=.7)
    axes[1].set(xlabel='Test date',ylabel='Paired-mask IC',title='Frozen validation-selected pool: held-out diagnostics')
    axes[1].legend(loc='best',fontsize=9)
    fig.suptitle('AlphaQCM / CSI300 / seed 0 / corrected release / shortened budget',fontsize=13)
    fig.savefig(run/'diagnostics.png',dpi=160)
    plt.close(fig)
    hashes={str(f.relative_to(run)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(run.rglob('*')) if f.is_file() and f.name!='artifact-sha256.json'}
    (run/'artifact-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')

if __name__=='__main__':
    main()
