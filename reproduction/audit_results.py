"""Independently replay position CSVs, then compare means/SDs to Table 1.

Default input is the AUTHOR-SHIPPED CSVs, not a fresh trained reproduction.
Use --results <fresh-directory> --label fresh_training for new executions.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    40: {'Sharpe': (.6717,.0628), 'Sortino': (.8232,.0792),
         'MaxDrawdown': (.3611,.0037), 'FinalWealthFactor': (1.7903,.1023)},
    50: {'Sharpe': (.7715,.0627), 'Sortino': (.9743,.0998),
         'MaxDrawdown': (.3271,.0094), 'FinalWealthFactor': (1.9215,.1792)},
}

def metrics(rets):
    eq = np.r_[1., np.cumprod(1.+rets)]
    return dict(Sharpe=rets.mean()/(rets.std()+1e-12)*math.sqrt(252),
                Sortino=rets.mean()/(rets[rets<0].std()+1e-12)*math.sqrt(252),
                MaxDrawdown=np.max((np.maximum.accumulate(eq)-eq)/(np.maximum.accumulate(eq)+1e-16)),
                FinalWealthFactor=eq[-1])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--label',default='author_shipped_arithmetic_only')
    args=parser.parse_args()
    prices=pd.read_csv(ROOT/'asset_data/full_dataset.csv',parse_dates=['Date']).set_index('Date')
    out=ROOT/'reproduction'/args.label
    out.mkdir(parents=True,exist_ok=True)
    audits=[];comparisons=[]
    for pool,targets in TARGETS.items():
        runs=[]
        files=sorted(args.results.glob(f'dp{pool}_*test_positions.csv'))
        for path in files:
            positions=pd.read_csv(path,parse_dates=['Date'])
            raw=prices.iloc[:,:pool].values.astype(np.float32)
            indices=prices.index.get_indexer(positions.Date)
            assert (indices>=0).all()
            returns=(raw[indices+1]-raw[indices])/(raw[indices]+1e-8)
            weights=positions.filter(like='Weight_').values.astype(np.float32)
            rets=np.array([float(np.dot(w,r)) for w,r in zip(weights,returns)])
            previous=np.r_[10000.,positions.CumulativeCapital.values[:-1]]
            saved_rets=positions.DailyPnL.values/previous
            max_error=float(np.max(np.abs(rets-saved_rets)))
            recomputed=metrics(rets)
            reported=pd.read_csv(str(path).replace('positions.csv','metrics.csv')).iloc[0]
            audits.append(dict(file=path.name,pool=pool,days=len(rets),
                               max_daily_return_error=max_error,
                               weight_sum_max_error=float(np.max(np.abs(weights.sum(1)-1))),
                               metric_max_error=max(abs(recomputed[k]-reported[k]) for k in recomputed),
                               first_date=str(positions.Date.iloc[0].date()),
                               last_date=str(positions.Date.iloc[-1].date())))
            runs.append(recomputed)
        if not runs:continue
        vals=pd.DataFrame(runs)
        for metric,(mean,sd) in targets.items():
            for kind,target,fresh in [('mean',mean,vals[metric].mean()),('sample_sd',sd,vals[metric].std(ddof=1))]:
                comparisons.append(dict(pool=pool,metric=metric,statistic=kind,runs=len(runs),paper=target,
                                        replayed=fresh,absolute_error=abs(fresh-target),
                                        printed_match=bool(abs(fresh-target)<=.00005+1e-12)))
    pd.DataFrame(audits).to_csv(out/'daily_replay_audit.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'table1_comparison.csv',index=False)
    summary=dict(label=args.label,runs_audited=len(audits),cells=len(comparisons),
                 printed_matches=sum(r['printed_match'] for r in comparisons),
                 max_daily_return_error=max((r['max_daily_return_error'] for r in audits),default=None),
                 note='Shipped-artifact arithmetic does not establish fresh training reproduction.' if args.label.startswith('author') else 'See training logs and environment deviations.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
