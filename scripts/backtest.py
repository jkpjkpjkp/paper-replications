from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.portfolio import backtest
from scripts.build_data import write_json,sha
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='main')
    p.add_argument('--universe',default='csi300',choices=['csi300','csi500'])
    p.add_argument('--seed',type=int,default=0);p.add_argument('--r',type=float,default=.1)
    a=p.parse_args();folder=ROOT/'results'/a.run_name/a.universe/f'seed{a.seed}'
    panel=Panel(ROOT/'data'/a.universe)
    pred=pd.read_parquet(folder/'test_predictions.parquet')
    benchmark=pd.read_csv(ROOT/'sources/benchmarks'/f'{a.universe}.csv').set_index('date')['return']
    benchmark=benchmark.reindex(panel.dates).to_numpy()
    if not np.isfinite(benchmark[panel.split('test')+2]).all():raise ValueError('Missing benchmark returns')
    metrics={}
    for strategy in ['equal','risk_parity','rank','apo','top50']:
        daily,metric=backtest(panel,pred,strategy=strategy,r=a.r,benchmark=benchmark)
        daily.to_csv(folder/f'backtest_{strategy}.csv',index=False)
        metrics[strategy]=metric
    write_json(folder/'backtest_metrics.json',dict(metrics=metrics,
        engine_sha256=sha(ROOT/'limt/portfolio.py'),prediction_sha256=sha(folder/'test_predictions.parquet'),
        benchmark_sha256=sha(ROOT/'sources/benchmarks'/f'{a.universe}.csv'),conventions=dict(
        capital_each_side=1e9,fee_per_traded_value=.001,participation=.02,r=a.r,
        volume_multiplier=100,valuation='Signal t; trade at t+1 close; accrue to t+2 close; fixed CNY targets, NAV denominator.',
        capital_ratios='Turnover and gross_exposure use initial allocation: 2C for long-short, C for long-only. Return uses current equity.',
        selection='Top/bottom deciles for long-short; top50 max20 replacements for long-only.',
        caps='Eq26 target cap and independent execution trade cap; all sizing baselines use both.',
        order_success='Fraction of nonzero stock-level orders fully filled; partial orders count as failures.',
        shorting='Assumed available; no borrow fee, locate data, or funding cost.',
        other_limits='No board-lot rounding or price-limit enforcement; not the unavailable author execution engine.',
        missing='Missing close carried for valuation, zero/missing volume prevents fills; missing delisting proceeds not available.')))
    print(metrics)


if __name__=='__main__':main()
