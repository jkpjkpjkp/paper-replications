"""Score completed models with strictly matched dollar long/short positions."""
from pathlib import Path
import argparse
import json
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.neutral import neutral_backtest
from scripts.build_data import write_json,sha
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-name',default='main')
    p.add_argument('--buy-bp',type=float,default=5.)
    p.add_argument('--sell-bp',type=float,nargs='+',default=[5.,10.,15.])
    args=p.parse_args();root=ROOT/'results'/args.run_name
    out=root/'market_neutral';out.mkdir(exist_ok=True)
    records=[]
    for u in ['csi300','csi500']:
        panel=Panel(ROOT/'data'/u)
        for seed in range(3):
            folder=root/u/f'seed{seed}'
            status=folder/'status.json'
            if not status.exists() or json.loads(status.read_text())['phase']!='complete':continue
            predictions=pd.read_parquet(folder/'test_predictions.parquet')
            for sell_bp in args.sell_bp:
                name=f'{u}_seed{seed}_buy{args.buy_bp:g}_sell{sell_bp:g}'
                daily,metrics=neutral_backtest(panel,predictions,buy_fee=args.buy_bp/10000,sell_fee=sell_bp/10000)
                daily.to_csv(out/f'{name}.csv',index=False)
                write_json(out/f'{name}.json',dict(metrics=metrics,universe=u,seed=seed,
                    prediction_sha256=sha(folder/'test_predictions.parquet'),engine_sha256=sha(ROOT/'limt/neutral.py'),
                    conventions=dict(capital_each_side=1e9,initial_equity=2e9,participation=.02,volume_multiplier=100,
                        targets='Equal-weighted top/bottom return deciles; larger capped leg scaled down to the smaller.',
                        execution='Signal t, execute t+1 close, earn return to t+2 close. Project feasible fills to zero net dollars.',
                        holdings='Retain unfilled positions; non-candidates can only liquidate. Enforce trade caps and candidate position bounds.',
                        fees='Positive signed trades pay buy fee, including short covers; negative trades pay sell fee, including short sales.',
                        cash='Unallocated capital earns zero; returns use current equity. Both legs have equal executed dollar notional.',
                        limitations='Short availability assumed; no borrow fees, lot rounding or price-limit modeling.')))
                records.append(dict(universe=u,seed=seed,buy_bp=args.buy_bp,sell_bp=sell_bp,**metrics))
                print(json.dumps(dict(universe=u,seed=seed,buy_bp=args.buy_bp,sell_bp=sell_bp,
                    annualized_return=metrics['annualized_return'],sharpe=metrics['sharpe'],max_net=metrics['max_absolute_net_exposure'])),flush=True)
    summary=pd.DataFrame(records);summary.to_csv(out/'summary.csv',index=False)
    grouped=summary.groupby(['universe','buy_bp','sell_bp']).agg(seeds=('seed','count'),
        annualized_return=('annualized_return','mean'),annualized_return_std=('annualized_return','std'),
        sharpe=('sharpe','mean'),max_drawdown=('max_drawdown','mean'),
        gross_notional_ratio=('mean_gross_notional_ratio','mean'),max_net_exposure=('max_absolute_net_exposure','max'))
    grouped.to_csv(out/'aggregate.csv')
    print(grouped.to_string(),flush=True)


if __name__=='__main__':main()
