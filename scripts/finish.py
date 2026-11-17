"""Finish completed runs: backtest, sensitivity, verify and assemble results."""
from pathlib import Path
import json
import subprocess
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.portfolio import backtest
from scripts.build_data import write_json,sha
from scripts.verify import verify
from scripts.report import generate
ROOT=Path(__file__).resolve().parents[1]


def main(run_name='main'):
    for u in ['csi300','csi500']:
        panel=Panel(ROOT/'data'/u)
        bench=pd.read_csv(ROOT/'sources/benchmarks'/f'{u}.csv').set_index('date')['return'].reindex(panel.dates).to_numpy()
        for seed in range(3):
            folder=ROOT/'results'/run_name/u/f'seed{seed}'
            assert json.loads((folder/'status.json').read_text())['phase']=='complete',folder
            with (folder/'backtest.log').open('w') as log:
                subprocess.run([sys.executable,'scripts/backtest.py','--run-name',run_name,
                    '--universe',u,'--seed',str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            pred=pd.read_parquet(folder/'test_predictions.parquet')
            sensitivity={}
            for tenth in range(1,10):
                r=tenth/10
                _,metric=backtest(panel,pred,strategy='apo',r=r,benchmark=bench)
                sensitivity[str(r)]=metric
            write_json(folder/'sensitivity.json',sensitivity)
            baseline=ROOT/'results/baseline'/u/f'seed{seed}'
            bpred=pd.read_parquet(baseline/'test_predictions.parquet')
            daily,metric=backtest(panel,bpred,strategy='top50',benchmark=bench)
            daily.to_csv(baseline/'backtest_top50.csv',index=False)
            write_json(baseline/'backtest_metrics.json',dict(top50=metric,
                engine_sha256=sha(ROOT/'limt/portfolio.py'),
                prediction_sha256=sha(baseline/'test_predictions.parquet'),
                benchmark_sha256=sha(ROOT/'sources/benchmarks'/f'{u}.csv')))
            print(json.dumps(dict(universe=u,seed=seed,phase='backtests_complete')),flush=True)
    subprocess.run([sys.executable,'scripts/interpret.py','--run-name',run_name],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'scripts/market_neutral.py','--run-name',run_name,
        '--buy-bp','5','--sell-bp','5','10','15'],cwd=ROOT,check=True)
    verify(run_name,require_complete=True)
    generate(run_name)


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else 'main')
