from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]


def mean_sd(values,percent=False):
    values=np.asarray(values,dtype=float)*(100 if percent else 1)
    return f'{values.mean():.4f} ± {values.std(ddof=1):.4f}'+('%' if percent else '')


def paired_interval(values,block=20,repetitions=2000):
    rng=np.random.default_rng(20261106);values=np.asarray(values)
    starts=rng.integers(0,len(values),(repetitions,int(np.ceil(len(values)/block))))
    ix=(starts[:,:,None]+np.arange(block)[None,None,:])%len(values)
    draws=values[ix.reshape(repetitions,-1)[:,:len(values)]].mean(axis=1)
    return [float(np.quantile(draws,q)) for q in [.025,.975]]


def generate(run_name='main'):
    folder=ROOT/'results'/run_name;figure=folder/'figures';figure.mkdir(exist_ok=True)
    rows=[];portrows=[];sensrows=[];paired=[];long_only_rows=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    navfig,navaxes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    icfig,icaxes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    daily_frames={}
    for ui,u in enumerate(['csi300','csi500']):
        primary_ic=[];baseline_ic=[]
        curves={s:[] for s in ['equal','risk_parity','rank','apo']}
        for seed in range(3):
            out=folder/u/f'seed{seed}'
            m=json.loads((out/'metrics.json').read_text())['test']
            status=json.loads((out/'status.json').read_text())
            assert status['phase']=='complete'
            rows.append(dict(model='LiMT',universe=u,seed=seed,**m['return'],volume_IC=m['volume']['IC'],
                volatility_IC=m['volatility']['IC'],epochs=status['epochs'],best_epoch=status['best_epoch']))
            bm=json.loads((ROOT/'results/baseline'/u/f'seed{seed}'/'metrics.json').read_text())['test']['return_metrics']
            rows.append(dict(model='LightGBM',universe=u,seed=seed,**bm))
            h=json.loads((out/'history.json').read_text())
            axes[ui].plot([v['epoch'] for v in h],[v['validation']['return']['IC'] for v in h],label=f'Seed{seed}')
            best=h[status['best_epoch']-1]
            axes[ui].scatter(best['epoch'],best['validation']['return']['IC'],s=22)
            pm=json.loads((out/'backtest_metrics.json').read_text())['metrics']
            long_only_rows.append(dict(universe=u,seed=seed,model='LiMT',**pm['top50']))
            baseline_pm=json.loads((ROOT/'results/baseline'/u/f'seed{seed}'/'backtest_metrics.json').read_text())['top50']
            long_only_rows.append(dict(universe=u,seed=seed,model='LightGBM',**baseline_pm))
            for strategy,metric in pm.items():
                portrows.append(dict(universe=u,seed=seed,strategy=strategy,**metric))
                if strategy in curves:
                    f=pd.read_csv(out/f'backtest_{strategy}.csv',parse_dates=['date']).set_index('date')
                    curves[strategy].append((1+f.return_net).cumprod())
            sensitivity=json.loads((out/'sensitivity.json').read_text())
            for r,metric in sensitivity.items():sensrows.append(dict(universe=u,seed=seed,r=float(r),**metric))
            for model,predpath,target in [('LiMT',out/'test_predictions.parquet',primary_ic),
                ('LightGBM',ROOT/'results/baseline'/u/f'seed{seed}'/'test_predictions.parquet',baseline_ic)]:
                pred=pd.read_parquet(predpath)
                values=pred.groupby('date').apply(lambda f:f.pred_return.corr(f.label_return),include_groups=False)
                target.append(values)
                daily_frames[(u,model,seed)]=values
        axes[ui].set(title=u.upper(),xlabel='Epoch',ylabel='Validation return IC');axes[ui].legend()
        for strategy,curveset in curves.items():
            mean_nav=pd.concat(curveset,axis=1).mean(axis=1)
            navaxes[ui].plot(mean_nav.index,(mean_nav-1)*100,label=strategy)
        navaxes[ui].set(title=u.upper(),ylabel='Mean cumulative return (%)',xlabel='Realization date')
        navaxes[ui].legend(fontsize=8);navaxes[ui].tick_params(axis='x',rotation=25)
        a=pd.concat(primary_ic,axis=1).mean(axis=1);b=pd.concat(baseline_ic,axis=1).mean(axis=1)
        difference=(a-b).dropna()
        lo,hi=paired_interval(difference)
        paired.append(dict(universe=u,mean_daily_IC_difference=float(difference.mean()),
            bootstrap_95_low=lo,bootstrap_95_high=hi,days=len(difference),block=20,repetitions=2000))
        for label,series in [('LiMT',a),('LightGBM',b)]:
            series.index=pd.to_datetime(series.index)
            icaxes[ui].plot(series.index,series.rolling(60,min_periods=60).mean(),label=label)
        icaxes[ui].axhline(0,color='gray',lw=.6)
        icaxes[ui].set(title=u.upper(),xlabel='Signal date',ylabel='60-day mean return IC')
        icaxes[ui].legend();icaxes[ui].tick_params(axis='x',rotation=25)
    fig.savefig(figure/'learning_curves.png',dpi=160);plt.close(fig)
    navfig.savefig(figure/'portfolio_nav.png',dpi=160);plt.close(navfig)
    icfig.savefig(figure/'rolling_ic.png',dpi=160);plt.close(icfig)
    forecast=pd.DataFrame(rows);portfolio=pd.DataFrame(portrows);sensitivity=pd.DataFrame(sensrows)
    long_only=pd.DataFrame(long_only_rows)
    forecast.to_csv(folder/'forecast_summary.csv',index=False)
    portfolio.to_csv(folder/'portfolio_summary.csv',index=False)
    sensitivity.to_csv(folder/'sensitivity_summary.csv',index=False)
    long_only.to_csv(folder/'long_only_summary.csv',index=False)
    pd.DataFrame(paired).to_csv(folder/'paired_ic_bootstrap.csv',index=False)
    neutral=pd.read_csv(folder/'market_neutral/aggregate.csv')
    text=['# LiMT reproduction results','',
        '**Status: completed adaptation on the local 2010-start CSI dataset. Exact numerical reproduction of the paper is not established.**','',
        'Six full-date LiMT runs and six LightGBM comparator runs were completed. Each LiMT model uses up to 100 epochs with 21-epoch validation early stopping, and only its validation-selected checkpoint is scored on test. All results below come from saved predictions; the short smoke run is excluded.','',
        'The author core is pinned at [`c023262`](https://github.com/YHY-10/LiMT/tree/c0232623d50db26b02275b6f77b682f25effeec5). The [paper](https://arxiv.org/html/2609.25617v1) uses 2008-start data; this snapshot begins in 2010, and the public model differs from several equations. [PROTOCOL.md](PROTOCOL.md) specifies the complete setup and execution limitations.','',
        '## Forecasting','',
        'Mean ± sample standard deviation over seeds 0, 1, 2. ICIR is daily IC mean divided by its sample standard deviation, without annualization. The paper references use different data and cannot serve as an apples-to-apples performance comparison.','',
        '| Universe | Model | IC | ICIR | RankIC | Paper IC reference |','| --- | --- | --- | --- | --- | --- |']
    for u,paper in [('csi300',.0575),('csi500',.0496)]:
        for model in ['LiMT','LightGBM']:
            f=forecast[(forecast.universe==u)&(forecast.model==model)]
            text.append(f'| {u.upper()} | {model} | {mean_sd(f.IC)} | {mean_sd(f.ICIR)} | {mean_sd(f.RankIC)} | {paper:.4f} |' if model=='LiMT' else
                        f'| {u.upper()} | {model} | {mean_sd(f.IC)} | {mean_sd(f.ICIR)} | {mean_sd(f.RankIC)} | — |')
    text+=['','The LightGBM comparator uses a declared independent configuration and validation-MSE early stopping. It does not reconstruct all fourteen tuned paper baselines.','',
        '| Universe | Volume-shock IC | Range/VWAP IC | Selected epochs | Epochs trained |','| --- | --- | --- | --- | --- |']
    for u in ['csi300','csi500']:
        f=forecast[(forecast.universe==u)&(forecast.model=='LiMT')]
        text.append(f'| {u.upper()} | {mean_sd(f.volume_IC)} | {mean_sd(f.volatility_IC)} | '+', '.join(str(int(x)) for x in f.best_epoch)+' | '+', '.join(str(int(x)) for x in f.epochs)+' |')
    text+=['','Paired daily IC differences below average the three seeds within each model before subtraction. Intervals use a deterministic circular 20-trading-day block bootstrap with 2000 draws. They are exploratory, conditional on this dataset and protocol, and do not measure future investment performance.','',
        '| Universe | LiMT minus LightGBM IC | 95% block-bootstrap interval |','| --- | --- | --- |']
    for row in paired:text.append(f"| {row['universe'].upper()} | {row['mean_daily_IC_difference']:.4f} | [{row['bootstrap_95_low']:.4f}, {row['bootstrap_95_high']:.4f}] |")
    text+=['','![Validation learning curves](results/'+run_name+'/figures/learning_curves.png)',
        '','![Rolling held-out IC](results/'+run_name+'/figures/rolling_ic.png)','',
        '## Independent liquidity-constrained backtest','',
        'Initial CNY 1 billion per side, 10 bp per traded value, 2% target and trade caps, volume converted from lots to shares, and a t+1 close execution delay. Return uses current total equity, initially CNY 2 billion. Turnover and gross notional ratios use the initial allocation. Unfilled allocation remains cash. Short borrowing is assumed; locate availability, borrow fees, board-lot rules and price limits are not modeled. These statistics are not the authors’ unreleased execution-engine results.','',
        '| Universe | Sizing | Annualized return | Sharpe | Max drawdown | Turnover | Full-order success | Gross notional / initial capital |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for u in ['csi300','csi500']:
        for s in ['equal','risk_parity','rank','apo']:
            f=portfolio[(portfolio.universe==u)&(portfolio.strategy==s)]
            text.append(f'| {u.upper()} | {s} | {mean_sd(f.annualized_return,True)} | {mean_sd(f.sharpe)} | {mean_sd(f.max_drawdown,True)} | {mean_sd(f.mean_turnover,True)} | {mean_sd(f.order_success_rate,True)} | {mean_sd(f.mean_gross_exposure,True)} |')
    text+=['']
    for u in ['csi300','csi500']:
        a=portfolio[(portfolio.universe==u)&(portfolio.strategy=='apo')]
        e=portfolio[(portfolio.universe==u)&(portfolio.strategy=='equal')]
        text.append(f'On {u.upper()}, APO changes mean annualized return by {(a.annualized_return.mean()-e.annualized_return.mean())*100:+.2f} percentage points, mean Sharpe by {a.sharpe.mean()-e.sharpe.mean():+.2f}, and mean deployed gross notional / initial capital by {(a.mean_gross_exposure.mean()-e.mean_gross_exposure.mean())*100:+.2f} percentage points relative to equal weighting. These are descriptive seed averages under the declared engine.')
        text.append('')
    text+=['','APO uses the fixed paper parameter r=0.1. Its published CSI300 AR 10.01% and Sharpe 1.86 are reference claims, not matched results. The report includes gross exposure because cash left by liquidity caps materially changes risk and return.','',
        '![Portfolio NAV](results/'+run_name+'/figures/portfolio_nav.png)','',
        '## Long-only reference','',
        'This independent top-50/20-replacement reference applies the same 10 bp cost and trade cap to LiMT and LightGBM forecasts. Excess figures subtract archived price-index returns. It is not the paper’s unspecified Qlib exchange configuration.','',
        '| Universe | Forecast model | Excess annualized return | Excess information ratio |','| --- | --- | --- | --- |']
    for u in ['csi300','csi500']:
        for model in ['LiMT','LightGBM']:
            f=long_only[(long_only.universe==u)&(long_only.model==model)]
            text.append(f'| {u.upper()} | {model} | {mean_sd(f.excess_annualized_return,True)} | {mean_sd(f.excess_information_ratio)} |')
    text+=['','## Dollar-neutral portfolios with a 5 bp buy cost','',
        'This additional evaluation uses equal-weighted top/bottom decile targets and enforces equal executed long and short dollar notionals after every rebalance. Buys, including short covers, cost 5 bp. The sell-cost scenarios below are explicit alternatives because the user specified only the buy cost. Liquidity and trade caps remain at 2%, with CNY 1 billion allocated to each leg and returns divided by current equity, initially CNY 2 billion. Caps can leave much of the allocation unused. The executable neutral portfolio is a box-constrained projection of the capped targets; an infeasible neutral rebalance raises an error.','',
        '| Universe | Buy / sell cost | Seeds | Net annualized return | Mean Sharpe | Mean max drawdown | Deployed gross / initial equity |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for row in neutral.itertuples(index=False):
        text.append(f'| {row.universe.upper()} | {row.buy_bp:g} / {row.sell_bp:g} bp | {row.seeds} | {row.annualized_return*100:.2f}% | {row.sharpe:.2f} | {row.max_drawdown*100:.2f}% | {row.gross_notional_ratio*100:.2f}% |')
    text+=['',f'Maximum absolute post-trade net dollar exposure / equity across these cases is {neutral.max_net_exposure.max():.3g}. Directional fees, retained positions and daily neutrality are audited in `results/{run_name}/verification.json`. These remain hypothetical portfolios with the same short-availability and execution limitations described above.','']
    text+=['','The complete seed-level records, including the baseline long-only runs, are in `results/'+run_name+'/`. Sensitivity at r=0.1,...,0.9 is descriptive and never selects an optimal test parameter.','',
        '## Attention and expert-routing case study','',
        'The paper’s March 20, 2020 case study is recreated from the frozen CSI300 seed-0 checkpoint. The target is 600000.SH and its paired source is 600004.SH. The calculation uses the released attention scaling, head averages and Eq. 27 propagation products. CPU forecasts are checked against the saved test forecasts. These are descriptive learned weights; residual and feed-forward paths are omitted, so the plots do not establish causal effects or complete feature attribution. The data and sampled peers differ from the authors’ figures.','',
        '![Cross-stock attention and propagation](results/'+run_name+'/csi300/seed0/interpretability/cross_stock_and_propagation.png)','',
        '![Directional attention propagation](results/'+run_name+'/csi300/seed0/interpretability/directional_propagation.png)','',
        '![Task-specific expert routing](results/'+run_name+'/csi300/seed0/interpretability/expert_routing.png)','',
        'Raw attention arrays, sampled peer identifiers, figure hashes and the reconstruction check are saved alongside the images.','',
        '## Validation and remaining gaps','',
        '- Output, input-gradient and parameter-gradient equivalence to the released model is tested with dropout disabled. Label offsets, split purging, as-of eligibility, window fill, APO causality and partial-fill caps have dedicated tests.',
        '- Artifact verification checks preserved upstream state, dataset and benchmark hashes, 100 real source-label calculations and 400 raw-source feature calculations per universe, all test dates, duplicate keys, finite predictions, recomputed IC, validation checkpoint selection and recomputed portfolio metrics. See `results/'+run_name+'/verification.json`.',
        '- Exact replication still needs the authors’ processed 2008-start dataset, selected configurations, and execution engine/settings. The present work does not reproduce the complete hyperparameter search, fourteen baseline suite, or all ablations.',
        '- The local provider does not establish short availability, corporate-action accuracy, complete delisting proceeds, or point-in-time vendor revisions. Model performance and simulated profitability must be read with these limitations.','',
        'Run commands and file descriptions are in [README.md](README.md). Raw sources, benchmark responses, per-file hashes, environment lock, configurations, checkpoints and prediction files remain in the project.','']
    (ROOT/'REPORT.md').write_text('\n'.join(text))
    print(f'Wrote {ROOT / "REPORT.md"}',flush=True)


if __name__=='__main__':generate(sys.argv[1] if len(sys.argv)>1 else 'main')
