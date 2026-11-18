"""Save forecast comparisons and explicitly approximate monthly portfolio results."""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from build_panel import DEFAULT, BASE, FIN, save_json

PAPER={'QFM':(.65,.140,.52),'AutoReg':(.58,.142,.56),'Linear':(.52,.155,.64),
       'mlp':(.48,.161,.68),'lstm':(.48,.162,.68),'uq_lstm':(.48,.177,.84),'uq_mlp':(.47,.173,.83)}


def evaluate(out):
    meta=pd.read_parquet(out/'samples.parquet')
    y=np.load(out/'y.npy',mmap_mode='r');t=FIN.index('ebit')
    scale=json.loads((out/'scaling.json').read_text());sd=scale['std'][t];mean=scale['mean'][t]
    c=duckdb.connect();c.execute('SET threads=4')
    returns=c.sql(f"""SELECT permno,year(mthcaldt)*12+month(mthcaldt) AS month_id,mthret,
        mthdelflg, mthprcdt=mthcaldt AS has_end_price
        FROM '{BASE}/crsp-monthly/parquet/*.parquet'
        WHERE mthcaldt BETWEEN DATE '1999-12-01' AND DATE '2020-01-31'""").df()
    rf=c.sql(f"""SELECT year(date)*12+month(date) AS month_id,exp(sum(ln(1+rf)))-1 AS rf
        FROM '{BASE}/factors-daily/ff3_daily.parquet' GROUP BY 1""").df()
    results={};portfolio_dir=out/'portfolios';portfolio_dir.mkdir(exist_ok=True)
    for kind in PAPER:
        if kind in ['QFM','AutoReg','Linear']:
            files=list((out/'baselines').glob(f'{kind}_predictions.npz'))
        else:files=[p for p in sorted((out/kind).glob('seed-*/predictions.npz')) if (p.parent/'complete.json').exists()]
        if not files:continue
        pr=None;mu=None;second=None
        for f in files:
            with np.load(f) as a:
                if pr is None:pr=a['indices'];mu=np.zeros(len(pr));second=np.zeros(len(pr))
                else:assert np.array_equal(pr,a['indices'])
                mu+=a['mean']/len(files)
                second+=(a['mean']**2+(a['variance'] if 'variance' in a else 0))/len(files)
        variance=np.maximum(second-mu**2,1e-8)
        test=meta.iloc[pr].split.eq('test').to_numpy()&np.isfinite(y[pr,-1,t])
        mse=float(np.mean((mu[test]-y[pr[test],-1,t])**2))
        frame=meta.iloc[pr].copy()
        frame['forecast_ebit']=(mu*sd+mean)*frame.market_cap
        frame['variance_scaled_ebit']=variance*sd**2
        frame['score']=frame.forecast_ebit/frame.ev
        frame=frame.merge(returns[['permno','month_id','mthdelflg','has_end_price']],
                          on=['permno','month_id'],how='left',validate='one_to_one')
        # Variance of the market-cap-normalized forecast: avoid adding a size
        # penalty by changing variance units to dollars squared.
        if kind.startswith('uq_'):frame['score']/=frame.variance_scaled_ebit
        formation=frame.month_id.between(1999*12+12,2019*12+11)&frame.ev.gt(0)&np.isfinite(frame.score)
        # Exclude delistings already completed in the formation month. CRSP
        # tradingstatusflg may still reflect the last trading day's active status.
        # Do NOT apply this screen to next-month held-security returns.
        formation &= frame.mthdelflg.eq('N') & frame.has_end_price.eq(True)
        selected=frame.loc[formation].sort_values(['month_id','score','permno'],ascending=[True,False,True]).groupby('month_id').head(50)
        selected['month_id']=selected.month_id+1
        holdings=selected.merge(returns[['permno','month_id','mthret']],on=['permno','month_id'],how='left',validate='one_to_one')
        holdings.to_parquet(portfolio_dir/f'{kind}_holdings.parquet',index=False)
        monthly=holdings.groupby('month_id').agg(gross_return=('mthret','mean'),holdings=('permno','size'),missing_returns=('mthret',lambda s:int(s.isna().sum())))
        # Never silently renormalize away missing held-security returns.
        monthly.loc[monthly.missing_returns.gt(0),'gross_return']=np.nan
        monthly=monthly.reset_index().merge(rf,on='month_id',how='left')
        monthly.to_csv(portfolio_dir/f'{kind}_monthly.csv',index=False)
        complete=len(monthly)==240 and monthly.gross_return.notna().all() and monthly.rf.notna().all()
        car=sharpe=None
        if complete:
            r=monthly.gross_return.to_numpy();riskfree=monthly.rf.to_numpy()
            car=float(np.expm1(np.log1p(r).mean()*12))
            rf_ann=float(np.expm1(np.log1p(riskfree).mean()*12))
            sharpe=float((car-rf_ann)/(np.std(r,ddof=1)*np.sqrt(12)))
        results[kind]={'ensemble_members_completed':len(files),'test_samples':int(test.sum()),
            'standardized_ebit_mse':mse,'paper_mse':PAPER[kind][0],
            'mse_absolute_difference':abs(mse-PAPER[kind][0]),
            'gross_monthly_proxy_car':car,'gross_monthly_proxy_sharpe':sharpe,
            'paper_net_car':PAPER[kind][1],'paper_net_sharpe':PAPER[kind][2],
            'portfolio_months':len(monthly),'missing_held_returns':int(monthly.missing_returns.sum()),
            'portfolio_complete':bool(complete)}
    save_json(out/'comparison.json',{'scope':'Current/restated-data Table 2 reconstruction',
        'exact_replication':False,'portfolio_matches_paper_execution':False,
        'portfolio_limitations':['Month-end rebalance using CRSP next-month total returns',
        'Gross returns: no first-ten-day execution, liquidity constraints, commission, slippage or 300 randomized initial portfolios',
        'Positive enterprise value filter and current GICS mapping are reconstruction choices',
        'MSE scaling depends on reconstructed training population and is not numerically identical to author preprocessing'],
        'models':results})
    lines=['CURRENT/RESTATED-DATA RECONSTRUCTION — completed results only',
           'These results do not establish an exact original-paper reproduction.',
           'MSE uses the reconstructed training population and scale.',
           'Portfolio figures are gross monthly diagnostics, not original net returns.','',
           'Model       Completed members    Local EBIT MSE    Paper MSE']
    for name,r in results.items():
        lines.append(f"{name:10s}  {r['ensemble_members_completed']:>17d}    {r['standardized_ebit_mse']:>14.6f}    {r['paper_mse']:.2f}")
    lines+=['','Neural ensemble target: 10 members for each of MLP, LSTM, UQ-MLP and UQ-LSTM.',
            'See run_status.json for the active stage and logs/ for progress.',
            'See comparison.json for portfolio completeness, missing returns and limitations.']
    (out/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT)
    out=parser.parse_args().out
    evaluate(out)
    from verify import verify
    verify(out)
    from refresh_progress import refresh
    refresh(out)
