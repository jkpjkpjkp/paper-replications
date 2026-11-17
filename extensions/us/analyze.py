"""Validate US forecasts, neutral portfolios, serial inference and comparators."""
from pathlib import Path
import json
import hashlib
import os
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from usdata import ROOT,Panel,config,sha,write_json
from portfolio import backtest

OUT=ROOT/'analysis'
INPUTS={}
ROWS=[]
PAIRED=[]
CHECKS=[]


def source(path):
    path=Path(path)
    INPUTS[str(path)]=sha(path)
    return path


def hac(x,lag=20):
    x=np.asarray(x,dtype=float)
    fit=sm.OLS(x,np.ones((len(x),1)),missing='raise').fit(cov_type='HAC',
        cov_kwds={'maxlags':lag,'kernel':'bartlett','use_correction':True},use_t=True)
    u=x-x.mean();n=len(x);variance=u@u
    for k in range(1,lag+1):variance+=2*(1-k/(lag+1))*(u[k:]@u[:-k])
    np.testing.assert_allclose(fit.bse[0],np.sqrt(variance/(n*(n-1))),atol=1e-13,rtol=1e-9)
    a,b=fit.conf_int()[0]
    return dict(mean=float(x.mean()),p=float(fit.pvalues[0]),t=float(fit.tvalues[0]),ci_low=float(a),ci_high=float(b))


def bootstrap(x,block=20,seed=260925617):
    x=np.asarray(x,dtype=float);n=len(x);rng=np.random.default_rng(seed);means=[]
    for first in range(0,5000,250):
        starts=rng.integers(0,n,size=(250,(n+block-1)//block))
        ix=((starts[:,:,None]+np.arange(block))%n).reshape(250,-1)[:,:n]
        means.extend(x[ix].mean(1))
    return np.quantile(means,[.025,.975]).tolist()


def load_predictions(model,seed,panel):
    folder=ROOT/('results' if model=='LiMT' else 'baseline')/f'seed{seed}'
    if model=='LiMT':
        status=json.loads(source(folder/'status.json').read_text())
        assert status['phase']=='complete'
        lock=json.loads(source(folder/'lock.json').read_text())
        assert lock['protocol_sha256']==sha(ROOT/'protocol.json')
        assert lock['data_manifest_sha256']==sha(ROOT/'data/manifest.json')
    else:source(folder/'complete.json')
    pred=pd.read_parquet(source(folder/'test_predictions.parquet')).sort_values(['date','symbol']).reset_index(drop=True)
    assert not pred[['date','symbol']].duplicated().any()
    dates={d:i for i,d in enumerate(panel.dates)};symbols={s:i for i,s in enumerate(panel.symbols)}
    counts=pred.groupby('date',sort=True).size()
    test_days=panel.split('test')
    np.testing.assert_array_equal(counts.index.to_numpy(),panel.dates[test_days])
    np.testing.assert_array_equal(counts.to_numpy(),panel.member[:,test_days].sum(0))
    ii=pred.symbol.map(symbols).to_numpy();tt=pred.date.map(dates).to_numpy()
    assert np.asarray(panel.member[ii,tt]).all()
    np.testing.assert_array_equal(pred.realized_date,panel.dates[tt+2])
    np.testing.assert_allclose(pred.label_return,panel.price_return[ii,tt+2],equal_nan=True,atol=1e-7)
    assert np.isfinite(pred[['pred_return','pred_volume','pred_volatility']]).all().all()
    return pred


def ensemble(parts):
    result=parts[0].copy()
    for part in parts[1:]:
        pd.testing.assert_frame_equal(result[['date','symbol']],part[['date','symbol']])
    columns=['pred_return','pred_volume','pred_volatility']
    result[columns]=np.mean([p[columns].to_numpy() for p in parts],axis=0)
    return result


def windows(frame):
    p=config()
    yield 'full-2017-2025',np.ones(len(frame),bool)
    for name,(a,b) in zip(['paper-period-2017-2020','recent-2021-2025'],p['report_windows']):
        yield name,frame.date.between(a,b).to_numpy()


def summarize(ledger,model,seed,strategy,buy_bp):
    for window,mask in windows(ledger):
        f=ledger.loc[mask];x=f.return_net.to_numpy();n=len(x)
        assert n>100 and np.isfinite(x).all()
        result=hac(x)
        wealth=np.r_[1.,np.cumprod(1+x)]
        boot20=np.array(bootstrap(x))*252
        boot60=np.array(bootstrap(x,60))*252
        rec=dict(model=model,seed=str(seed),strategy=strategy,buy_bp=buy_bp,window=window,
            start=f.date.iloc[0],end=f.date.iloc[-1],days=n,
            annual_mean=float(x.mean()*252),cagr=float(wealth[-1]**(252/n)-1),
            sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(252)),
            max_drawdown=float((wealth/np.maximum.accumulate(wealth)-1).min()),
            p=result['p'],t=result['t'],ci_low=result['ci_low']*252,ci_high=result['ci_high']*252,
            bootstrap20_low=boot20[0],bootstrap20_high=boot20[1],bootstrap60_low=boot60[0],bootstrap60_high=boot60[1],
            mean_gross_exposure=float(f.gross_exposure.mean()),max_abs_net_exposure=float(f.net_exposure.abs().max()),
            max_abs_end_net_drift=float(f.end_net_drift.abs().max()),
            mean_daily_buy_turnover=float((f.buy_value/f.start_nav).mean()),
            mean_cost_bp=float((f.fees/f.start_nav).mean()*1e4),
            missing_held_count=int(f.missing_held_count.sum()),max_missing_weight=float(f.missing_return_gross_weight.max()))
        for lag in [5,60]:
            sensitivity=hac(x,lag)
            rec.update({f'hac{lag}_{k}':value*252 if k in ['mean','ci_low','ci_high'] else value for k,value in sensitivity.items()})
        ROWS.append(rec)


def forecast_metrics(pred):
    rows=[]
    for date,g in pred.groupby('date',sort=True):
        y=g.label_return.to_numpy();x=g.pred_return.to_numpy();valid=np.isfinite(y)&np.isfinite(x)
        rows.append(dict(signal_date=date,date=g.realized_date.iloc[0],
            ic=float(np.corrcoef(x[valid],y[valid])[0,1]),
            rank_ic=float(st.spearmanr(x[valid],y[valid]).statistic),stocks=len(g),observed_targets=int(valid.sum())))
    return pd.DataFrame(rows)


def main():
    import subprocess
    subprocess.run([os.environ.get('LIMT_MODEL_PYTHON', sys.executable),str(ROOT/'verify_predictions.py')],check=True,cwd=ROOT)
    source(ROOT/'prediction-replay-verification.json')
    OUT.mkdir(exist_ok=True);panel=Panel();p=config()
    original=json.loads(source(ROOT/'data/manifest.json').read_text())
    for name,expected in original['output_sha256'].items():
        actual=sha(ROOT/'data'/name)
        assert actual==expected,('Data changed',name)
        INPUTS[str(ROOT/'data'/name)]=actual
    source(ROOT/'protocol.json');source(ROOT/'portfolio.py');source(ROOT/'usdata.py')
    complete={}
    for model in ['LiMT','LightGBM']:
        parts=[load_predictions(model,seed,panel) for seed in [0,1,2]]
        complete[model]=ensemble(parts)
        all_cases=[('ensemble',complete[model])]+[(str(seed),part) for seed,part in enumerate(parts)]
        for seed,pred in all_cases:
            metrics=forecast_metrics(pred)
            metrics.to_csv(OUT/f'{model}_{seed}_daily_forecasts.csv',index=False)
            for strategy in (['equal','apo'] if model=='LiMT' else ['equal']):
                for buy in [0.,5.]:
                    ledger,weights=backtest(panel,pred,strategy=strategy,buy_bp=buy,save_weights=seed=='ensemble' and buy==0)
                    tag=f'{model}_{seed}_{strategy}_buy{buy:g}'
                    ledger.to_csv(OUT/f'{tag}_ledger.csv',index=False)
                    if weights is not None:
                        np.savez_compressed(OUT/f'{tag}_weights.npz',weights=weights,
                            dates=ledger.date.to_numpy(dtype=str),symbols=panel.symbols)
                        # Weighted observed source total returns independently
                        # reconcile the gross, zero-fee portfolio payoff stream.
                        lookup={d:i for i,d in enumerate(panel.dates)}
                        ix=np.array([lookup[d] for d in ledger.date])
                        realized=np.nan_to_num(np.asarray(panel.total_return[:,ix]).T,nan=0.)
                        np.testing.assert_allclose(np.sum(weights*realized,axis=1),ledger.return_gross,atol=1e-12)
                    summarize(ledger,model,seed,strategy,buy)
                    print('ANALYZED',tag,flush=True)
        # Missing-payoff stress does not reselect or remove affected stocks.
        for strategy in (['equal','apo'] if model=='LiMT' else ['equal']):
            stress,_=backtest(panel,complete[model],strategy=strategy,adverse_missing=True)
            stress.to_csv(OUT/f'{model}_ensemble_{strategy}_missing_stress.csv',index=False)
    left=forecast_metrics(complete['LiMT']);right=forecast_metrics(complete['LightGBM'])
    pd.testing.assert_series_equal(left.date,right.date)
    left['baseline_ic']=right.ic;left['delta_ic']=left.ic-right.ic
    left.to_csv(OUT/'paired_forecasts.csv',index=False)
    for window,mask in windows(left):
        g=left.loc[mask];x=g.delta_ic.to_numpy();stats=hac(x);low,high=bootstrap(x)
        PAIRED.append(dict(window=window,days=len(g),limt_ic=float(g.ic.mean()),baseline_ic=float(g.baseline_ic.mean()),
            paired_ic_gain=float(x.mean()),paired_p=stats['p'],ci_low=stats['ci_low'],ci_high=stats['ci_high'],
            bootstrap_low=low,bootstrap_high=high))
    summary=pd.DataFrame(ROWS);paired=pd.DataFrame(PAIRED)
    main_rows=summary.index[summary.seed=='ensemble']
    family=np.r_[summary.loc[main_rows,'p'].to_numpy(),paired.paired_p.to_numpy()]
    adjusted=multipletests(family,method='holm')[1]
    summary.loc[main_rows,'p_holm']=adjusted[:len(main_rows)]
    paired['p_holm']=adjusted[len(main_rows):]
    summary.to_csv(OUT/'summary.csv',index=False);paired.to_csv(OUT/'forecast_comparison.csv',index=False)
    CHECKS.extend(['All three neural and all three baseline seeds completed with unchanged locked data/protocol.',
        'Selected neural checkpoints independently replay first/middle/last full test cross-sections for all three seeds.',
        'Every forecast universe row matches contemporaneous membership; all date offsets and direct CRSP return labels reconcile.',
        'All frozen data arrays match pretraining SHA256 hashes.',
        'All portfolio ledgers have equal long/short dollars and reconcile drifted holdings, gross P&L, fees and NAV.',
        'Zero-cost ensemble portfolios independently dot held weights against raw CRSP total returns.',
        'Intercept-only HAC standard errors independently match Bartlett arithmetic at all declared lags.',
        'Seeds are separate diagnostics; all significance sampling units are unique calendar dates.'])
    write_json(OUT/'verification.json',dict(status='passed',checks=CHECKS,inputs_sha256=INPUTS,
        code_sha256=sha(__file__),statsmodels=statsmodels.__version__,summary_rows=len(summary),
        portfolio_aggregate_tests=len(main_rows),paired_forecast_tests=len(paired),
        family_tests=len(family),created_utc=datetime.now(timezone.utc).isoformat()))
    report(summary,paired,original)


def report(summary,paired,manifest):
    main=summary[summary.seed=='ensemble'].copy()
    columns=['model','strategy','buy_bp','window','days','annual_mean','cagr','sharpe','ci_low','ci_high','p','p_holm','mean_gross_exposure','missing_held_count']
    lines=['LiMT US TRANSFER — COMPLETED RESULTS','',
        'Primary design: changing historical US 500-stock universe, one issuer/share class, 2008-2014 training,',
        '2015-2016 validation, held-out predictions 2017-2025. Three full-size LiMT seeds and three LightGBM seeds.',
        'Scores are averaged across seeds before forming the ensemble portfolio. Individual seeds are diagnostics.',
        'Prices/volume come from local verified CRSP. Observed VWAP is unavailable: OHLC4 is explicitly substituted.',
        'A US method transfer, not a numerical replication of the paper\'s Chinese benchmark.',
        '', 'ENSEMBLE PORTFOLIOS (annual mean=252*daily mean; CAGR is separate)',main[columns].to_string(index=False),
        '', 'FORECAST COMPARISON (paired daily IC on identical stocks/dates)',paired.to_string(index=False),
        '', 'Interpretation: a positive annual estimate is significant at the nominal level only when its HAC',
        '95% CI excludes zero. Holm adjustment jointly covers all 18 ensemble portfolio/window tests plus',
        'three paired IC comparisons. These corrections do not account for all prior method choices across papers.',
        '', 'Dollar-neutrality: equal executed long/short dollars at each rebalance. The primary equal-weight',
        'strategy targets +50%/-50% NAV. APO uses USD1bn each side with 2% target and trade caps; unallocated',
        'capital earns zero, and realized deployed exposure is reported. Equal dollars do not imply beta neutrality.',
        '', 'Signal at t; assumed execution at t+1 close; realized CRSP total payoff t+2. Initial trades and final',
        'short covers are included in cost accounting. Previous positions drift by CRSP price returns;',
        'total returns include cash income/dividends, including short obligations. Terminal stock positions',
        'convert to cash after the supplied terminal return, without a second legacy delisting adjustment.',
        '', 'The primary zero-cost case assumes available shorts and ideal closes, with no borrowing, financing,',
        'slippage, impact or collateral yield. The buy5 case charges 5bp on long purchases/short covers only.',
        'APO execution caps use realized execution-session volume, an ideal fill model rather than observed fills.',
        'Missing held payoffs are preserved, treated as zero in the primary ledger, counted/weighted in summaries,',
        'and stressed adversely without removing those assets or reselecting portfolios. Stress files are saved.',
        '', 'The data are a retrospective current-vintage CRSP snapshot. Dynamic membership avoids a present-day',
        'survivor universe, but cannot certify historical as-published data or real stock-loan availability.',
        'One adjustment discrepancy larger than 20bp occurs among approximately 2.4m eligible price checks:',
        'PERMNO12060, 2019-02-26. It is retained; direct CRSP return fields drive labels/P&L, not price ratios.',
        '', 'All hyperparameters and date/basket/cost/window variants were specified before training/test evaluation.',
        'The complete source core is pinned to author commit c0232623d50db26b02275b6f77b682f25effeec5.',
        'Public core/code differences and unavailable author preprocessing/execution remain as in the Chinese',
        'reproduction protocol. No new test-return hyperparameter search, direction flip, or winning-seed choice.',
        '', 'INFERENCE: Bartlett HAC20, small-sample correction and two-sided t-reference intervals; HAC5/HAC60',
        'and 5000-replicate circular block20/block60 bootstrap intervals are saved in summary.csv.',
        'Independent seeds on common dates are not independent trading samples.',
        '', 'ARTIFACTS: analysis/summary.csv, forecast_comparison.csv, *_ledger.csv, *_weights.npz,',
        'daily_forecasts.csv, paired_forecasts.csv, *_missing_stress.csv, verification.json, performance.png.',
        'The runtime/model/protocol and data input hashes are in each seed\'s lock/provenance files.',
        '', 'REPRODUCE: from this directory, $LIMT_MODEL_PYTHON run.py',
        'Select pinned model and analysis environments with LIMT_MODEL_PYTHON and LIMT_ANALYSIS_PYTHON.',
        '', 'SOURCES: https://arxiv.org/html/2609.25617v1 ; https://github.com/YHY-10/LiMT ; local CRSP CIZ daily',
        'source validation in ../crsp-daily/validation.json and schema in its data/crsp_a_stock/dsf_v2/schema.json.',
        'Statistical methods reference: https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html',
        'The statsmodels skill informed robust inference and multiplicity reporting; workflow attribution:',
        'Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026). Scientific Agent Skills:',
        'A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065',
        'Current arXiv v2 metadata verified during this conversation; attribution is not profitability evidence.']
    (ROOT/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True,layout='constrained')
    for model,strategy,label in [('LiMT','equal','LiMT equal dollars'),('LiMT','apo','LiMT capped APO'),('LightGBM','equal','LightGBM equal dollars')]:
        f=pd.read_csv(OUT/f'{model}_ensemble_{strategy}_buy0_ledger.csv',parse_dates=['date'])
        axes[0].plot(f.date,np.cumprod(1+f.return_net),label=label)
        axes[1].plot(f.date,f.gross_exposure,label=label,alpha=.75)
    axes[0].set_title('US transfer: zero-cost, equal-dollar long/short portfolios',loc='left')
    axes[0].set_ylabel('Compounded unit capital');axes[0].legend()
    axes[1].set_ylabel('Actual gross exposure / NAV');axes[1].set_xlabel('Payoff date')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'performance.png',dpi=180);plt.close(fig)
    import html
    pretty=main[columns].copy()
    for c in ['annual_mean','cagr','ci_low','ci_high','mean_gross_exposure']:pretty[c]=pretty[c].map(lambda x:f'{x:.2%}')
    for c in ['p','p_holm']:pretty[c]=pretty[c].map(lambda x:f'{x:.5g}')
    pretty.sharpe=pretty.sharpe.map(lambda x:f'{x:.2f}')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LiMT US transfer</title><style>body{font:16px/1.6 system-ui;max-width:1400px;margin:32px auto;padding:0 22px;color:#172934}table{font-size:12px;border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}.table{overflow-x:auto}pre{white-space:pre-wrap;font:14px/1.7 system-ui}img{max-width:100%}</style><h1>LiMT on US daily CRSP data</h1><p>Three seeds; equal-dollar long/short; zero-cost and buy5 sensitivities.</p><img src="analysis/performance.png" alt="Calendar-time portfolio wealth and gross exposures"><div class="table">'''+pretty.to_html(index=False)+'''</div><pre>'''+html.escape('\n'.join(lines))+'''</pre></html>'''
    (ROOT/'report.html').write_text(page)
    print(main[columns].to_string(index=False),flush=True)


if __name__=='__main__':main()
