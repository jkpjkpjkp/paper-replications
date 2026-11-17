#!/usr/bin/env python3
"""Paired calendar-time inference; inference never treats stock rows as IID."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/factor-overlay-mpl')
import json,html,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import duckdb
from statsmodels.stats.multitest import multipletests
from scipy.stats import skew
from data import HERE,ROOT,FACTORS
from models import METHODS

OUT=HERE/'results'
HORIZON=1
PERIODS={'validation':(2005,2014),'holdout':(2015,2025),'all_oos':(2005,2025)}

def hac(y,x=None,lag=60,annual=252):
    if x is None:x=np.ones((len(y),1))
    else:x=np.column_stack([np.ones(len(y)),x])
    f=sm.OLS(y,x,missing='raise').fit(cov_type='HAC',cov_kwds={'maxlags':lag})
    ci=f.conf_int()[0]*annual
    return dict(estimate=float(f.params[0]*annual),lo=float(ci[0]),hi=float(ci[1]),p=float(f.pvalues[0]),se=float(f.bse[0]*annual))

def main():
    z=dict(np.load(OUT/'daily.npz'));spec=json.loads((OUT/'models.json').read_text())
    dates=pd.DatetimeIndex(z['dates']);years=dates.year;lookup={s['id']:i for i,s in enumerate(spec)}
    market=duckdb.sql(f"SELECT * FROM read_parquet('{ROOT}/results/panel/market_*.parquet') WHERE year(date) BETWEEN 2005 AND 2025 ORDER BY date").df()
    print('Market columns',list(market.columns),flush=True)
    market=market.set_index('date').reindex(dates)
    mcol=next(k for k in ['vw_market','market_vw','vw','mkt_vw','ret_vw'] if k in market)
    mr=market[mcol].to_numpy(float);assert np.isfinite(mr).all()
    rows=[];tests=[];risk=[];annual=[]
    for ci,cost in enumerate(z['costs']):
        for period,(lo,hi) in PERIODS.items():
            mask=(years>=lo)&(years<=hi);rr=z['returns'][mask,ci];dd=dates[mask]
            for m,s in enumerate(spec):
                r=rr[:,m];mean=r.mean()*252;variance=r.var(ddof=1)*252;vol=np.sqrt(variance)
                nav=np.cumprod(1+r);peak=np.maximum.accumulate(np.r_[1,nav])[1:]
                row=dict(**s,cost_bps=cost*10000,period=period,annual_mean=mean,annual_vol=vol,sharpe=mean/vol if vol>0 else np.nan,cagr=nav[-1]**(252/len(r))-1,max_drawdown=float((nav/peak-1).min()),cer3=mean-1.5*variance,skew=float(skew(r)),lag1_autocorr=float(pd.Series(r).autocorr()),active=z['active'][mask,ci,m].mean(),turnover=z['turnover'][mask,ci,m].mean()*252,fees=z['fees'][mask,ci,m].mean()*252,borrow=z['borrow'][mask,ci,m].mean()*252,mean_long=z['n_long'][mask,ci,m].mean(),mean_short=z['n_short'][mask,ci,m].mean(),mean_gross=z['gross_exposure'][mask,ci,m].mean(),mean_net=z['net_exposure'][mask,ci,m].mean(),forced_exits=z['forced_exit'][mask,ci,m].sum(),missing=z['missing'][mask,ci,m].sum())
                rows.append(row)
                if not s['cycle']:continue
                base=lookup[s['id'].replace('__cycle','__base')];plain=lookup[s['factor']+'__plain']
                for label,comp in [('paired',base),('plain',plain)]:
                    delta=r-rr[:,comp];h=hac(delta)
                    own=hac(r)
                    mm=(pd.DataFrame({'a':r,'b':rr[:,comp]},index=dd)+1).resample('ME').prod()-1
                    mh=hac((mm.a-mm.b).to_numpy(),lag=3,annual=12)
                    vbase=rr[:,comp].var(ddof=1)*252
                    tests.append(dict(id=s['id'],factor=s['factor'],method=s['method'],cost_bps=cost*10000,period=period,comparator=label,delta_mean=h['estimate'],ci_low=h['lo'],ci_high=h['hi'],p=h['p'],own_mean=mean,own_p=own['p'],delta_cer3=delta.mean()*252-1.5*(variance-vbase),delta_sharpe=mean/vol-rr[:,comp].mean()*252/np.sqrt(vbase),monthly_delta=mh['estimate'],monthly_p=mh['p'],delta_turnover=(z['turnover'][mask,ci,m]-z['turnover'][mask,ci,comp]).mean()*252))
                controls=np.column_stack([rr[:,base],mr[mask],rr[:,-1]])
                h=hac(r,controls)
                risk.append(dict(id=s['id'],factor=s['factor'],method=s['method'],cost_bps=cost*10000,period=period,alpha=h['estimate'],ci_low=h['lo'],ci_high=h['hi'],p=h['p']))
        for year in range(2005,2026):
            rr=z['returns'][years==year,ci]
            for m,s in enumerate(spec):annual.append(dict(id=s['id'],year=year,cost_bps=cost*10000,return_compounded=float(np.prod(1+rr[:,m])-1)))
    stats=pd.DataFrame(rows);tests=pd.DataFrame(tests);risk=pd.DataFrame(risk)
    for _,g in tests.groupby(['cost_bps','period','comparator']):
        tests.loc[g.index,'holm_p']=multipletests(g.p,method='holm')[1]
        tests.loc[g.index,'own_holm_p']=multipletests(g.own_p,method='holm')[1]
        tests.loc[g.index,'monthly_holm_p']=multipletests(g.monthly_p,method='holm')[1]
    for _,g in risk.groupby(['cost_bps','period']):risk.loc[g.index,'holm_p']=multipletests(g.p,method='holm')[1]
    tests['joint_success']=(tests.delta_mean>0)&(tests.holm_p<.05)&(tests.own_mean>0)&(tests.own_holm_p<.05)
    stats.to_csv(OUT/'performance.csv',index=False);tests.to_csv(OUT/'paired_inference.csv',index=False);risk.to_csv(OUT/'spanning_alpha.csv',index=False);pd.DataFrame(annual).to_csv(OUT/'annual_returns.csv',index=False)
    # Human-readable daily series for every matched portfolio and scenario.
    daily=pd.DataFrame({'date':dates})
    daily=pd.concat([daily,pd.DataFrame({f"{s['id']}__{cost*10000:g}bp":z['returns'][:,ci,m] for ci,cost in enumerate(z['costs']) for m,s in enumerate(spec)})],axis=1)
    daily.to_csv(OUT/'daily_returns.csv',index=False)
    training=dict(np.load(HERE/('data/train.npz' if HORIZON==1 else 'data/train5.npz')));clock=np.round(training['Xcycle'][:,11]*65)
    clockbin=pd.cut(clock,[-np.inf,-20,-10,-.5,10,20,np.inf],labels=['pre21plus','pre11_20','pre1_10','post0_10','post11_20','post21plus'])
    moments=pd.DataFrame({'factor':np.array(FACTORS)[training['factor']],'year':training['year'],'idx':training['idx'],'leg':training['Xbase'][:,6],'clock_bin':clockbin,'momentum_bin':pd.cut(training['Xbase'][:,1],[-np.inf,-1/3,1/3,np.inf],labels=['low','middle','high']),'signed_payoff':training['y']})
    moments=moments[moments.year>=2015]
    moments.groupby(['factor','leg','clock_bin','momentum_bin'],observed=True).signed_payoff.agg(['count','mean','std']).to_csv(OUT/'cycle_moments_descriptive.csv')
    # Report complete family; best row per factor is explicitly ex post.
    t=tests.query("period=='holdout' and cost_bps==5 and comparator=='paired'").copy()
    best=t.sort_values('delta_mean',ascending=False).groupby('factor',sort=False).head(1).set_index('factor').reindex(FACTORS).reset_index()
    best.to_csv(OUT/'best_per_factor_expost.csv',index=False)
    fit=json.loads((OUT/'fit_audit.json').read_text())
    diagnostics=dict(number_fits=len(fit),nonconverged=[r for r in fit if not r['converged']],maximum_parameters=max(r['n_parameters'] for r in fit),positive_increment=int((t.delta_mean>0).sum()),significant_increment=int(((t.delta_mean>0)&(t.holm_p<.05)).sum()),joint_success=int(t.joint_success.sum()))
    (OUT/'inference_audit.json').write_text(json.dumps(diagnostics,indent=2))
    render(stats,tests,risk,best,diagnostics)

def table(df):return df.to_html(index=False,border=0,classes='data',float_format=lambda x:f'{x:.3f}')
def render(stats,tests,risk,best,diag):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=tests.query("period=='holdout' and cost_bps==5 and comparator=='paired'")
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    for f,ax in zip(FACTORS,axes.flat):
        g=t[t.factor==f].set_index('method').reindex(METHODS)
        yy=np.arange(len(METHODS));ax.errorbar(g.delta_mean*100,yy,xerr=np.vstack([(g.delta_mean-g.ci_low)*100,(g.ci_high-g.delta_mean)*100]),fmt='o',color='#226677',capsize=3)
        ax.axvline(0,color='#999',lw=1);ax.set_yticks(yy,METHODS);ax.set_title(f+(' (concentrated)' if f=='RDIPO' else ''));ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Annual incremental net return (percentage points)')
    fig.suptitle('Adding announcement-cycle features: 2015–2025\nSame model without cycle; 5 bp per dollar traded + 1% annual short borrow. HAC60 95% intervals.')
    fig.tight_layout();fig.savefig(OUT/'incremental_returns.png',dpi=160);plt.close(fig)
    view=[]
    for r in best.itertuples():
        base=stats.query("id==@r.id.replace('__cycle','__base')") if False else stats[(stats.id==r.id.replace('__cycle','__base'))&(stats.period=='holdout')&(stats.cost_bps==5)].iloc[0]
        cyc=stats[(stats.id==r.id)&(stats.period=='holdout')&(stats.cost_bps==5)].iloc[0]
        plain=stats[(stats.id==r.factor+'__plain')&(stats.period=='holdout')&(stats.cost_bps==5)].iloc[0]
        a=risk[(risk.id==r.id)&(risk.period=='holdout')&(risk.cost_bps==5)].iloc[0]
        view.append({'Factor':r.factor,'Model (ex post)':r.method,'Plain %/yr':plain.annual_mean*100,'No-cycle %/yr':base.annual_mean*100,'Cycle %/yr':cyc.annual_mean*100,'Increment pp/yr':r.delta_mean*100,'HAC60 95% interval':f'[{100*r.ci_low:.2f}, {100*r.ci_high:.2f}]','Holm p':r.holm_p,'Cycle Sharpe':cyc.sharpe,'Increment CER pp':r.delta_cer3*100,'Spanning alpha %/yr':a.alpha*100,'Alpha Holm p':a.holm_p})
    view=pd.DataFrame(view);view.to_csv(OUT/'headline_table.csv',index=False)
    plain=stats[(stats.method=='plain')&(stats.factor!='Mom12m')&(stats.period=='holdout')&(stats.cost_bps==5)].copy()
    plain=plain[['factor','annual_mean','cagr','sharpe','annual_vol','turnover','fees','mean_short','active']]
    summary=f"2015–2025, 5bp+1% borrow: {diag['positive_increment']}/36 announcement-cycle models have positive paired net increments; {diag['significant_increment']}/36 have positive increments significant after Holm36; {diag['joint_success']}/36 meet both adjusted positive-increment and positive-own-net-return tests. These are retrospective walk-forward tests, not untouched or live performance."
    (OUT/'summary.txt').write_text(summary+'\n\n'+view.to_string(index=False)+'\n')
    sources=(HERE/'sources.txt').read_text()
    sensitivity_note='' if HORIZON==1 else '<p class="callout"><b>Post-result targeted robustness:</b> this run fits exact five-session held-share returns divided by5, with calendar-year boundary purging. Conditional residual variance is converted to a daily equivalent by multiplying residual-square by5. The general methodology text below describes the primary run; the one-day-target persistence limitation is replaced here by direct five-day labels. This sensitivity was added after viewing primary results, is not a new untouched test, and must be read alongside the combined Holm72 correction.</p>'
    body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Announcement cycle: incremental factor alpha</title><style>body{{font:16px/1.55 system-ui;max-width:1400px;margin:40px auto;padding:0 28px;color:#18303a}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:7px;border-bottom:1px solid #d6e0e4;text-align:right}}th{{background:#edf3f5}}pre{{white-space:pre-wrap;background:#edf3f5;padding:18px}}img{{width:100%}}.callout{{padding:20px;background:#edf3f5;border-left:5px solid #226677}}a{{color:#126383}}</style></head><body>
    {sensitivity_note}
    <h1>Does announcement-cycle information improve these six factors?</h1><p class="callout">{summary}</p>
    <p>Each of six regression families is fitted twice: common information, then common information plus the clock. Historical means, lagged volatility, factor strength, 12-month momentum, size, price and liquidity determine bounded position sizes. Costs are charged to actual stock trades. The previously quoted historical monthly return ranges are not verified comparable net returns; our local baselines below are the relevant hurdle.</p>
    <h2>Best observed incremental model for each factor</h2><p>These rows are selected <b>after observing 2015–2025</b>, solely to summarize the full 36-row experiment. They are not six independently selected strategies. All tests retain the 36-comparison Holm adjustment, including the concentrated RDIPO sensitivity. Returns are annual arithmetic means, not monthly returns or CAGR.</p>{table(view)}
    <img src="incremental_returns.png" alt="All36 paired net increments and HAC intervals">
    <h2>What was held constant</h2><p>Exact same factor universe and missingness; fixed factor signs and source percentile thirds (binary raw0/raw1 for RDIPO); same noncycle inputs, training rows, optimizer, rebalance schedule, +100%/−100% target exposure, allocation bounds, and fee assumptions. Positions drift between five-session rebalances. First five factors require20 stocks on each side. RDIPO is separately concentrated: minimum5, no more than20% of NAV in one stock at rebalance. If a leg fails its minimum, the entire factor book goes to cash at that rebalance.</p>
    <p>We use source-universe percentile thirds for a matched local factor construction, not an exact paper replication. For example, original Cash portfolios use more extreme quantiles and original CompEquIss includes a price filter. MS and RIO_Disp retain the source's embedded growth/disagreement screens. Returns include the restricted regular-announcement, analyst-covered universe.</p>
    <h2>Mathematical sizing rule</h2><pre>For each leg, let a_i be positive dollar weights, b_i=1/n equal weights,
p_i be the preceding portfolio's drifted weights, μ_i predicted signed return,
and σ_i² conditional variance. Solve at each rebalance:

max_a  H Σ μ_i a_i − (H γ/2) Σ σ_i² (a_i−b_i)² / b_i − c Σ |a_i−p_i|
subject to Σ a_i=1, 0.5b_i ≤ a_i ≤ min(1.5b_i, 0.20).
H=5, γ=10, c=0 / 0.0005 / 0.001 one-way fee per dollar traded.

Interior no-trade condition, with equality multiplier ν:
|H μ_i − H γ σ_i²(p_i−b_i)/b_i − ν| ≤ c.

The conditional mean must justify risk and cost before moving a holding.
Risk is a regularized diagonal tracking penalty, not full portfolio covariance.
H=5 extrapolates a one-day forecast across the next five sessions; no future
five-day labels or future actual announcement dates enter the model.</pre>
    <p>Mean models: OLS; ridge; lasso; elastic net; Huber robust regression; quadratic ridge (squares plus factor, momentum, risk and leg interactions). Fixed penalties: ridge0.1; lasso0.0001; elastic L1=0.00005/L2=0.02; Huber ridge0.05 with1.345 residual threshold. Weighted training objective uses one unit of weight per sampled date/leg. Features are imputed/standardized on training data and standardized coordinates bounded at±6. Mean training labels cap at±20%/day, forecasts at±1%/day. <b>Risk calibration uses uncapped residuals; realized portfolio returns are never capped.</b></p>
    <p>Conditional variance is a ridge prediction of log residual-square, calibrated to arithmetic residual variance and shrunk50% toward the stock's prior63-session realized variance. Model output variance is bounded between10⁻⁵ and0.04 daily. The clock contributes four quarterly Fourier terms, signed forecast distance, pre/post10 flags and age; each interacts with factor rank and portfolio leg.</p>
    <h2>Chronology and inference</h2><p>2005–2025 is annual walk-forward evaluation. Each annual model sees only the preceding ten calendar years. There is no hyperparameter selection on evaluation returns. Monthly factor and momentum snapshots are used only in the next calendar month; volatility excludes the current return; earnings state is known before the return day. Training samples deterministically select at most30 names per factor/leg every fifth session, before observing outcomes. Features predict signed stock returns. No stock-level IID significance claims are made.</p>
    <p>The primary statistic is the paired daily net difference, with Newey–West HAC60 uncertainty and Holm36 adjustment. We also report own-net-return significance, monthly HAC3, Sharpe, mean/variance certainty equivalent (CER=annual mean−3/2 annual variance), and spanning intercept against the corresponding no-cycle strategy, same-universe momentum portfolio, and local raw value-weight market. The spanning intercept is <b>not an official Fama–French alpha</b>; financing/risk-free returns and further risk controls are absent. Unadjusted95% intervals shown in the chart are not simultaneous confidence bands.</p>
    <h2>Plain factor baselines (2015–2025, primary costs)</h2><p>Decimal returns below; turnover counts every dollar bought and sold. Cash earns zero.</p>{table(plain)}
    <h2>Limits and execution</h2><p>Each cost scenario has its own optimizer, positions and NAV. Daily total return earns P&amp;L; ex-distribution return marks shares, leaving distributions in cash. Missing held returns are zero and counted; disappearing rows force an exit at last mark with fees. End-of-study liquidation is charged. The5bp and10bp scenarios include1% annual short borrow; gross charges neither. These are cost scenarios, not historical fee estimates. No market impact, stock-specific borrow availability, financing or rebates. RDIPO and RIO_Disp are particularly exposed to borrow limitations.</p>
    <p>All six factor papers were published after2004; earlier related work examined these holdouts. Annual chronology prevents mechanical future leakage but cannot undo publication hindsight, prior search, revised fundamentals or retrospective linkage. Holm applies to this36-pair family, not all earlier research. Significant findings need genuinely new data and measured execution costs before a deployment claim. A negative finding only rejects this bounded set of models and sizing policies.</p>
    <h2>Files and reproduction</h2><p><a href="paired_inference.csv">All paired tests</a> · <a href="performance.csv">All portfolio statistics</a> · <a href="spanning_alpha.csv">Risk-control intercepts</a> · <a href="predictive_scores.csv">Forecast MSE and risk score</a> · <a href="cycle_moments_descriptive.csv">Cycle × momentum descriptive moments</a> · <a href="daily_returns.csv">Daily return series</a> · <a href="execution_audit.json">Execution audit</a> · <a href="portfolio_validation.json">Independent accounting/optimizer checks</a></p>
    <pre>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_overlay/data.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_overlay/run.py all
OPENBLAS_NUM_THREADS=1 .venv/bin/python factor_overlay/report.py</pre>
    <h2>Verified primary sources and procedural references</h2><pre>{html.escape(sources)}</pre>
    <p>Statistical workflow applied the statsmodels Scientific Agent Skill: Kassis et al. (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a>. Package versions are saved with this study.</p></body></html>'''
    (OUT/'report.html').write_text(body)
    print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--horizon',type=int,choices=[1,5],default=1);args=parser.parse_args();HORIZON=args.horizon
    if HORIZON==5:OUT=HERE/'results_h5'
    main()
