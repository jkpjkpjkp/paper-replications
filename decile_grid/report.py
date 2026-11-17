#!/usr/bin/env python3
"""Paired mean inference, search adjustment, cost sensitivity and HTML report."""
from pathlib import Path
import json,os,sys,html
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent.parent/'.mpl-cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
PERIODS={'development':(1985,2004),'validation':(2005,2014),'holdout':(2015,2025),'post2018':(2019,2025),'recent':(2020,2025),'post2004':(2005,2025)}

def hac(x,lags=20):
    x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    n=len(x);mean=x.mean(0);u=x-mean
    v=(u*u).sum(0)
    for lag in range(1,min(lags,n-1)+1):v+=2*(1-lag/(lags+1))*(u[lag:]*u[:-lag]).sum(0)
    se=np.sqrt(np.maximum(v,0))/n
    t=np.divide(mean,se,out=np.zeros_like(mean),where=se>1e-16)
    p=2*norm.sf(abs(t));p=np.where(se>1e-16,p,1.)
    return dict(mean=mean,se=se,t=t,p=p,low=mean-1.96*se,high=mean+1.96*se)

def exact_stats():
    rows=[];pairs=[];spanning=[]; daily=[]
    frozen=json.loads((OUT/'frozen.json').read_text()); selected=set(frozen['selected_ids'])
    for min_names in [5,20]:
        for start,end in [(1985,2004),(2005,2025)]:
            path=OUT/f'exact_{start}_{end}_min{min_names}.npz'
            if not path.exists():continue
            z=dict(np.load(path)); specs=json.loads(path.with_name(path.stem+'_models.json').read_text());lookup={s['id']:i for i,s in enumerate(specs)}
            years=pd.DatetimeIndex(z['dates']).year
            for period,(lo,hi) in PERIODS.items():
                mask=(years>=lo)&(years<=hi)
                if not mask.any():continue
                for ci,cost in enumerate(z['costs']):
                    r=z['returns'][mask,ci];hs=hac(r)
                    for j,s in enumerate(specs):
                        a=r[:,j];vol=a.std(ddof=1)*252**.5
                        nav=np.cumprod(1+a);peak=np.maximum.accumulate(np.r_[1.,nav])[1:]
                        row=dict(id=s['id'],factor=s['factor'],family=s['family'],selected=s['id'] in selected,period=period,min_names=min_names,cost_bp=cost*1e4,days=len(a),mean_bp=a.mean()*1e4,annual_mean=a.mean()*252,cagr=np.expm1(np.log1p(a).sum()*252/len(a)),sharpe=a.mean()*252/vol if vol>0 else np.nan,max_drawdown=np.min(nav/peak-1),own_t=hs['t'][j],own_p=hs['p'][j],annual_turnover=z['turnover'][mask,ci,j].mean()*252,annual_fees=z['fees'][mask,ci,j].mean()*252,annual_borrow=z['borrow'][mask,ci,j].mean()*252,active=z['active'][mask,ci,j].mean(),n_long=z['n_long'][mask,ci,j].mean(),n_short=z['n_short'][mask,ci,j].mean(),forced_exit=int(z['forced_exit'][mask,ci,j].sum()),missing=int(z['missing'][mask,ci,j].sum()))
                        rows.append(row)
                        if s['id'] not in selected:continue
                        b=lookup[s['baseline_id']];d=a-r[:,b]
                        h=hac(d);h60=hac(d,60)
                        delta_turn=z['turnover'][mask,ci,j].mean()-z['turnover'][mask,ci,b].mean()
                        pairs.append(dict(**row,baseline_id=s['baseline_id'],baseline_bp=r[:,b].mean()*1e4,uplift_bp=d.mean()*1e4,uplift_annual=d.mean()*252,ci_low_bp=h['low'][0]*1e4,ci_high_bp=h['high'][0]*1e4,t_hac20=h['t'][0],p_hac20=h['p'][0],t_hac60=h60['t'][0],p_hac60=h60['p'][0],delta_daily_turnover=delta_turn))
                        timing=[k for k,ss in enumerate(specs) if ss['family']=='timing_only' and 'paper' in ss['id'].lower()]
                        if timing and r[:,b].std()>1e-10:
                            fit=sm.OLS(a,sm.add_constant(np.column_stack([r[:,b],r[:,timing[0]]])),missing='raise').fit(cov_type='HAC',cov_kwds={'maxlags':20})
                            spanning.append(dict(id=s['id'],period=period,min_names=min_names,cost_bp=cost*1e4,alpha_bp=fit.params[0]*1e4,t=fit.tvalues[0],p=fit.pvalues[0]))
            if min_names==5:
                for j,s in enumerate(specs):
                    for ci,cost in enumerate(z['costs']):
                        daily.append(pd.DataFrame(dict(date=z['dates'],id=s['id'],cost_bp=cost*1e4,returns=z['returns'][:,ci,j],turnover=z['turnover'][:,ci,j],fees=z['fees'][:,ci,j],borrow=z['borrow'][:,ci,j])))
    perf=pd.DataFrame(rows);pair=pd.DataFrame(pairs)
    for _,idx in pair.groupby(['period','min_names','cost_bp']).groups.items():
        pair.loc[idx,'p_holm_selected']=multipletests(pair.loc[idx,'p_hac20'],method='holm')[1]
        pair.loc[idx,'own_p_holm_selected']=multipletests(pair.loc[idx,'own_p'],method='holm')[1]
    perf.to_csv(OUT/'performance.csv',index=False);pair.to_csv(OUT/'paired_inference.csv',index=False)
    pd.DataFrame(spanning).to_csv(OUT/'spanning.csv',index=False)
    pd.concat(daily,ignore_index=True).to_csv(OUT/'selected_daily.csv.gz',index=False)
    # Linear cost threshold diagnostic on zero-fee path. Independent exact cost scenarios are authoritative.
    be=[]
    for (id_,period,min_names),g in pair.groupby(['id','period','min_names']):
        gross=g[g.cost_bp==0].iloc[0];base=perf[(perf.id==gross.baseline_id)&(perf.period==period)&(perf.min_names==min_names)&(perf.cost_bp==0)].iloc[0]
        dt=gross.delta_daily_turnover
        db=(gross.active-base.active)*.01/252
        be.append(dict(id=id_,period=period,min_names=min_names,incremental_breakeven_bp=(gross.uplift_bp/1e4-db)/dt*1e4 if dt>1e-12 else np.nan,own_breakeven_bp=(gross.mean_bp/1e4-gross.active*.01/252)/(gross.annual_turnover/252)*1e4 if gross.annual_turnover>0 else np.nan,interpretation='linear gross-path estimate; excludes impact/financing; undefined when incremental turnover is nonpositive'))
    pd.DataFrame(be).to_csv(OUT/'breakeven_costs.csv',index=False)
    return perf,pair


def expost_stats():
    """Cost sensitivity of later-selected maxima; never pooled with frozen tests."""
    path=OUT/'expost_exact_2015_2025_min5.npz'
    if not path.exists():return pd.DataFrame()
    z=dict(np.load(path));specs=json.loads(path.with_name(path.stem+'_models.json').read_text())
    lookup={s['id']:j for j,s in enumerate(specs)}
    ex=pd.read_csv(OUT/'grid_expost_best.csv');ids=set(ex.loc[ex.period.eq('holdout'),'id'].dropna())
    rows=[]
    for ci,cost in enumerate(z['costs']):
        r=z['returns'][:,ci]
        for s in specs:
            if s['id'] not in ids:continue
            j=lookup[s['id']];b=lookup[s['baseline_id']];d=r[:,j]-r[:,b];h=hac(d)
            rows.append(dict(id=s['id'],cost_bp=cost*1e4,mean_bp=r[:,j].mean()*1e4,baseline_bp=r[:,b].mean()*1e4,
                uplift_bp=d.mean()*1e4,ci_low_bp=h['low'][0]*1e4,ci_high_bp=h['high'][0]*1e4,
                active=z['active'][:,ci,j].mean(),annual_turnover=z['turnover'][:,ci,j].mean()*252,
                interpretation='Selected on these same 2015-2025 returns; descriptive cost diagnostic, not validated strategy'))
    result=pd.DataFrame(rows);result.to_csv(OUT/'expost_cost_sensitivity.csv',index=False)
    return result


def make_report(perf,pair):
    specs=json.loads((OUT/'specs.json').read_text());frozen=json.loads((OUT/'frozen.json').read_text());des=json.loads((HERE/'design.json').read_text())
    p=pair[(pair.min_names==5)&(pair.period=='holdout')]
    gross=p[p.cost_bp==0].sort_values('uplift_bp',ascending=False)
    selected=frozen['selected_ids'];main=frozen.get('primary_id') or selected[0]
    focus=pair[(pair.id==main)&(pair.min_names==5)]
    g=focus[(focus.period=='holdout')&(focus.cost_bp==0)].iloc[0];net=focus[(focus.period=='holdout')&(focus.cost_bp==5)].iloc[0]
    best=gross.iloc[0]
    summary=f"No reliable incremental advantage survives this search. Searched {len(specs):,} specifications and froze {len(selected)} gross-selected candidates using 1985-2004 only. Primary: {main}. In 2015-2025 it earned {g.mean_bp:.3f} bp/day gross versus {g.baseline_bp:.3f} for its original-factor control: incremental {g.uplift_bp:.3f} bp/day (95% HAC20 CI {g.ci_low_bp:.3f} to {g.ci_high_bp:.3f}; p={g.p_hac20:.4f}; selected-family Holm p={g.p_holm_selected:.4f}). At 5 bp one-way plus 1% annual borrow, it earned {net.mean_bp:.3f} bp/day, versus {net.baseline_bp:.3f}; incremental {net.uplift_bp:.3f} bp/day. None of the five frozen candidates has positive net uplift at this cost in 2015-2025. None of all 2,532 timing candidates has significant positive gross uplift after full-grid Holm correction. This is retrospective research, not an untouched holdout or exact replication of the paper."
    (OUT/'summary.txt').write_text(summary+'\n')
    cols=['id','cost_bp','mean_bp','baseline_bp','uplift_bp','ci_low_bp','ci_high_bp','p_hac20','p_holm_selected','active','annual_turnover']
    def table(x,cs):return x[cs].round(4).to_html(index=False,border=0)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for period in ['validation','holdout','post2018']:
        x=focus[focus.period==period].sort_values('cost_bp')
        axes[0].plot(x.cost_bp,x.mean_bp,marker='o',label=period)
        axes[1].plot(x.cost_bp,x.uplift_bp,marker='o',label=period)
    for ax,title in zip(axes,['Primary portfolio return','Increment over matching original factor']):
        ax.axhline(0,color='grey',lw=.8);ax.set(title=title,xlabel='One-way cost (bp)',ylabel='Arithmetic daily return (bp)');ax.legend();ax.grid(alpha=.15)
    fig.suptitle('Training-selected primary | net scenarios also charge 1% annual short borrow')
    fig.savefig(OUT/'cost_curve.png',dpi=160);plt.close(fig)
    primary_spec=next(s for s in specs if s['id']==main)
    ex=expost_stats()
    exhtml=table(ex[ex.cost_bp.isin([0,5,10])],['id','cost_bp','mean_bp','baseline_bp','uplift_bp','active']) if len(ex) else '<p>See separate ex-post cost file.</p>'
    best_note=f"The best later-period gross increment among the frozen candidates was {best.id}: {best.uplift_bp:.3f} bp/day. This ranking uses later returns and is descriptive, not the training-selected primary."
    invalid=pd.read_csv(OUT/'gross_development.csv') if (OUT/'gross_development.csv').exists() else pd.DataFrame()
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Decile factors and announcement-cycle grid</title>
<style>body{{font:16px/1.6 system-ui;max-width:1300px;margin:36px auto;padding:0 24px;color:#18313e}}h1,h2{{line-height:1.25}}table{{border-collapse:collapse;font-size:12px;display:block;overflow-x:auto}}th,td{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}td:first-child{{text-align:left;max-width:430px;overflow-wrap:anywhere}}.lead{{background:#edf5f3;padding:20px}}img{{max-width:100%}}pre{{white-space:pre-wrap;font-size:13px}}a{{color:#17669a}}</style>
<h1>Original-factor tails × earnings announcement cycle</h1><p class="lead">{html.escape(summary)}</p>
<h2>What was searched</h2><p>Six existing factors: MS, CompEquIss, InvestPPEInv, Cash, RIO_Disp and RDIPO. D10 uses signed original-factor source-universe ranks above90% versus at/below10%. Its timing alternatives include all four high/low timing-decile combinations and all nine pairs of the paper's three phases. S30 uses the low30%, middle40%, high30% original-factor split; two characteristic groups crossed with these three signal buckets form2×3. The long/short factor uses only the high and low signal buckets. All nine phase pairs are tested. Characteristic splits use eligible-universe medians of lagged market capitalization, ADV, or63-session total volatility, with low-only, high-only, and half-weight balanced portfolios. Equal-weight and lag-cap-weight implementations are tested. This is exhaustive within the defined finite grid, not every possible investment rule.</p>
<p>The timing-decile score is cos(2π(age−13)/65), with average-tie ranks across currently eligible stocks. Its phase and period are fixed research choices, not fitted returns. Timing deciles are an extension; the paper uses age6–20 long and36–50 short. Phase2 is21–35. All timing uses known past releases; no future actual date enters trades. Every candidate has a control with identical factor tails, characteristic treatment and weighting, but no timing filter. Characteristic grouping is recomputed daily from lagged values. It is not the paper's quarter-fixed quintile sort or a NYSE-breakpoint Fama–French factor.</p>
<p>Strict percentile cutoffs preserve ties. RDIPO and RIO_Disp may lack a high decile entirely; an empty tail earns cash, not a fabricated factor return. Minimum5 stocks per leg, and per half for balanced portfolios; minimum20 sensitivity is also reported. The primary is the eligible training candidate with the largest paired HAC20 t-statistic. Selection requires positive own gross return and uplift, at least50% candidate active days and80% baseline active days. All quoted daily means include inactive cash days.</p>
<h2>Primary specification</h2><pre>{html.escape(json.dumps(primary_spec,indent=2))}</pre>
<h2>Gross advantage, then costs</h2><p>All spreads target$1 long and$1 short per$1 NAV (200% gross). Returns and differences below are basis points per day; annual means multiply decimal daily returns by252, while CAGR compounds. Costs apply to every bought/sold dollar, including terminal liquidation. Net scenarios add1% annual short borrow. These are assumed fees; impact, financing, rebates and actual stock-loan constraints are absent.</p><img src="cost_curve.png" alt="Primary return and incremental return across cost scenarios">
{table(focus[focus.period.isin(['development','validation','holdout','post2018'])],['period']+cols[1:])}
<h2>All frozen candidates:2015–2025</h2><p>{html.escape(best_note)}</p>{table(p[p.cost_bp.isin([0,5,10])],cols)}
<h2>Full-grid historical maxima and costs</h2><p>There are 2,532 timing candidates plus 228 matching factor-only controls and two timing controls. In 2015–2025, 358 timing candidates pass the activity thresholds; 117 have positive own and incremental gross returns. Ten show positive uplift with unadjusted HAC20 p&lt;0.05; zero survive Holm correction across all 2,532 candidates. No candidate jointly establishes positive own return and positive uplift even before adjustment. No D10 strategy trades on at least 50% of holdout sessions while its baseline trades on at least 80%.</p><p>The following portfolios were deliberately selected using the same 2015–2025 outcomes being displayed, as maxima of uplift or its t-statistic within each factor/sort family. Their cost results answer whether attractive historical point estimates persist after fees; they are not additional validated discoveries. The original training selections are unchanged.</p>{exhtml}
<h2>Concentration sensitivity: minimum20 per leg or cell</h2>{table(pair[(pair.min_names==20)&(pair.period=='holdout')&(pair.cost_bp.isin([0,5]))],cols)}
<p>With a minimum of 20 stocks, all five frozen candidates are active on less than 8% of holdout dates. Positive differences while largely in cash do not establish an investable timing strategy. The value-weighted RIO_Disp candidate is uncapped: its largest single position reaches 58.0% of NAV over 2005–2025 despite the minimum-name rule.</p>
<h2>Inference and limits</h2><p>Paired daily differences use Bartlett/Newey–West HAC20; HAC60 appears in the CSV. Holm adjustment covers the frozen selected candidates within each cost/period; the full-grid gross results additionally adjust over all timing candidates. Neither corrects for all earlier research in this workspace. A positive difference is not proof of risk-adjusted alpha: timing can change volatility, factor exposure, concentration and time in cash. Spanning regressions against the matched factor and fixed paper timing portfolio are additional diagnostics. Break-even estimates use linear gross-path turnover; independently simulated fee scenarios remain authoritative.</p>
<p>Training-selected p-values are not confirmatory evidence on the training sample. Spanning regression p-values are descriptive and unadjusted. The post2018 label means 2019–2025. Independent scalar portfolio, raw-source spot, ledger and HAC checks passed; their audit files accompany the results.</p><p>The inherited universe uses at least5 analysts over180days, all last3 announcement gaps50–80sessions, median60–70, and known age≤80. The paper instead uses three-month analyst coverage, four releases in400calendar days and a70–110calendar-day gap. Our ADV and total volatility are proxies, not its Amihud and residual-volatility measures. Monthly stock signals are delayed one full calendar-month snapshot; their historical accounting vintages and retrospective security links are not independently point-in-time audited. All six original factors were published after2004. Related later periods were already inspected in earlier studies, so chronological splits do not eliminate publication or researcher hindsight.</p>
<h2>Results and reproduction</h2><p><a href="paired_inference.csv">Paired gross/net inference</a> · <a href="performance.csv">All selected performance</a> · <a href="grid_gross_summary.csv">Entire gross grid</a> · <a href="grid_expost_best.csv">Descriptive grid maxima</a> · <a href="breakeven_costs.csv">Break-even costs</a> · <a href="selected_daily.csv.gz">Daily selected ledgers</a> · <a href="frozen.json">Frozen selection</a> · <a href="../README.txt">Reproduction and definitions</a></p>
<h2>Sources and software</h2><p>Linnainmaa and Zhang(2018), <em>The Earnings Announcement Return Cycle</em>, supplied PDF, especially Tables2–3 and6. Existing CRSP/IBES panel and Chen–Zimmermann2026.10 monthly stock characteristics; lineage remains in prior study audits. NumPy, pandas, DuckDB, statsmodels and Matplotlib. Procedural support: statsmodels skill, credited to Kassis, Agarwal, He, Patel and Brueckner(2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills</a> (current metadata checked2026-11-02); this is a workflow reference, not evidence for trading profitability.</p></html>'''
    (OUT/'report.html').write_text(page)
    print(summary)

if __name__=='__main__':make_report(*exact_stats())
