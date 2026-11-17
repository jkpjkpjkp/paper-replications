#!/usr/bin/env python3
"""Reports and paired inference for frozen entry/exit-horizon portfolios.

Consumes horizons.py outputs; never changes strategy choices or input caches.
The exhaustive event regressions are explicitly distinguished from exact
calendar-time portfolio results. Run only after final/sensitivity files exist.
"""
from __future__ import annotations
import gzip
import html
import json
import math
import os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import horizons as h

OUT=h.OUT


def mean_hac(values, lag=60):
    """Intercept-only Bartlett HAC, matching statsmodels' default HAC fit.

    Uses exact scalar sufficient statistics to make thousands of prespecified
    paired tests economical. A statsmodels comparison is recorded for primary.
    """
    x=np.asarray(values,float)
    assert np.isfinite(x).all()
    n=len(x)
    if n<5:
        return dict(observations=n,annual_mean=np.nan,lo=np.nan,hi=np.nan,t=np.nan,p=1.)
    mean=float(x.mean())
    if np.max(np.abs(x))<1e-15:
        return dict(observations=n,annual_mean=0.,lo=0.,hi=0.,t=0.,p=1.)
    e=x-mean;lag=min(int(lag),n-1)
    varsum=float(e@e)
    for k in range(1,lag+1):
        varsum+=2*(1-k/(lag+1))*float(e[k:]@e[:-k])
    se=math.sqrt(max(0.,varsum))/n
    if se<=1e-18:
        t=math.copysign(float('inf'),mean) if mean else 0.
        p=0. if mean else 1.
    else:
        t=mean/se;p=float(2*norm.sf(abs(t)))
    z=float(norm.ppf(.975))
    return dict(observations=n,annual_mean=mean*252,lo=(mean-z*se)*252,hi=(mean+z*se)*252,t=t,p=p)


def _cost_index(array,cost=.0005):
    indices=np.flatnonzero(np.isclose(array['costs'],cost,atol=1e-12))
    assert len(indices)==1,(array['costs'],cost)
    return int(indices[0])


def _load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def inference(array,models,frozen):
    dates=array['dates'];ci=_cost_index(array);ret=array['returns'][:,ci];active=array['active'][:,ci].astype(bool)
    ix={m['id']:j for j,m in enumerate(models)};lookup={m['id']:m for m in models}
    rows=[]
    for period,(lo,hi) in h.base.PERIODS.items():
        days=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
        if not days.any():continue
        for name in frozen['candidates']:
            m=lookup[name];j=ix[name];bid=h.model(m['factor'],1,1)['id'];b=ix[bid]
            common=active[:,j]&active[:,b]
            for scope,mask in [('all_days',days),('common_active',days&common)]:
                for benchmark in (['same_factor_N01_M01','zero'] if scope=='all_days' else ['same_factor_N01_M01']):
                    diff=ret[mask,j] if benchmark=='zero' else ret[mask,j]-ret[mask,b]
                    for lag in [20,60]:
                        rows.append(dict(model=name,factor=m['factor'],buy_before=m['n'],sell_after=m['m'],baseline=bid,period=period,scope=scope,benchmark=benchmark,hac_lags=lag,common_active_fraction=float(common[days].mean()),candidate_active_fraction=float(active[days,j].mean()),baseline_active_fraction=float(active[days,b].mean()),**mean_hac(diff,lag)))
    tests=pd.DataFrame(rows);tests['p_holm']=np.nan
    for _,group in tests.groupby(['period','scope','benchmark','hac_lags'],sort=False):
        tests.loc[group.index,'p_holm']=multipletests(group.p.fillna(1).to_numpy(),method='holm')[1]
    tests.to_csv(OUT/'paired_inference.csv',index=False)
    # Independent verification of the optimized HAC against the named library.
    primary=lookup[frozen['primary']];j=ix[primary['id']];b=ix[h.model(primary['factor'],1,1)['id']]
    validation=[]
    for lag in [20,60]:
        mask=dates>=np.datetime64('2005-01-01');x=ret[mask,j]-ret[mask,b]
        if len(x)<5 or np.std(x)<1e-15:continue
        fit=sm.OLS(x,np.ones((len(x),1)),missing='raise').fit(cov_type='HAC',cov_kwds={'maxlags':lag})
        own=mean_hac(x,lag);interval=fit.conf_int()[0]*252
        error=max(abs(own['annual_mean']-float(fit.params[0]*252)),abs(own['lo']-float(interval[0])),abs(own['hi']-float(interval[1])),abs(own['p']-float(fit.pvalues[0])))
        assert error<1e-10,error
        validation.append(dict(hac_lags=lag,max_statsmodels_difference=error))
    (OUT/'inference_validation.json').write_text(json.dumps(dict(primary_hac_checks=validation,holm_candidates=len(frozen['candidates']),same_factor_baseline_ids_verified=True),indent=2))
    return tests


def export_returns(array,models):
    dates=pd.to_datetime(array['dates']);years=dates.year.to_numpy();annual=[]
    # Streaming avoids retaining a multi-million-row object table in memory.
    with gzip.open(OUT/'daily_returns.csv.gz','wt',encoding='utf-8',newline='') as stream:
        first=True
        for ci,cost in enumerate(array['costs']):
            for j,m in enumerate(models):
                returns=array['returns'][:,ci,j]
                frame=pd.DataFrame(dict(date=dates,model=m['id'],factor=m['factor'],buy_before=m['n'],sell_after=m['m'],cost_bps=cost*10000,returns=returns,turnover=array['turnover'][:,ci,j],fees=array['fees'][:,ci,j],active=array['active'][:,ci,j],n_long=array['n_long'][:,ci,j],n_short=array['n_short'][:,ci,j]))
                frame.to_csv(stream,index=False,header=first);first=False
                for year in np.unique(years):
                    r=returns[years==year]
                    annual.append(dict(model=m['id'],factor=m['factor'],buy_before=m['n'],sell_after=m['m'],cost_bps=cost*10000,year=int(year),return_compounded=float(np.prod(1+r)-1)))
    annual=pd.DataFrame(annual);annual.to_csv(OUT/'annual_returns.csv',index=False)
    return annual


def log_growth_inference(array,models,frozen):
    """Supplement arithmetic-profit tests with paired compounded-growth tests."""
    dates=array['dates'];ret=array['returns'][:,_cost_index(array)]
    ix={m['id']:j for j,m in enumerate(models)};lookup={m['id']:m for m in models};rows=[]
    for period,(lo,hi) in h.base.PERIODS.items():
        keep=(dates>=np.datetime64(f'{lo}-01-01'))&(dates<=np.datetime64(f'{hi}-12-31'))
        if not keep.any():continue
        for cid in frozen['candidates']:
            bid=h.model(lookup[cid]['factor'],1,1)['id'];x=ret[keep,ix[cid]];y=ret[keep,ix[bid]]
            valid=bool((x>-1).all() and (y>-1).all())
            for lag in [20,60]:
                stats=mean_hac(np.log1p(x)-np.log1p(y),lag) if valid else dict(observations=len(x),annual_mean=np.nan,lo=np.nan,hi=np.nan,t=np.nan,p=1.)
                rows.append(dict(model=cid,baseline=bid,period=period,hac_lags=lag,finite_log_paths=valid,**stats))
    table=pd.DataFrame(rows);table['p_holm']=np.nan
    for _,g in table.groupby(['period','hac_lags']):table.loc[g.index,'p_holm']=multipletests(g.p,method='holm')[1]
    table.to_csv(OUT/'paired_log_growth.csv',index=False)
    return table


def _plain_json(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _percent_table(frame,percentage=(),decimals=3):
    out=frame.copy()
    for c in out:
        if c in percentage:
            out[c]=out[c].map(lambda v:f'{v:.2%}' if pd.notna(v) else '—')
        elif pd.api.types.is_float_dtype(out[c]):
            out[c]=out[c].map(lambda v:f'{v:.{decimals}g}' if pd.notna(v) else '—')
    return out.to_html(index=False,escape=True,border=0)


def _paired_table(tests,names=None,periods=('validation','holdout','post2004'),scope='all_days'):
    keep=(tests.hac_lags==60)&tests.period.isin(periods)&tests.scope.eq(scope)&tests.benchmark.eq('same_factor_N01_M01')
    if names is not None:keep&=tests.model.isin(names)
    cols=['model','period','observations','annual_mean','lo','hi','p','p_holm','common_active_fraction']
    return _percent_table(tests.loc[keep,cols],['annual_mean','lo','hi','common_active_fraction'])


def figures(array,models,frozen,tests,screen):
    ci=_cost_index(array);dates=array['dates'];ix={m['id']:j for j,m in enumerate(models)}
    primary=next(m for m in models if m['id']==frozen['primary']);bid=h.model(primary['factor'],1,1)['id']
    keep=dates>=np.datetime64('2005-01-01')
    fig,axes=plt.subplots(2,1,figsize=(11,8),layout='constrained')
    for name,label in [(primary['id'],f"Frozen primary: {primary['factor']} N={primary['n']}, M={primary['m']}"),(bid,'Same factor baseline: N=1, M=1')]:
        axes[0].plot(dates[keep],np.cumprod(1+array['returns'][keep,ci,ix[name]]),label=label)
    axes[0].set(yscale='log',ylabel='Growth of $1 (log scale)',title='Exact calendar-time dollar-neutral portfolios; 5 bp + 1% annual short borrow')
    axes[0].legend(fontsize=9)
    t=tests[(tests.model==primary['id'])&tests.period.isin(['validation','holdout','post2004'])&tests.scope.eq('all_days')&tests.benchmark.eq('same_factor_N01_M01')&(tests.hac_lags==60)].copy()
    axes[1].errorbar(np.arange(len(t)),t.annual_mean*100,yerr=np.vstack([(t.annual_mean-t.lo)*100,(t.hi-t.annual_mean)*100]),fmt='o',capsize=5)
    axes[1].set_xticks(np.arange(len(t)),t.period);axes[1].axhline(0,color='gray',lw=1)
    axes[1].set(ylabel='Annual mean difference (percentage points)',title='Paired primary minus -1/+1 baseline: HAC60 95% intervals')
    fig.savefig(OUT/'primary_comparison.png',dpi=165);plt.close(fig)
    if screen is not None and len(screen):
        s=screen[screen.factor==primary['factor']]
        matrix=s.pivot(index='buy_before',columns='sell_after',values='sharpe_proxy').sort_index().sort_index(axis=1)
        fig,ax=plt.subplots(figsize=(8.5,7),layout='constrained')
        a=matrix.to_numpy(float);finite=a[np.isfinite(a)];limit=float(np.quantile(np.abs(finite),.98)) if len(finite) else 1.
        if limit<=0:limit=1.
        im=ax.imshow(a,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
        xs=matrix.columns.to_numpy();ys=matrix.index.to_numpy();ticks=np.arange(0,len(xs),5)
        ax.set_xticks(ticks,xs[ticks]);ticks=np.arange(0,len(ys),5);ax.set_yticks(ticks,ys[ticks])
        ax.scatter(np.flatnonzero(xs==primary['m']),np.flatnonzero(ys==primary['n']),marker='x',s=100,c='black',linewidths=2,label='Exact training-selected primary')
        ax.set(xlabel='M: sessions after actual announcement (availability-clamped)',ylabel='N: sessions before forecast announcement',title=f"{primary['factor']}: 1985-2004 EVENT screening Sharpe proxy")
        ax.legend(fontsize=8,loc='best');fig.colorbar(im,ax=ax,label='Event-weighted proxy; not calendar strategy Sharpe')
        fig.savefig(OUT/'training_N_M_heatmap.png',dpi=165);plt.close(fig)


def report():
    required=['final.npz','final_models.json','frozen_choices.json','final_statistics.csv','regression_screen.csv']
    for name in required:
        if not (OUT/name).exists():raise FileNotFoundError(f'Run horizons final evaluation first: {OUT/name}')
    frozen=_plain_json(OUT/'frozen_choices.json');models=_plain_json(OUT/'final_models.json');array=_load(OUT/'final.npz')
    stats=pd.read_csv(OUT/'final_statistics.csv');stats=stats[np.isclose(stats.cost_bps,5)].copy()
    # The final evaluation may start with fresh capital in 2005, after training
    # choices and training-end liquidation. Development values then come from
    # the separate exact training book, never from a warm-started final path.
    if not stats.period.eq('development').any() and (OUT/'exact_train_statistics.csv').exists():
        development=pd.read_csv(OUT/'exact_train_statistics.csv')
        development=development[development.period.eq('development')&np.isclose(development.cost_bps,5)&development.id.isin({m['id'] for m in models})]
        stats=pd.concat([development,stats],ignore_index=True,sort=False)
    screen=pd.read_csv(OUT/'regression_screen.csv');tests=inference(array,models,frozen)
    log_tests=log_growth_inference(array,models,frozen)
    export_returns(array,models);figures(array,models,frozen,tests,screen)
    lookup={m['id']:m for m in models};primary=lookup[frozen['primary']];baseline=h.model(primary['factor'],1,1)['id']
    post=stats[stats.period.eq('holdout')&stats.id.isin(frozen['candidates'])]
    if 'bankrupt' in post:post=post[~post.bankrupt]
    best=post.sort_values(['annual_mean','id'],ascending=[False,True]).iloc[0] if len(post) else None
    # Direction is essential: a statistically significant negative difference
    # must never be labeled an improved strategy.
    counts=[]
    for period in ['validation','holdout','post2004']:
        t=tests[(tests.period==period)&tests.scope.eq('all_days')&(tests.hac_lags==60)]
        paired=t[t.benchmark.eq('same_factor_N01_M01')].set_index('model')
        own=t[t.benchmark.eq('zero')].set_index('model')
        better=set(paired[(paired.annual_mean>0)&(paired.p_holm<.05)].index)
        profitable=set(own[(own.annual_mean>0)&(own.p_holm<.05)].index)
        viable=(~stats.bankrupt) if 'bankrupt' in stats else np.ones(len(stats),bool)
        positive=set(stats[(stats.period==period)&(stats.annual_mean>0)&(stats.cagr>0)&viable].id)
        profitable&=positive
        counts.append(dict(period=period,positive_significant_improvement=len(better),positive_return_and_significant_improvement=len(better&positive),positive_significant_profit_and_improvement=len(better&profitable)))
    count=pd.DataFrame(counts);count.to_csv(OUT/'significance_counts.csv',index=False)
    ph=stats[(stats.id==primary['id'])&stats.period.eq('holdout')].iloc[0]
    ptest=tests[(tests.model==primary['id'])&tests.period.eq('holdout')&tests.scope.eq('all_days')&tests.benchmark.eq('same_factor_N01_M01')&(tests.hac_lags==60)].iloc[0]
    number=int(count.loc[count.period.eq('holdout'),'positive_significant_improvement'].iloc[0])
    verified=int(count.loc[count.period.eq('holdout'),'positive_significant_profit_and_improvement'].iloc[0])
    conclusion=(f'完整训练筛选覆盖 {len(screen):,} 个 factor×N×M 事件回归。实际逐日多空回测仅覆盖训练筛出的 '
      f'{frozen.get("exact_shortlist_count","若干")} 个规格；之后冻结 {len(frozen["candidates"])} 个候选进行后期检验。'
      f'2015–2025 年有 {number} 个候选在 Holm 校正后显著优于同 factor 的 −1/+1 基准；'
      f'其中 {verified} 个同时具备显著正的净收益。'
      f'训练选定主策略为 {primary["factor"]}、N={primary["n"]}、M={primary["m"]}，'
      f'2015–2025 年算术年均净收益 {ph.annual_mean:.2%}、CAGR {ph.cagr:.2%}、Sharpe {ph.sharpe:.2f}；'
      f'相对同 factor 基准的算术年均差为 {ptest.annual_mean:.2%}，HAC60 95% 区间 '
      f'[{ptest.lo:.2%}, {ptest.hi:.2%}]，Holm p={ptest.p_holm:.3g}。')
    if verified==0:
        conclusion+='本次搜索尚未找到同时满足“自身显著盈利”和“显著优于基准”的后期结果。'
    else:
        conclusion+='显著性计数属于既有数据上的回顾检验，后续实时数据仍需确认。'
    if best is not None:
        conclusion+=(f' 后期算术收益最高的冻结候选为 {best.factor}、N={int(best.n)}、M={int(best.m)}，'
                     f'年均 {best.annual_mean:.2%}；这是查看后期结果后的描述标签。')
    (OUT/'conclusion.txt').write_text(conclusion+'\n')
    focus_names={primary['id'],baseline}
    if best is not None:
        focus_names|={best.id,h.model(best.factor,1,1)['id']}
    cols=['id','factor','n','m','period','annual_mean','cagr','sharpe','max_drawdown','active','mean_longs','mean_shorts','annual_turnover','annual_cost']
    if 'bankrupt' in stats:cols.append('bankrupt')
    focus=stats[stats.id.isin(focus_names)&stats.period.isin(['development','validation','holdout','post2004'])][cols]
    focus.to_csv(OUT/'primary_and_expost_comparison.csv',index=False)
    # Every per-factor frozen winner, including winners that equal their own baseline.
    winners=stats[stats.id.isin(frozen.get('per_factor_winners',frozen['candidates']))&stats.period.isin(['validation','holdout','post2004'])].copy()
    paired=tests[(tests.hac_lags==60)&tests.scope.eq('all_days')&tests.benchmark.eq('same_factor_N01_M01')]
    winner_table=winners.merge(paired[['model','period','annual_mean','lo','hi','p','p_holm','common_active_fraction']],left_on=['id','period'],right_on=['model','period'],suffixes=('','_difference'),validate='one_to_one')
    bcols=['id','period','annual_mean','cagr','sharpe','active']+(['bankrupt'] if 'bankrupt' in stats else [])
    baseline_stats=stats[bcols].rename(columns={'id':'baseline','annual_mean':'baseline_annual_mean','cagr':'baseline_cagr','sharpe':'baseline_sharpe','active':'baseline_active','bankrupt':'baseline_bankrupt'})
    winner_table['baseline']=[h.model(f,1,1)['id'] for f in winner_table.factor]
    winner_table=winner_table.merge(baseline_stats,on=['baseline','period'],validate='many_to_one')
    winner_table.to_csv(OUT/'per_factor_comparison.csv',index=False)
    audit=_plain_json(OUT/'final_audit.json');reg_audit=_plain_json(OUT/'regression_screen_audit.json');independent=_plain_json(OUT/'independent_validation.json')
    common=_paired_table(tests,[primary['id']],scope='common_active')
    sensitivity=''
    for file,title in [('cost_sensitivity_statistics.csv','主策略与基准：0/5/10 bp 成本'),('min5_sensitivity_statistics.csv','每边至少5只股票的敏感性'),('holdout_reset_statistics.csv','2015年重新以1美元开始：主策略与基准'),('holdout_reset2015_statistics.csv','2015年重新以1美元开始：主策略与基准')]:
        p=OUT/file
        if p.exists():
            d=pd.read_csv(p);d=d[d.period.isin(['validation','holdout','post2004'])]
            useful=[c for c in ['id','period','cost_bps','annual_mean','cagr','sharpe','max_drawdown','active','mean_longs','mean_shorts','bankrupt'] if c in d]
            sensitivity+=f'<h3>{title}</h3>'+_percent_table(d[useful],['annual_mean','cagr','max_drawdown','active'])
    oracle=''
    op=Path(__file__).resolve().parent/'horizon_oracle_results'/'regression_screen.csv'
    if op.exists():
        o=pd.read_csv(op);o=o[o.eligible] if 'eligible' in o else o
        o=o.sort_values('sharpe_proxy',ascending=False).head(10)
        useful=[c for c in ['factor','buy_before','sell_after','annual_mean_proxy','sharpe_proxy','cluster_t','cluster_p'] if c in o]
        oracle=('<h2>实际公告日期的事后事件诊断</h2><p>此诊断在进场时使用未来实际公告日，存在事前不可知信息。'
                '以下仅为事件回归/收益代理，不能视为可交易组合，也没有用于主策略选择。</p>'
                +_percent_table(o[useful],['annual_mean_proxy'])+'<p><a href="../horizon_oracle_results/regression_screen.csv">全部事后诊断回归</a></p>')
    screen_link='<a href="regression_screen.csv">188,100个训练事件回归（完整CSV）</a>'
    surfaces=''
    if (OUT/'surface_summary.json').exists():
        ss=_plain_json(OUT/'surface_summary.json')
        coeff=pd.read_csv(OUT/'surface_coefficients.csv')
        surfaces=('<h2>三个参数的共同响应曲面回归</h2><p>补充回归为：事件年化收益代理 = factor固定效应 + N + M + N² + M² + N×M，N/M以15.5日为中心。'
          '该回归仅使用训练期网格；系数、标准误及p值是相关性描述，不能替代组合盈利检验。不同factor共享股票，按factor聚类仍不能完全消除其相关性。</p>'
          +_percent_table(coeff)+'<p><a href="surface_factor_effects.csv">factor固定效应与各因子最佳窗口</a> · <a href="surface_fitted_grid.csv">共同拟合曲面</a> · <a href="surface_summary.json">解释率、边界检查和多重检验计数</a></p>')
    diagnostics=''
    if (OUT/'entry_timing_diagnostics.csv').exists():
        dd=pd.read_csv(OUT/'entry_timing_diagnostics.csv')
        diagnostics=('<h3>预测日期与实际公告日的偏差</h3><p>下表是入场时因子位于上下三分之一的计划机会诊断，包含因整本组合无法配平而未实际成交的计划。实际公告发生后、记录尚不可知时，仍可能按旧预测计划入场；这些情况没有用未来信息删除。该表不能当作真实成交次数。</p>'
          +_percent_table(dd,[c for c in dd.columns if c.startswith('fraction_')]))
    log_primary=log_tests[(log_tests.model==primary['id'])&(log_tests.hac_lags==60)]
    log_count=int(((log_tests.period=='holdout')&(log_tests.hac_lags==60)&(log_tests.annual_mean>0)&(log_tests.p_holm<.05)).sum())
    log_html=(f'<h3>复利增长差的补充检验</h3><p>另对逐日log(1+收益)之差做HAC检验。2015–2025年有{log_count}个候选在Holm校正后显著改善复利增长。下表是年化对数增长差，不是算术收益差；净值曾清零的路径无法取有限对数，单列而不作显著发现。</p>'
      +_percent_table(log_primary,['annual_mean','lo','hi'])+'<p><a href="paired_log_growth.csv">全部复利增长检验</a></p>')
    perf_cols=['factor','n','m','period','annual_mean','baseline_annual_mean','annual_mean_difference','lo','hi','p_holm','cagr','sharpe','max_drawdown','active','baseline_active']
    perf_cols.extend(c for c in ['bankrupt','baseline_bankrupt'] if c in winner_table)
    pct=['annual_mean','baseline_annual_mean','annual_mean_difference','lo','hi','cagr','max_drawdown','active','baseline_active']
    primary_table=_percent_table(focus,['annual_mean','cagr','max_drawdown','active','annual_cost'])
    best_html=''
    if best is not None and best.id!=primary['id']:
        best_html='<h3>查看后期结果后标记的最高收益候选</h3><p>该行来自冻结候选集合，但不是训练选定的主策略。后期最大值不能单独作为成功发现。</p>'+_paired_table(tests,[best.id])
    document=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>Factor × 公告前买入 × 公告后卖出</title>
    <style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1400px;margin:36px auto;padding:0 24px;color:#173044}}h1{{font-size:30px}}h2{{margin-top:2em}}table{{font-size:12px;border-collapse:collapse;display:block;overflow-x:auto}}td,th{{padding:6px 9px;border-bottom:1px solid #d9e1e8;text-align:right;white-space:nowrap}}td:first-child,th:first-child{{text-align:left}}a{{color:#17649b}}img{{max-width:100%;height:auto}}.finding{{background:#eef5fa;padding:18px;border-left:4px solid #246b98}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:15px;font-size:12px}}.small{{font-size:13px;color:#435b70}}</style></head><body>
    <h1>Factor、公告前 N 天买入、公告后 M 天卖出</h1><div class="finding">{html.escape(conclusion)}</div>
    <p>所有组合在完整交易日序列上计算收益，空仓日计零。主口径每边至少1只股票；每天按费用后NAV持有100%多头和100%空头，目标美元净敞口为零、总敞口200%。5 bp 按每一美元买入及卖出收费，净口径另含1%年化空头借券费。表中收益与区间已显示为百分比；算术年均收益与复利CAGR分列。</p>
    <h2>三个参数和 baseline 的精确定义</h2>
    <p>Factor 为 Chen–Zimmermann 的209个可下载股票特征，按原论文 Sign 统一方向。先在每个完整月份计算全样本百分位，入场时上三分之一做多、下三分之一做空；因子分组在整个持仓期间冻结。月度快照只使用首个收益日前已经完成的月份，因此月末收盘入场可以使用刚完成的月末快照。</p>
    <p>N=1…30：先用当时已知的四次季度公告记录预测下一公告 F，再在 F−N 的收盘建立持仓，从下一交易日赚取收益。M=1…30：在下一实际公告 E 后第 M 个交易日收盘退出；若该公告尚未可知则延迟到信息可知收盘 K，最迟在 F+40 收盘强制退出。公式为 X=min(F+40,max(E+M,K))。实际未来 E/K 只决定进场后的停止规则，绝不用于筛选或安排主回测的提前买入。</p>
    <p><b>Baseline：完全相同 factor 的 N=1、M=1</b>，使用同样的预测日期、信息延迟、费用、选股资格、冻结信号及多空配平方式。实际提前或延迟公告意味着它不必在实际公告日前1天进场。缺少事前历史公告日历时，不能把事后实际日期直接用于可交易进场。</p>
    <p>只能在规定的进场收盘建立新仓。若一边没有股票，整本组合清仓，遗留持仓计划同时终止，之后只能由新的、当日到期的进场机会重新启动。两边均按等美元权重每天调整，因此“冻结”指因子方向及持仓计划，不代表固定股数。持仓后覆盖或规律性下降不自动退出；股票消失在上次可用价格退出。所有终点清仓计费。</p>
    {diagnostics}<h2>主策略与同因子基准</h2><p>训练选定：<b>{html.escape(primary['factor'])}，N={primary['n']}，M={primary['m']}</b>。{screen_link}；<a href="frozen_choices.json">冻结选择记录</a>。</p>
    <img src="primary_comparison.png" alt="Primary strategy versus same factor -1/+1 baseline">
    {primary_table}<h3>主检验：完整交易日配对差，HAC60</h3>{_paired_table(tests,[primary['id']])}
    {log_html}<h3>共同持仓日敏感性</h3><p>只保留候选与基准同时持有多空仓的日子。这里的252倍均值属于条件活跃日统计，不是另一条完整可交易策略的年化收益；主结论仍使用上面的完整日序列。</p>{common}
    {best_html}<h2>显著性计数</h2><p>Holm 校正覆盖全部 {len(frozen['candidates'])} 个冻结候选，包含等于自身baseline的候选；分别在期间、比较对象、HAC滞后和统计口径内校正。只有正方向且校正p&lt;0.05才计为显著改善。负差异即便显著也不属于成功。第一列检验改善；后两列还要求正算术收益、正CAGR及全程存活；最后一列同时要求自身净收益和改善均显著。bankrupt为全样本路径是否曾清零，清零后保持空仓。</p>{count.to_html(index=False,border=0)}
    <h2>每个 factor 的训练选定窗口：后期完整结果</h2><p>该表保留每个因子的冻结赢家，包括后来亏损、或窗口等于baseline的规则。<a href="per_factor_comparison.csv">完整比较CSV</a>。</p>{_percent_table(winner_table[perf_cols],pct)}
    <h2>训练窗口曲面</h2><p>以下是主因子全部30×30的训练事件收益Sharpe代理，不能当成逐日多空回测Sharpe。黑色标记为训练精确回测选出的主窗口。</p><img src="training_N_M_heatmap.png" alt="Training event proxy N by M heatmap">
    <h2>成本和持仓集中度敏感性</h2><p>每边至少1只股票会产生集中风险；下表报告每边至少5只股票的独立回测。每日平均持仓数包含空仓日。无法观察实际借券、冲击、融资及历史可成交性。基准两次2016年单日大亏已用原始CRSP和独立绝对金额账本复核；当时只有一只多头与一只空头，主要亏损来自多头分别下跌54.76%和50.93%。<a href="extreme_day_audit.json">极端日期原始数据及账本审计</a>。</p>{sensitivity}
    {surfaces}{oracle}<h2>回归、选择与证据边界</h2>
    <p>对每个 factor、N、M，用单股事件累计总收益回归于截距及“带原论文方向的入场因子百分位−0.5”，按预测公告交易日聚类计算标准误。训练1985–2004，统一清除可能读取训练期以外收益的事件标签。事件筛选另用计入往返成本和借券的尾部分组收益代理。它们忽略日历重叠和逐日配平，仅用于缩小精确回测名单。</p>
    <p>精确训练之后冻结每因子和整体候选，再评估2005–2014、2015–2025以及合并的2005–2025。若主评估从2005年开始，则训练末已清仓，2005年以新的1美元净值和空仓启动；发展期数值来自独立训练回测。2015–2025主表是同一后期路径的后半段，若另有2015年重新开始的敏感性，则在上表单列。先前相关搜索已经看过这些历史区间，因此这些是回顾性检验，不能称为从未接触的前瞻性holdout。209个信号还包含发表晚于训练结束的因子；主策略仅从2004年前已发表的因子中选出。已发表因子也不消除训练样本中的论文或数据修订事后偏差。</p>
    <p>美元中性不保证市场beta、行业或风险因子中性。日期预测误差、记录激活延迟、会计重述及历史证券匹配的版本问题仍存在。未来实际公告时间在主交易进场中没有使用。研究使用上一月度特征快照，无法检验公告当天实时更新的因子。</p>
    <h2>复现文件和检查</h2><p>{screen_link} · <a href="paired_inference.csv">全部配对检验</a> · <a href="annual_returns.csv">年度复利收益</a> · <a href="daily_returns.csv.gz">全部逐日收益CSV（gzip）</a> · <a href="final_models.json">模型参数</a> · <a href="independent_validation.json">独立持仓账本及事件累计收益校验</a> · <a href="inference_validation.json">HAC实现与statsmodels比较</a>。</p>
    <details><summary>精确回测审计</summary><pre>{html.escape(json.dumps(audit,indent=2))}</pre></details><details><summary>独立校验</summary><pre>{html.escape(json.dumps(independent,indent=2))}</pre></details><details><summary>完整事件回归审计</summary><pre>{html.escape(json.dumps(reg_audit,indent=2))}</pre></details>
    <p class="small">Chen, Andrew Y., and Tom Zimmermann (2022), Open Source Cross-Sectional Asset Pricing, Critical Finance Review 11(2), 207–264. <a href="https://huggingface.co/datasets/OpenSourceAP/CrossSection">官方股票特征数据与时间定义</a>，版本 Data-Release-2026.10；<a href="https://www.openassetpricing.com/">项目</a>。统计工作流程使用 Kassis, Timothy, Vinayak Agarwal, Yuhuan He, Darshil Patel, and Aubrey M. Brueckner (2026), <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>，已核验当前arXiv v2。报告生成时间 {pd.Timestamp.now(tz='UTC').isoformat()}。</p></body></html>'''
    (OUT/'report.html').write_text(document)
    print(conclusion,flush=True)
    return dict(conclusion=conclusion,counts=counts,primary=primary,report=str(OUT/'report.html'))


if __name__=='__main__':report()
