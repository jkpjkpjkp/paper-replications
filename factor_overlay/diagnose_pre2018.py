#!/usr/bin/env python3
"""Read-only split of existing forecasts at 2018; no models are refitted."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from diagnose_forecasts import OUT,HERE,mean_hac,joint_calibration


def main():
    f=pd.read_csv(OUT/'forecast_daily_diagnostics.csv')
    f=f[f.year.between(2005,2025)].copy()
    f['period']=np.where(f.year<=2017,'2005-2017','2018-2025')
    cal=pd.read_csv(HERE/'data/calendar.csv',parse_dates=['date'])[['idx','date']]
    f=f.merge(cal,on='idx',validate='many_to_one')
    assert f.date.dt.year.eq(f.year).all()
    method_rows=[]
    for (factor,method,horizon,period),g in f.groupby(['factor','method','horizon','period']):
        g=g.sort_values('idx')
        row=dict(factor=factor,method=method,horizon=horizon,period=period,dates=len(g),
                 mean_incremental_ic=float(g.increment_ic.mean()),
                 mean_baseline_ic=float(g.base_ic.mean()),mean_cycle_ic=float(g.cycle_ic.mean()),
                 relative_mse_improvement=float(1-g.cycle_mse.mean()/g.base_mse.mean()),
                 variance_score_improvement=float(-g.variance_only_score_change.mean()),
                 **joint_calibration(g))
        method_rows.append(row)
    methods=pd.DataFrame(method_rows)
    for _,ix in methods.groupby(['period','horizon']).groups.items():
        methods.loc[ix,'calibration_increment_p_holm36']=multipletests(methods.loc[ix,'calibration_increment_p'],method='holm')[1]
    methods.to_csv(OUT/'pre2018_forecast_by_model.csv',index=False)

    keys=['idx','date','year','horizon','factor','period']
    numeric=[x for x in f.select_dtypes('number').columns if x not in keys and x!='model_year']
    daily=f.groupby(keys,as_index=False)[numeric].mean()
    daily['method_count']=f.groupby(keys).method.nunique().to_numpy()
    assert daily.method_count.eq(6).all()
    daily=daily.rename(columns={'increment_ic':'mean_method_incremental_ic',
                                'base_ic':'mean_method_baseline_ic','cycle_ic':'mean_method_cycle_ic'})
    daily['variance_score_improvement']=-daily.variance_only_score_change
    daily['mse_improvement']=daily.base_mse-daily.cycle_mse
    daily.to_csv(OUT/'pre2018_forecast_daily.csv',index=False,date_format='%Y-%m-%d')

    summaries=[]
    for (factor,horizon,period),g in daily.groupby(['factor','horizon','period']):
        g=g.sort_values('idx')
        mm=methods[(methods.factor==factor)&(methods.horizon==horizon)&(methods.period==period)]
        row=dict(factor=factor,horizon=horizon,period=period,dates=len(g),
                 first_date=str(g.date.iloc[0].date()),last_date=str(g.date.iloc[-1].date()),methods=6,
                 mean_method_relative_mse_improvement=float(mm.relative_mse_improvement.mean()),
                 mean_method_incremental_calibration=float(mm.calibration_increment.mean()),
                 positive_calibration_methods=int((mm.calibration_increment>0).sum()),
                 significant_positive_calibration_methods_unadjusted=int(((mm.calibration_increment>0)&(mm.calibration_increment_p<.05)).sum()),
                 significant_positive_calibration_methods_holm36=int(((mm.calibration_increment>0)&(mm.calibration_increment_p_holm36<.05)).sum()),
                 improving_mse_methods=int((mm.relative_mse_improvement>0).sum()),
                 positive_incremental_ic_methods=int((mm.mean_incremental_ic>0).sum()))
        for col in ['mean_method_baseline_ic','mean_method_cycle_ic','mean_method_incremental_ic','variance_score_improvement','mse_improvement']:
            m,se,p,lo,hi=mean_hac(g[col])
            row.update({col:m,col+'_se':se,col+'_p':p,col+'_ci_low':lo,col+'_ci_high':hi})
        # A pooled calibration is supplied separately from the requested mean
        # of the six individual model slopes. The estimates need not coincide.
        row.update({'pooled_'+k:v for k,v in joint_calibration(g).items()})
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    for _,ix in summary.groupby(['period','horizon']).groups.items():
        for p in ['mean_method_incremental_ic_p','variance_score_improvement_p']:
            summary.loc[ix,p+'_holm6']=multipletests(summary.loc[ix,p],method='holm')[1]
    summary.to_csv(OUT/'pre2018_forecast_summary.csv',index=False)

    changes=[]
    for (factor,horizon),g in daily.groupby(['factor','horizon']):
        g=g.sort_values('idx');post=(g.year>=2018).to_numpy(float)
        x=np.column_stack([np.ones(len(g)),post])
        row=dict(factor=factor,horizon=horizon)
        for col in ['mean_method_incremental_ic','variance_score_improvement','mse_improvement']:
            fit=sm.OLS(g[col].to_numpy(),x).fit(cov_type='HAC',cov_kwds={'maxlags':12})
            row.update({col+'_post_minus_pre':float(fit.params[1]),col+'_difference_p':float(fit.pvalues[1])})
        changes.append(row)
    pd.DataFrame(changes).to_csv(OUT/'pre2018_forecast_changes.csv',index=False)

    agg=[]
    for (period,horizon),q in methods.groupby(['period','horizon']):
        agg.append(dict(period=period,horizon=horizon,comparisons=len(q),
                        mean_incremental_ic=float(q.mean_incremental_ic.mean()),
                        positive_incremental_ic=int((q.mean_incremental_ic>0).sum()),
                        mean_incremental_calibration=float(q.calibration_increment.mean()),
                        median_incremental_calibration=float(q.calibration_increment.median()),
                        positive_calibration=int((q.calibration_increment>0).sum()),
                        significant_positive_calibration_unadjusted=int(((q.calibration_increment>0)&(q.calibration_increment_p<.05)).sum()),
                        significant_positive_calibration_holm36=int(((q.calibration_increment>0)&(q.calibration_increment_p_holm36<.05)).sum()),
                        mean_relative_mse_improvement=float(q.relative_mse_improvement.mean()),
                        improving_mse=int((q.relative_mse_improvement>0).sum()),
                        mean_variance_score_improvement=float(q.variance_score_improvement.mean())))
    aggregate=pd.DataFrame(agg);aggregate.to_csv(OUT/'pre2018_forecast_aggregate.csv',index=False)
    audit=dict(status='post-result requested date split of saved forecasts; no refitting or selection',
                periods=['2005-2017 inclusive','2018-2025 inclusive'],
                models='primary annual one-day-trained models for both evaluation horizons',
                horizon1='signed next-day stock payoff',horizon5='signed exact five-session fixed-share gross payoff divided by5; these are primary forecasts, not horizon-matched-model forecasts',
                factor_summary='equal arithmetic mean across six regression methods; correlations are averaged, not the correlation of an ensemble forecast',
                dates='date is actual first return/forecast-evaluation trading date; idx is original market session index; sampled every fifth market session. Five-day labels run date session through idx+4 and exclude source-year boundaries.',
                sign_conventions='positive relative_mse_improvement means lower forecast MSE with the cycle; positive variance_score_improvement means better variance scoring with baseline mean held fixed; positive incremental IC means better ranking by added forecast',
                calibration='mean_method_incremental_calibration averages six joint-projection slopes; pooled_calibration_increment uses the pooled moments instead. Slope0=no added linear direction;1=correct scale. These differ from portfolio returns.',
                inference='HAC12 on equally method-averaged weekly dates. Holm6 for displayed IC/risk tests separately by period/horizon; Holm36 for individual-method calibration separately by period/horizon. Change tests unadjusted diagnostics.',
                limitation='The split is chosen after previous results. Counts across correlated methods are descriptive. RDIPO sparse-short caveat remains. No claim of causal change at2018.')
    (OUT/'pre2018_forecast_audit.json').write_text(json.dumps(audit,indent=2))
    notes='''The earlier period is descriptively brighter on some directional measures, especially the five-session outcome, but the evidence does not establish a profitable predictive regime before 2018.

Across the six factors and six methods, average incremental one-day rank IC changes from +0.00075 in 2005–2017 to -0.00465 in 2018–2025. Cash, CompEquIss and InvestPPEInv have positive factor-average one-day incremental IC before 2018; all six factor averages are negative afterward. No individual factor's IC is significantly positive after Holm6 in either period, and no factor's pre/post IC change is significant at 5% even without adjustment.

For five-session average outcomes, average incremental rank IC is +0.00337 before 2018 versus +0.00094 later. More notably, mean incremental calibration becomes modestly positive earlier: +0.054 before 2018 versus -0.062 later; 26/36 model slopes are positive earlier versus 10/36 later. Calibration 1 would mean correctly scaled additional forecasts, while 0 means no extra linear direction after conditioning on baseline forecasts. The earlier positive slope is small, and no positive individual slope survives Holm36. Cash's factor-average five-day slope is +0.164 then -0.067; RIO_Disp's is +0.074 then -0.023. MS remains negative in both periods.

One-day calibration is not broadly positive earlier: averaging across all 36 models gives -0.071 before 2018 and -0.050 afterward. Therefore the earlier-period advantage depends on which forecast property and horizon are inspected.

Prediction error does not improve in the earlier period. All 36 cycle models worsen MSE before 2018 at both horizons. Mean relative MSE deterioration is 0.304% for one-day and 1.078% for five-day outcomes before 2018, versus 0.060% and 0.473% later. These percentages measure forecast error, not investment returns. A positive but weak ranking relationship can coexist with worse MSE when forecasts have the wrong magnitude or unstable offsets.

The clock's variance information remains clearer than its return direction: keeping the baseline mean forecast fixed, all six factor-average variance forecasts improve risk scoring in both periods at both horizons, surviving Holm6. This result supports risk forecasting in each subperiod, not an automatically profitable timing trade.

All forecasts are the saved primary one-day-trained models; five-day diagnostics evaluate those same forecasts against exact five-session fixed-share payoffs divided by five. No models were refitted for this split. The split at 2018 was requested after prior results, so the comparison is exploratory and does not identify a causal market change in 2018.

For plots, pre2018_forecast_daily.csv contains actual date (the first return/forecast-evaluation date), idx (market-session index), year, factor, horizon and period. Each row averages diagnostics equally across the six methods; observations occur every fifth trading session. Positive mean_method_incremental_ic means better ranking by the extra forecast. Positive mean_method_relative_mse_improvement in the period summary means lower MSE. Positive variance_score_improvement means better risk scoring. Correlations are averages of model ICs, not the IC of an averaged forecast.
'''
    (OUT/'pre2018_forecast_notes.txt').write_text(notes)
    print(summary[['factor','horizon','period','mean_method_incremental_ic','mean_method_relative_mse_improvement','mean_method_incremental_calibration','variance_score_improvement']].to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__=='__main__':main()
