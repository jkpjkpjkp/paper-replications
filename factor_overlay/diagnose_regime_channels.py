#!/usr/bin/env python3
"""Aggregate existing channel/weight diagnostics around the user-specified2018 split.

No new fit, portfolio, or strategy selection. The ridge crossed mean/risk
experiment uses original one-day forecasts; the strict local fee gate uses
direct-five-day forecasts. Keep these provenance distinctions explicit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent;OUT=HERE/'diagnostics'
PERIODS={'2005-2017':(2005,2017),'2018-2025':(2018,2025)}


def main():
    OUT.mkdir(exist_ok=True)
    folder=HERE/'results/risk_mean_ablation'
    z=np.load(folder/'daily.npz');spec=json.loads((folder/'models.json').read_text())
    years=pd.DatetimeIndex(z['dates']).year
    channels=[]
    for period,(lo,hi) in PERIODS.items():
        mask=(years>=lo)&(years<=hi)
        for j in range(0,len(spec),4):
            base=z['returns'][mask,j]
            for k in range(1,4):
                model=j+k;r=z['returns'][mask,model];delta=r-base
                net=252*delta.mean()
                gross=252*(z['gross_return'][mask,model]-z['gross_return'][mask,j]).mean()
                fees=252*(z['fees'][mask,model]-z['fees'][mask,j]).mean()
                borrow=252*(z['borrow'][mask,model]-z['borrow'][mask,j]).mean()
                assert abs(net-(gross-fees-borrow))<1e-12
                channels.append(dict(period=period,factor=spec[model]['factor'],variant=spec[model]['variant'],
                    forecast_target='original_one_day_ridge',sessions=int(mask.sum()),
                    net_increment_bp_year=net*10000,gross_increment_bp_year=gross*10000,
                    extra_fees_bp_year=fees*10000,extra_borrow_bp_year=borrow*10000,
                    baseline_annual_mean=252*base.mean(),augmented_annual_mean=252*r.mean(),
                    baseline_annual_vol=np.sqrt(252)*base.std(ddof=1),augmented_annual_vol=np.sqrt(252)*r.std(ddof=1),
                    incremental_tracking_vol=np.sqrt(252)*delta.std(ddof=1),
                    increment_cer3_bp_year=(net-1.5*252*(r.var(ddof=1)-base.var(ddof=1)))*10000))
    channels=pd.DataFrame(channels);channels.to_csv(OUT/'pre2018_channels.csv',index=False)
    w=pd.read_csv(folder/'rebalance_diagnostics.csv')
    weightrows=[]
    for period,(lo,hi) in PERIODS.items():
        q=w[w.year.between(lo,hi)]
        columns=['n','abs_multiplier_tilt','lower_fraction','upper_fraction','optimizer_no_trade_fraction',
            'allocation_l1_to_base','forecast_rms_mean_change','forecast_mean_variance','forecast_rms_logvariance_change']
        # Equal weight per active rebalance/leg. Each source cell is already a
        # within-leg equal-stock average; do not size-weight the portfolio again.
        for (factor,variant),g in q.groupby(['factor','variant']):
            weightrows.append(dict(period=period,factor=factor,variant=variant,
                forecast_target='original_one_day_ridge',active_rebalance_leg_cells=len(g),
                **{col:float(g[col].mean()) for col in columns}))
    weights=pd.DataFrame(weightrows);weights.to_csv(OUT/'pre2018_weight_diagnostics.csv',index=False)
    guard=pd.read_csv(HERE/'results_cost_guard/decisions.csv')
    gz=np.load(HERE/'results_cost_guard/daily.npz');gy=pd.DatetimeIndex(gz['dates']).year
    accept=[]
    for period,(lo,hi) in PERIODS.items():
        n=int(((gy>=lo)&(gy<=hi)).sum())
        q=guard[guard.year.between(lo,hi)]
        for (factor,method),g in q.groupby(['factor','method']):
            active=g[g.active]
            assert (g.selected_fee<=g.fallback_fee+1e-12).all()
            assert (g.loc[g.accepted,'enhanced_fee']<g.loc[g.accepted,'fallback_fee']-1e-12).all()
            accept.append(dict(period=period,factor=factor,method=method,forecast_target='direct_five_day',
                active_rebalances=len(active),accepted_rebalances=int(active.accepted.sum()),
                acceptance_fraction=float(active.accepted.mean()),
                annual_local_fee_saving_bp=float(g.local_fee_saving.sum()*252/n*10000),
                mean_candidate_target_l1_difference=float(active.target_l1_difference.mean()),
                strict_local_veto_violations=0))
    pd.DataFrame(accept).to_csv(OUT/'pre2018_guard_acceptance.csv',index=False)
    both=channels[channels.variant.eq('cycle_both')]
    text='USER-SPECIFIED REGIME SPLIT: 2005–2017 / 2018–2025\n\n'
    text+='No new model fits or simulations. This is an exploratory split after prior results were inspected.\n'
    text+='pre2018_channels.csv is the ORIGINAL ONE-DAY ridge mean/risk ablation, with unchanged five-session execution. It is not the direct-five-day experiment.\n'
    text+='Positive gross increment means the extra positions earned more before fees/borrow; positive extra_fees means added cost. Net = gross - extra_fees - extra_borrow.\n'
    text+='pre2018_weight_diagnostics.csv averages each active rebalance/leg equally. The allocator no-trade fraction is at target-weight level, before common post-fee NAV scaling.\n'
    text+='pre2018_guard_acceptance.csv uses DIRECT-FIVE-DAY fits. Its local savings compare candidate targets starting from identical guard holdings; they do not equal cumulative savings versus a separately evolved baseline.\n'
    text+='Variance-only input attribution is not causal: each variance model was trained using its own corresponding mean residuals; nonlinear allocation creates an interaction.\n\n'
    text+=both[['period','factor','net_increment_bp_year','gross_increment_bp_year','extra_fees_bp_year']].to_string(index=False)+'\n'
    (OUT/'pre2018_channel_notes.txt').write_text(text)
    print(text)


if __name__=='__main__':main()
