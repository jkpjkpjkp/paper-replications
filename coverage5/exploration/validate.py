#!/usr/bin/env python3
"""Chronology, raw-return ledgers, neutrality and selection verification."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import explore as e

def load(name):
    with np.load(e.OUT/f"{name}.npz") as z:return {k:z[k] for k in z.files}

def independent_target(f,m):
    """Direct mathematical specification, without the engine's target function."""
    if m["type"]=="window":
        values=f[m["clock"]]
        lm=values.between(m["long_lo"],m["long_hi"]).to_numpy()
        ss=values.between(m["short_lo"],m["short_hi"]).to_numpy()
        w=np.zeros(len(f))
        if min(lm.sum(),ss.sum())>=20:w[lm]=1/lm.sum();w[ss]=-1/ss.sum()
        return w
    period=f.period_sessions.to_numpy(dtype=float)
    if m["phase"]=="seasonal":
        period=np.where(f.seasonal_period.between(50,80),f.seasonal_period.to_numpy(),period)
    theta=2*np.pi*f.age.to_numpy()/period
    if m["type"]=="fixed":score=np.cos(theta-2*np.pi*m["offset_sessions"]/65)
    else:
        score=np.zeros(len(f))
        for j in range(m["harmonics"]):
            score+=m["coefficients"][2*j]*np.sin((j+1)*theta)+m["coefficients"][2*j+1]*np.cos((j+1)*theta)
    w=np.zeros(len(f))
    if m["weighting"]=="score":
        score-=score.mean();lm=score>1e-16;ss=score< -1e-16
        if min(lm.sum(),ss.sum())>=20:w[lm]=score[lm]/score[lm].sum();w[ss]=-score[ss]/score[ss].sum()
    else:
        low,high=np.quantile(score,[1/3,2/3]);lm=score>=high;ss=score<=low
        if high-low>1e-15 and min(lm.sum(),ss.sum())>=20:w[lm]=1/lm.sum();w[ss]=-1/ss.sum()
    return w

def main():
    final=load("final");train=load("stage2_train")
    models=json.loads((e.OUT/"final_models.json").read_text());trainmodels=json.loads((e.OUT/"stage2_models.json").read_text())
    choices=json.loads((e.OUT/"frozen_choices.json").read_text())
    assert choices["selection_end"]=="2004-12-31" and not choices["later_returns_used"]
    checks={"cash_and_neutrality_closed_form":True}
    dummy=[dict(id="test",adjustment=1.)]
    old=np.zeros((len(e.COSTS),1,40));w=np.r_[np.full(20,.05),np.full(20,-.05)][None,:]
    r=np.r_[np.full(20,.01),np.full(20,.004)]
    ans=e.execution(old,w,np.ones(40,bool),r,r,dummy)
    for ci,(_,cost,borrow) in enumerate(e.COSTS):
        invest=1/(1+2*cost)
        expected=invest*.006-2*cost*invest-borrow/252*invest
        assert abs(ans[0][ci,0]-expected)<1e-12
    dummy[0]["adjustment"]=.1
    slow=e.execution(ans[1],w,np.ones(40,bool),-r,-r,dummy)
    assert np.max(np.abs(slow[5]-slow[6]))<1e-12
    cash=e.execution(slow[1],np.zeros_like(w),np.ones(40,bool),r,r,dummy)
    assert np.max(np.abs(cash[1]))==0
    assert np.allclose(cash[0],-cash[3])
    checks["slow_adjustment_daily_neutrality_and_cash_exit"]=True
    trainmask=final["dates"]<=np.datetime64("2004-12-31")
    assert np.array_equal(final["dates"][trainmask],train["dates"])
    for j,m in enumerate(models):
        ti=next(k for k,x in enumerate(trainmodels) if x["id"]==m["id"])
        for field in ["returns","turnover","cost","borrow","n_long","n_short"]:
            assert np.allclose(final[field][trainmask,:,j],train[field][:,:,ti],atol=1e-11)
    checks["full_evaluation_training_prefix_identical"]=True
    stats=pd.read_csv(e.OUT/"stage2_train_statistics.csv")
    for scenario,key in [("10bps","family_winners_10bps"),("1bp","family_winners_1bp")]:
        z=stats.loc[stats.period.eq("development")&stats.scenario.eq(scenario)&stats.active_fraction.ge(.30)]
        expected=z.sort_values(["sharpe","model"],ascending=[False,True]).groupby("family").head(1)
        assert set(expected.model)=={x["model"] for x in choices[key]}
    checks["selection_recomputed_on_development_only"]=True
    checks["max_abs_target_net_exposure"]=float(final["max_net_error"])
    assert checks["max_abs_target_net_exposure"]<1e-10
    assert np.isfinite(final["returns"]).all() and (final["returns"]> -1).all()
    for ci,(_,cost,borrow) in enumerate(e.COSTS):
        assert np.allclose(final["cost"][:,ci],cost*final["turnover"][:,ci],atol=1e-12)
        expected=borrow/252*(1-final["cost"][:,ci])*final["active"][:,ci]
        assert np.allclose(expected,final["borrow"][:,ci],atol=1e-12)
    checks["fees_and_actual_short_borrow"]=True
    c=e.connection();cal=e.calendar();ledgers=[]
    raw=e.STUDY.parent.parent/"crsp-daily/data/crsp_a_stock/dsf_v2"
    # Independently recompute primary and cheap-cost champion targets and ledgers
    # on multiple dates, with new reads of raw total and ex-dividend returns.
    selected={choices["overall_10bps"],choices["overall_1bp"]}
    for day in ["2008-10-10","2019-06-14","2025-12-31"]:
        dt=pd.Timestamp(day);previous=cal.loc[cal.date<dt,"date"].max()
        frames=[]
        for date in [previous,dt]:
            f=c.execute(f"""SELECT p.permno,p.idx,p.age,p.period_sessions,
              p.clock AS nearest_median,p.seasonal_clock AS nearest_seasonal,p.age AS since_known,
              p.seasonal_forecast_idx-p.event_idx AS seasonal_period,
              CAST(r.dlyret AS DOUBLE) AS ret,CAST(r.dlyretx AS DOUBLE) AS retx
              FROM read_parquet('{e.PANEL}/{date.year}.parquet') p
              JOIN read_parquet('{raw}/{date.year}.parquet') r
              ON p.permno=r.permno AND p.date=r.dlycaldt
              WHERE p.eligible AND p.date=DATE '{date.date()}' ORDER BY p.permno""").df()
            frames.append(f)
        for modelid in selected:
            j=next(k for k,x in enumerate(models) if x["id"]==modelid);m=models[j]
            if m["adjustment"]!=1:continue
            targets=[pd.Series(independent_target(f,m),index=f.permno) for f in frames]
            ix=int(np.where(final["dates"]==np.datetime64(dt.date()))[0][0])
            for ci in [0,1,3,4]:
                cost=e.COSTS[ci][1];borrow=e.COSTS[ci][2]
                union=targets[0].index.union(targets[1].index)
                prior_rx=frames[0].set_index("permno").retx.fillna(frames[0].set_index("permno").ret).fillna(0).reindex(union).fillna(0)
                old=targets[0].reindex(union).fillna(0)*(1-final["cost"][ix-1,ci,j])*(1+prior_rx)/(1+final["returns"][ix-1,ci,j])
                target=targets[1].reindex(union).fillna(0)
                lo=0.;hi=1.
                for _ in range(60):
                    x=(lo+hi)/2
                    if x+cost*np.abs(x*target-old).sum()>1:hi=x
                    else:lo=x
                x=(lo+hi)/2;turn=float(np.abs(x*target-old).sum());fee=cost*turn
                returns=frames[1].set_index("permno").ret.reindex(union).fillna(0)
                borrowing=borrow/252*float(-np.minimum(x*target,0).sum())
                pnl=float((x*target*returns).sum())-fee-borrowing
                assert abs(pnl-final["returns"][ix,ci,j])<1e-11,(day,modelid,ci)
                assert abs(turn-final["turnover"][ix,ci,j])<1e-10
                ledgers.append(dict(date=day,model=modelid,scenario=e.COSTS[ci][0],returns=pnl))
        # Metamorphic check: radically changed current returns cannot change targets.
        f=frames[1];before=e.target_weights(f,models);changed=f.copy()
        changed["ret"]=np.linspace(-.9,10,len(f));changed["retx"]=10
        assert np.allclose(before,e.target_weights(changed,models))
    checks["independent_raw_return_bisection_ledgers"]=ledgers
    checks["current_returns_do_not_affect_signal_targets"]=True
    perf=pd.read_csv(e.OUT/"performance.csv")
    for _,z in perf.loc[perf.primary_family_winner].groupby(["period","scenario"]):
        assert np.allclose(multipletests(z.p,method="holm")[1],z.p_holm_family)
    j=next(i for i,m in enumerate(models) if m["id"]==choices["overall_10bps"])
    mask=final["dates"]>=np.datetime64("2015-01-01")
    r=final["returns"][mask,3,j]
    f=sm.OLS(r,np.ones((len(r),1))).fit(cov_type="HAC",cov_kwds={"maxlags":20})
    row=perf.loc[perf.model.eq(choices["overall_10bps"])&perf.period.eq("holdout")&perf.scenario.eq("10bps")].iloc[0]
    assert abs(row.t-f.tvalues[0])<1e-10 and abs(row.annual_mean-r.mean()*252)<1e-12
    checks["independent_holdout_HAC_and_family_multiplicity"]=True
    checks["missing_held_returns_across_final_models_all_scenarios"]=int(final["missing_returns"].sum())
    e.save(e.OUT/"validation_checks.json",checks)
    print(json.dumps(checks,indent=2))

if __name__=="__main__":main()
