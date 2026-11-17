#!/usr/bin/env python3
"""Exploratory annual curve refits using strictly preceding calendar years."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import explore as e

DESIGN={"created_utc":pd.Timestamp.now(tz="UTC").isoformat(),
 "scope":"Added after inspecting frozen-rule later-period results. All eight variants are specified here and reported; no winner is selected on evaluation returns. Retrospective exploratory evidence, not an untouched holdout.",
 "features":"Same date-only first Fourier harmonic of known age / median or seasonal expected release period, same regular-65 five-analyst universe and availability.",
 "fit":"At the start of each year 2005..2025, calendar-day-demeaned cross-sectional ridge fit using only years strictly before that year. Both expanding history from 1985 and trailing 10 full calendar years. Ridge=0.01*mean diagonal X'X. No same-year data enters its fitted curve.",
 "targets":"Centered predicted scores, positive-score weights normalized to $1 long and negative-score weights to $1 short, >=20 each; cash otherwise. Report 100% or 10% daily adjustment toward this target, with daily equal-dollar neutrality. Eight combinations, no parameter tuning on evaluation returns.",
 "costs":e.COSTS,"inference":"HAC20 mean inference, Holm across eight adaptive variants per period/cost. Also provide Holm across these eight plus the original six frozen family winners. All other execution assumptions unchanged."}

def main():
    p=e.OUT/"adaptive_design.json"
    if p.exists():
        a=json.loads(p.read_text());a.pop("created_utc");b=DESIGN.copy();b.pop("created_utc")
        assert a==json.loads(json.dumps(b))
    else:e.save(p,DESIGN)
    c=e.connection();moments={}
    for year in range(1985,2025):
        e.log(f"Annual date-only cross-sectional fitting moments: {year}")
        d=e.frame(c,year)
        by={kind:dict(xx=np.zeros((2,2)),xy=np.zeros(2),days=0) for kind in ["median","seasonal"]}
        for _,f in d.groupby("idx",sort=False):
            if len(f)<100:continue
            y=f.ret.fillna(0).to_numpy();y=y-y.mean()
            for kind,z in by.items():
                x=e.basis(f,kind,1);x-=x.mean(axis=0)
                z["xx"]+=x.T@x/len(f);z["xy"]+=x.T@y/len(f);z["days"]+=1
        moments[year]=by
    models=[];chronology=[]
    for kind in ["median","seasonal"]:
        for window in ["expanding","rolling10"]:
            coefficients={}
            for year in range(2005,2026):
                start=1985 if window=="expanding" else year-10
                prior=[moments[y][kind] for y in range(start,year)]
                xx=sum((z["xx"] for z in prior),np.zeros((2,2)))
                xy=sum((z["xy"] for z in prior),np.zeros(2))
                penalty=.01*np.trace(xx)/2
                coefficients[str(year)]=np.linalg.solve(xx+penalty*np.eye(2),xy).tolist()
                chronology.append(dict(phase=kind,fit_window=window,evaluation_year=year,
                    fit_start_year=start,fit_last_year=year-1,fit_days=sum(z["days"] for z in prior)))
            for rho in [1.,.1]:
                name=f"adaptive_{kind}_{window}_adjust{rho}"
                models.append(dict(id=name,family=f"adaptive_{kind}_{window}",type="learned",phase=kind,
                    harmonics=1,coefficients=coefficients["2005"],annual_coefficients=coefficients,
                    weighting="score",adjustment=rho,fit_window=window))
    e.save(e.OUT/"adaptive_fit_chronology.json",chronology)
    e.simulate(models,2005,2025,"adaptive")
    stats,_=e.simple_stats("adaptive")
    with np.load(e.OUT/"adaptive.npz") as z:a={k:z[k] for k in z.files}
    inference=[]
    for period,(start,end) in e.PERIODS.items():
        mask=(a["dates"]>=np.datetime64(start))&(a["dates"]<=np.datetime64(end))
        if not mask.any():continue
        for j,m in enumerate(models):
            for ci,(scenario,_,_) in enumerate(e.COSTS):
                r=a["returns"][mask,ci,j];f=sm.OLS(r,np.ones((len(r),1))).fit(cov_type="HAC",cov_kwds={"maxlags":20})
                lo,hi=f.conf_int()[0]*252
                inference.append(dict(model=m["id"],period=period,scenario=scenario,t=float(f.tvalues[0]),p=float(f.pvalues[0]),ci_low=float(lo),ci_high=float(hi)))
    stats=stats.merge(pd.DataFrame(inference),on=["model","period","scenario"],validate="one_to_one")
    for _,z in stats.groupby(["period","scenario"]):stats.loc[z.index,"p_holm_adaptive8"]=multipletests(z.p,method="holm")[1]
    original=pd.read_csv(e.OUT/"performance.csv");original=original.loc[original.primary_family_winner]
    for (period,scenario),z in stats.groupby(["period","scenario"]):
        other=original.loc[original.period.eq(period)&original.scenario.eq(scenario)]
        pvalues=np.r_[z.p.to_numpy(),other.p.to_numpy()]
        stats.loc[z.index,"p_holm_combined14"]=multipletests(pvalues,method="holm")[1][:len(z)]
    stats.to_csv(e.OUT/"adaptive_performance.csv",index=False)
    d=[]
    for j,m in enumerate(models):
        for ci,(scenario,_,_) in enumerate(e.COSTS):
            for k,dt in enumerate(pd.to_datetime(a["dates"])):
                d.append(dict(date=dt,model=m["id"],scenario=scenario,returns=a["returns"][k,ci,j],
                    turnover=a["turnover"][k,ci,j],cost=a["cost"][k,ci,j],borrow=a["borrow"][k,ci,j]))
    pd.DataFrame(d).to_csv(e.OUT/"adaptive_daily_returns.csv",index=False)
    assert all(x["fit_last_year"]<x["evaluation_year"] for x in chronology)
    assert float(a["max_net_error"])<1e-10
    for ci,(_,cost,borrow) in enumerate(e.COSTS):
        assert np.allclose(a["cost"][:,ci],cost*a["turnover"][:,ci],atol=1e-12)
        expected=borrow/252*(1-a["cost"][:,ci])*a["active"][:,ci]
        assert np.allclose(a["borrow"][:,ci],expected,atol=1e-12)
    e.save(e.OUT/"adaptive_validation.json",dict(all_training_years_strictly_precede_returns=True,
        fitted_year_models=len(chronology),max_abs_target_net_exposure=float(a["max_net_error"]),
        all_eight_variants_reported=True,later_returns_used_to_select_variant=False,
        cost_and_actual_short_borrow_identities=True))
    print(stats.loc[stats.period.eq("holdout")&stats.scenario.isin(["1bp","10bps"]),
       ["model","scenario","annual_mean","cagr","sharpe","t","p_holm_combined14"]].round(4).to_string(index=False))

if __name__=="__main__":main()
