#!/usr/bin/env python3
"""Date-only dollar-neutral strategy exploration. Raw inputs are read-only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
OUT=HERE/"results"
PANEL=STUDY/"results_regular65/panel"
os.environ.setdefault("MPLCONFIGDIR",str(STUDY.parent/".mpl-cache"))
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COSTS=[("gross",0.,0.),("1bp",.0001,.01),("5bps",.0005,.01),("10bps",.001,.01),("25bps",.0025,.02)]
PERIODS={"development":("1985-01-01","2004-12-31"),"validation":("2005-01-01","2014-12-31"),
 "holdout":("2015-01-01","2025-12-31"),"recent":("2020-01-01","2025-12-31"),"all":("1985-01-01","2025-12-31")}
DESIGN={"created_utc":pd.Timestamp.now(tz="UTC").isoformat(),
 "request":"Explore dollar-neutral long/short strategies using only time from known or forecast quarterly earnings releases.",
 "universe":"Unchanged >=5 active annual-EPS-analyst proxy, known-release gaps each 50..80 market sessions with median 60..70, four known releases, age<=80; same conservative availability and lagged US common-stock eligibility as prior validated study. No extra price/size/liquidity alpha filters.",
 "clocks":["nearest_median","nearest_seasonal","since_known"],
 "windows":"All 5/10/15/20/30-session windows on a five-session grid: nearest clocks -35..34, known age 0..64, plus explicit earlier-study/post-event windows. Both directions of every disjoint same-clock long/short pair are screened. >=20 stocks per leg; otherwise both legs cash.",
 "screen":"Development 1985-2004 only. Cache individual leg returns and turnover with ex-dividend drift but without pair-specific NAV normalization. Approximate pair costs include joint cash entry/exit and 1% short borrow. Per clock retain top three net-Sharpe candidates at 1,5,10 bps, plus highest gross Sharpe and highest net-mean at 10 bps; union/deduplicate. Require >=30% joint active days. This is a screening approximation, not a claim to an exhaustive exact optimum.",
 "curves":"Fit calendar-day-demeaned cross-sectional return on first 1/2/3 Fourier harmonics of known age / median or seasonal expected period, using 1985-2004 only. Each day gets equal total weight; ridge=0.01*mean diagonal(X'X). Also fixed cosine peaks at phase offsets 0,10,..60 sessions of a 65-session cycle. Score-weighted centered positions and equal-weight top/bottom terciles, ties included, all date features only. No current returns or forecast EPS values enter targets.",
 "execution":"Daily equal long/short dollars, each 100% of after-cost NAV, hence 200% gross and zero net target at each rebalance. Individual short and long legs each require >=20 stocks. Cost-specific holdings/NAV, ex-dividend share drift, distributions in cash, exact after-cost sizing. All dollars bought/sold charged. Missing returns zero and counted; names leaving universe close at prior mark. No financing/cash interest/collateral rebate/terminal liquidation or measured stock-specific borrow/impact.",
 "slow_adjustment":"For the development-best model in each family, also test 25% and 10% daily adjustment toward the date-implied portfolio. First re-normalize eligible carried positions to equal long/short dollars, blend with today's base target, then re-normalize each leg. Both legs cash if current base signal cannot form >=20 stocks per side. This carries some names outside the current signal window; it is a different, slower holding rule. Price drift is accounting, not an alpha predictor.",
 "selection":"Exact development net Sharpe at 10 bps +1% annual borrow, active fraction >=30%. Freeze initial per-family winners before later evaluation, then compare their fixed 25%/10% variants in development only. Report one final winner per family and the overall winner. If the best development return is negative, cash dominates that choice. Selection on 1 bp also reported as an independent cost objective.",
 "costs":COSTS,"periods":PERIODS,
 "inference":"Calendar-time HAC20, HAC60, monthly HAC3; Holm across frozen per-family winners within period/scenario. Include raw market-controlled intercept, not official multifactor alpha. Full search results are exploratory; unadjusted maxima do not establish significance.",
 "scope":"Retrospective exploratory search after prior related validation/holdout results were inspected. No new rule is selected using 2005+ returns. The calendar split does not remove prior researcher/publication hindsight; future prospective data are needed for confirmation."}

def log(s):print(time.strftime("%H:%M:%S"),s,flush=True)
def save(p,x):p.write_text(json.dumps(x,indent=2,default=str,allow_nan=False))
def sha(p):
    with p.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def freeze():
    OUT.mkdir(exist_ok=True,parents=True)
    p=OUT/"design.json"
    if p.exists():
        a=json.loads(p.read_text());a.pop("created_utc",None)
        b=DESIGN.copy();b.pop("created_utc",None)
        assert a==json.loads(json.dumps(b)),"Existing frozen design differs."
    else:save(p,DESIGN)
def connection():
    c=duckdb.connect();c.execute("SET threads=2");c.execute("SET memory_limit='1000MB'");return c
def calendar():return pd.read_csv(STUDY.parent/"results/trading_calendar.csv",parse_dates=["date"]).query("date>='1985-01-01'").reset_index(drop=True)
def frame(c,year):
    return c.execute(f"""SELECT permno,idx,ret,retx,age,period_sessions,clock AS nearest_median,
      seasonal_clock AS nearest_seasonal,age AS since_known,
      seasonal_forecast_idx-event_idx AS seasonal_period FROM read_parquet('{PANEL}/{year}.parquet')
      WHERE eligible ORDER BY idx,permno""").df()

def windows():
    out=[]
    for clock,lo,hi in [("nearest_median",-35,35),("nearest_seasonal",-35,35),("since_known",0,65)]:
        ranges={(a,a+w-1) for w in [5,10,15,20,30] for a in range(lo,hi-w+1,5)}
        ranges.update([(2,5),(2,10),(2,20),(6,20),(21,35),(36,50),(51,65)] if clock=="since_known"
          else [(-5,-1),(-10,-1),(-20,-1),(0,5),(6,20),(21,35),(-5,5),(-5,10),(-10,5),(-10,10)])
        for a,b in sorted(ranges):out.append(dict(id=f"{clock}_{a}_{b}",clock=clock,lo=a,hi=b))
    return out

def basis(f,kind,harmonics=3):
    age=f.age.to_numpy(dtype=float);period=f.period_sessions.to_numpy(dtype=float)
    if kind=="seasonal":
        alt=f.seasonal_period.to_numpy(dtype=float)
        period=np.where((alt>=50)&(alt<=80),alt,period)
    angle=2*np.pi*age/period
    return np.column_stack([g(k*angle) for k in range(1,harmonics+1) for g in [np.sin,np.cos]])

def prepare():
    freeze();c=connection();cal=calendar();wins=windows();nw=len(wins);nd=len(cal)
    save(OUT/"windows.json",wins)
    n=np.zeros((nd,nw),np.int32);ret=np.zeros((nd,nw));rxmean=ret.copy();turn=ret.copy()
    old_ids=np.array([],int);oldmark=np.zeros((nw,0));row_of={int(x):i for i,x in enumerate(cal.idx)}
    fits={k:dict(xx=np.zeros((6,6)),xy=np.zeros(6),days=0,stockdays=0) for k in ["median","seasonal"]}
    audits=[]
    for year in range(1985,2026):
        log(f"Caching date-window returns, drift and training curve moments: {year}")
        audits.append(dict(path=str(PANEL/f"{year}.parquet"),sha256=sha(PANEL/f"{year}.parquet")))
        d=frame(c,year);groups={int(k):g for k,g in d.groupby("idx",sort=False)}
        for idx in cal.loc[cal.date.dt.year.eq(year),"idx"]:
            f=groups.get(int(idx),d.iloc[:0]);ids=f.permno.to_numpy(dtype=int);di=row_of[int(idx)]
            union=np.union1d(old_ids,ids);loc=np.searchsorted(union,ids)
            w=np.zeros((nw,len(ids)))
            for clock in ["nearest_median","nearest_seasonal","since_known"]:
                positions=[j for j,x in enumerate(wins) if x["clock"]==clock]
                a=np.array([wins[j]["lo"] for j in positions]);b=np.array([wins[j]["hi"] for j in positions])
                values=f[clock].to_numpy(dtype=float,na_value=np.nan)
                mask=(values[None,:]>=a[:,None])&(values[None,:]<=b[:,None])
                counts=mask.sum(axis=1);n[di,positions]=counts
                w[positions]=mask/np.maximum(counts,1)[:,None]
            r=f.ret.fillna(0).to_numpy(dtype=float);rx=f.retx.fillna(f.ret).fillna(0).to_numpy(dtype=float)
            ret[di]=w@r;rxmean[di]=w@rx
            marked=np.zeros((nw,len(union)));marked[:,np.searchsorted(union,old_ids)]=oldmark
            target=np.zeros_like(marked);target[:,loc]=w
            turn[di]=np.abs(target-marked).sum(axis=1)
            oldmark=w*(1+rx);old_ids=ids
            if year<=2004 and len(f)>=100:
                y=r-r.mean()
                for kind,z in fits.items():
                    x=basis(f,kind);x-=x.mean(axis=0)
                    z["xx"]+=x.T@x/len(f);z["xy"]+=x.T@y/len(f)
                    z["days"]+=1;z["stockdays"]+=len(f)
    np.savez_compressed(OUT/"leg_cache.npz",counts=n,returns=ret,rxmean=rxmean,turnover=turn,
        dates=cal.date.to_numpy(dtype="datetime64[D]"),indices=cal.idx.to_numpy())
    curves=[]
    for kind,z in fits.items():
        for h in [1,2,3]:
            xx=z["xx"][:2*h,:2*h];xy=z["xy"][:2*h]
            penalty=.01*np.trace(xx)/(2*h)
            coeff=np.linalg.solve(xx+penalty*np.eye(2*h),xy)
            for weighting in ["score","tercile"]:
                curves.append(dict(id=f"learned_{kind}_H{h}_{weighting}",family=f"learned_{kind}",
                    type="learned",phase=kind,harmonics=h,coefficients=coeff.tolist(),weighting=weighting,
                    fit_days=z["days"],fit_stockdays=z["stockdays"],fit_end="2004-12-31",adjustment=1.))
    for offset in range(0,61,10):
        for weighting in ["score","tercile"]:
            curves.append(dict(id=f"fixed_cos_{offset}_{weighting}",family="fixed_periodic",type="fixed",
                offset_sessions=offset,phase="median",weighting=weighting,adjustment=1.))
    save(OUT/"curve_models.json",curves);save(OUT/"input_audit.json",dict(files=audits,
        inherited_source_audit=str(STUDY/"results_regular65/input_audit.json"),
        inherited_validation=str(STUDY/"results_regular65/validation_checks.json")))

def screen():
    freeze();cache=np.load(OUT/"leg_cache.npz");wins=json.loads((OUT/"windows.json").read_text())
    train=cache["dates"]<=np.datetime64("2004-12-31")
    n=cache["counts"][train];r=cache["returns"][train];rx=cache["rxmean"][train];t=cache["turnover"][train]
    pairs=[]
    for i,a in enumerate(wins):
        for j,b in enumerate(wins):
            if a["clock"]==b["clock"] and (a["hi"]<b["lo"] or b["hi"]<a["lo"]):pairs.append((i,j,a["clock"]))
    rows=[]
    for start in range(0,len(pairs),256):
        batch=pairs[start:start+256];li=np.array([x[0] for x in batch]);si=np.array([x[1] for x in batch])
        active=(n[:,li]>=20)&(n[:,si]>=20);prev=np.vstack([np.zeros((1,len(batch)),bool),active[:-1]])
        prev_rx=np.vstack([np.zeros((1,len(batch))),rx[:-1,li]+rx[:-1,si]])
        turnover=np.where(active&prev,t[:,li]+t[:,si],np.where(active,2.,np.where(prev,2+prev_rx,0.)))
        gross=np.where(active,r[:,li]-r[:,si],0.)
        for label,cost,borrow in COSTS[:4]:
            net=gross-cost*turnover-borrow/252*active
            mean=net.mean(axis=0)*252;sd=net.std(axis=0,ddof=1)*np.sqrt(252)
            score=np.divide(mean,sd,out=np.full_like(mean,-999.),where=sd>1e-10)
            for k,(i,j,family) in enumerate(batch):
                rows.append(dict(long_index=i,short_index=j,family=family,scenario=label,
                    annual_mean=float(mean[k]),sharpe=float(score[k]),active_fraction=float(active[:,k].mean()),
                    annual_turnover=float(turnover[:,k].mean()*252)))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/"window_screen.csv",index=False)
    chosen=set()
    for family,z in scores.loc[scores.active_fraction>=.30].groupby("family"):
        for scenario in ["1bp","5bps","10bps"]:
            top=z.loc[z.scenario.eq(scenario)].nlargest(3,"sharpe")
            chosen.update((int(x.long_index),int(x.short_index)) for x in top.itertuples())
        for scenario,metric in [("gross","sharpe"),("10bps","annual_mean")]:
            top=z.loc[z.scenario.eq(scenario)].nlargest(1,metric)
            chosen.update((int(x.long_index),int(x.short_index)) for x in top.itertuples())
    models=[]
    for i,j in sorted(chosen):
        l,s=wins[i],wins[j]
        models.append(dict(id=f"W_{i}_{j}",family=l["clock"],type="window",clock=l["clock"],
             long_lo=l["lo"],long_hi=l["hi"],short_lo=s["lo"],short_hi=s["hi"],adjustment=1.))
    models+=json.loads((OUT/"curve_models.json").read_text())
    save(OUT/"stage1_models.json",models)
    save(OUT/"screen_audit.json",dict(window_count=len(wins),ordered_window_pairs=len(pairs),
        screened_cost_objectives=4,shortlisted_window_pairs=len(chosen),curve_models=len(models)-len(chosen),
        selection_years="1985-2004",later_returns_used_for_selection=False))
    log(f"Screened {len(pairs)} ordered window pairs; exact simulation shortlist: {len(models)} models")

def target_weights(f,models):
    w=np.zeros((len(models),len(f)));basis_cache={}
    for j,m in enumerate(models):
        if m["type"]=="window":
            clock=f[m["clock"]].to_numpy(dtype=float,na_value=np.nan)
            long=(clock>=m["long_lo"])&(clock<=m["long_hi"])
            short=(clock>=m["short_lo"])&(clock<=m["short_hi"])
            if min(long.sum(),short.sum())>=20:w[j,long]=1/long.sum();w[j,short]=-1/short.sum()
            continue
        if len(f)<100:continue
        kind=m["phase"]
        if kind not in basis_cache:basis_cache[kind]=basis(f,kind)
        if m["type"]=="learned":score=basis_cache[kind][:,:2*m["harmonics"]]@np.asarray(m["coefficients"])
        else:
            x=basis_cache[kind];a=2*np.pi*m["offset_sessions"]/65
            score=x[:,1]*np.cos(a)+x[:,0]*np.sin(a)
        if m["weighting"]=="score":
            score=score-score.mean();long=score>1e-16;short=score< -1e-16
            if min(long.sum(),short.sum())>=20:
                w[j,long]=score[long]/score[long].sum();w[j,short]=-score[short]/score[short].sum()
        else:
            lo,hi=np.quantile(score,[1/3,2/3])
            if hi-lo<1e-15:continue
            long=score>=hi;short=score<=lo
            if min(long.sum(),short.sum())>=20:w[j,long]=1/long.sum();w[j,short]=-1/short.sum()
    return w

def execution(v,wbase,eligible,r,rx,models):
    """Cost-specific exact NAV and daily neutralization, including slow adjustment."""
    nc,nm,nn=v.shape
    w=np.broadcast_to(wbase,(nc,nm,nn)).copy()
    slow=np.array([m["adjustment"]<1 for m in models])
    if slow.any():
        rho=np.array([m["adjustment"] for m in models])[slow][None,:,None]
        prev=np.where(eligible[None,None,:],v[:,slow],0.)
        pl=np.maximum(prev,0);ps=-np.minimum(prev,0)
        sl=pl.sum(axis=2,keepdims=True);ss=ps.sum(axis=2,keepdims=True)
        carried=pl/np.maximum(sl,1e-300)-ps/np.maximum(ss,1e-300)
        valid=(sl>1e-12)&(ss>1e-12)
        old_norm=np.where(valid,carried,w[:,slow])
        mixed=(1-rho)*old_norm+rho*w[:,slow]
        ml=np.maximum(mixed,0);ms=-np.minimum(mixed,0)
        good=((ml>1e-14).sum(axis=2,keepdims=True)>=20)&((ms>1e-14).sum(axis=2,keepdims=True)>=20)
        base_active=np.abs(w[:,slow]).sum(axis=2,keepdims=True)>1
        final=ml/np.maximum(ml.sum(axis=2,keepdims=True),1e-300)-ms/np.maximum(ms.sum(axis=2,keepdims=True),1e-300)
        w[:,slow]=np.where(good&base_active,final,0.)
    cost=np.array([x[1] for x in COSTS])[:,None]
    x=np.ones((nc,nm))
    for _ in range(7):x=1-cost*np.abs(x[:,:,None]*w-v).sum(axis=2)
    positions=x[:,:,None]*w
    turn=np.abs(positions-v).sum(axis=2);fee=cost*turn
    assert np.max(np.abs(x-(1-fee)))<1e-10
    long=np.maximum(positions,0).sum(axis=2);short=-np.minimum(positions,0).sum(axis=2)
    assert np.max(np.abs(long-short))<1e-10
    borrow=np.array([x[2] for x in COSTS])[:,None]/252*short
    pnl=(positions*r[None,None,:]).sum(axis=2)-fee-borrow
    assert (1+pnl>0).all(),"Portfolio insolvency"
    after=positions*(1+rx[None,None,:])/(1+pnl[:,:,None])
    return pnl,after,turn,fee,borrow,long,short,positions

def simulate(models,start,end,name):
    c=connection();cal=calendar();cal=cal.loc[cal.date.dt.year.between(start,end)].reset_index(drop=True)
    shape=(len(cal),len(COSTS),len(models))
    arrays={k:np.zeros(shape) for k in ["returns","turnover","cost","borrow","n_long","n_short","active","missing_returns"]}
    oldids=np.array([],int);hold=np.zeros((len(COSTS),len(models),0));di=0;net_error=0.
    for year in range(start,end+1):
        log(f"Exact {name}, {len(models)} models x {len(COSTS)} costs: {year}")
        active_models=[dict(m,coefficients=m["annual_coefficients"][str(year)]) if "annual_coefficients" in m else m for m in models]
        d=frame(c,year);groups={int(k):g for k,g in d.groupby("idx",sort=False)}
        for idx in cal.loc[cal.date.dt.year.eq(year),"idx"]:
            f=groups.get(int(idx),d.iloc[:0]);ids=f.permno.to_numpy(dtype=int)
            union=np.union1d(oldids,ids);loc=np.searchsorted(union,ids)
            v=np.zeros((len(COSTS),len(models),len(union)));v[:,:,np.searchsorted(union,oldids)]=hold
            target=np.zeros((len(models),len(union)));target[:,loc]=target_weights(f,active_models)
            eligible=np.zeros(len(union),bool);eligible[loc]=True
            r=np.zeros(len(union));rx=r.copy();missing=np.zeros(len(union),bool)
            r[loc]=f.ret.fillna(0).to_numpy(dtype=float)
            rx[loc]=f.retx.fillna(f.ret).fillna(0).to_numpy(dtype=float)
            missing[loc]=f.ret.isna().to_numpy()
            pnl,after,turn,fee,borrow,long,short,positions=execution(v,target,eligible,r,rx,models)
            net_error=max(net_error,float(np.max(np.abs(long-short))))
            arrays["returns"][di]=pnl;arrays["turnover"][di]=turn
            arrays["cost"][di]=fee;arrays["borrow"][di]=borrow
            arrays["n_long"][di]=(positions>1e-14).sum(axis=2);arrays["n_short"][di]=(positions< -1e-14).sum(axis=2)
            arrays["active"][di]=long>0
            arrays["missing_returns"][di]=((positions!=0)&missing[None,None,:]).sum(axis=2)
            hold=after[:,:,loc];oldids=ids;di+=1
    np.savez_compressed(OUT/f"{name}.npz",**arrays,dates=cal.date.to_numpy(dtype="datetime64[D]"),
        indices=cal.idx.to_numpy(),max_net_error=np.array(net_error))
    save(OUT/f"{name}_models.json",models)
    return arrays,cal

def simple_stats(name):
    with np.load(OUT/f"{name}.npz") as loaded:a={k:loaded[k] for k in loaded.files}
    models=json.loads((OUT/f"{name}_models.json").read_text())
    rows=[]
    for period,(start,end) in PERIODS.items():
        mask=(a["dates"]>=np.datetime64(start))&(a["dates"]<=np.datetime64(end))
        if not mask.any():continue
        for ci,(scenario,_,_) in enumerate(COSTS):
            r=a["returns"][mask,ci,:];mean=r.mean(axis=0)*252;sd=r.std(axis=0,ddof=1)*np.sqrt(252)
            sharpe=np.divide(mean,sd,out=np.full_like(mean,-999),where=sd>1e-10)
            nav=np.cumprod(1+r,axis=0);peak=np.maximum.accumulate(np.vstack([np.ones((1,len(models))),nav]),axis=0)[1:]
            dd=(nav/peak-1).min(axis=0)
            for j,m in enumerate(models):
                rows.append(dict(model=m["id"],family=m["family"],adjustment=m["adjustment"],period=period,
                    scenario=scenario,annual_mean=float(mean[j]),cagr=float(nav[-1,j]**(252/len(r))-1),
                    annual_vol=float(sd[j]),sharpe=float(sharpe[j]),max_drawdown=float(dd[j]),
                    active_fraction=float(a["active"][mask,ci,j].mean()),
                    annual_turnover=float(a["turnover"][mask,ci,j].mean()*252),
                    annual_cost=float(a["cost"][mask,ci,j].mean()*252),annual_borrow=float(a["borrow"][mask,ci,j].mean()*252),
                    mean_longs=float(a["n_long"][mask,ci,j].mean()),mean_shorts=float(a["n_short"][mask,ci,j].mean())))
    stats=pd.DataFrame(rows);stats.to_csv(OUT/f"{name}_statistics.csv",index=False);return stats,models

def winners(stats,scenario):
    z=stats.loc[stats.period.eq("development")&stats.scenario.eq(scenario)&stats.active_fraction.ge(.30)]
    assert len(z),"No sufficiently active training strategies"
    # Deterministic id tiebreak if performance is numerically tied.
    selected=z.sort_values(["sharpe","model"],ascending=[False,True]).groupby("family",sort=True).head(1)
    return selected

def train():
    freeze();models=json.loads((OUT/"stage1_models.json").read_text())
    p=OUT/"stage1_train.npz";meta=OUT/"stage1_train_models.json"
    if p.exists() and meta.exists():assert json.loads(meta.read_text())==models
    else:simulate(models,1985,2004,"stage1_train")
    stats,models=simple_stats("stage1_train")
    chosen=pd.concat([winners(stats,"10bps"),winners(stats,"1bp")]).model.unique()
    stage2=[]
    for m in models:
        if m["id"] in chosen:
            for rho in [1.,.25,.10]:
                v=dict(m);v["adjustment"]=rho
                if rho!=1:v["id"]+=f"_adjust{rho}"
                stage2.append(v)
    dest=OUT/"stage2_models.json"
    if dest.exists():assert json.loads(dest.read_text())==stage2
    else:save(dest,stage2)
    p=OUT/"stage2_train.npz";meta=OUT/"stage2_train_models.json"
    if p.exists() and meta.exists():assert json.loads(meta.read_text())==stage2
    else:simulate(stage2,1985,2004,"stage2_train")
    stats,models=simple_stats("stage2_train")
    primary=winners(stats,"10bps");low=winners(stats,"1bp")
    choices=dict(primary_cost="10bps",selection_end="2004-12-31",later_returns_used=False,
        family_winners_10bps=primary[["family","model","sharpe","annual_mean"]].to_dict("records"),
        family_winners_1bp=low[["family","model","sharpe","annual_mean"]].to_dict("records"),
        overall_10bps=primary.sort_values("sharpe",ascending=False).iloc[0].model,
        overall_1bp=low.sort_values("sharpe",ascending=False).iloc[0].model)
    dest=OUT/"frozen_choices.json"
    if dest.exists():assert json.loads(dest.read_text())==choices
    else:save(dest,choices)
    ids=set(primary.model)|set(low.model)
    save(OUT/"final_models.json",[m for m in models if m["id"] in ids])
    log("Frozen choices before later-period simulation: "+json.dumps(choices))

def evaluate():
    freeze();models=json.loads((OUT/"final_models.json").read_text())
    simulate(models,1985,2025,"final")
    stats,_=simple_stats("final")
    with np.load(OUT/"final.npz") as loaded:a={k:loaded[k] for k in loaded.files}
    choices=json.loads((OUT/"frozen_choices.json").read_text())
    rows=[];robust=[];daily=[]
    chosen_primary={x["model"] for x in choices["family_winners_10bps"]}
    # Read the broad local VW market only for post-construction diagnostics.
    c=connection()
    market=c.execute(f"SELECT date,vw_market FROM read_parquet('{STUDY.parent}/results/panel/market_*.parquet') ORDER BY date").df().set_index("date").vw_market
    dates=pd.to_datetime(a["dates"]);vw=market.reindex(dates).to_numpy()
    for j,m in enumerate(models):
        for ci,(scenario,_,_) in enumerate(COSTS):
            for k,dt in enumerate(dates):
                daily.append(dict(date=dt,model=m["id"],family=m["family"],scenario=scenario,
                    returns=a["returns"][k,ci,j],turnover=a["turnover"][k,ci,j],
                    cost=a["cost"][k,ci,j],borrow=a["borrow"][k,ci,j],
                    active=bool(a["active"][k,ci,j]),n_long=int(a["n_long"][k,ci,j]),n_short=int(a["n_short"][k,ci,j])))
            for period,(start,end) in PERIODS.items():
                mask=(a["dates"]>=np.datetime64(start))&(a["dates"]<=np.datetime64(end))
                r=a["returns"][mask,ci,j]
                f=sm.OLS(r,np.ones((len(r),1))).fit(cov_type="HAC",cov_kwds={"maxlags":20})
                lo,hi=f.conf_int()[0]*252
                controlled=sm.OLS(r,sm.add_constant(vw[mask])).fit(cov_type="HAC",cov_kwds={"maxlags":20})
                rows.append(dict(model=m["id"],period=period,scenario=scenario,t=float(f.tvalues[0]),p=float(f.pvalues[0]),
                    ci_low=float(lo),ci_high=float(hi),vw_intercept=float(controlled.params[0]*252),
                    vw_t=float(controlled.tvalues[0]),vw_beta=float(controlled.params[1]),primary_family_winner=m["id"] in chosen_primary))
                if period in ["validation","holdout"]:
                    h=sm.OLS(r,np.ones((len(r),1))).fit(cov_type="HAC",cov_kwds={"maxlags":60})
                    robust.append(dict(model=m["id"],period=period,scenario=scenario,frequency="daily",lags=60,
                        t=float(h.tvalues[0]),p=float(h.pvalues[0]),annual_mean=float(r.mean()*252)))
                    s=pd.Series(r,index=dates[mask]).resample("ME").apply(lambda z:np.prod(1+z)-1)
                    h=sm.OLS(s.to_numpy(),np.ones((len(s),1))).fit(cov_type="HAC",cov_kwds={"maxlags":3})
                    robust.append(dict(model=m["id"],period=period,scenario=scenario,frequency="monthly",lags=3,
                        t=float(h.tvalues[0]),p=float(h.pvalues[0]),annual_mean=float(s.mean()*12)))
    stats=stats.merge(pd.DataFrame(rows),on=["model","period","scenario"],validate="one_to_one")
    stats["p_holm_family"]=np.nan
    for _,z in stats.loc[stats.primary_family_winner].groupby(["period","scenario"]):
        stats.loc[z.index,"p_holm_family"]=multipletests(z.p,method="holm")[1]
    stats.to_csv(OUT/"performance.csv",index=False)
    pd.DataFrame(robust).to_csv(OUT/"inference_robustness.csv",index=False)
    daily=pd.DataFrame(daily);daily.to_csv(OUT/"daily_returns.csv",index=False)
    daily.assign(year=daily.date.dt.year).groupby(["model","scenario","year"]).agg(
        returns=("returns",lambda x:np.prod(1+x)-1),turnover=("turnover","sum"),
        cost=("cost","sum"),borrow=("borrow","sum")).to_csv(OUT/"annual_returns.csv")
    log("Finished chronological evaluation of frozen rules")

def main():
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["prepare","screen","train","evaluate"]);a=p.parse_args()
    if a.stage=="prepare":prepare()
    elif a.stage=="screen":screen()
    elif a.stage=="train":train()
    elif a.stage=="evaluate":evaluate()

if __name__=="__main__":main()
