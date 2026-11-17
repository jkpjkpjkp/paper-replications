"""Exploratory real-data screen; all candidates and outcomes retained.

Public snapshots fixed by commit; point forecasts strictly lagged.
Pareto evaluation is an oracle description of finite-sample outcomes, not a
claim that tuning on evaluation observations is possible in deployment.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller,metrics

SOURCES = {
    "btc": dict(url="https://raw.githubusercontent.com/coinmetrics/data/f1a36afb962731c387bb03982758ab0103063da5/csv/btc.csv",
                page="https://github.com/coinmetrics/data", name="Coin Metrics BTC daily PriceUSD", license="CC BY-NC 4.0"),
    "covid_us": dict(url="https://raw.githubusercontent.com/nytimes/covid-19-data/62ef34cfcb60214be873a38d73619da9ea57d50b/us.csv",
                     page="https://github.com/nytimes/covid-19-data", name="NYT US COVID cumulative cases", license="See source repository LICENSE"),
    "nyc_taxi": dict(url="https://raw.githubusercontent.com/numenta/NAB/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/realKnownCause/nyc_taxi.csv",
                     page="https://github.com/numenta/NAB/tree/master/data/realKnownCause", name="NAB NYC taxi half-hourly passenger counts", license="See source repository LICENSE"),
    "machine_temperature": dict(url="https://raw.githubusercontent.com/numenta/NAB/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/realKnownCause/machine_temperature_system_failure.csv",
                                page="https://github.com/numenta/NAB/tree/master/data/realKnownCause", name="NAB machine temperature before system failure", license="See source repository LICENSE")
}


def download(item):
    name,source=item
    p=HERE/"raw"/f"{name}.csv"
    p.parent.mkdir(exist_ok=True)
    if not p.exists():
        print("Downloading",name,flush=True)
        urllib.request.urlretrieve(source["url"],p)
    return name,dict(**source,path=str(p.relative_to(HERE)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def prepare():
    with ThreadPoolExecutor(max_workers=4) as pool:
        manifest=dict(pool.map(download,SOURCES.items()))
    arrays={}
    for name in SOURCES:
        df=pd.read_csv(HERE/"raw"/f"{name}.csv")
        if name=="btc":
            df["date"]=pd.to_datetime(df["time"],utc=True)
            df=df[(df.date>="2013-01-01")&(df.date<"2026-01-01")]
            df=df.dropna(subset=["PriceUSD"])
            y=df.PriceUSD.astype(float).to_numpy()
            dates=df.date.to_numpy(dtype=str)
            lag=1
            forecast="y[t-1] (random walk in original USD price units)"
        elif name=="covid_us":
            y=np.maximum(0,np.diff(df.cases.to_numpy(),prepend=df.cases.iloc[0]))
            dates=df.date.to_numpy(dtype=str)
            lag=7
            forecast="new cases seven days earlier; negative cumulative-count revisions clipped to zero new cases"
        else:
            y=df.value.astype(float).to_numpy()
            dates=df.timestamp.to_numpy(dtype=str)
            lag=48 if name=="nyc_taxi" else 1
            forecast=f"y[t-{lag}]"
        score=np.abs(y[lag:]-y[:-lag])
        burnin=min(300,max(100,len(score)//10))
        scale=float(np.median(score[:burnin]))
        if scale<=0:
            scale=float(np.mean(score[:burnin])) or 1.0
        arrays[name]=dict(scores=score/scale,raw_scores=score,dates=dates[lag:],scale=scale,burnin=burnin)
        np.savez_compressed(HERE/f"{name}_scores.npz",**arrays[name])
        manifest[name].update(n=len(score),date_start=str(dates[lag]),date_end=str(dates[-1]),
                              forecast=forecast,normalization="divide all scores by median burn-in score; fixed before evaluation",scale=scale,burnin=burnin)
    (HERE/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return arrays


def main():
    data=prepare()
    alpha_grid=np.array([.0025,.005,.0075,.01,.015,.02,.025,.03,.035,.04,.045,.05,.06,.075,.1,.125,.15,.2,.25,.3])
    sf_lrs=[.01,.1,.25,1,2.5,10,25,100]
    p_lrs=[.01,.05,.1,.25,.5,1,2]
    all_rows=[]
    for name,info in data.items():
        s=info["scores"];b=info["burnin"]
        # These are candidate screens, not independent replications.
        for method,lrs in [("UP",[0]),("UP-paper",[0]),("KT",[0]),("DtACI",[0]),("SFOGD",sf_lrs),("P",p_lrs)]:
            for lr in lrs:
                for alpha in alpha_grid:
                    q=controller(s,alpha,lr,b)[b:] if method=="P" else run(s[b:],method,alpha,lr)
                    for period,slc in [("all",slice(51,None)),("discovery",slice(51,len(q)*3//5)),("confirmation",slice(len(q)*3//5,None))]:
                        m=metrics(s[b:][slc],q[slc])
                        # Widths are converted back into original units.
                        for key in ["mean_width","median_width","q75_width","q90_width","q95_width"]:
                            m[key]*=info["scale"]
                        all_rows.append(dict(dataset=name,method=method,lr=lr,alpha=alpha,period=period,**m))
        print(name,"finished",flush=True)
    frame=pd.DataFrame(all_rows)
    frame.to_csv(HERE/"all_runs.csv",index=False)
    matches=[]
    # For each evaluated period, minimum width among configurations with at
    # least target coverage. This oracle frontier grants baselines LR tuning.
    for (dataset,period,method),part in frame.groupby(["dataset","period","method"]):
        for target in [.9,.95,.975]:
            feasible=part[(part.coverage>=target)&np.isfinite(part.mean_width)]
            if len(feasible):
                r=feasible.loc[feasible.mean_width.idxmin()]
                matches.append(dict(dataset=dataset,period=period,method=method,target=target,
                                    achieved_coverage=r.coverage,width=r.mean_width,alpha=r.alpha,lr=r.lr,
                                    longest_error_streak=r.longest_error_streak))
            else:
                matches.append(dict(dataset=dataset,period=period,method=method,target=target,
                                    achieved_coverage=np.nan,width=np.inf,alpha=np.nan,lr=np.nan,
                                    longest_error_streak=np.nan))
    matched=pd.DataFrame(matches)
    matched.to_csv(HERE/"oracle_frontiers.csv",index=False)
    # Genuine holdout test: choose nominal 95% configs only on discovery,
    # then read that same config's untouched confirmation outcomes.
    held=[]
    for dataset in data:
        for method in frame.method.unique():
            d=frame[(frame.dataset==dataset)&(frame.method==method)&(frame.period=="discovery")]
            feasible=d[(d.coverage>=.95)&np.isfinite(d.mean_width)]
            if not len(feasible): continue
            chosen=feasible.loc[feasible.mean_width.idxmin()]
            test=frame[(frame.dataset==dataset)&(frame.method==method)&(frame.period=="confirmation")&
                       (frame.lr==chosen.lr)&(frame.alpha==chosen.alpha)].iloc[0]
            held.append(dict(dataset=dataset,method=method,lr=chosen.lr,alpha=chosen.alpha,
                             discovery_coverage=chosen.coverage,discovery_width=chosen.mean_width,
                             confirmation_coverage=test.coverage,confirmation_width=test.mean_width,
                             confirmation_longest_error_streak=test.longest_error_streak))
    pd.DataFrame(held).to_csv(HERE/"discovery_selected_holdout.csv",index=False)
    print(matched[(matched.period=="all")&(matched.target==.95)].to_string(index=False))


if __name__=="__main__":main()
