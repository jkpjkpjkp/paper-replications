"""Post-confirmation robustness: select expanded controller grids on discovery only."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run,controller
from confirm import generate,bootstrap
from round2 import generate as round2_generate
OUT=HERE/"expanded_robustness"
OUT.mkdir(exist_ok=True)
ALPHAS=sorted(set(np.round(np.arange(.02,.0601,.002),3))|{.035,.045})
WINDOWS=[25,50,100,200,300]
LRS=[.01,.025,.05,.1,.25,.5,1,2]


def get_score(name,seed,p):
    if name=="sigma15":return generate(seed,15,p)
    if name=="sigma20":return generate(seed,20,p)
    if name=="original_sinusoid":return round2_generate(name,seed)
    raise ValueError(name)


def configs():
    for a in ALPHAS:
        yield dict(method="UP-paper-expanded",alpha=a,lr=1.,window=100,ki_multiplier=0.,csat=1.)
        for lr in [.01,.1,1,10,25,100,300,1000,3000]:
            yield dict(method="SFOGD-expanded",alpha=a,lr=float(lr),window=100,ki_multiplier=0.,csat=1.)
        for w in WINDOWS:
            for lr in LRS:
                yield dict(method="P-expanded",alpha=a,lr=lr,window=w,ki_multiplier=0.,csat=1.)
    # A declared smaller PI grid; do not imply exhaustive controller optimization.
    for a in [.025,.03,.035,.04,.045,.05]:
        for w in [50,100,300]:
            for lr in [.05,.1,.5,1]:
                for csat in [1.,5.,10.]:
                    for mult in [.1,1.]:
                        yield dict(method="PI-expanded",alpha=a,lr=lr,window=w,ki_multiplier=mult,csat=csat)


def predict(s,c):
    if c["method"] in ["P-expanded","PI-expanded"]:
        ki=c["ki_multiplier"]*float(np.max(s[:300]))
        return controller(s,alpha=c["alpha"],lr=c["lr"],window=c["window"],ki=ki,csat=c["csat"])
    method="UP-paper" if c["method"]=="UP-paper-expanded" else "SFOGD"
    return run(s,method,c["alpha"],c["lr"])


def measure(s,q):
    score=s[300:];q=q[300:]
    return dict(coverage=float(np.mean(score<=q)),mean_width=float(np.mean(2*np.maximum(q,0))),
                infinite_fraction=float(np.mean(np.isinf(q))))


def main():
    p=json.loads((HERE/"confirmation/locked_protocol.json").read_text())
    selected=[];raw=[]
    candidate_configs=list(configs())
    for name in ["sigma15","original_sinusoid"]:
        paths=[get_score(name,seed,p) for seed in range(5)]
        for c in candidate_configs:
            for seed,s in zip(range(5),paths):raw.append(dict(scenario=name,seed=seed,**c,**measure(s,predict(s,c))))
        print(name,"discovery grid done",flush=True)
    raw=pd.DataFrame(raw)
    raw.to_csv(OUT/"discovery_per_seed.csv",index=False)
    group=["scenario","method","alpha","lr","window","ki_multiplier","csat"]
    agg=raw.groupby(group,as_index=False).agg(coverage=("coverage","mean"),mean_width=("mean_width","mean"))
    agg.to_csv(OUT/"discovery_grid.csv",index=False)
    for (name,method),part in agg.groupby(["scenario","method"]):
        valid=part[(part.coverage>=.95)&np.isfinite(part.mean_width)]
        if len(valid):selected.append(valid.sort_values("mean_width").iloc[0].to_dict())
    lock=dict(locked_at=datetime.now(timezone.utc).isoformat(),
        label="Post-confirmation robustness, selected on discovery seeds0..4 only; original locked results unchanged",
        target=.95,discovery_seeds=list(range(5)),evaluation_seeds=list(range(100,130)),
        methods=selected,parameters=dict(alphas=ALPHAS,p_windows=WINDOWS,p_lrs=LRS,
            pi_windows=[50,100,300],pi_lrs=[.05,.1,.5,1],pi_alphas=[.025,.03,.035,.04,.045,.05],
            pi_csat=[1,5,10],pi_ki_multipliers=[.1,1],pi_ki="multiplier times max score in first300 observed calibration steps"))
    lock_path=OUT/"locked_expanded_configs.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text())["methods"]==lock["methods"],"Changed selection on rerun"
    else:lock_path.write_text(json.dumps(lock,indent=2))
    print("Expanded settings locked",flush=True)
    records=[]
    for name in ["sigma15","sigma20","original_sinusoid"]:
        source="sigma15" if name=="sigma20" else name
        chosen=[c for c in selected if c["scenario"]==source]
        for seed in range(100,130):
            s=get_score(name,seed,p)
            for c in chosen:
                cfg={k:c[k] for k in ["method","alpha","lr","window","ki_multiplier","csat"]}
                records.append(dict(scenario=name,seed=seed,**cfg,**measure(s,predict(s,cfg))))
            # Retain the original primary alpha=.035 as an explicit reference.
            records.append(dict(scenario=name,seed=seed,method="UP-paper",alpha=.035,lr=1,
                window=100,ki_multiplier=0,csat=1,**measure(s,run(s,"UP-paper",.035))))
        print(name,"expanded holdout evaluated",flush=True)
    results=pd.DataFrame(records);results.to_csv(OUT/"holdout_per_seed.csv",index=False)
    summary=results.groupby(["scenario","method"],as_index=False).agg(
        coverage=("coverage","mean"),mean_width=("mean_width","mean"),
        infinite_fraction=("infinite_fraction","mean"))
    summary.to_csv(OUT/"holdout_summary.csv",index=False)
    pairs=[]
    for name,part in results.groupby("scenario"):
        for method in part.method.unique():
            if method!="UP-paper":pairs.append(dict(scenario=name,**bootstrap(part,method,p)))
    pd.DataFrame(pairs).to_csv(OUT/"paired_vs_original_up.csv",index=False)
    print(pd.DataFrame(selected).to_string(index=False))
    print(summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__=="__main__":main()
