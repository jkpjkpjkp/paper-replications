"""Exploratory scenario screen. Selection is discovery-only; no confirmation seeds."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ocp import run, controller

N = 3000
BURN = 300
SEEDS = list(range(5))
ALPHAS = [.005,.01,.015,.02,.025,.03,.035,.04,.045,.05,.06,.075,.1,.125,.15,.175,.2,.225,.25]
SF_LRS = [.01,.1,1,10,25,100,300,1000,3000]
P_LRS = [.005,.01,.025,.05,.1,.25,.5,1,2]
TARGETS = [.90,.95]
SCENARIOS = [
    "jump_and_recovery", "growing_volatility", "growing_cycles", "contamination_drift",
    "t3_garch", "t3_garch_drift", "on_off_activity", "sawtooth_workload",
    "rare_shocks", "quiet_then_bursts", "lognormal_drift", "multiscale_regimes",
    "gradual_then_jump", "ramp_reset", "gaussian_cycles", "spiky_cycles"
]


def generate(name, seed):
    r = np.random.default_rng(seed)
    t = np.arange(N)
    normal = np.abs(r.normal(size=N))
    if name == "jump_and_recovery":
        scale = np.repeat([1,20,1,50,1],600)
        s = scale * normal
    elif name == "growing_volatility":
        s = np.exp(np.log(100)*t/(N-1)) * normal
    elif name == "growing_cycles":
        s = np.exp(np.log(50)*t/(N-1)) * (2+1.8*np.sin(2*np.pi*t/200)) * normal
    elif name == "contamination_drift":
        s = np.where(t<1500,1,20) * (normal + (r.random(N)<.02)*r.exponential(100,N))
    elif name.startswith("t3_garch"):
        z=r.standard_t(3,N)/np.sqrt(3)
        h=np.ones(N); residual=np.zeros(N)
        for j in range(N):
            if j: h[j]=.02+.12*residual[j-1]**2+.86*h[j-1]
            residual[j]=np.sqrt(h[j])*z[j]
        s=np.abs(residual)
        if name.endswith("drift"): s*=np.exp(np.log(30)*t/(N-1))
    elif name == "on_off_activity":
        s=np.where((t//200)%2==0,0,r.gamma(2,10,N))
    elif name == "sawtooth_workload":
        s=np.maximum(0,20*(t%300)/299+r.normal(0,.3,N))
    elif name == "rare_shocks":
        s=normal.copy()
        for start in range(450,N,500): s[start:start+10]*=100
    elif name == "quiet_then_bursts":
        s=np.where((t%500)<400,.1,50)*normal
    elif name == "lognormal_drift":
        s=np.exp(np.log(30)*t/(N-1))*r.lognormal(0,1.5,N)
    elif name == "multiscale_regimes":
        levels=np.array([1,10,2,100,3,30,1,300,10,1000])
        s=levels[t//300]*normal
    elif name == "gradual_then_jump":
        s=(1+5*t/(N-1))*np.where(t<2000,1,100)*normal
    elif name == "ramp_reset":
        s=(1+49*(t%500)/499)*normal
    elif name == "gaussian_cycles":
        s=(2+1.9*np.sin(2*np.pi*t/200))*normal
    elif name == "spiky_cycles":
        s=(2+1.9*np.sin(2*np.pi*t/200))*(normal+(r.random(N)<.01)*r.exponential(100,N))
    else:
        raise ValueError(name)
    assert np.isfinite(s).all() and (s>=0).all()
    return s


def main():
    protocol = dict(stage="discovery only", seeds=SEEDS, n=N, burn=BURN, alphas=ALPHAS,
                    sfogd_lrs=SF_LRS, p_lrs=P_LRS, p_window=100,
                    initialization="All methods initialized at t=0; evaluate t=300..2999",
                    comparisons="Each method's minimum mean width among grid configurations with average empirical coverage >= target; no interpolation.",
                    targets=TARGETS, scenarios=SCENARIOS,
                    caveat="All alpha/lr choices are oracle discovery choices. They must be locked and evaluated on disjoint seeds for confirmation.")
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2))
    configs=[(m,1) for m in ["UP","UP-paper","KT","DtACI"]]
    configs += [("SFOGD",lr) for lr in SF_LRS]+[("P",lr) for lr in P_LRS]
    records=[]
    start=time.time()
    for name in SCENARIOS:
        for seed in SEEDS:
            scores=generate(name,seed)
            for method,lr in configs:
                for a in ALPHAS:
                    q=controller(scores,alpha=a,lr=lr,window=100) if method=="P" else run(scores,method,a,lr)
                    q=q[BURN:]; s=scores[BURN:]
                    records.append(dict(scenario=name,seed=seed,method=method,lr=lr,alpha=a,
                        coverage=np.mean(s<=q),mean_width=np.mean(2*np.maximum(q,0)),
                        infinite_fraction=np.mean(np.isinf(q))))
        print(name,round(time.time()-start,1),flush=True)
        pd.DataFrame(records).to_csv(OUT/"per_seed.csv",index=False)
    df=pd.DataFrame(records)
    agg=df.groupby(["scenario","method","lr","alpha"],as_index=False).agg(
        coverage=("coverage","mean"),mean_width=("mean_width","mean"),
        infinite_fraction=("infinite_fraction","mean"))
    agg.to_csv(OUT/"aggregate.csv",index=False)
    best=[]
    for (scenario,method),part in agg.groupby(["scenario","method"]):
        for target in TARGETS:
            valid=part[(part.coverage>=target)&np.isfinite(part.mean_width)]
            if len(valid):
                row=valid.sort_values("mean_width").iloc[0].to_dict();row["target"]=target;best.append(row)
    best=pd.DataFrame(best);best.to_csv(OUT/"best_at_coverage.csv",index=False)
    ranking=[]
    for (scenario,target),part in best.groupby(["scenario","target"]):
        by=part.set_index("method")
        baselines=part[part.method.isin(["KT","DtACI","SFOGD","P"])]
        if len(baselines) and "UP" in by.index:
            base=baselines.sort_values("mean_width").iloc[0]
            ranking.append(dict(scenario=scenario,target=target,up_width=by.loc["UP","mean_width"],
                best_baseline=base.method,baseline_width=base.mean_width,
                up_vs_best_ratio=by.loc["UP","mean_width"]/base.mean_width,
                up_paper_width=by.loc["UP-paper","mean_width"] if "UP-paper" in by.index else np.nan,
                up_paper_vs_best_ratio=by.loc["UP-paper","mean_width"]/base.mean_width if "UP-paper" in by.index else np.nan,
                up_vs_kt_ratio=by.loc["UP","mean_width"]/by.loc["KT","mean_width"] if "KT" in by.index else np.nan))
    ranking=pd.DataFrame(ranking).sort_values("up_vs_best_ratio")
    ranking.to_csv(OUT/"ranking.csv",index=False)
    print(ranking.to_string(index=False))


if __name__=="__main__": main()
