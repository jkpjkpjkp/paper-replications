"""Locked unseen-seed confirmation plus explicitly descriptive test-oracle sensitivity."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ocp import run, controller
OUT=HERE/"confirmation"
OUT.mkdir(exist_ok=True)


def lock():
    path=OUT/"locked_protocol.json"
    if path.exists():
        return json.loads(path.read_text())
    protocol=dict(locked_at=datetime.now(timezone.utc).isoformat(),
        primary_scenario="smooth_correlated_lognormal_sigma15_amplitude8",
        transfer_scenario="smooth_correlated_lognormal_sigma20_amplitude8",
        primary_method="UP-paper",n=3000,burn=300,seeds=list(range(100,130)),
        generation=dict(rng="NumPy default_rng/PCG64",order="Draw and discard N N(0,.3) values, then draw N independent N(0,1) latent values, preserving exact round2.generate stream order.",
            score="exp(8 * scipy.ndimage.gaussian_filter1d(z, sigma))",boundary="reflect (SciPy default)",
            sigma_primary=15,sigma_transfer=20,amplitude=8,
            interpretation="Stationary smooth, persistent lognormal nonconformity scores; jointly generated latent path. Calibration only receives past scores."),
        fixed_configs={"UP-paper":{"alpha":.035,"lr":1},"UP":{"alpha":.035,"lr":1},
            "P":{"alpha":.04,"lr":.5,"window":100},"SFOGD":{"alpha":.045,"lr":25},
            "KT":{"alpha":.03,"lr":1},"DtACI":{"alpha":.05,"lr":1}},
        dtaci_note="No finite 95%-coverage configuration existed on discovery grid. Alpha=.05 is retained as a fixed nominal diagnostic; not a selected finite optimum.",
        primary_outcomes=["mean interval width","empirical coverage","paired UP-paper/baseline ratio of mean widths","paired coverage differences"],
        bootstrap=dict(unit="independent seed/path",n_resamples=10000,seed=72026,interval="percentile 95%",paired=True),
        descriptive_oracle=dict(label="Test-oracle sensitivity; selection on confirmation outcomes, not an independent confirmation",
            alphas=[.005,.01,.015,.02,.025,.03,.035,.04,.045,.05,.06,.075,.1,.125,.15,.175,.2,.225,.25],
            sfogd_lrs=[.01,.1,1,10,25,100,300,1000,3000],
            p_lrs=[.005,.01,.025,.05,.1,.25,.5,1,2],p_windows=[100],target=.95),
        frozen_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/"round2.py",HERE/"screen.py",ROOT/"ocp.py"]})
    path.write_text(json.dumps(protocol,indent=2))
    return protocol


def generate(seed,sigma,p):
    r=np.random.default_rng(seed)
    r.normal(0,.3,p["n"])
    z=r.normal(size=p["n"])
    return np.exp(8*gaussian_filter1d(z,sigma=sigma,mode="reflect"))


def predict(s,method,alpha,lr=1,window=100):
    return controller(s,alpha=alpha,lr=lr,window=window) if method=="P" else run(s,method,alpha,lr)


def bootstrap(part,baseline,p):
    pivot=part.pivot(index="seed",columns="method",values="mean_width")
    cov=part.pivot(index="seed",columns="method",values="coverage")
    a=pivot["UP-paper"].to_numpy();b=pivot[baseline].to_numpy()
    if not np.isfinite(b).all():
        return dict(baseline=baseline,ratio=None,ratio_ci_low=None,ratio_ci_high=None,
                    coverage_difference=float(np.mean(cov["UP-paper"]-cov[baseline])),
                    note="At least one baseline mean width is infinite; finite width ratio CI not reported.")
    r=np.random.default_rng(p["bootstrap"]["seed"])
    ix=r.integers(0,len(a),size=(p["bootstrap"]["n_resamples"],len(a)))
    ratios=a[ix].mean(axis=1)/b[ix].mean(axis=1)
    diff=(cov["UP-paper"]-cov[baseline]).to_numpy()
    diffs=diff[ix].mean(axis=1)
    return dict(baseline=baseline,ratio=float(a.mean()/b.mean()),
        ratio_ci_low=float(np.quantile(ratios,.025)),ratio_ci_high=float(np.quantile(ratios,.975)),
        saving_percent=float(100*(1-a.mean()/b.mean())),
        saving_ci_low=float(100*(1-np.quantile(ratios,.975))),saving_ci_high=float(100*(1-np.quantile(ratios,.025))),
        coverage_difference=float(diff.mean()),coverage_difference_ci_low=float(np.quantile(diffs,.025)),
        coverage_difference_ci_high=float(np.quantile(diffs,.975)))


def main():
    p=lock()  # Must exist on disk before any unseen path is generated.
    print("Locked at",p["locked_at"],flush=True)
    records=[];oracle=[];traces={}
    for scenario,sigma in [(p["primary_scenario"],15),(p["transfer_scenario"],20)]:
        series=[]
        for seed in p["seeds"]:
            s=generate(seed,sigma,p);series.append(s)
            for method,cfg in p["fixed_configs"].items():
                q=predict(s,method,**cfg)[p["burn"]:];score=s[p["burn"]:]
                records.append(dict(scenario=scenario,seed=seed,method=method,**cfg,
                    coverage=float(np.mean(score<=q)),mean_width=float(np.mean(2*np.maximum(q,0))),
                    infinite_fraction=float(np.mean(np.isinf(q)))))
        traces[scenario]=np.array(series)
        pd.DataFrame(records).to_csv(OUT/"fixed_per_seed.csv",index=False)
        print(scenario,"fixed completed",flush=True)
        grid=p["descriptive_oracle"]
        configs=[(m,1) for m in ["UP-paper","UP","KT","DtACI"]]+[("SFOGD",v) for v in grid["sfogd_lrs"]]+[("P",v) for v in grid["p_lrs"]]
        for method,lr in configs:
            for alpha in grid["alphas"]:
                for seed,s in zip(p["seeds"],series):
                    q=predict(s,method,alpha,lr)[p["burn"]:]
                    oracle.append(dict(scenario=scenario,seed=seed,method=method,lr=lr,alpha=alpha,
                        coverage=float(np.mean(s[p["burn"]:]<=q)),mean_width=float(np.mean(2*np.maximum(q,0)))))
        print(scenario,"oracle completed",flush=True)
    np.savez_compressed(OUT/"scores.npz",**traces,seeds=p["seeds"])
    fixed=pd.DataFrame(records);fixed.to_csv(OUT/"fixed_per_seed.csv",index=False)
    fixed_summary=fixed.groupby(["scenario","method"],as_index=False).agg(
        coverage=("coverage","mean"),coverage_sem=("coverage",lambda x:x.std(ddof=1)/np.sqrt(len(x))),
        mean_width=("mean_width","mean"),width_sem=("mean_width",lambda x:x.std(ddof=1)/np.sqrt(len(x))),
        infinite_fraction=("infinite_fraction","mean"))
    fixed_summary.to_csv(OUT/"fixed_summary.csv",index=False)
    pairs=[]
    for scenario,part in fixed.groupby("scenario"):
        for method in ["P","SFOGD","KT","DtACI"]:
            pairs.append(dict(scenario=scenario,**bootstrap(part,method,p)))
    pd.DataFrame(pairs).to_csv(OUT/"paired_comparisons.csv",index=False)
    oracle=pd.DataFrame(oracle);oracle.to_csv(OUT/"oracle_per_seed.csv",index=False)
    agg=oracle.groupby(["scenario","method","lr","alpha"],as_index=False).agg(
        coverage=("coverage","mean"),mean_width=("mean_width","mean"))
    agg.to_csv(OUT/"oracle_grid.csv",index=False)
    best=[]
    for (scenario,method),part in agg.groupby(["scenario","method"]):
        valid=part[(part.coverage>=.95)&np.isfinite(part.mean_width)]
        if len(valid):best.append(valid.sort_values("mean_width").iloc[0].to_dict())
    pd.DataFrame(best).to_csv(OUT/"oracle_best_95.csv",index=False)
    print(fixed_summary.to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))
    print("TEST-ORACLE, DESCRIPTIVE ONLY")
    print(pd.DataFrame(best).to_string(index=False))


if __name__=="__main__":main()
