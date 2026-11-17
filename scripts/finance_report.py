#!/usr/bin/env python3
"""Finance-focused result tables, price-path plots, and window sensitivity."""
import csv
import html
import json
from pathlib import Path
import sys
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.reproduce import sde_features,metrics,write_json
from scripts.predict import predict
from chaos.data import heston

OUT=ROOT/"report"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})


def save(fig,name):
    for extension in ["png","svg"]:fig.savefig(OUT/f"{name}.{extension}",bbox_inches="tight")
    plt.close(fig)


def mean_sd(values):
    a=np.asarray(values,float)
    return {"mean":float(a.mean()),"seed_sd":float(a.std(ddof=1)) if len(a)>1 else None,"seeds":len(a),"values":a.tolist()}


def main():
    torch.set_num_threads(2)
    groups=defaultdict(list);runs=[];controls=[]
    for p in sorted((ROOT/"results").glob("*/metrics.json")):
        d=json.loads(p.read_text())
        if d["experiment"]!="heston":continue
        cfg=d["config"];window=cfg["train_steps"]
        a=np.load(p.parent/"predictions.npz")
        truth=a["truth"];known=a["prediction"];unknown=a["prediction_unknown_noise"]
        if window==cfg["steps"]:
            controls.append({"run":p.parent.name,"seed":d["seed"],
                             "full_interval_test_price":metrics(truth[:,1:,0],known[:,1:,0])})
            continue
        after=slice(window+1,None)
        steps=np.arange(1,cfg["steps"]-window+1)
        conditional=truth[:,window,0,None]*(1+cfg["heston"]["mu"]*cfg["dt"])**steps[None]
        persistence=np.repeat(truth[:,window,0,None],len(steps),axis=1)
        full_price=metrics(truth[:,after,0],known[:,after,0])
        unknown_price=metrics(truth[:,after,0],unknown[:,after,0])
        oracle=metrics(truth[:,after,0],conditional)
        persistence_metric=metrics(truth[:,after,0],persistence)
        run={"run":p.parent.name,"seed":d["seed"],"train_steps":window,
             "future_price_full_noise":full_price,"future_price_unknown_noise":unknown_price,
             "shared_81_100_price_full_noise":metrics(truth[:,81:,0],known[:,81:,0]),
             "shared_81_100_price_unknown_noise":metrics(truth[:,81:,0],unknown[:,81:,0]),
             "future_price_conditional_mean_oracle":oracle,"future_price_persistence":persistence_metric,
             "future_variance_full_noise_rmse":float(np.sqrt(np.mean((truth[:,after,1]-known[:,after,1])**2))),
             "negative_predicted_prices_fraction":float(np.mean(known[:,:,0]<0)),
             "negative_predicted_variances_fraction":float(np.mean(known[:,:,1]<0)),
             "terminal_price_distribution_wasserstein_1":float(wasserstein_distance(truth[:,-1,0],known[:,-1,0])),
             "terminal_price_mean_ground_truth":float(truth[:,-1,0].mean()),
             "terminal_price_mean_prediction":float(known[:,-1,0].mean()),
             "best_epoch":d["best_epoch"],"config":cfg}
        runs.append(run);groups[window].append((run,d,a))
    aggregates={}
    for window,values in groups.items():
        aggregates[str(window)]={name:mean_sd([r[name]["rmse"] for r,_,_ in values]) for name in
            ["future_price_full_noise","future_price_unknown_noise","future_price_conditional_mean_oracle","future_price_persistence",
             "shared_81_100_price_full_noise","shared_81_100_price_unknown_noise"]}
    write_json(OUT/"finance_summary.json",{"scope":"Heston extrapolation, paper F.5 and Figure 13",
             "claim":"independent implementation; published extrapolation accuracy not established",
             "window_aggregates":aggregates,"runs":runs,"full_horizon_supervised_controls":controls})
    with (OUT/"finance_summary.csv").open("w") as f:
        writer=csv.writer(f);writer.writerow(["train_steps","seeds","full_noise_price_rmse","seed_sd",
               "unknown_noise_price_rmse","conditional_mean_oracle_rmse","persistence_rmse","shared_81_100_full_noise_price_rmse"])
        for window,a in sorted(aggregates.items(),key=lambda x:int(x[0])):
            writer.writerow([window,a["future_price_full_noise"]["seeds"],a["future_price_full_noise"]["mean"],a["future_price_full_noise"]["seed_sd"],
                a["future_price_unknown_noise"]["mean"],a["future_price_conditional_mean_oracle"]["mean"],a["future_price_persistence"]["mean"],
                a["shared_81_100_price_full_noise"]["mean"]])

    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    styles=["-","--","-.",":","-"]
    markers=["o","s","^","d","x"]
    for curve_index,(window,values) in enumerate(sorted(groups.items())):
        for ax,field in zip(axes,["prediction","prediction_unknown_noise"]):
            curves=[]
            for _,_,a in values:
                curves.append(np.sqrt(np.mean((a[field][:,window+1:,0]-a["truth"][:,window+1:,0])**2,axis=0)))
            curves=np.stack(curves)
            t=a["time"][window+1:]
            ax.plot(t,curves.mean(0),linestyle=styles[curve_index%5],marker=markers[curve_index%5],
                    markersize=3,markevery=5,label=f"{window} training steps")
    for ax,title in zip(axes,["Future noise supplied (paper setting)","Future noise unknown (zero-filled)"]):
        ax.set(xlabel="Time",ylabel="Asset-price RMSE",title=title);ax.legend(fontsize=8)
    fig.tight_layout();save(fig,"finance_window_sensitivity")

    base=groups.get(75,[])
    if base:
        run,d,a=sorted(base,key=lambda x:x[0]["seed"])[0]
        cfg=d["config"];ntr=cfg["train_paths"];seed=d["seed"]
        rng=np.random.default_rng(seed+1000)
        dw=rng.normal(size=(ntr+cfg["validation_paths"]+cfg["test_paths"],cfg["steps"],2))*np.sqrt(cfg["dt"])
        # Show first paths, with no selection by prediction error.
        same_truth=heston(dw[:3],cfg["dt"],**cfg["heston"])
        same_prediction=predict(ROOT/"results"/run["run"]/"model.pt",{"dw":dw[:3]})
        fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
        for row,(truth,pred,title) in enumerate([(same_truth,same_prediction,"Same training noise paths"),
                                               (a["truth"][:3],a["prediction"][:3],"Independent test noise paths")]):
            for k,ax in enumerate(axes[row]):
                ax.plot(a["time"],truth[k,:,0],"-",label="Euler–Maruyama truth")
                ax.plot(a["time"],pred[k,:,0],"--",label="SDENO, full noise")
                ax.axvline(.75,color="gray",ls=":")
                ax.set(title=f"{title}, path {k+1}",xlabel="Time",ylabel="Asset price S(t)")
                if k==0:ax.legend(fontsize=7)
        fig.tight_layout();save(fig,"finance_price_paths")
        fig,axes=plt.subplots(1,2,figsize=(11,3.6))
        for field,label,ls in [("prediction","Full supplied noise","-"),("prediction_unknown_noise","Prefix noise only","--")]:
            curves=[]
            for _,_,array in base:
                curves.append(np.sqrt(np.mean((array[field][:,:,0]-array["truth"][:,:,0])**2,axis=0)))
            curves=np.stack(curves);axes[0].plot(a["time"],curves.mean(0),ls,label=label)
            axes[0].fill_between(a["time"],np.maximum(curves.mean(0)-curves.std(0,ddof=1),0),curves.mean(0)+curves.std(0,ddof=1),alpha=.12)
        conditional=[]
        for _,_,array in base:
            steps=np.arange(1,26);ref=array["truth"][:,75,0,None]*(1+cfg["heston"]["mu"]*cfg["dt"])**steps[None]
            conditional.append(np.sqrt(np.mean((ref-array["truth"][:,76:,0])**2,axis=0)))
        axes[0].plot(a["time"][76:],np.mean(conditional,0),":",color="black",label="Known-parameter conditional mean")
        axes[0].axvline(.75,color="gray",ls=":");axes[0].legend(fontsize=8)
        axes[0].set(xlabel="Time",ylabel="Asset-price RMSE",title="75-step training; three independent seeds")
        bins=np.linspace(min(a["truth"][:,-1,0].min(),a["prediction"][:,-1,0].min()),
                         max(a["truth"][:,-1,0].max(),a["prediction"][:,-1,0].max()),25)
        axes[1].hist(a["truth"][:,-1,0],bins=bins,alpha=.5,label="Truth")
        axes[1].hist(a["prediction"][:,-1,0],bins=bins,histtype="step",lw=2,label="SDENO")
        axes[1].set(xlabel="Terminal price S(1)",ylabel="Test paths",title="Seed 0, 200 test paths");axes[1].legend()
        fig.tight_layout();save(fig,"finance_forecast_diagnostics")

    tr=[]
    for window,a in sorted(aggregates.items(),key=lambda x:int(x[0])):
        x=a["future_price_full_noise"]
        dispersion=f" ± {x['seed_sd']:.4f}" if x["seed_sd"] is not None else " (one seed)"
        tr.append(f'<tr><td>{window}</td><td>{x["seeds"]}</td><td>{x["mean"]:.4f}{dispersion}</td>'
                  f'<td>{a["future_price_unknown_noise"]["mean"]:.4f}</td><td>{a["future_price_conditional_mean_oracle"]["mean"]:.4f}</td>'
                  f'<td>{a["shared_81_100_price_full_noise"]["mean"]:.4f}</td></tr>')
    control_text=""
    if controls:
        c=mean_sd([x["full_interval_test_price"]["rmse"] for x in controls])
        control_text=f'<p><strong>Full-horizon supervised control:</strong> the same Heston architecture trained on all 100 steps gives mean independent-test price RMSE {c["mean"]:.4f} across {c["seeds"]} seeds over t=0.01–1. This is a solution-operator control, not an extrapolation result. Its checkpoints and metrics are also retained.</p>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Heston finance reproduction</title>
<style>body{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172535}h1{font-size:32px}h2{margin-top:32px}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd5df;text-align:left}.note{background:#fff3d6;padding:16px;border-left:4px solid #ae7700}code{background:#eef2f7;padding:3px}a{color:#12629a}</style>
<h1>Heston: the finance-focused reproduction</h1>
<p>Expanding the Chaos — Shi et al., ICML 2026. Primary targets: Section 5.2 / Figure 6 and Appendix F.5 / Figure 13.</p>
<div class="note"><strong>Verdict:</strong> The financial extrapolation protocol has been implemented and run. This independent implementation does not reproduce the high-fidelity example shown in the paper’s Figure 6. The paper does not tabulate Heston errors, provide the underlying Figure 13 values, or specify its Heston parameters, and the official source repository currently contains only a README. These runs therefore cannot establish an exact numerical match.</div>
<h2>What was reproduced</h2><p>The simulator follows dS = μS dt + √V S dWˢ and dV = κ(θ−V)dt + ζ√V dWⱽ, with correlation ρ between price and variance increments. Each run uses 500 training paths, 100 Euler–Maruyama steps over T=1, MSE supervision on the initial training window, 2,000 AdamW epochs, and learning rate 0.001. We evaluate training windows of 50, 60, 70, 75, and 80 steps, with three seeds per window. Identical generated paths are held fixed across windows within each seed. Additional 100 validation and 200 independent test paths prevent selection using future outcomes.</p>
<p>The network encodes normalized Wick–Hermite features up to degree 2 from 12 cosine modes per independent Brownian driver. A separate time MLP predicts propagators; their inner product gives price and variance predictions. These architectural values and the Heston parameters are explicit local choices where the paper omits details. Joint price/variance supervision with separate training-prefix RMS scales, full-batch updates, and AdamW weight decay 0.01 are also local choices. The same noise prior is used throughout; the experiment is synthetic and is not calibrated to market data.</p>
<h2>Finance-specific results</h2><p>Table entries are asset-price RMSE over unseen times only. The ± value is sample SD across three seeds. Windows have different future intervals, so their averaged RMSEs are not directly comparable as identical-horizon measurements; compare the time curves at shared times below. No window is selected as optimal using test results.</p>
<table><tr><th>Training steps</th><th>Seeds</th><th>Full future noise supplied</th><th>Future noise unknown</th><th>Conditional-mean reference</th><th>Full noise: shared t=0.81–1</th></tr>'''+''.join(tr)+'''</table>
<p>The conditional-mean reference starts from the true observed S at the last supervised time and uses the known simulator drift: E[Sₙ₊ₕ | Sₙ] = Sₙ(1+μΔt)ʰ. It is an oracle with known parameters and observed state, not a trained competitor from the paper. A persistence forecast and variance errors are also stored in <a href="finance_summary.json">raw finance metrics</a>.</p>'''+control_text+'''
<h2>Future noise changes the task</h2><p>The paper supplies the complete future Brownian trajectory at evaluation. That evaluates a learned stochastic solution operator conditional on future noise. In a financial forecasting setting, future Brownian increments are unknown. We also evaluate zero-filled future increments, and compare that setting with the conditional-mean oracle. The latter cannot recover a particular realized future shock. Neither setting measures trading returns.</p>
<img src="finance_forecast_diagnostics.png" alt="Price RMSE by time and terminal-price histogram"><p>Shading shows one sample SD across seeds. In this implementation, changing from prefix-only features during training to full-path features at evaluation can also change predictions for past times. This sensitivity and the generic propagator network’s extrapolation are important unresolved differences from the unavailable author implementation. We do not attribute these results to the authors’ code.</p>
<h2>Price-path comparison</h2><p>The first three generated paths are shown without selection by error. The upper row uses the same noise paths used in training; the lower row uses independent test paths. Future noise is supplied in both rows. The vertical line marks t=0.75.</p><img src="finance_price_paths.png" alt="Ground truth and predicted Heston asset prices for the first three training and test paths">
<h2>Training-window sensitivity</h2><p>This follows the training-window procedure behind Appendix Figure 13 and applies it to Heston. The original caption does not identify which of the OU/Heston processes its plotted RMSE represents. It tests how the available time interval affects the same implemented architecture. It is an exploratory sensitivity study, not a numerical duplication of the original figure, whose underlying values are unavailable.</p><img src="finance_window_sensitivity.png" alt="Heston price RMSE for different training windows, with and without future noise">
<h2>Parameters, assumptions, and reproducibility</h2><p>Local parameters: μ=0.05, κ=2, θ=0.04, ζ=0.2, ρ=−0.7, S₀=1, V₀=0.04. Variance uses full-truncation Euler and is floored at zero. Drivers are independent before the simulator applies correlation; this preserves orthonormal Gaussian coordinates for Wick features. The two states are scaled using training-prefix RMS only. Time is the paper’s normalized simulator interval; no market horizon or instrument calibration is inferred.</p>
<p>The neural output does not enforce positive asset prices or nonnegative variances. Fractions of negative predictions are recorded per run in the finance JSON, alongside terminal-price distribution errors; the simulator itself applies variance truncation.</p>
<p>Use <code>.venv/bin/python scripts/run_finance.py</code> and <code>.venv/bin/python scripts/finance_report.py</code> to repeat the 15-run window study and three additional full-horizon controls. Run <code>scripts/verify.py</code> to compare regenerated inference with saved predictions and recompute metrics. Checkpoints, complete configs, all seed results, future-noise handling, data hashes, and measured parameter counts are preserved under <code>results/</code>. See <a href="finance_summary.csv">CSV</a>, <a href="finance_summary.json">JSON</a>, <a href="../verification.json">verification record</a>, <a href="index.html">full synthetic report and workflow references</a>, and <a href="../README.md">instructions</a>.</p>
<p>Sources: <a href="https://proceedings.mlr.press/v306/shi26i.html">published paper</a>, Section 5.2 and Appendix F.5; <a href="https://github.com/EEthanShi/SDENO">official repository</a>, commit 6b3973ddf6d27bd12adb7d83669bfe6e089a211f. Full source provenance and the Scientific Agent Skills workflow citation are included in the linked synthetic report. Computational report produced with AI assistance; accountable human scientific review is pending.</p></html>'''
    (OUT/"finance.html").write_text(page)
    print(OUT/"finance.html")


if __name__=="__main__":main()
