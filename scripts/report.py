#!/usr/bin/env python3
"""Generate figures and an auditable HTML report directly from saved results."""
from collections import defaultdict
import csv
import html
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chaos.basis import cosine_basis

OUT=ROOT/"report"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":150,"savefig.bbox":"tight"})


def save(fig,name):
    fig.savefig(OUT/f"{name}.png")
    fig.savefig(OUT/f"{name}.svg")
    plt.close(fig)


def grouped_stats(rows,key):
    values=np.array([key(d) for d in rows],float)
    return {"mean":float(values.mean()),"seed_sd":float(values.std(ddof=1)) if len(values)>1 else None,
            "seeds":len(values),"values":values.tolist()}


def main():
    records=[]
    for path in sorted((ROOT/"results").glob("*/metrics.json")):
        d=json.loads(path.read_text());d["run"]=path.parent.name
        records.append(d)
    groups=defaultdict(list)
    for d in records:
        key=d.get("label",d["experiment"])
        if d["experiment"]=="phi4":key=f"phi4_{d['condition']}_M{d['config']['hermite_order']}"
        groups[key].append(d)
    aggregate={}
    table=[]
    for name,rows in groups.items():
        if rows[0]["experiment"]=="brownian":continue
        if rows[0]["experiment"]=="phi4":
            stat=grouped_stats(rows,lambda d:d["test"]["relative_l2_mean"])
            aggregate[name]={"mean_test_relative_l2":stat}
            reported=0.012 if rows[0]["condition"]=="fixed" else 0.015
            table.append([name,stat["seeds"],stat["mean"],stat["seed_sd"],reported,"relative L2"])
        else:
            stat=grouped_stats(rows,lambda d:d["future_full_noise"]["rmse"])
            unknown=grouped_stats(rows,lambda d:d["future_unknown_noise"]["rmse"])
            aggregate[name]={"test_rmse":stat,"unknown_noise_rmse":unknown}
            table.append([name,stat["seeds"],stat["mean"],stat["seed_sd"],None,
                          "full-interval RMSE" if "full_horizon" in name else "future RMSE"])
    (OUT/"summary.json").write_text(json.dumps(aggregate,indent=2)+"\n")
    with (OUT/"summary.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["experiment","seeds","mean","seed_sd","paper_value","metric"]);w.writerows(table)

    brown=groups.get("brownian",[])
    if brown:
        b=brown[0];arr=np.load(ROOT/"results"/b["run"]/"predictions.npz")
        fig,axes=plt.subplots(1,2,figsize=(11,3.6))
        axes[0].plot(arr["time"],arr["truth"][0],color="black",lw=1,label="Original Brownian path")
        for k,ls in [(4,"--"),(16,"-."),(64,":")]:
            axes[0].plot(arr["time"],arr[f"prediction_{k}"][0],ls=ls,label=f"{k} modes")
        axes[0].set(xlabel="Time",ylabel="W(t)",title="Held fixed: first generated path")
        axes[0].legend(fontsize=8)
        axes[1].loglog([r["modes"] for r in b["results"]],[r["rmse"] for r in b["results"]],"o-")
        axes[1].set(xlabel="Retained Gaussian coordinates",ylabel="RMSE across 1,000 paths",title="Exact reconstruction at full discrete rank")
        fig.tight_layout();save(fig,"brownian")

    fig,axes=plt.subplots(1,2,figsize=(11,3.6))
    for ax,kind in zip(axes,["ou","heston"]):
        if kind not in groups:continue
        for field,label,ls in [("prediction","Full supplied noise","-"),("prediction_unknown_noise","Prefix noise only","--")]:
            curves=[]
            for d in groups[kind]:
                a=np.load(ROOT/"results"/d["run"]/"predictions.npz")
                curves.append(np.sqrt(np.mean((a[field][:,:,0]-a["truth"][:,:,0])**2,axis=0)))
            curves=np.stack(curves)
            ax.plot(a["time"],curves.mean(0),ls,label=label)
            if len(curves)>1:
                ax.fill_between(a["time"],np.maximum(curves.mean(0)-curves.std(0,ddof=1),0),
                                curves.mean(0)+curves.std(0,ddof=1),alpha=.12)
        if kind=="ou":
            d=groups[kind][0];a=np.load(ROOT/"results"/d["run"]/"predictions.npz")
            ax.plot(a["time"],np.sqrt(np.mean((a["oracle"][:,:,0]-a["truth"][:,:,0])**2,0)),":",color="black",label="Exact 32-mode OU reference")
        ax.axvline(.75,color="gray",ls=":")
        ax.set(xlabel="Time",ylabel="State RMSE" if kind=="ou" else "Price RMSE",title=f"{kind.upper()}: independent test paths")
        ax.legend(fontsize=8)
    fig.tight_layout();save(fig,"sde_errors")

    phi=[d for d in records if d["experiment"]=="phi4"]
    if phi:
        fig,axes=plt.subplots(1,2,figsize=(11,3.6))
        for ax,condition in zip(axes,["fixed","random"]):
            for d in phi:
                if d["condition"]!=condition:continue
                hist=d["history"]
                ax.semilogy([h["epoch"] for h in hist],[h["val_mse_normalized"] for h in hist],
                            label=f"M={d['config']['hermite_order']}, seed={d['seed']}")
            ax.set(xlabel="Epoch",ylabel="Validation normalized MSE",title=f"Phi4: {condition} initial condition")
            ax.legend(fontsize=7)
        fig.tight_layout();save(fig,"phi4_learning")
        chosen=sorted(phi,key=lambda d:(d["seed"]!=0,d["condition"]!="fixed",d["config"]["hermite_order"]!=2))[0]
        a=np.load(ROOT/"results"/chosen["run"]/"predictions.npz")
        truth,pred=a["truth"][0],a["prediction"][0]
        fig,axes=plt.subplots(1,3,figsize=(12,3.2),layout="constrained")
        v=float(np.max(np.abs(truth)))
        for ax,z,title in zip(axes,[truth,pred,pred-truth],["Ground truth","F-SPDENO","Signed error"]):
            im=ax.imshow(z,aspect="auto",origin="lower",extent=[0,1,.001,.05],cmap="RdBu_r",vmin=-v,vmax=v)
            ax.set(xlabel="Space",ylabel="Time",title=title)
        fig.colorbar(im,ax=list(axes),label="Solution value (common scale)")
        fig.suptitle(f"First test path, {chosen['run']}; no path selection by error",fontsize=10)
        save(fig,"phi4_example")

    def fmt(x):return "unreported" if x is None else f"{x:.5g}"
    rows_html="".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in
                     [name,n,fmt(mean),fmt(sd),fmt(paper),metric])+"</tr>" for name,n,mean,sd,paper,metric in table)
    links="".join(f'<li><a href="../results/{d["run"]}/metrics.json">{d["run"]}: raw metrics</a></li>' for d in records)
    # This is a computational report, not a submission-ready scientific manuscript.
    page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Expanding the Chaos — Reproduction</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;max-width:1120px;margin:45px auto;padding:0 24px;color:#172535}}h1{{font-size:32px}}h2{{margin-top:35px}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #d6dde5;text-align:left}}img{{max-width:100%;height:auto}}code{{background:#eef2f7;padding:3px}}.status{{background:#fff3d6;border-left:4px solid #ae7700;padding:16px}}a{{color:#12629a}}</style>
<h1>Expanding the Chaos: independent reproduction</h1>
<p>Shi et al., ICML 2026. Computational report generated from local checkpoints and measured results. Sources were inspected by the coding agent; accountable human scientific review is pending.</p>
<div class="status"><strong>Scope and verdict:</strong> Brownian reconstruction, OU and Heston SDEs, and the 1D stochastic Phi4 model are implemented and executed. These experiments do not establish reproduction of the paper’s complete benchmark suite or published accuracy. Numerical differences are reported below; no paper number is substituted for a measurement.</div>
<h2>Measured results</h2><p>Each run uses independent training, validation, and test paths. The checkpoint is selected using validation loss. SD is across random seeds; a single-run SD is unreported. Phi4 values average per-path relative L2 over positive-time spatial trajectories. Paper values are Table 1 at N=1,000, with methodological differences listed below. OU full-horizon is a supervised control over the entire interval and is not an extrapolation result. Heston RMSE in the table pools price and variance; both state-specific RMSEs are retained in raw metrics.</p>
<table><thead><tr><th>Experiment</th><th>Seeds</th><th>Measured mean</th><th>Seed SD</th><th>Paper target</th><th>Metric</th></tr></thead><tbody>{rows_html}</tbody></table>
<h2>Brownian reconstruction</h2><p>Orthogonal projections of Brownian increments onto a discrete cosine basis reconstruct the original discrete Brownian path at full rank. Normalized probabilists’ Hermite products include mixed terms. This checks the construction in Section 3, rather than claiming an exact duplication of Figure 1’s unspecified basis and random realization.</p><img src="brownian.png" alt="Brownian path reconstructions and RMSE against number of retained modes">
<h2>SDE extrapolation and controls</h2><p>Appendix F.5 specifies 500 paths, 100 Euler–Maruyama steps, supervision through step 75, 2,000 AdamW epochs at learning rate 0.001. Those counts are retained. Only increments up to step 75 enter training features. Evaluation supplies all 100 noise increments, as described in the paper, and also evaluates zero-filled unknown future increments. A fixed temporal basis is used for both intervals. No future states enter training or checkpoint selection.</p>
<p>The two input settings answer different questions: the full-path setting evaluates a solution operator conditional on future randomness, while the prefix-only setting cannot recover a particular realized future path. For OU, the exact Euler–Maruyama first-chaos coefficients separate neural approximation error from projection error. Higher chaos coefficients for additive OU are exactly zero (Appendix C.3).</p>
<p>The implemented generic MLP can produce sensitivity to future coordinates even at earlier times. Its training covariance on prefix-projected coordinates is ill-conditioned, whereas full-horizon Gaussian coordinates have identity covariance. Eigenvalues are recorded per run. This is a diagnosis of this implementation’s training/representation choice; it does not establish the behavior of the authors’ unavailable code. Same-training-path evaluations are also stored, because the paper describes time extrapolation without an explicit independent path split.</p>
<p>Shaded bands show plus/minus one sample standard deviation across seeds. The vertical line marks the last supervised time.</p><img src="sde_errors.png" alt="OU and Heston errors versus time for full and prefix noise, with training cutoff at 0.75">
<h2>SPDE training and examples</h2><p>The Phi4 experiment uses 1,000 training paths, 100 additional validation paths, 200 test paths, dx=1/128, dt=0.001, T=0.05, noise intensity 0.1, Adam at 0.001, batch size 64, and 200 epochs. The periodic finite-difference diffusion solve is implicit, including its action on noise. Drift is explicit. The first 128 spatial nodes are retained, without a duplicated periodic endpoint. A linearized solver supplied with the same driving noise is an additional reference, not a neural baseline from the paper.</p>
<img src="phi4_learning.png" alt="Validation learning curves for both initial-condition settings"><img src="phi4_example.png" alt="Ground truth, learned solution, and signed error on first held-out SPDE path">
<h2>Explicit implementation choices and limitations</h2><ul>
<li>The official SDENO repository contained only README.md at commit 6b3973ddf6d27bd12adb7d83669bfe6e089a211f. This code is an independent implementation, not the authors’ code.</li>
<li>Phi4 FNO: six layers, width 24, 12 Fourier modes, 64 sampled continuous Haar output functions. Width and Fourier modes are local choices because Appendix F.1 omits them. The output basis has rank 50 on the 50 positive-time samples; 64 output channels are therefore redundant.</li>
<li>Phi4 noise coordinates: all 50 orthonormal discrete Haar projections, standardized by known pointwise noise variance. Complete degree-2 products at each spatial node are used, including all mixed temporal coordinates. These are pointwise temporal features, not a complete tensor-product chaos over all spatial KL modes. The paper does not give its implemented channel/index ordering. The M=1 runs are feature ablations, not a separate published baseline.</li>
<li>The spatial noise follows the cited code’s sqrt(2) sin(j pi x) prescription, j=0,...,128. This sine prescription is not a translation-invariant Fourier covariance on the torus. The cited generator has a duplicated endpoint and a different endpoint matrix; this implementation uses a standard 128-node periodic Laplacian.</li>
<li>For random initial conditions, eight periodic sine modes with Gaussian amplitudes proportional to 0.3/k^2 are a local choice. Appendix F.1 does not specify its initial-field amplitude/decay values.</li>
<li>OU parameters are theta=2, mu=1, sigma=0.3, x0=0; Heston parameters are mu=0.05, kappa=2, theta=0.04, zeta=0.2, rho=-0.7, S0=1, V0=0.04. Appendix F.5 does not supply them. Heston uses full-truncation Euler and independent coordinates before correlating the driving increments. Variance is a second supervised state, and per-state training scales are fitted only on training prefixes.</li>
<li>OU uses 32 cosine noise modes and degree 1. Heston uses 12 modes per independent driver and complete degree 2, encoded by an MLP, then contracted with a time MLP. Network width is 128. These unreported hyperparameters are local choices. No positional encoding or dynamics constraint is imposed.</li>
<li>Reported inference timing excludes construction of Wick features and comes from a shared RTX 4060; it is not comparable to the paper’s H200 timings. Complex weights are counted as two real parameters.</li>
<li>Not executed: 2D Navier–Stokes, diffusion images/FID, fMRI graph interpolation, meta-parameter estimation, manifold flood likelihoods, the N=10,000 Phi4 setting, and the other paper baseline architectures. No claims are made for their results.</li>
</ul>
<h2>Reproducibility</h2><p>Configurations, exact dependency versions, dataset hashes, environments, validation histories, selected checkpoint epochs, model weights, predictions, and test metrics are retained. See <a href="../README.md">README</a>, <a href="summary.json">aggregate JSON</a>, <a href="summary.csv">aggregate CSV</a>, <a href="../sources/manifest.json">source manifest</a>, and <a href="../verification.json">checkpoint verification</a>.</p><ul>{links}</ul>
<h2>Sources and provenance</h2><ol>
<li><a href="https://proceedings.mlr.press/v306/shi26i.html">Shi et al., published paper record</a>; Sections 3–5, Appendix C.3, D, E, F.1, F.5. Local PDF SHA-256: d533ef4802d02725d7137828bfd378a4309834802a2fd52ee2693d78fd0f7e36.</li>
<li><a href="https://github.com/EEthanShi/SDENO">Official SDENO repository</a>, inspected 2026-11-06.</li>
<li><a href="https://github.com/crispitagorico/torchspde">Cited torchspde reference</a>; data/general_solver.ipynb, data/Noise.ipynb, data/data_generation_ginzburg_landau.ipynb.</li>
<li>Workflow assistance: Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026). <a href="https://doi.org/10.48550/arXiv.2609.00065">Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents</a>. Metadata checked against arXiv v2 (2026-09-02); scientific-writing and scientific-critical-thinking procedures informed provenance and limitations reporting.</li>
</ol><p>Implementation and analysis produced with AI assistance. This computational report has not been approved as a scientific manuscript.</p></html>'''
    (OUT/"index.html").write_text(page)
    print(OUT/"index.html")


if __name__=="__main__":main()
