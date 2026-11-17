"""Generate standalone scientific figures from this run's numeric outputs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/figures"
OUT.mkdir(exist_ok=True)
COLORS = {"UP":"#d33b32", "KT":"#346fc2", "DtACI":"#23875d", "SFOGD":"#a27709",
          "P":"#8057ad", "PI-reconstructed":"#888888", "UP-paper":"#dc7a2c",
          "UP-current":"#656565", "UP-corrected":"#102c4a"}
LABELS = {"UP":"UP (author experiment)", "KT":"KT", "DtACI":"DtACI", "SFOGD":"SF-OGD",
          "P":"P control", "PI-reconstructed":"PI (reconstructed)",
          "UP-paper":"UP (Algorithm 1)", "UP-current":"UP (current default)",
          "UP-corrected":"UP (alpha correction)"}
METHODS = ["UP", "KT", "DtACI", "SFOGD", "P"]
plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                     "axes.grid":True, "grid.alpha":0.18, "savefig.dpi":180})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def tradeoff(ax, sweep, dataset):
    for method in METHODS:
        d = sweep[(sweep.dataset == dataset) & (sweep.method == method)].sort_values("alpha")
        x = d.mean_width.to_numpy(copy=True)
        y = d.coverage.to_numpy()
        x[~np.isfinite(x)] = np.nan
        ax.plot(x, y, color=COLORS[method], label=LABELS[method], lw=1.8)
        if dataset == "sinusoid":
            ax.errorbar(x[::7], y[::7], xerr=d.mean_width_sem.to_numpy()[::7],
                        yerr=d.coverage_sem.to_numpy()[::7], fmt="none", alpha=.5,
                        color=COLORS[method], capsize=2, lw=.8)
    ax.set(xlabel="Mean interval width", ylabel="Realized coverage", title=dataset.upper() + ": coverage vs. size")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.axhline(.95, color="black", ls=":", lw=1)
    ax.text(.02,.03,"Infinite mean widths are outside this plot.", transform=ax.transAxes, fontsize=8)


def main():
    sweep = pd.read_csv(ROOT / "results/sweep_summary.csv")
    summary = pd.read_csv(ROOT / "results/summary.csv")
    fig, axs = plt.subplots(2,2,figsize=(13,9), layout="constrained")
    tradeoff(axs[0,0], sweep, "AXP")
    tradeoff(axs[0,1], sweep, "sinusoid")
    for method in METHODS:
        d = sweep[(sweep.dataset == "AXP") & (sweep.method == method)]
        axs[1,0].plot(1-d.alpha,d.coverage,label=LABELS[method], color=COLORS[method])
    axs[1,0].plot([.75,.95],[.75,.95],"k--",lw=1)
    axs[1,0].set(xlabel="Target coverage",ylabel="Realized coverage",title="AXP: target tracking")
    axs[1,0].xaxis.set_major_formatter(PercentFormatter(1))
    axs[1,0].yaxis.set_major_formatter(PercentFormatter(1))
    trace = np.load(ROOT / "results/traces/sinusoid.npz")
    b = int(trace["burnin"])
    t = np.arange(2000,2501)
    axs[1,1].plot(t,trace["scores"][:,t-1].mean(axis=0),color="black",lw=1,label="Score")
    for method in ["UP","KT","DtACI","P"]:
        radius = trace[method][:,t-1-b].mean(axis=0)
        radius[~np.isfinite(radius)] = np.nan
        axs[1,1].plot(t,radius,color=COLORS[method],label=LABELS[method],lw=1.3)
    axs[1,1].set(xlabel="Time t",ylabel="Radius (half-width)", title="Sinusoid: mean of 10 independent runs")
    axs[1,1].text(.02,.94,"DtACI gaps indicate infinite radii.",transform=axs[1,1].transAxes, fontsize=8)
    axs[0,0].legend(fontsize=8,loc="best")
    axs[1,1].legend(fontsize=8,loc="upper right")
    fig.suptitle("UP-OCP reproduction: main experiments\n50 target levels; uncertainty bars are one standard error",fontsize=14)
    save(fig,"main_results")

    fig, axs = plt.subplots(2,1,figsize=(12,7),layout="constrained",sharex=True)
    tr = np.load(ROOT / "results/traces/AXP.npz")
    b = int(tr["burnin"]); offset = int(tr["eval_offset"])
    dates = pd.to_datetime(np.load(ROOT / "data/scores/AXP.npz")["dates"])[b+offset:]
    s = tr["scores"][0,b+offset:]
    for method in ["UP","UP-corrected","KT","SFOGD"]:
        q = tr[method][0,offset:]
        local = pd.Series(s<=q).rolling(100,min_periods=100).mean()
        axs[0].plot(dates,local,color=COLORS[method],lw=1,label=LABELS[method])
        axs[1].plot(dates,2*np.maximum(q,0),color=COLORS[method],lw=.65,alpha=.8)
    axs[0].axhline(.95,color="black",ls="--",lw=1)
    axs[0].yaxis.set_major_formatter(PercentFormatter(1))
    axs[0].set(ylabel="Coverage over preceding 100 days",title="AXP: local coverage and interval width")
    axs[1].set(ylabel="Interval width (price units)",xlabel="Original trading dates")
    axs[0].legend(ncol=4,fontsize=9)
    save(fig,"axp_dynamics")

    fig, axs = plt.subplots(2,4,figsize=(15,7),layout="constrained")
    datasets = ["AXP","AAPL","AMZN","GOOGL","elec2","sinusoid","stationary","quadratic"]
    for ax,name in zip(axs.flat,datasets):
        d = summary[summary.dataset==name].set_index("method")
        methods = ["UP","UP-corrected","KT","DtACI","SFOGD","P"]
        for i,m in enumerate(methods):
            r = d.loc[m]
            ax.errorbar(i,r.coverage,yerr=r.coverage_sem if np.isfinite(r.coverage_sem) else 0,
                        fmt="o",color=COLORS[m],capsize=3)
        ax.axhline(.95,color="black",ls="--",lw=1)
        ax.set(xticks=range(len(methods)),xticklabels=["UP","UP+c","KT","DtACI","SF","P"],title=name,ylim=(.89,.98))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("Nominal 95% coverage across all datasets\nUP+c uses alpha - 1/sqrt(T); synthetic runs: 10 seeds, one standard error",fontsize=14)
    save(fig,"all_datasets_coverage")

    fig, axs = plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    d = summary[summary.dataset=="AXP"].set_index("method")
    variants = ["UP","UP-paper","UP-current","UP-corrected"]
    for ax,key,title in zip(axs,["coverage","mean_width"],["Realized coverage","Mean interval width"]):
        for i,m in enumerate(variants):
            v=d.loc[m,key]
            ax.bar(i,v,color=COLORS[m],width=.65)
            ax.text(i,v,f"{v:.3%}" if key=="coverage" else f"{v:.3f}",ha="center",va="bottom",fontsize=9)
        ax.set(xticks=range(4),xticklabels=["Experimental\nuncorrected","Literal\nAlgorithm 1","Current default\nalpha - 0.015","Correction\nalpha - 1/sqrt(T)"],title=title)
        if key=="coverage":
            ax.set_ylim(.9,.97)
            ax.axhline(.95,color="black",ls="--",lw=1)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        else:
            ax.set_ylim(0,19)
    fig.suptitle("AXP: distinguish the four UP variants",fontsize=14)
    save(fig,"up_variants")
    print(f"Wrote PNG and SVG figures to {OUT}")


if __name__ == "__main__":
    main()
