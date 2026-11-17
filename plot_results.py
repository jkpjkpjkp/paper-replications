"""Generate the Figure 3 and Figure 7 reproductions from computed returns."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"BASPE": "#984ea3", "BASPP": "#4daf4a", "BDRO": "#ff7f00",
          "DowJones": "#a65628", "Markowitz": "#f05bad"}
LABELS = {"BASPE": r"DRO-BAS$_{PE}$", "BASPP": r"DRO-BAS$_{PP}$", "BDRO": "BDRO"}


def plot_results(output):
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    weekly = pd.read_csv(output / "weekly_returns.csv")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180})
    dro = summary[summary.method.isin(LABELS)].copy()
    values = dro[["variance_weekly_return", "mean_weekly_return"]].to_numpy()
    efficient = [not np.any((values[:, 0] <= v) & (values[:, 1] >= m)
                            & ((values[:, 0] < v) | (values[:, 1] > m))) for v, m in values]
    dro["pareto_efficient"] = efficient
    dro.to_csv(output / "pareto.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5.7), layout="constrained")
    for method, marker in [("BASPE", "o"), ("BASPP", "s"), ("BDRO", "^")]:
        group = dro[dro.method == method].sort_values("radius_above_offset")
        ax.plot(group.variance_weekly_return, group.mean_weekly_return, ":",
                color=COLORS[method], label=LABELS[method])
        for efficient, points in group.groupby("pareto_efficient"):
            ax.scatter(points.variance_weekly_return, points.mean_weekly_return,
                       s=32, marker=marker, edgecolors=COLORS[method],
                       facecolors=COLORS[method] if efficient else "white", zorder=3)
        for radius in (1e-5, .001, 1.):
            point = group[np.isclose(group.radius_above_offset, radius)]
            if len(point):
                row = point.iloc[0]
                label = ("G + " if method == "BASPE" else "") + f"{radius:g}"
                offset = {"BASPE": (-60, 8), "BASPP": (7, -12), "BDRO": (8, 4)}[method]
                ax.annotate(label, (row.variance_weekly_return, row.mean_weekly_return),
                            xytext=offset, textcoords="offset points", fontsize=8,
                            color=COLORS[method])
    ax.set(xlabel="Out-of-sample weekly return variance",
           ylabel="Out-of-sample mean weekly return",
           title="Portfolio mean–variance tradeoff (Figure 3 reproduction)")
    ax.grid(alpha=.16)
    ax.margins(x=.16, y=.12)
    ax.legend(frameon=False, loc="upper left")
    fig.savefig(output / "figure3_mean_variance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    for method in ("BDRO", "BASPE", "BASPP"):
        for radius, style in [(1e-5, ":"), (.001, "--"), (1., "-")]:
            points = weekly[(weekly.method == method)
                            & np.isclose(weekly.radius_above_offset, radius)].sort_values("week")
            if len(points):
                epsilon = ("G + " if method == "BASPE" else "") + f"{radius:g}"
                ax.plot(points.week, points["return"].cumsum(), linestyle=style,
                        color=COLORS[method], lw=1.3, label=f"{LABELS[method]} ({epsilon})")
    for method in ("DowJones", "Markowitz"):
        points = weekly[weekly.method == method].sort_values("week")
        ax.plot(points.week, points["return"].cumsum(), color=COLORS[method],
                lw=1.3, label="Dow Jones index" if method == "DowJones" else method)
    ax.axvline(52, linestyle=":", color="black", lw=.8)
    ax.set(xlabel="Week index", ylabel="Sum of out-of-sample weekly returns",
           title="Portfolio cumulative returns (Figure 7 reproduction)", xlim=(0, None))
    ax.legend(frameon=True, fontsize=9, loc="upper left")
    ax.grid(alpha=.15)
    fig.savefig(output / "figure7_cumulative_returns.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    plot_results(parser.parse_args().output)
