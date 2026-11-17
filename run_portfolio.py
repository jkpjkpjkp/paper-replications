"""Run all 109 windows and 16 radii; checkpoint each completed window."""
import os
# Small dense matrices should not create a BLAS thread pool per worker.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.io import loadmat

from portfolio import (METHODS, RADII, fit_posterior, gaussian_objective,
                       predictive_objective, sample_covariances,
                       sample_predictive, solve_simplex)

ROOT = Path(__file__).resolve().parent


def load_data(data_dir):
    files = list(Path(data_dir).rglob("DowJones.mat"))
    if len(files) != 1:
        raise FileNotFoundError("Extract the Bruni mmc2 archive under data/ (see README.md)")
    raw = loadmat(files[0])
    returns = np.asarray(raw["Assets_Returns"], dtype=float)
    index = np.asarray(raw["Index_Returns"], dtype=float).ravel()
    if returns.shape != (1363, 28) or index.shape != (1363,):
        raise ValueError(f"Unexpected data dimensions: {returns.shape}, {index.shape}")
    if not np.isfinite(returns).all() or not np.isfinite(index).all():
        raise ValueError("Nonfinite input returns")
    baseline_file, = Path(data_dir).rglob("OutofSamplePortReturns_MeanVar_DowJones_List.txt")
    baseline = np.loadtxt(baseline_file).ravel()
    return returns, index, baseline, files[0]


def window_job(job):
    j, returns, radii, pp_count, bdro_count, checkpoint = job
    start = 12 * j
    train = returns[start:start + 52]
    test = returns[start + 52:start + 64]
    before = perf_counter()
    post = fit_posterior(train)
    posterior_seconds = perf_counter() - before
    samples = {}
    sample_seconds = {}
    for method in METHODS:
        before = perf_counter()
        if method == "BASPE":
            samples[method] = post.covariance_pe[None, :, :]
        elif method == "BASPP":
            samples[method] = sample_predictive(post, pp_count, j)
        else:
            samples[method] = sample_covariances(post, bdro_count, j)
        sample_seconds[method] = perf_counter() - before
    records = []
    for method in METHODS:
        initial = None
        for radius in sorted(radii, reverse=True):
            if method == "BASPP":
                objective = lambda x: predictive_objective(x, samples[method], radius)
            else:
                objective = lambda x: gaussian_objective(x, post.mean, samples[method], radius)
            result = solve_simplex(objective, returns.shape[1], initial)
            initial = result.pop("weights")
            record = {"window": j, "train_start": start, "train_stop": start + 52,
                      "test_start": start + 52, "test_stop": start + 64,
                      "method": method, "radius_above_offset": float(radius),
                      "epsilon": float(radius + post.offset()) if method == "BASPE" else float(radius),
                      "G": post.offset(), "G_upstream": post.offset(True),
                      "samples": 0 if method == "BASPE" else len(samples[method]),
                      "posterior_seconds": posterior_seconds,
                      "sampling_seconds": sample_seconds[method],
                      "weights": initial.tolist(), "returns": (test @ initial).tolist(), **result}
            records.append(record)
    target = Path(checkpoint)
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(records, indent=2) + "\n")
    temp.replace(target)
    return j


def aggregate(output, records, index, baseline, windows):
    frame = pd.DataFrame(records)
    summary = []
    weekly = []
    weights = []
    for (method, radius), group in frame.groupby(["method", "radius_above_offset"]):
        group = group.sort_values("window")
        values = np.concatenate(group["returns"].to_numpy())
        summary.append({"method": method, "radius_above_offset": radius,
                        "epsilon": group["epsilon"].iloc[0],
                        "mean_weekly_return": values.mean(),
                        "variance_weekly_return": values.var(ddof=1),
                        "sum_weekly_returns": values.sum(),
                        "n_oos_weeks": len(values), "max_optimality_gap": group["gap"].max()})
        weekly.extend({"method": method, "radius_above_offset": radius,
                       "week": 52 + k, "return": value} for k, value in enumerate(values))
    for method, values in [("DowJones", index[52:52 + 12 * windows]),
                           ("Markowitz", baseline[:12 * windows])]:
        summary.append({"method": method, "radius_above_offset": np.nan,
                        "epsilon": np.nan, "mean_weekly_return": values.mean(),
                        "variance_weekly_return": values.var(ddof=1),
                        "sum_weekly_returns": values.sum(), "n_oos_weeks": len(values),
                        "max_optimality_gap": np.nan})
        weekly.extend({"method": method, "radius_above_offset": np.nan,
                       "week": 52 + k, "return": value} for k, value in enumerate(values))
    for record in records:
        weights.append({"window": record["window"], "method": record["method"],
                        "radius_above_offset": record["radius_above_offset"],
                        **{f"asset_{i:02d}": w for i, w in enumerate(record["weights"])}})
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    pd.DataFrame(weekly).to_csv(output / "weekly_returns.csv", index=False)
    pd.DataFrame(weights).to_csv(output / "weights.csv", index=False)
    frame.drop(columns=["weights", "returns"]).to_csv(output / "diagnostics.csv", index=False)
    timing = frame.groupby("method").agg(
        solve_mean_seconds=("solve_seconds", "mean"), solve_std_seconds=("solve_seconds", "std"),
        sample_mean_seconds=("sampling_seconds", "mean"), sample_std_seconds=("sampling_seconds", "std"),
        posterior_mean_seconds=("posterior_seconds", "mean"), max_optimality_gap=("gap", "max"))
    timing.to_csv(output / "timing.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/full")
    parser.add_argument("--windows", type=int, default=109)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pp-samples", type=int, default=3600)
    parser.add_argument("--bdro-samples", type=int, default=900)
    parser.add_argument("--radii", type=float, nargs="+", default=RADII.tolist())
    args = parser.parse_args()
    if not 1 <= args.windows <= 109 or min(args.radii) <= 0:
        parser.error("Use 1..109 windows and positive radii")
    if args.pp_samples < 2 or args.bdro_samples < 2 or args.workers < 1:
        parser.error("Sample counts must be >=2 and workers >=1")
    args.radii = sorted(set(args.radii))
    returns, index, baseline, data_file = load_data(args.data_dir)
    output = args.output
    checkpoints = output / "windows"
    checkpoints.mkdir(parents=True, exist_ok=True)
    config = {"windows": args.windows, "radii": args.radii,
              "pp_samples": args.pp_samples, "bdro_samples": args.bdro_samples,
              "data_sha256": sha256(data_file.read_bytes()).hexdigest(),
              "implementation_sha256": sha256((ROOT / "portfolio.py").read_bytes()
                                               + Path(__file__).read_bytes()).hexdigest(),
              "seed": "window index, independently reset per method",
              "python": platform.python_version(),
              "packages": {p: importlib.metadata.version(p)
                           for p in ("numpy", "scipy", "cvxpy", "clarabel", "ecos", "matplotlib", "pandas")}}
    config_file = output / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError("Output contains a different configuration; choose a new --output")
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    started = perf_counter()
    jobs = [(j, returns, args.radii, args.pp_samples, args.bdro_samples,
             str(checkpoints / f"window_{j:03d}.json"))
            for j in range(args.windows) if not (checkpoints / f"window_{j:03d}.json").exists()]
    print(f"Running {len(jobs)} pending windows, {len(args.radii)} radii, 3 methods", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(window_job, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            j = future.result()
            print(f"Window {j:03d} complete ({completed}/{len(jobs)}), "
                  f"elapsed {perf_counter() - started:.1f}s", flush=True)
    records = []
    for j in range(args.windows):
        rows = json.loads((checkpoints / f"window_{j:03d}.json").read_text())
        if len(rows) != len(args.radii) * len(METHODS):
            raise ValueError(f"Incomplete window {j}")
        records.extend(rows)
    aggregate(output, records, index, baseline, args.windows)
    from plot_results import plot_results
    plot_results(output)
    print(f"Saved {len(records)} certified portfolios and figures to {output}", flush=True)


if __name__ == "__main__":
    main()
