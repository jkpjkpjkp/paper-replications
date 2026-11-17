#!/usr/bin/env python3
"""Run proposed methods and transparent baselines on the original datasets."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from stco.market import (DATASETS, load_dataset, backtest, ubah_backtest,
                         best_stock, bcrp_weights, fixed_weight_backtest, metrics)
from stco.solver import PSGAConfig


def slug(name):
    return {"NYSE(O)": "nyse-o", "NYSE(N)": "nyse-n"}.get(name, name.lower())


def run(args):
    output = ROOT/args.output
    output.mkdir(parents=True, exist_ok=True)
    (output/"runs").mkdir(exist_ok=True)
    cfg = PSGAConfig(k=args.k, penalty=args.penalty, accelerate=not args.direct)
    rates = sorted(set([0., .0025, .005] +
                        ([i/1000 for i in range(11)] if args.sweep else [])))
    rows, bcrp_info = [], {}
    started = time.time()
    for name in args.datasets:
        x = load_dataset(name)
        print(f"{name} {x.shape}", flush=True)
        best = best_stock(x)
        bcrp, kkt = bcrp_weights(x)
        bcrp_info[name] = {"weights": bcrp.tolist(), "kkt_residual": kkt}
        for gamma in rates:
            market = ubah_backtest(x, gamma)
            for strategy in ["UBAH", "BEST", "BCRP", "STCO1", "STCO2"]:
                windows = args.windows if strategy == "STCO2" else [args.windows[0]]
                for window in windows:
                    tick = time.time()
                    if strategy == "UBAH":
                        result = market
                    elif strategy == "BEST":
                        result = fixed_weight_backtest(x, best, gamma)
                    elif strategy == "BCRP":
                        result = fixed_weight_backtest(x, bcrp, gamma)
                    else:
                        result = backtest(x, gamma, int(strategy[-1]), window, cfg)
                    row = {"dataset": name, "strategy": strategy, "gamma": gamma,
                           "window": window if strategy == "STCO2" else 0,
                           **metrics(result, market), "seconds": time.time()-tick}
                    if strategy.startswith("STCO"):
                        row.update({"not_converged": int((~result["converged"]).sum()),
                                    "max_iterations": int(result["iterations"].max()),
                                    "total_iterations": int(result["iterations"].sum()),
                                    "total_prox_evaluations": int(result["prox_evaluations"].sum()),
                                    "days_over_k_after_initial": int(np.sum(
                                        (result["weights"][1:] > 1e-8).sum(axis=1) > cfg.k))})
                    rows.append(row)
                    # Save complete daily trajectories for table rates.
                    if gamma in [0., .0025, .005]:
                        filename = f"{slug(name)}_{strategy}_g{gamma:.4f}_w{row['window']}.npz"
                        np.savez_compressed(output/"runs"/filename, **result)
                    print(f"  {strategy:5s} w={row['window']} gamma={gamma:.4f} "
                          f"S={row['wealth']:.8g} support={row['support_mean']:.4f} "
                          f"fail={row.get('not_converged', 0)} {row['seconds']:.2f}s", flush=True)
                    pd.DataFrame(rows).to_csv(output/"metrics.csv", index=False)
    df = pd.DataFrame(rows)
    targets = pd.read_csv(ROOT/"sources/paper_targets.csv")
    main = df[(df.strategy != "STCO2") | (df.window == args.windows[0])]
    comparison = targets.merge(main, on=["dataset", "strategy", "gamma"], how="inner")
    comparison["wealth_ratio_to_paper"] = comparison.wealth/comparison.paper_wealth
    comparison["wealth_relative_error"] = comparison.wealth_ratio_to_paper-1
    comparison["within_1pct"] = comparison.wealth_relative_error.abs() <= .01
    comparison.to_csv(output/"paper_comparison.csv", index=False)
    metadata = {"algorithm": "Equation (11) PSGA with exact scalar-dual prox",
                "config": asdict(cfg), "windows": args.windows, "rates": rates,
                "python": platform.python_version(), "platform": platform.platform(),
                "elapsed_seconds": time.time()-started, "bcrp": bcrp_info,
                "upstream_commit": "a2e04730cbdfb6ccb0017456d4f835a4d7446ec3",
                "datasets": {name: {"shape": list(load_dataset(name).shape),
                    "sha256": hashlib.sha256((ROOT/"upstream/STCO-code"/DATASETS[name]).read_bytes()).hexdigest()}
                    for name in args.datasets},
                "paper_sha256": hashlib.sha256((ROOT/"sources/paper.pdf").read_bytes()).hexdigest()}
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Finished in {time.time()-started:.1f}s; {comparison.within_1pct.sum()}"
          f"/{len(comparison)} proposed-method wealth cells within 1% of paper.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--windows", nargs="+", type=int, default=[4])
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--penalty", type=float, default=8e-6)
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--direct", action="store_true", help="disable algebraic event skipping")
    p.add_argument("--output", default="results/main")
    run(p.parse_args())
