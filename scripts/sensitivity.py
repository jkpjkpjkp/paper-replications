#!/usr/bin/env python3
"""Re-run the proposed-method parameter settings of paper Tables 2 and 3."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from stco.market import load_dataset, backtest, ubah_backtest, metrics
from stco.solver import PSGAConfig


def main():
    output = ROOT/"results/sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(ROOT/"results/main/metrics.csv")
    cfg = PSGAConfig()
    cases = []
    k_targets = {("TSE", 0., 1): 1216.92, ("TSE", 0., 2): 95.92,
                 ("MSCI", 0., 1): 11.48, ("MSCI", 0., 2): 15.17,
                 ("TSE", .0025, 1): 5.60, ("TSE", .0025, 2): 3.97,
                 ("MSCI", .0025, 1): 2.65, ("MSCI", .0025, 2): 2.05}
    for name in ["TSE", "MSCI"]:
        for rate in [0., .0025]:
            for k in [10, 15]:
                for strategy in [1, 2]:
                    cases.append({"paper_table": 2, "dataset": name, "gamma": rate,
                                  "k": k, "penalty": cfg.penalty, "strategy": f"STCO{strategy}",
                                  "paper_wealth": k_targets[name, rate, strategy]})
    for name, rate in [("NYSE(N)", .0025), ("TSE", 0.)]:
        for c in [2e-6, 4e-6, 8e-6, 1.6e-5]:
            for strategy in [1, 2]:
                paper = (274.15 if c == 2e-6 else 274.18) if strategy == 1 else 6e5
                if name == "TSE":
                    paper = 1216.92 if strategy == 1 else 95.92
                cases.append({"paper_table": 3, "dataset": name, "gamma": rate,
                              "k": cfg.k, "penalty": c, "strategy": f"STCO{strategy}",
                              "paper_wealth": paper})
    rows, new_runs = [], 0
    started = time.time()
    for case in cases:
        name, rate, strategy = case["dataset"], case["gamma"], case["strategy"]
        if case["k"] == cfg.k and case["penalty"] == cfg.penalty:
            selection = baseline[(baseline.dataset == name) & np.isclose(baseline.gamma, rate)
                                 & (baseline.strategy == strategy)
                                 & (baseline.window == (4 if strategy == "STCO2" else 0))].iloc[0]
            result_metrics = {key: selection[key] for key in
                              ["wealth", "support_mean", "support_std", "simplex_error", "not_converged"]}
            reused = True
        else:
            x = load_dataset(name)
            run_cfg = replace(cfg, k=case["k"], penalty=case["penalty"])
            result = backtest(x, rate, int(strategy[-1]), 4, run_cfg)
            result_metrics = metrics(result, ubah_backtest(x, rate))
            result_metrics["not_converged"] = int((~result["converged"]).sum())
            reused = False
            new_runs += 1
        row = {**case, **result_metrics, "window": 4 if strategy == "STCO2" else 0,
               "reused_main_run": reused}
        row["relative_wealth_error"] = row["wealth"]/row["paper_wealth"]-1
        rows.append(row)
        pd.DataFrame(rows).to_csv(output/"paper_comparison.csv", index=False)
        print(f"Table {case['paper_table']} {name} {strategy} gamma={rate} "
              f"K={case['k']} c={case['penalty']:g} S={row['wealth']:.8g}", flush=True)
    summary = {"table_cells": len(rows), "new_backtests": new_runs,
               "reused_main_cells": len(rows)-new_runs, "elapsed_seconds": time.time()-started,
               "window_assumption_for_STCO2": 4,
               "not_converged": int(sum(r["not_converged"] for r in rows))}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
