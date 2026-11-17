"""Audit the full run, numerical reference outputs, and independent conic solves."""
import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio import (fit_posterior, sample_covariances, sample_predictive,
                       solve_conic, gaussian_objective, predictive_objective)
from run_portfolio import ROOT, load_data


def validate(output, conic=True):
    config = json.loads((output / "config.json").read_text())
    records = [row for j in range(config["windows"])
               for row in json.loads((output / "windows" / f"window_{j:03d}.json").read_text())]
    returns, index, baseline, _ = load_data(ROOT / "data")
    weights = np.array([row["weights"] for row in records])
    errors = []
    for row in records:
        j = row["window"]
        assert (row["train_start"], row["train_stop"], row["test_start"], row["test_stop"]) == (
            j*12, j*12+52, j*12+52, j*12+64)
        errors.append(np.max(np.abs(returns[j*12+52:j*12+64] @ row["weights"] - row["returns"])))
    assert len(records) == config["windows"] * len(config["radii"]) * 3
    assert weights.min() >= -1e-10 and np.max(np.abs(weights.sum(axis=1)-1)) < 1e-10
    assert max(row["gap"] for row in records) <= 1e-7
    baseline_weights = np.loadtxt(ROOT / "data/Solutions/DowJones/OptPortfolios_MeanVar_DowJones.txt").T
    baseline_check = np.concatenate([returns[j*12+52:j*12+64] @ baseline_weights[j]
                                     for j in range(config["windows"])])
    baseline_error = float(np.max(np.abs(baseline_check - baseline[:len(baseline_check)])))
    assert baseline_error < 1e-10
    summary = pd.read_csv(output / "summary.csv")
    report = {"portfolios": len(records), "windows": config["windows"],
              "oos_weeks": config["windows"]*12, "minimum_weight": float(weights.min()),
              "max_weight_sum_error": float(np.max(np.abs(weights.sum(axis=1)-1))),
              "max_return_recalculation_error": float(max(errors)),
              "max_optimality_gap": max(row["gap"] for row in records),
              "max_markowitz_return_recalculation_error": baseline_error}
    if config["windows"] == 109:
        reference = json.loads((ROOT / "data/authors_reference.json").read_text())
        comparison = summary.merge(pd.DataFrame(reference["aggregate"]),
                                   on=["method", "radius_above_offset"], suffixes=("", "_authors"))
        for metric in ("mean_weekly_return", "variance_weekly_return"):
            comparison[f"{metric}_error"] = comparison[metric] - comparison[f"{metric}_authors"]
            report[f"max_abs_{metric}_difference_vs_authors"] = float(comparison[f"{metric}_error"].abs().max())
        comparison.to_csv(output / "authors_comparison.csv", index=False)
        endpoints = summary.merge(pd.DataFrame(reference["cumulative"]),
                                  on=["method", "radius_above_offset"], suffixes=("", "_authors"))
        endpoints["absolute_difference"] = endpoints.sum_weekly_returns - endpoints.sum_weekly_returns_authors
        endpoints["relative_difference_percent"] = 100 * endpoints.absolute_difference / endpoints.sum_weekly_returns_authors
        endpoints.to_csv(output / "authors_endpoints_comparison.csv", index=False)
        report["max_abs_endpoint_relative_difference_percent"] = float(endpoints.relative_difference_percent.abs().max())
    conic_records = []
    if conic:
        for j in sorted(set([0, config["windows"]//2, config["windows"]-1])):
            post = fit_posterior(returns[j*12:j*12+52])
            covs = sample_covariances(post, config["bdro_samples"], j)
            samples = sample_predictive(post, config["pp_samples"], j)
            for row in records:
                if row["window"] != j or row["radius_above_offset"] not in (1e-5, .001, 1.):
                    continue
                radius = row["radius_above_offset"]
                c = post.covariance_pe[None] if row["method"] == "BASPE" else covs
                s = samples if row["method"] == "BASPP" else None
                ref = solve_conic(post.mean, c, s, radius)
                objective_error = abs(row["objective"] - ref["objective"])
                assert objective_error < 1e-7, (j, row["method"], radius, objective_error)
                conic_records.append({"window": j, "method": row["method"], "radius": radius,
                                      "slsqp_objective": row["objective"], "conic_objective": ref["objective"],
                                      "absolute_difference": objective_error, "conic_status": ref["status"],
                                      "conic_solver": ref["solver"]})
            print(f"Independent conic checks passed for window {j}", flush=True)
        pd.DataFrame(conic_records).to_csv(output / "conic_validation.csv", index=False)
        report["conic_checks"] = len(conic_records)
        report["max_conic_objective_difference"] = max(row["absolute_difference"] for row in conic_records)
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/full")
    parser.add_argument("--skip-conic", action="store_true")
    args = parser.parse_args()
    validate(args.output, not args.skip_conic)
