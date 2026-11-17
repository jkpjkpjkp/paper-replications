"""Fresh Table 5 baseline calculation with independent accounting and solver checks.

No archived predictions, portfolios or wealth series are loaded. BCRP optimizes
gross log wealth in hindsight, then uses the same portfolio at each cost rate.
Paper targets are used only after optimization. This is not STCO replication.
"""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy
from scipy.io import loadmat
from scipy.optimize import brentq, minimize


# Visually verified against the original PDF, page 22, Table 5 (2026-11-17).
TARGETS = {
    "NYSE(O)": {"UBAH": [14.50, 14.46, 14.43], "BEST": [54.14, 54.01, 53.87],
                "BCRP": [252.07, 181.03, 133.52]},
    "NYSE(N)": {"UBAH": [18.06, 18.01, 17.97], "BEST": [83.51, 83.30, 83.09],
                "BCRP": [119.71, 98.02, 80.23]},
    "TSE": {"UBAH": [1.61, 1.61, 1.60], "BEST": [6.28, 6.26, 6.25],
            "BCRP": [6.70, 6.38, 6.12]},
    "MSCI": {"UBAH": [0.91, 0.90, 0.90], "BEST": [1.50, 1.50, 1.50],
             "BCRP": [1.51, 1.49, 1.48]},
}
FILES = {"NYSE(O)": "nyse-o.mat", "NYSE(N)": "nyse-n.mat",
         "TSE": "tse.mat", "MSCI": "msci.mat"}
SHAPES = {"NYSE(O)": (5651, 36), "NYSE(N)": (6431, 23),
          "TSE": (1259, 88), "MSCI": (1043, 24)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def optimize_bcrp(x):
    n, m = x.shape
    result = minimize(lambda w: -np.log(x @ w).sum(), np.ones(m) / m,
                      jac=lambda w: -(x / (x @ w)[:, None]).sum(axis=0),
                      constraints=[{"type": "eq", "fun": lambda w: w.sum()-1,
                                    "jac": lambda w: np.ones(m)}],
                      bounds=[(0, 1)] * m, method="SLSQP",
                      options={"maxiter": 2000, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(result.message)
    w = np.maximum(result.x, 0)
    w /= w.sum()
    # Concavity makes this a global upper bound on average log-wealth gap.
    gradient = np.mean(x / (x @ w)[:, None], axis=0)
    gap = float(np.max(gradient)-np.dot(gradient, w))
    assert gap < 1e-7, gap

    # Exponential-cone optimization is independent of SLSQP derivatives.
    v = cp.Variable(m)
    # Since sum(v)=1, centering removes a large equality-multiplier component.
    problem = cp.Problem(cp.Maximize(cp.sum(cp.log1p((x-1) @ v))),
                         [v >= 0, cp.sum(v) == 1])
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                  tol_feas=1e-10, max_iter=300)
    assert problem.status in {"optimal", "optimal_inaccurate"}, problem.status
    other = np.maximum(v.value, 0)
    other /= other.sum()
    delta = float(abs(np.log(x @ w).sum()-np.log(x @ other).sum()))
    # Check average log objective; the four datasets have different lengths.
    assert delta/n < 1e-9, delta
    return w, other, {"slsqp_iterations": result.nit, "global_log_gap_bound": gap*n,
                      "mean_log_gap_bound": gap, "conic_status": problem.status,
                      "independent_total_log_objective_difference": delta,
                      "weight_max_difference": float(abs(w-other).max())}


def rebalance(x, weights, gamma):
    """Equation (5), independently solved with Brent's root method."""
    gross = x @ weights
    previous = np.zeros(x.shape[1])
    fractions = []
    residual = 0.0
    for day in range(len(x)):
        f = lambda z: z + gamma*np.abs(previous-z*weights).sum()-1
        theta = brentq(f, 0., 1., xtol=5e-15, rtol=1e-14)
        residual = max(residual, abs(f(theta)))
        fractions.append(theta)
        previous = weights*x[day]/gross[day]
    fractions = np.array(fractions)
    assert residual < 1e-12
    return np.cumprod(gross*fractions), fractions, residual


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--paper", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, details = [], {}
    for dataset, filename in FILES.items():
        path = args.data/filename
        try:
            x = loadmat(path)["data"]
        except NotImplementedError:
            import h5py
            with h5py.File(path) as handle:
                x = handle["data"][:].T
        x = np.ascontiguousarray(x, dtype=float)
        assert x.shape == SHAPES[dataset] and np.all(np.isfinite(x)) and x.min() > 0
        w, conic_w, diagnostics = optimize_bcrp(x)
        buyhold = np.cumprod(x, axis=0).mean(axis=1)
        best = np.zeros(x.shape[1])
        best[np.log(x).sum(axis=0).argmax()] = 1
        arrays = {"bcrp_weights": w, "conic_weights": conic_w}
        diagnostics.update({"shape": list(x.shape), "sha256": sha(path)})
        for idx, gamma in enumerate([0., .0025, .005]):
            bcrp, fractions, residual = rebalance(x, w, gamma)
            conic_path, _, _ = rebalance(x, conic_w, gamma)
            best_path, _, _ = rebalance(x, best, gamma)
            np.testing.assert_allclose(best_path, np.cumprod(x @ best)/(1+gamma),
                                       rtol=1e-10)
            arrays[f"bcrp_wealth_{idx}"] = bcrp
            arrays[f"bcrp_theta_{idx}"] = fractions
            diagnostic_delta = float(abs(bcrp[-1]-conic_path[-1]))
            assert diagnostic_delta < 1e-3, (dataset, gamma, diagnostic_delta)
            diagnostics[f"accounting_residual_{idx}"] = residual
            diagnostics[f"independent_net_wealth_difference_{idx}"] = diagnostic_delta
            for method, value in [("BCRP", bcrp[-1]), ("BEST", best_path[-1]),
                                  ("UBAH", buyhold[-1]/(1+gamma))]:
                target = TARGETS[dataset][method][idx]
                error = float(value-target)
                rows.append({"dataset": dataset, "method": method,
                             "transaction_rate": gamma, "paper_wealth": target,
                             "fresh_wealth": float(value), "absolute_error": abs(error),
                             "relative_error": error/target, "rounding_tolerance": .005,
                             "matches_rounding": abs(error) <= .005})
        details[dataset] = diagnostics
        np.savez_compressed(args.output/f"{filename}.npz", **arrays)
        print(dataset, diagnostics, flush=True)
    with (args.output/"comparison.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"date": "2026-11-17", "scope": "Fresh traditional baseline optimization; no STCO claim",
               "paper_sha256": sha(args.paper), "script_sha256": sha(__file__),
               "python": platform.python_version(), "numpy": np.__version__,
               "scipy": scipy.__version__, "cvxpy": cp.__version__, "datasets": details,
               "matches": sum(row["matches_rounding"] for row in rows), "cells": len(rows)}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(summary["matches"], "of", summary["cells"], "match rounding", flush=True)


if __name__ == "__main__":
    main()
