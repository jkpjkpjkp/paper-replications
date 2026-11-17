#!/usr/bin/env python3
"""Independent solver checks, event-skipping audit, and a descent trace."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from stco.market import load_dataset, predict
from stco.solver import (PSGAConfig, _topk, solve_psga, prox_admm,
                         prox_exact, objective)


def lp_objective(q, h, lam, subset, penalty=0., restrict=False):
    m = len(q)
    selected = np.zeros(m); selected[list(subset)] = 1
    a = np.block([[np.eye(m), -np.eye(m)], [-np.eye(m), -np.eye(m)]])
    bounds = [(0, None) if not restrict or selected[i] else (0, 0)
              for i in range(m)] + [(0, None)]*m
    result = linprog(np.r_[-q-penalty*selected, np.full(m, lam)],
                     A_ub=a, b_ub=np.r_[h, -h],
                     A_eq=np.r_[np.ones(m), np.zeros(m)][None, :], b_eq=[1.],
                     bounds=bounds, method="highs")
    assert result.success
    return float(result.fun), result.x[:m]


def small_problem_audit():
    rng = np.random.default_rng(2026)
    q = rng.uniform(.8, 1.2, 7)
    h = rng.dirichlet(np.ones(7))
    k, lam = 3, .025
    bound = 2*np.sqrt(2)*(np.linalg.norm(q-q.mean())+lam*np.sqrt(len(q)))
    c = 1.1*bound
    subsets = list(itertools.combinations(range(len(q)), k))
    hard = min((lp_objective(q, h, lam, s, restrict=True) for s in subsets),
               key=lambda a: a[0])
    dc = min((lp_objective(q, h, lam, s, penalty=c) for s in subsets),
             key=lambda a: a[0])
    w, info = solve_psga(q, h, lam, PSGAConfig(k=k, penalty=c))
    return {"m": len(q), "k": k, "penalty": c, "sufficient_bound": bound,
            "hard_global_objective": hard[0], "dc_global_objective_plus_c": dc[0]+c,
            "global_equivalence_error": abs(dc[0]+c-hard[0]),
            "psga_global_objective_gap": objective(w, q, h, lam, c, k)-dc[0],
            "psga_support": int((w > 1e-8).sum()), "solver": info,
            "note": "Finite oracle check of exact penalty; not a proof or a global PSGA guarantee."}


def trace_descent(output):
    rng = np.random.default_rng(114)
    q, h = rng.uniform(.8, 1.2, 8), rng.dirichlet(np.ones(8))
    cfg = PSGAConfig(k=3, penalty=.001)
    lam, w = .025, np.ones(8)/8
    final, info = solve_psga(q, h, lam, cfg)
    rows = []
    last_obj = objective(w, q, h, lam, cfg.penalty, cfg.k)
    max_increase = 0.
    for step in range(200_000):
        obj = objective(w, q, h, lam, cfg.penalty, cfg.k)
        new = prox_exact(w+cfg.alpha*(q+cfg.penalty*_topk(w, cfg.k)), h, cfg.alpha*lam)
        rel = np.linalg.norm(new-w)/np.linalg.norm(new)
        if step % 10 == 0 or rel <= cfg.tol:
            rows.append({"iteration": step, "objective": obj,
                         "distance_to_limit": np.linalg.norm(w-final),
                         "relative_step": rel})
        max_increase = max(max_increase, obj-last_obj)
        last_obj = obj
        w = new
        if rel <= cfg.tol:
            break
    pd.DataFrame(rows).to_csv(output/"convergence_trace.csv", index=False)
    return {"iterations": step+1, "max_objective_increase": max_increase,
            "distance_to_accelerated_limit": float(np.linalg.norm(w-final)),
            "configuration": {"seed": 114, "m": 8, "k": 3, "c": .001,
                               "lambda": .025, "alpha": .0025},
            "note": "Numerical descent/stationarity check; does not prove the KL exponent."}


def market_direct_audit(results, output):
    records = []
    for name, file_slug in [("NYSE(O)", "nyse-o"), ("NYSE(N)", "nyse-n"),
                            ("TSE", "tse"), ("MSCI", "msci")]:
        x = load_dataset(name)
        for strategy in [1, 2]:
            path = results/"runs"/f"{file_slug}_STCO{strategy}_g0.0025_w{4 if strategy==2 else 0}.npz"
            saved = np.load(path)
            # Bound the direct audit workload; large paper-step runs can
            # represent tens of millions of iterations per trading day.
            candidates = np.flatnonzero((saved["iterations"] > 1)
                                        & (saved["iterations"] <= 250_000))
            selected = candidates[np.linspace(0, len(candidates)-1, 4, dtype=int)]
            for t in selected:
                prev = saved["weights"][t-1]
                h = prev*x[t-1]/np.dot(prev, x[t-1])
                q = predict(x, t-1, strategy, 4)
                w, direct = solve_psga(q, h, .025, PSGAConfig(accelerate=False))
                fast, info = solve_psga(q, h, .025)
                records.append({"dataset": name, "strategy": strategy, "day_zero_based": t,
                                "max_weight_error": float(np.abs(w-fast).max()),
                                "stored_weight_error": float(np.abs(w-saved["weights"][t]).max()),
                                "direct_iterations": direct["iterations"],
                                "accelerated_iterations": info["iterations"],
                                "prox_evaluations": info["prox_evaluations"],
                                "direct_converged": direct["converged"],
                                "accelerated_converged": info["converged"]})
                print(f"audit {name} STCO{strategy} t={t} error={records[-1]['max_weight_error']:.2g}", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(output/"direct_market_audit.csv", index=False)
    if df.max_weight_error.max() > 2e-7 or not df.direct_converged.all():
        raise RuntimeError("Direct PSGA audit failed")
    return {"cases": len(df), "max_weight_error": float(df.max_weight_error.max()),
            "max_stored_weight_error": float(df.stored_weight_error.max()),
            "prox_evaluation_reduction": float(df.direct_iterations.sum()/df.prox_evaluations.sum()),
            "sampling": "Four evenly spaced eligible days per dataset/strategy at gamma=.0025; direct iterations <=250000."}


def stationarity_audit(results, output):
    """Fixed-subgradient convex gap; checks every saved proposed-method day.

    With a=q+c*v, move h_i to the maximum-a asset only when
    a_max-a_i > 2*lambda. This solves the convex linear/L1 model exactly.
    A zero convex gap certifies DC criticality for the selected v, not
    global optimality of the nonconvex DC model.
    """
    rows = []
    for name, slug in [("NYSE(O)", "nyse-o"), ("NYSE(N)", "nyse-n"),
                       ("TSE", "tse"), ("MSCI", "msci")]:
        x = load_dataset(name)
        for strategy in [1, 2]:
            for gamma in [0., .0025, .005]:
                windows = [0] if strategy == 1 else [4, 5]
                for window in windows:
                    path = results/"runs"/f"{slug}_STCO{strategy}_g{gamma:.4f}_w{window}.npz"
                    if not path.exists():
                        continue
                    saved = np.load(path)
                    weights = saved["weights"]
                    gaps, fixed_residuals = [], []
                    for t in range(1, len(x)):
                        prev, w = weights[t-1], weights[t]
                        h = prev*x[t-1]/np.dot(prev, x[t-1])
                        q = predict(x, t-1, strategy, max(window, 4))
                        v = _topk(w, 10)
                        a, lam = q+8e-6*v, 10*gamma
                        minimum = -a@h - np.sum(h*np.maximum(a.max()-a-2*lam, 0))
                        actual = -a@w + lam*np.abs(w-h).sum()
                        gaps.append(actual-minimum)
                        if t % 100 == 0:
                            updated = prox_exact(w+.0025*a, h, .0025*lam)
                            fixed_residuals.append(np.linalg.norm(updated-w)/np.linalg.norm(updated))
                    rows.append({"dataset": name, "strategy": f"STCO{strategy}",
                                 "gamma": gamma, "window": window,
                                 "max_fixed_v_convex_gap": float(max(gaps)),
                                 "mean_fixed_v_convex_gap": float(np.mean(gaps)),
                                 "days_gap_over_1e_6": int(np.sum(np.array(gaps) > 1e-6)),
                                 "sample_max_relative_fixed_point_residual": float(max(fixed_residuals))})
    df = pd.DataFrame(rows)
    df.to_csv(output/"stationarity_audit.csv", index=False)
    return {"runs": len(df), "max_convex_gap": float(df.max_fixed_v_convex_gap.max()),
            "days_gap_over_1e_6": int(df.days_gap_over_1e_6.sum()),
            "max_sampled_fixed_point_residual": float(df.sample_max_relative_fixed_point_residual.max())}


def main():
    output = ROOT/"results/validation"
    output.mkdir(parents=True, exist_ok=True)
    summary = {"small_exact_penalty_oracle": small_problem_audit(),
               "convergence": trace_descent(output),
               "direct_market_audit": market_direct_audit(ROOT/"results/main", output),
               "stationarity_audit": stationarity_audit(ROOT/"results/main", output)}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
