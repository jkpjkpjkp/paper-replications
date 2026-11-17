#!/usr/bin/env python3
"""Execute the released vanilla CVaR cell and audit it independently.

Run from any directory with the environment recorded in requirements.lock.txt.
Outputs are isolated; an existing completed output directory is never replaced.
"""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

os.environ.setdefault("MPLBACKEND", "Agg")
import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["Sharpe", "Sortino", "MaxDrawdown", "FinalWealthFactor"]
PAPER = {
    30: [0.2883, 0.3707, 0.3499, 1.1915],
    40: [0.1531, 0.2001, 0.3516, 1.0569],
    50: [0.2165, 0.2858, 0.3086, 1.1170],
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def independent_metrics(returns):
    equity = np.r_[1., np.cumprod(1. + returns)]
    return dict(zip(METRICS, [
        returns.mean() / returns.std(ddof=0) * np.sqrt(252),
        returns.mean() / returns[returns < 0].std(ddof=0) * np.sqrt(252),
        (1 - equity / np.maximum.accumulate(equity)).max(),
        equity[-1],
    ]))


def verify_pool(pool, output, data, rebal_dates, solve_records):
    """Reconstruct every return and solve each LP with independent HiGHS."""
    positions = pd.read_csv(output / "results_trad" / f"CVaR_{pool}" /
                            "test_positions.csv", parse_dates=["Date"]).set_index("Date")
    frame = data.iloc[:, :pool].loc["2019-10-09":"2024-12-30"].dropna()
    weights = positions[[f"Weight_{c}" for c in frame.columns]].to_numpy()
    returns = (weights * frame.loc[positions.index].to_numpy()).sum(axis=1)
    capital = np.r_[10000., positions.CumulativeCapital.to_numpy()]
    stored_returns = positions.DailyPnL.to_numpy() / capital[:-1]
    np.testing.assert_allclose(returns, stored_returns, rtol=0, atol=1e-12)
    np.testing.assert_allclose(capital[1:], 10000 * np.cumprod(1 + returns), rtol=1e-12)
    assert np.isfinite(weights).all() and weights.min() >= -1e-7
    assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-7
    valid = [d for d in rebal_dates if d in frame.index and frame.index.get_loc(d) >= 60]
    assert len(valid) == len(solve_records), (len(valid), len(solve_records))
    assert positions.index.equals(frame.loc[valid[0]:].index)
    periods = []
    for i, date in enumerate(valid):
        t = frame.index.get_loc(date)
        stop = frame.index.get_loc(valid[i + 1]) if i + 1 < len(valid) else len(frame)
        history = frame.iloc[t - 60:t]
        assert history.index[-1] < date
        w = positions.loc[date, [f"Weight_{c}" for c in frame.columns]].to_numpy(float)
        held_weights = positions.loc[frame.index[t:stop], [f"Weight_{c}" for c in frame.columns]]
        np.testing.assert_allclose(held_weights.to_numpy(), np.tile(w, (stop - t, 1)), atol=0, rtol=0)
        # Variables: [asset weights, unrestricted VaR, nonnegative excess losses].
        scenarios = history.to_numpy()
        m = len(history)
        objective = np.r_[np.zeros(pool), 1., np.full(m, 1 / ((1 - .95) * m))]
        inequalities = np.column_stack([-scenarios, -np.ones(m), -np.eye(m)])
        equality = np.r_[np.ones(pool), np.zeros(1 + m)][None, :]
        check = linprog(objective, A_ub=inequalities, b_ub=np.zeros(m),
                        A_eq=equality, b_eq=[1.],
                        bounds=[(0, None)] * pool + [(None, None)] + [(0, None)] * m,
                        method="highs")
        assert check.success, check.message
        # Exactly three worst observations for alpha=.95 and 60 scenarios.
        empirical_cvar = np.sort(-scenarios @ w)[-3:].mean()
        gap = float(empirical_cvar - check.fun)
        assert abs(gap) < 1e-7, (pool, str(date), gap)
        assert abs(solve_records[i]["objective"] - empirical_cvar) < 1e-7
        periods.append({
            "pool": pool, "rebalance_date": str(date.date()),
            "history_start": str(history.index[0].date()),
            "history_end": str(history.index[-1].date()),
            "history_days": m, "holding_days": stop - t,
            "holding_end": str(frame.index[stop - 1].date()),
            "cvar_from_weights": float(empirical_cvar),
            "highs_optimal_objective": float(check.fun), "objective_gap": gap,
            **solve_records[i],
        })
    metrics = independent_metrics(returns)
    saved = pd.read_csv(output / "results_trad" / f"CVaR_{pool}" / "test_metrics.csv").iloc[0]
    metric_error = max(abs(float(saved[k]) - metrics[k]) for k in METRICS)
    assert metric_error < 1e-8, metric_error
    return periods, {
        "pool": pool, "return_days": len(positions), "allocation_decisions": len(valid),
        "first_return_date": str(positions.index[0].date()),
        "last_return_date": str(positions.index[-1].date()),
        "min_weight": float(weights.min()),
        "max_weight_sum_error": float(np.abs(weights.sum(axis=1) - 1).max()),
        "max_daily_return_replay_error": float(np.abs(returns - stored_returns).max()),
        "max_independent_metric_error": metric_error,
        "max_highs_objective_gap_abs": max(abs(p["objective_gap"]) for p in periods),
        "solver_counts": dict(Counter(p["solver"] for p in periods)),
        "status_counts": dict(Counter(p["status"] for p in periods)),
        "metrics": {k: float(saved[k]) for k in saved.index},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reproduction/vanilla_cvar_20261117")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results_trad").exists():
        raise SystemExit("Existing results: choose a fresh --output directory.")
    start = time.monotonic()
    started = datetime.now(timezone.utc).isoformat()
    notebook = ROOT / "TS_Traditional/run_benchmark.ipynb"
    data_path = ROOT / "TS_Traditional/dataset/full_dataset.csv"
    source = "".join(json.loads(notebook.read_text())["cells"][0]["source"])
    (output / "executed_author_cell.py").write_text(source)
    namespace = {"__name__": "released_baseline"}
    # Do not run the notebook's __main__ block (it also launches other methods).
    exec(compile(source, str(notebook) + ":cell0", "exec"), namespace)
    rebalance_node = next(n for n in ast.walk(ast.parse(source))
                          if isinstance(n, ast.Assign) and
                          any(isinstance(t, ast.Name) and t.id == "rebal_dates_str" for t in n.targets))
    rebalance_dates = list(pd.to_datetime(ast.literal_eval(rebalance_node.value)))
    data = pd.read_csv(data_path, parse_dates=["Date"]).set_index("Date")
    prices = pd.read_csv(ROOT / "asset_data/full_dataset.csv", parse_dates=["Date"]).set_index("Date")
    derived = prices.pct_change(fill_method=None).iloc[1:]
    assert data.index.equals(derived.index) and data.columns.equals(derived.columns)
    data_difference = float(np.abs(data.to_numpy() - derived.to_numpy()).max())
    assert data_difference < 1e-12, data_difference
    save_json(output / "protocol.json", {
        "kind": "fresh optimization from released returns; no saved weights replayed",
        "pools": [30, 40, 50], "confidence": .95, "lookback_daily_returns": 60,
        "constraints": ["sum(w)=1", "w>=0"], "objective": "minimum historical empirical CVaR",
        "return_target_constraint": None, "risk_free_rate": 0, "transaction_costs": 0,
        "source_date_slice": ["2019-10-09", "2024-12-30"],
        "scheduled_rebalance_dates": [str(d.date()) for d in rebalance_dates],
        "execution": "Unmodified author cell functions; only CVaR strategy invoked",
        "holdings_convention": "Same target weight vector applied to every daily return until next listed rebalance",
        "metrics_convention": "252 annualization; population std; Sortino uses std of negative returns",
        "rounding_tolerance_absolute": .00005, "paper_targets": PAPER,
        "no_seed_search": "Deterministic convex baseline; paper baseline rows have no seed SD",
        "validation": "All decisions independently solved using scipy.optimize.linprog / HiGHS",
        "data_return_vs_released_price_pct_change_max_abs": data_difference,
    })
    original_solve = cp.Problem.solve
    records = []

    def logged_solve(problem, *a, **kw):
        # Instrumentation only: delegate exact original solver arguments.
        value = original_solve(problem, *a, **kw)
        records.append({"solver": problem.solver_stats.solver_name,
                        "status": problem.status, "objective": float(problem.value),
                        "solve_seconds": float(problem.solver_stats.solve_time)})
        return value

    cp.Problem.solve = logged_solve
    previous_cwd = Path.cwd()
    all_periods, summaries, comparisons = [], [], []
    try:
        os.chdir(output)
        for pool in [30, 40, 50]:
            records.clear()
            namespace["run_one"](pool, "CVaR", csv_path=str(data_path), lookback=60)
            periods, summary = verify_pool(pool, output, data, rebalance_dates, records)
            all_periods.extend(periods)
            summaries.append(summary)
            for metric, target in zip(METRICS, PAPER[pool]):
                actual = summary["metrics"][metric]
                comparisons.append({"universe": "S&P100", "pool": pool,
                    "paper_table": 5 if pool == 30 else 1, "metric": metric,
                    "paper": target, "fresh": actual, "absolute_error": abs(actual - target),
                    "relative_error_percent": abs(actual - target) / abs(target) * 100,
                    "matches_4dp": f"{actual:.4f}" == f"{target:.4f}"})
            print(json.dumps(summary), flush=True)
    finally:
        cp.Problem.solve = original_solve
        os.chdir(previous_cwd)
    pd.DataFrame(comparisons).to_csv(output / "paper_comparison.csv", index=False)
    pd.DataFrame(all_periods).to_csv(output / "solver_and_period_audit.csv", index=False)
    save_json(output / "summary.json", {
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - start,
        "matched_cells": sum(c["matches_4dp"] for c in comparisons),
        "total_cells": len(comparisons), "pools": summaries,
        "scope_limitations": [
            "Reproduces released S&P100 protocol; effective dates differ from paper prose.",
            "Released data defects documented in ASSET_DATA_AUDIT.txt are retained.",
            "DOW30 10/20 and CSI300 50/100 pools not reproduced: corresponding verified inputs unavailable.",
            "Matching this traditional baseline does not resolve fresh SIT training mismatches.",
        ],
    })
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    (output / "requirements.lock.txt").write_text("".join(f"{k}=={v}\n" for k, v in sorted(packages.items())))
    save_json(output / "environment.json", {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": packages, "installed_solvers": cp.installed_solvers(),
        "git_head": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    })
    paths = [notebook, data_path, ROOT / "asset_data/full_dataset.csv", ROOT / "reproduction/paper.pdf",
             Path(__file__).resolve(), output / "executed_author_cell.py"]
    save_json(output / "input_manifest.json", {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
                                               {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths})
    artifacts = [p for p in output.rglob("*") if p.is_file() and ".venv" not in p.parts
                 and p.name not in {"run.log", "output_manifest.json", "COMPLETED.json"}]
    save_json(output / "output_manifest.json", {str(p.relative_to(output)): sha256(p) for p in artifacts})
    save_json(output / "COMPLETED.json", {"completed": True, "verification_passed": True,
                                        "matched_cells": sum(c["matches_4dp"] for c in comparisons)})


if __name__ == "__main__":
    main()
