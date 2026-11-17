"""Run fresh conformal passes, nominal tables, baseline grids, and 50-alpha curves."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ocp import run, controller, metrics

BASE_METHODS = ["UP", "KT", "DtACI", "SFOGD", "P", "PI-reconstructed"]
VARIANTS = ["UP-paper", "UP-current", "UP-corrected"]
METRICS = list(metrics([1.0], [1.0]).keys())


def safe(value):
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf" if value < 0 else None
    return value


def predict(scores, burnin, method, alpha, cfg, lr_override=None):
    """Return post-burn-in predictions; method initialization is documented."""
    if method in ("P", "PI-reconstructed"):
        lr = cfg["p_lr" if method == "P" else "pi_lr"] if lr_override is None else lr_override
        ki = 0.0 if method == "P" else float(np.max(scores[:burnin]))
        return controller(scores, alpha, lr, burnin, ki, 1.0)[burnin:]
    lr = cfg["sfogd_lr"] if lr_override is None else lr_override
    effective_alpha = alpha
    algorithm = method
    if method == "UP-current":
        effective_alpha -= 0.015
        algorithm = "UP"
    elif method == "UP-corrected":
        effective_alpha -= 1 / np.sqrt(len(scores))
        algorithm = "UP"
    return run(scores[burnin:], algorithm, effective_alpha, lr)


def aggregate(frame, groups):
    records = []
    for key, part in frame.groupby(groups, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, key))
        row["repeats"] = len(part)
        for metric in METRICS:
            values = part[metric].to_numpy(dtype=float)
            row[metric] = float(np.mean(values))
            row[metric + "_sem"] = (float(np.std(values, ddof=1) / np.sqrt(len(values)))
                                    if len(values) > 1 and np.isfinite(values).all()
                                    else 0.0 if len(values) == 1 and np.isfinite(values).all() else np.nan)
        records.append(row)
    return pd.DataFrame(records)


def compare(summary, config):
    reference = json.loads((ROOT / "paper/reference_values.json").read_text())
    records = []
    for dataset in config["datasets"]:
        for method, values in reference[dataset].items():
            result = summary[(summary.dataset == dataset) & (summary.method == method)].iloc[0]
            for metric, target in zip(reference["metrics"], values):
                if target is None:
                    continue
                target = float(target)
                actual = float(result[metric])
                same = (np.isposinf(actual) and np.isposinf(target)) or (
                    np.isfinite(actual) and np.isfinite(target) and float(f"{actual:.3g}") == target)
                records.append(dict(dataset=dataset, table=config["datasets"][dataset]["table"],
                                    method=method, metric=metric, paper=target, reproduced=actual,
                                    delta=actual - target if np.isfinite(actual) and np.isfinite(target) else np.nan,
                                    match_at_published_precision=bool(same)))
    result = pd.DataFrame(records)
    result.to_csv(ROOT / "results/paper_comparison.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sweep", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config.json").read_text())
    results = ROOT / "results"
    traces = results / "traces"
    traces.mkdir(parents=True, exist_ok=True)
    start = time.time()
    nominal, sweep, grid = [], [], []
    offset = config["evaluation_offset_after_burnin"]
    alpha_grid = np.linspace(**dict(start=config["alpha_grid"]["start"],
                                   stop=config["alpha_grid"]["stop"],
                                   num=config["alpha_grid"]["count"]))
    for dataset, cfg in config["datasets"].items():
        data = np.load(ROOT / "data/scores" / f"{dataset}.npz")
        burnin = int(data["burnin"])
        arrays = {"scores": data["scores"], "burnin": burnin, "eval_offset": offset, "seeds": data["seeds"]}
        for method in BASE_METHODS + VARIANTS:
            method_traces = []
            for scores, seed in zip(data["scores"], data["seeds"]):
                q = predict(scores, burnin, method, config["alpha"], cfg)
                method_traces.append(q)
                nominal.append(dict(dataset=dataset, method=method, seed=int(seed), alpha=config["alpha"],
                                    **metrics(scores[burnin + offset:], q[offset:])))
            arrays[method] = np.array(method_traces)
        np.savez_compressed(traces / f"{dataset}.npz", **arrays)
        for method, rates in [("SFOGD", config["sfogd_grid"]),
                              ("P", config["controller_grid"]),
                              ("PI-reconstructed", config["controller_grid"])]:
            for lr in rates:
                for scores, seed in zip(data["scores"], data["seeds"]):
                    q = predict(scores, burnin, method, config["alpha"], cfg, lr)
                    grid.append(dict(dataset=dataset, method=method, lr=lr, seed=int(seed), alpha=config["alpha"],
                                     **metrics(scores[burnin + offset:], q[offset:])))
        if not args.skip_sweep:
            for method in BASE_METHODS:
                for alpha in alpha_grid:
                    for scores, seed in zip(data["scores"], data["seeds"]):
                        q = predict(scores, burnin, method, alpha, cfg)
                        sweep.append(dict(dataset=dataset, method=method, alpha=alpha, seed=int(seed),
                                          **metrics(scores[burnin + offset:], q[offset:])))
        print(f"{dataset}: {len(data['seeds'])} series completed ({time.time()-start:.1f}s)", flush=True)
        pd.DataFrame(nominal).to_csv(results / "nominal_per_trial.csv", index=False)
    summary = aggregate(pd.DataFrame(nominal), ["dataset", "method", "alpha"])
    summary.to_csv(results / "summary.csv", index=False)
    aggregate(pd.DataFrame(grid), ["dataset", "method", "lr", "alpha"]).to_csv(results / "baseline_grid.csv", index=False)
    pd.DataFrame(grid).to_csv(results / "baseline_grid_per_trial.csv", index=False)
    if sweep:
        pd.DataFrame(sweep).to_csv(results / "sweep_per_trial.csv", index=False)
        aggregate(pd.DataFrame(sweep), ["dataset", "method", "alpha"]).to_csv(results / "sweep_summary.csv", index=False)
    comparison = compare(summary, config)
    exact_core = comparison[(comparison.dataset.isin(["AXP", "AAPL", "AMZN", "GOOGL"])) &
                            comparison.method.isin(["UP", "UP-corrected", "KT", "DtACI", "SFOGD"])]
    status = dict(completed_at=datetime.now(timezone.utc).isoformat(),
                  status="completed_with_documented_discrepancies",
                  upstream_commit=subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip(),
                  python=platform.python_version(), elapsed_seconds=time.time() - start,
                  datasets=list(config["datasets"]), nominal_runs=len(nominal), grid_runs=len(grid), sweep_runs=len(sweep),
                  exact_core_stock_cells=int(exact_core.match_at_published_precision.sum()),
                  total_core_stock_cells=len(exact_core),
                  scope="Conformal layer recomputed from author forecast caches; base models not refitted in the main run.",
                  config_sha256=hashlib.sha256((ROOT / "config.json").read_bytes()).hexdigest(),
                  code_sha256={p:hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in ["ocp.py", "scripts/reproduce.py"]})
    (ROOT / "STATUS.json").write_text(json.dumps(safe(status), indent=2, allow_nan=False))
    print(json.dumps(safe(status), indent=2))


if __name__ == "__main__":
    main()
