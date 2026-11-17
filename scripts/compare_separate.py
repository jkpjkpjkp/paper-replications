"""Run the independent-task comparison and automatically assemble its report."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_data import sha, write_json
from scripts.train_separate import TASKS

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["date", "realized_date", "symbol"]


def merge_predictions(frames):
    """Require exactly matching stock/date keys; never silently drop samples."""
    merged = None
    for task in TASKS:
        frame = frames[task].sort_values(KEYS).reset_index(drop=True)
        if frame.duplicated(["date", "symbol"]).any():
            raise ValueError(f"Duplicate prediction keys for {task}")
        if not np.isfinite(frame[f"pred_{task}"]).all():
            raise ValueError(f"Non-finite predictions for {task}")
        if merged is None:
            merged = frame.copy()
        else:
            if not frame[KEYS].equals(merged[KEYS]):
                raise ValueError(f"Prediction keys differ for {task}")
            merged[f"pred_{task}"] = frame[f"pred_{task}"]
            merged[f"label_{task}"] = frame[f"label_{task}"]
    return merged


def combine(run_name, universe, seed):
    folder = ROOT/"results"/run_name/universe/f"seed{seed}"
    metrics, sources, parameters = {}, {}, 0
    for task in TASKS:
        path = folder/task
        if json.loads((path/"status.json").read_text())["phase"] != "complete":
            raise ValueError(f"Task is incomplete: {path}")
        m = json.loads((path/"metrics.json").read_text())
        for split in ("valid", "test"):
            metrics.setdefault(split, {}).update(m[split])
        sources[task] = {name: sha(path/name) for name in (
            "config.json", "best.pt", "test_predictions.parquet", "valid_predictions.parquet")}
        parameters += json.loads((path/"provenance.json").read_text())["parameters"]
    for split in ("valid", "test"):
        frame = merge_predictions({task: pd.read_parquet(folder/task/f"{split}_predictions.parquet")
                                   for task in TASKS})
        frame.to_parquet(folder/f"{split}_predictions.parquet", index=False)
    write_json(folder/"metrics.json", metrics)
    write_json(folder/"provenance.json", dict(task_artifacts=sources, parameters=parameters,
        test_prediction_sha256=sha(folder/"test_predictions.parquet"),
        assembler_sha256=sha(__file__)))
    write_json(folder/"status.json", dict(phase="complete", tasks=list(TASKS),
        test=metrics["test"], updated_utc=datetime.now(timezone.utc).isoformat()))


def run_command(command, logfile):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with logfile.open("a") as log:
        subprocess.run([sys.executable, "-u", *command], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def reference_command(config):
    command = ["scripts/train.py"]
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                command.append("--"+key.replace("_", "-"))
        else:
            command.append("--"+key.replace("_", "-"))
            command.extend(str(x) for x in value) if isinstance(value, list) else command.append(str(value))
    return command


def show_status(folder, reference):
    print("Universe  Seed Task        Phase       Epoch  Best")
    for seed in range(3):
        for universe in ("csi300", "csi500"):
            for task in ("joint", *TASKS):
                path = (reference if task == "joint" else folder)/universe/f"seed{seed}"
                if task != "joint":
                    path /= task
                state = json.loads((path/"status.json").read_text()) if (path/"status.json").exists() else {}
                print(f"{universe:9} {seed:<4} {task:11} {state.get('phase', 'queued'):11} "
                      f"{state.get('epoch', state.get('epochs', '—'))!s:6} {state.get('best_epoch', '—')}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="separate")
    p.add_argument("--reference-run", default="main")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--status", action="store_true")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args()
    if args.jobs < 1 or args.run_name == args.reference_run:
        p.error("Use positive --jobs and distinct run/reference names")
    folder, reference = [ROOT/"results"/name for name in (args.run_name, args.reference_run)]
    if args.status:
        show_status(folder, reference)
        return
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/"suite.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This comparison is already running")
        write_json(folder/"suite_status.json", dict(phase="reporting" if args.report_only else "training",
            pid=os.getpid(), reference_run=args.reference_run, jobs=args.jobs,
            started_utc=datetime.now(timezone.utc).isoformat()))
        try:
            if not args.report_only:
                # Complete interrupted joint references using their exact saved configuration.
                for seed in range(3):
                    for universe in ("csi300", "csi500"):
                        path = reference/universe/f"seed{seed}"
                        config = json.loads((path/"config.json").read_text())
                        if config["variant"] != "full" or config["smoke"]:
                            raise ValueError("Expected full-date joint reference models")
                        state = json.loads((path/"status.json").read_text())
                        if state["phase"] != "complete":
                            print(f"Resuming joint {universe} seed {seed}", flush=True)
                            run_command(reference_command(config), folder/f"reference-{universe}-seed{seed}.log")
                def train(item):
                    universe, seed, task = item
                    print(f"Training {universe} seed {seed} {task}", flush=True)
                    run_command(["scripts/train_separate.py", "--run-name", args.run_name,
                        "--reference-run", args.reference_run, "--universe", universe,
                        "--seed", str(seed), "--task", task], folder/f"{universe}-seed{seed}-{task}.log")
                    print(f"Completed {universe} seed {seed} {task}", flush=True)
                tasks = [(u, s, task) for s in range(3) for u in ("csi300", "csi500") for task in TASKS]
                with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                    list(pool.map(train, tasks))
            for seed in range(3):
                for universe in ("csi300", "csi500"):
                    combine(args.run_name, universe, seed)
            write_json(folder/"suite_status.json", dict(phase="reporting", pid=os.getpid()))
            from scripts.report_separate import generate
            generate(args.run_name, args.reference_run)
            write_json(folder/"suite_status.json", dict(phase="complete", reference_run=args.reference_run,
                report=str(folder/"comparison.md"), updated_utc=datetime.now(timezone.utc).isoformat()))
        except Exception as error:
            write_json(folder/"suite_status.json", dict(phase="failed", error=str(error),
                updated_utc=datetime.now(timezone.utc).isoformat()))
            raise


if __name__ == "__main__":
    main()
