from __future__ import annotations
import argparse
import fcntl
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from arboreal.data import DATASETS, load, prepare
from arboreal.experiment import run, tune_xgboost


def execute_job(job):
    data_path, dest, params, kwargs = job
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    # OS locks are released after crashes; concurrent resumptions cannot train
    # the same output. Separate processes isolate torch RNG and thread limits.
    with open(dest / ".run.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest / "result.json").exists():
            return {"skipped": str(dest)}
        result = run(data_path, dest, params, **kwargs)
        return {"completed": str(dest), "test": result["test"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--jobs", type=int, default=1,
                   help="Independent model processes; total CPU threads = jobs * threads")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", default="artifacts/benchmark")
    p.add_argument("--data-root", default="data")
    p.add_argument("--tau1", type=float, default=100)
    p.add_argument("--tau2", type=float, default=10)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(vars(args))
    settings.pop("jobs")  # Scheduling changes do not change scientific settings.
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.device != "cpu" and args.jobs != 1:
        raise ValueError("Use --jobs 1 for GPU experiments")
    settings_path = output / "suite_config.json"
    if settings_path.exists() and json.loads(settings_path.read_text()) != settings:
        raise ValueError("Suite configuration changed; use a fresh --output directory")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    jobs = []
    for name in args.datasets:
        data_path = prepare(name, args.data_root, args.split_seed)
        params = tune_xgboost(load(data_path), output / name / "tuning.json", args.threads)
        for seed in args.seeds:
            dest = output / name / f"seed{seed}"
            if (dest / "result.json").exists():
                print(f"SKIP completed {dest}", flush=True)
                continue
            jobs.append((str(data_path), str(dest), params,
                         {"seed": seed, "epochs": args.epochs, "patience": args.patience,
                          "batch_size": args.batch_size, "threads": args.threads,
                          "device": args.device, "tau1": args.tau1, "tau2": args.tau2}))
    if args.jobs == 1:
        for job in jobs:
            print(json.dumps(execute_job(job)), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs,
                                 mp_context=multiprocessing.get_context("spawn")) as pool:
            for future in as_completed([pool.submit(execute_job, job) for job in jobs]):
                print(json.dumps(future.result()), flush=True)
