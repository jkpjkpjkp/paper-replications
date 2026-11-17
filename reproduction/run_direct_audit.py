"""Run the untouched author entry point, with explicit paper/release settings.

This is a controlled check of the reproduction adapter, not a claim that the
undocumented published training protocol has been recovered.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=["pfl", "dfl"], default=["pfl", "dfl"])
    args = p.parse_args()
    data = args.data_root.resolve()
    out = args.output.resolve()
    if out.exists():
        raise SystemExit("Choose a new output directory; existing experiments are never overwritten")
    out.mkdir(parents=True)
    env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
    records = []
    for mode in args.modes:
        command = [sys.executable, str(ROOT / "upstream/src/main.py"), "--mode", mode,
                   "--dataset", "euro50", "--seed", "16", "--hidden_dims", "512", "256",
                   "--batchsize", "64", "--epochs", "300", "--cuda", "-1", "--alpha", "0.5",
                   "--splits", "test", "--k_list", "10", "--data_root", str(data),
                   "--save_dir", str(out), "--selector", "oscar"]
        if mode == "pfl":
            command.append("oscar_hard")
        record = dict(command=command, started_utc=datetime.now(timezone.utc).isoformat(),
                      data_sha256=hashlib.sha256((data / "euro50_160101_251231.csv").read_bytes()).hexdigest(),
                      entry_point="unmodified upstream/src/main.py", exact_paper_reproduction=False)
        with (out / f"{mode}.log").open("w") as log:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=run.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
        records.append(record)
        (out / "execution.json").write_text(json.dumps(records, indent=2))
        print(f"{mode}: exit {run.returncode}", flush=True)
        if run.returncode:
            return run.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
