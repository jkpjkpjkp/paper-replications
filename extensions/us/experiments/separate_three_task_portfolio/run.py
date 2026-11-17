"""One GPU trainer, then automatic paired profitability analysis."""
from pathlib import Path
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.separate_three_task_portfolio.common import HERE, input_hashes, now, protocol, source_hashes
from usdata import sha, write_json

MODEL = Path(os.environ.get("LIMT_MODEL_PYTHON", sys.executable))
ANALYSIS = Path(os.environ.get("LIMT_ANALYSIS_PYTHON", sys.executable))
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def status(stage, **kwargs):
    write_json(HERE / "status.json", dict(stage=stage, updated_utc=now(), **kwargs))


def freeze():
    path = HERE / "locked_before_training.json"
    if path.exists():
        frozen = json.loads(path.read_text())
        assert frozen["inputs_sha256"] == input_hashes()
        assert frozen["reused_sources_sha256"] == source_hashes()
        return
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        assert sha(ROOT / "data" / name) == expected, name
    write_json(path, dict(created_utc=now(), inputs_sha256=input_hashes(), reused_sources_sha256=source_hashes(),
        new_profitability_inspected_before_this_lock=False,
        scope="Retrospective follow-up; prior base profitability and two-task forecasting results already available."))


def main(checks_only=False):
    locks = [(ROOT / "run.lock").open("a"), (HERE / "run.lock").open("a")]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    try:
        status("preflight")
        with (HERE / "checks.log").open("w") as log:
            commands = [[str(MODEL), "-m", "pytest", str(ROOT / "tests"), "-q"],
                        [str(MODEL), str(HERE / "checks.py")],
                        [str(ANALYSIS), str(HERE / "analyze.py"), "--self-check"],
                        [str(MODEL), str(HERE / "train_volume.py"), "--smoke"]]
            for command in commands:
                subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        if checks_only:
            status("checks_passed")
            return
        freeze()
        status("baseline_profitability")
        with (HERE / "baseline.log").open("a") as log:
            subprocess.run([str(ANALYSIS), str(HERE / "analyze.py"), "--baseline-only"],
                           cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        complete = []
        for seed in [0, 1, 2]:
            folder = HERE / "results/volume_only" / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "train.log").open("a") as log:
                child = subprocess.Popen([str(MODEL), str(HERE / "train_volume.py"), "--seed", str(seed)],
                                         cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status("training_volume", running_seed=seed, pid=child.pid,
                           completed_seeds=complete, pending_seeds=list(range(seed + 1, 3)))
                    time.sleep(10)
                code = child.returncode
                child = None
                if code:
                    raise RuntimeError(f"Volume seed{seed} failed with exit {code}")
            complete.append(seed)
        status("analyzing", completed_volume_seeds=complete)
        with (HERE / "analysis.log").open("a") as log:
            subprocess.run([str(ANALYSIS), str(HERE / "analyze.py")], cwd=ROOT, env=ENV,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        verification = json.loads((HERE / "analysis/verification.json").read_text())
        assert verification["status"] == "passed"
        status("complete", verdict=verification["verdict"], report=str(HERE / "REPORT.txt"))
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        status("failed", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        result = json.loads((HERE / "status.json").read_text()) if (HERE / "status.json").exists() else {"stage": "not_started"}
        result["volume_fits"] = {str(p.parent.name): json.loads(p.read_text())
                                 for p in sorted((HERE / "results/volume_only").glob("seed*/status.json"))}
        print(json.dumps(result, indent=2))
    else:
        main(args.checks_only)
