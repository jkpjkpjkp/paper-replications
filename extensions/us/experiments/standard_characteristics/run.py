"""Isolated, resumable coordinator: preflight, freeze, three seeds, alpha report."""
from pathlib import Path
import sys
import argparse
import fcntl
import json
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from usdata import sha, write_json
from experiments.standard_characteristics.common import HERE, DATA, now, config, input_hashes

MODEL = Path(os.environ.get("LIMT_MODEL_PYTHON", sys.executable))
ANALYSIS = Path(os.environ.get("LIMT_ANALYSIS_PYTHON", sys.executable))
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def status(stage, **kwargs):
    write_json(HERE / "status.json", dict(stage=stage, updated_utc=now(), **kwargs))


def main(checks_only=False):
    lock = (HERE / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    try:
        status("preflight")
        with (HERE / "checks.log").open("a") as log:
            for command in [
                [str(MODEL), "-m", "pytest", str(HERE / "checks.py"), "-q"],
                [str(ANALYSIS), str(HERE / "analysis_checks.py")],
                [str(MODEL), str(HERE / "data.py")],
                [str(MODEL), str(HERE / "train.py"), "--smoke"]]:
                subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)
        if checks_only:
            status("checks_passed")
            return
        path = HERE / "locked_before_training.json"
        if path.exists():
            frozen = json.loads(path.read_text())
            assert frozen["inputs_sha256"] == input_hashes(), "Frozen source changed; refusing checkpoint reuse"
            assert frozen["data_manifest_sha256"] == sha(DATA / "manifest.json")
        else:
            for location in [ROOT / "data", DATA]:
                manifest = json.loads((location / "manifest.json").read_text())
                for name, expected in manifest["output_sha256"].items():
                    assert sha(location / name) == expected, name
            write_json(path, dict(created_utc=now(), inputs_sha256=input_hashes(),
                data_manifest_sha256=sha(DATA / "manifest.json"), preflight_sha256=sha(HERE / "preflight.json"),
                test_predictions_or_new_portfolio_results_inspected=False,
                historical_prior_results="Prior Alpha158/FF5+UMD results were available before this new retrospective specification."))
        completed = []
        start = time.monotonic()
        for seed in config()["training"]["seeds"]:
            folder = HERE / "results" / f"seed{seed}"
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "train.log").open("a") as log:
                child = subprocess.Popen([str(MODEL), str(HERE / "train.py"), "--seed", str(seed)],
                    cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status("training", running_seed=seed, pid=child.pid, completed_seeds=completed,
                        total_seeds=3, elapsed_seconds=time.monotonic() - start)
                    time.sleep(10)
                code, child = child.returncode, None
                if code:
                    raise RuntimeError(f"seed{seed} failed, exit {code}; see {folder / 'train.log'}")
            completed.append(seed)
        status("analyzing", completed_seeds=completed)
        with (HERE / "analysis.log").open("a") as log:
            subprocess.run([str(ANALYSIS), str(HERE / "analyze.py")], cwd=ROOT, env=ENV,
                stdout=log, stderr=subprocess.STDOUT, check=True)
        verification = json.loads((HERE / "analysis/verification.json").read_text())
        assert verification["status"] == "passed"
        status("complete", completed_seeds=completed, verdicts=verification["verdicts"], report=str(HERE / "REPORT.txt"))
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
        print((HERE / "status.json").read_text())
    else:
        main(args.checks_only)
