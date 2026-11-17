"""Resumable 21-fit coordinator, one trainer at a time, then audited analysis."""
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
from usdata import sha, write_json
from experiments.market_adjusted_monthly.common import HERE, folder, input_hashes, jobs, now

MODEL = Path(os.environ.get("LIMT_MODEL_PYTHON", sys.executable))
ANALYSIS = Path(os.environ.get("LIMT_ANALYSIS_PYTHON", sys.executable))
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def status(stage, **kwargs):
    write_json(HERE / "status.json", dict(stage=stage, updated_utc=now(), **kwargs))


def checks():
    with (HERE / "checks.log").open("w") as log:
        for command in [
            [str(MODEL), "-m", "pytest", str(HERE / "checks.py"), "-q"],
            [str(ANALYSIS), str(HERE / "analyze.py"), "--self-check"],
            [str(ANALYSIS), str(HERE / "analysis_checks.py")],
            [str(MODEL), str(HERE / "data.py")],
            [str(MODEL), str(HERE / "train.py"), "--smoke"]]:
            subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)


def freeze():
    path = HERE / "locked_before_training.json"
    if path.exists():
        old = json.loads(path.read_text())
        assert old["inputs_sha256"] == input_hashes()
        assert old["data_manifest_sha256"] == sha(HERE / "data/manifest.json")
        return
    for location in [ROOT / "data", HERE / "data"]:
        manifest = json.loads((location / "manifest.json").read_text())
        for name, expected in manifest["output_sha256"].items():
            assert sha(location / name) == expected, name
    write_json(path, dict(created_utc=now(), inputs_sha256=input_hashes(),
        data_manifest_sha256=sha(HERE / "data/manifest.json"), preflight_sha256=sha(HERE / "preflight.json"),
        new_test_forecasts_or_profitability_inspected=False,
        prior_results_available="Original raw-range joint/separate profitability and all earlier results are retrospective context."))


def main(checks_only=False):
    locks = [(ROOT / "run.lock").open("a"), (HERE / "run.lock").open("a")]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    try:
        status("checks_and_label_build")
        checks()
        if checks_only:
            status("checks_passed")
            return
        status("verifying_and_freezing")
        freeze()
        tasks = list(jobs())
        completed = []
        started = time.monotonic()
        for index, (regime, task, seed) in enumerate(tasks):
            out = folder(regime, task, seed)
            out.mkdir(parents=True, exist_ok=True)
            with (out / "train.log").open("a") as log:
                child = subprocess.Popen([str(MODEL), str(HERE / "train.py"), "--regime", regime,
                    "--task", task, "--seed", str(seed)], cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    status("training", running=dict(regime=regime, task=task, seed=seed, pid=child.pid),
                           completed=completed, remaining=len(tasks) - index, total=len(tasks),
                           elapsed_seconds=time.monotonic() - started)
                    time.sleep(10)
                code = child.returncode
                child = None
                if code:
                    raise RuntimeError(f"{regime}/{task}/seed{seed} failed, exit{code}; see {out / 'train.log'}")
            completed.append(dict(regime=regime, task=task, seed=seed))
        status("analyzing", completed=completed)
        with (HERE / "analysis.log").open("a") as log:
            subprocess.run([str(ANALYSIS), str(HERE / "analyze.py")], cwd=ROOT, env=ENV,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        verification = json.loads((HERE / "analysis/verification.json").read_text())
        assert verification["status"] == "passed"
        status("complete", completed=completed, verdicts=verification["verdicts"], report=str(HERE / "REPORT.txt"))
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
        result["fits"] = {str(path.parent.relative_to(HERE / "results")): json.loads(path.read_text())
                          for path in sorted((HERE / "results").glob("*/*/seed*/status.json"))}
        print(json.dumps(result, indent=2))
    else:
        main(args.checks_only)
