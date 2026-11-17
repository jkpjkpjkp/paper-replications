"""One-trainer coordinator; safely resumes all nine fits and final analysis."""
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

from experiments.joint_price_volatility.common import HERE, VARIANTS, input_hashes, now, protocol
from usdata import sha, write_json

MODEL_PYTHON = Path(os.environ.get("LIMT_MODEL_PYTHON", sys.executable))
ANALYSIS_PYTHON = Path(os.environ.get("LIMT_ANALYSIS_PYTHON", sys.executable))
ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")


def status(stage, **kwargs):
    write_json(HERE / "status.json", dict(stage=stage, updated_utc=now(), **kwargs))


def checks():
    with (HERE / "checks.log").open("w") as log:
        for command in [
            [str(MODEL_PYTHON), "-m", "pytest", str(ROOT / "tests"), "-q"],
            [str(ANALYSIS_PYTHON), str(HERE / "check_analysis.py")],
            [str(MODEL_PYTHON), str(HERE / "train.py"), "--smoke"],
        ]:
            subprocess.run(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT, check=True)


def freeze():
    hashes = input_hashes()
    path = HERE / "locked_before_training.json"
    if path.exists():
        assert json.loads(path.read_text())["inputs_sha256"] == hashes, "Frozen experiment inputs changed"
        return
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        assert sha(ROOT / "data" / name) == expected, f"Changed frozen data array: {name}"
    write_json(path, dict(created_utc=now(), inputs_sha256=hashes, data_array_hashes_verified=True,
                          checks_sha256=sha(HERE / "checks.log"), preflight_sha256=sha(HERE / "preflight.json"),
                          test_predictions_inspected_for_this_design=False,
                          holdout_status="Retrospective follow-up; the base experiment's test results were previously available."))


def main(checks_only=False):
    # Cooperate with the original pipeline as well as this experiment's runner.
    locks = [(ROOT / "run.lock").open("a"), (HERE / "run.lock").open("a")]
    for lock in locks:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    try:
        status("preflight")
        checks()
        if checks_only:
            status("checks_passed")
            print("All model, inference and real-panel CUDA checks passed.", flush=True)
            return
        status("verifying_and_freezing_inputs")
        freeze()
        jobs = [(variant, seed) for seed in protocol()["seeds"] for variant in VARIANTS]
        complete = []
        started = time.monotonic()
        for index, (variant, seed) in enumerate(jobs):
            output = HERE / "results" / variant / f"seed{seed}"
            output.mkdir(parents=True, exist_ok=True)
            with (output / "train.log").open("a") as log:
                command = [str(MODEL_PYTHON), str(HERE / "train.py"), "--variant", variant, "--seed", str(seed)]
                child = subprocess.Popen(command, cwd=ROOT, env=ENV, stdout=log, stderr=subprocess.STDOUT)
                print("RUNNING", variant, seed, "pid", child.pid, flush=True)
                while child.poll() is None:
                    status("training", running=dict(variant=variant, seed=seed, pid=child.pid),
                           completed=complete, pending=[dict(variant=v, seed=s) for v, s in jobs[index + 1:]],
                           elapsed_seconds=time.monotonic() - started)
                    time.sleep(10)
                code = child.returncode
                child = None
                if code:
                    raise RuntimeError(f"Fit failed: {variant} seed{seed}, exit {code}; see {output / 'train.log'}")
            complete.append(dict(variant=variant, seed=seed))
        status("analyzing", completed=complete)
        with (HERE / "analysis.log").open("a") as log:
            subprocess.run([str(ANALYSIS_PYTHON), str(HERE / "analyze.py")], cwd=ROOT, env=ENV,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        verification = json.loads((HERE / "analysis/verification.json").read_text())
        assert verification["status"] == "passed"
        status("complete", completed=complete, verdict=verification["verdict"],
               report=str(HERE / "REPORT.txt"), elapsed_seconds=time.monotonic() - started)
        print("COMPLETE", verification["verdict"], flush=True)
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
                          for path in sorted((HERE / "results").glob("*/seed*/status.json"))}
        print(json.dumps(result, indent=2))
    else:
        main(args.checks_only)
