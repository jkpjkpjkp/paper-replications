"""Run the preregistered literal-method Euro study under a shared resource cap."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/paper_method/study"
SEEDS=[16,37,47,214,1947]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    # Predeclared before examining any five-epoch test outputs. Baselines use
    # the identical raw panel, split, network and complete-pass training budget.
    for method in ["dfl","historic","pfl"]:
        for seed in ([0] if method=="historic" else SEEDS):
            for rho in [10,15,20]:
                jobs.append(dict(market="euro50",method=method,seed=seed,rho=rho,epochs=0 if method=="historic" else 5))
    # Breadth checks are labeled one-epoch/one-seed diagnostics, not Table1
    # five-seed aggregates. They cannot replace missing Euro repeats.
    for market in ["ftse100","nikkei225","kospi200"]:
        jobs.append(dict(market=market,method="dfl",seed=16,rho=10,epochs=1))
    spec={"created_utc":datetime.now(timezone.utc).isoformat(),"jobs":jobs,
          "shared_memory_max_gib":4,"max_workers":2,"cpu_quota_percent":400,
          "protocol_sha256":hashlib.sha256((ROOT/"reproduction/paper_method_protocol.json").read_bytes()).hexdigest(),
          "code_sha256":hashlib.sha256((ROOT/"reproduction/paper_method.py").read_bytes()).hexdigest(),
          "scope":"Five-seed/all-three-k Euro literal-method study; three other markets one-epoch diagnostics",
          "epoch_count_is_reconstruction_choice":True}
    manifest=OUT/"manifest.json"
    if manifest.exists():
        old=json.loads(manifest.read_text())
        for key in ["jobs","protocol_sha256"]:
            if old[key]!=spec[key]:
                raise RuntimeError("Cannot resume a study after changing its code, protocol, or schedule")
        if old["code_sha256"]!=spec["code_sha256"]:
            if "--resume-after-resource-fix" not in sys.argv or list(OUT.glob("*/result.json")):
                raise RuntimeError("Code changed without an explicitly reviewed resource-only restart")
            spec["resource_restart_parent_source_sha256"]=old["code_sha256"]
            manifest.write_text(json.dumps(spec,indent=2))
        elif old.get("max_workers")!=spec["max_workers"]:
            old["max_workers"]=spec["max_workers"]
            old["resource_note"]="Full GC and malloc_trim reduced measured single Euro worker peak below 0.6 GiB; two workers restored under the unchanged shared 4 GiB cap."
            manifest.write_text(json.dumps(old,indent=2))
    else:
        manifest.write_text(json.dumps(spec,indent=2))
    env=dict(os.environ,OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="2",MKL_NUM_THREADS="2")
    def one(job):
        name=f'{job["market"]}_{job["method"]}_rho{job["rho"]}_seed{job["seed"]}_epochs{job["epochs"]}'
        directory=OUT/name
        result_path=directory/"result.json"
        if result_path.exists():
            return json.loads(result_path.read_text())
        if directory.exists() and not (directory/"checkpoint.pt").exists():
            return {**job,"status":"interrupted_requires_review","output":str(directory)}
        command=[sys.executable,"-u",str(ROOT/"reproduction/paper_method.py"),
                 "--data",str(ROOT/f'data/daily/{job["market"]}_160101_251231.csv'),
                 "--output",str(directory),"--method",job["method"],"--seed",str(job["seed"]),
                 "--rho",str(job["rho"]),"--epochs",str(job["epochs"])]
        if directory.exists():
            command.append("--resume")
        print("START",name,flush=True)
        with (OUT/f'{name}.log').open("a") as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        observed=json.loads(result_path.read_text()) if result_path.exists() else {**job,"status":"process_failed","exit_code":result.returncode}
        print("FINISH",name,observed["status"],flush=True)
        return observed
    records=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(one,job) for job in jobs]):
            records.append(future.result())
            (OUT/"status.json").write_text(json.dumps({"completed_or_failed_jobs":len(records),"total_jobs":len(jobs),"results":records},indent=2))
    print("FINISHED",len(records),"jobs",flush=True)


if __name__=="__main__":
    main()
