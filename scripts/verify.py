#!/usr/bin/env python3
"""Recompute test predictions from saved checkpoints and regenerated noise."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from chaos.data import phi4_dataset
from scripts.predict import predict
from scripts.reproduce import digest,metrics,write_json


def main():
    torch.set_num_threads(2)
    checks=[]
    for p in sorted((ROOT/"results").glob("*/metrics.json")):
        d=json.loads(p.read_text())
        if d["experiment"]=="brownian":
            assert d["results"][-1]["rmse"]<1e-12
            checks.append({"run":p.parent.name,"full_rank_brownian_exact":True})
            continue
        cfg=d["config"];seed=d["seed"]
        first=cfg["train_paths"]+cfg["validation_paths"]
        total=first+cfg["test_paths"]
        if d["experiment"]=="phi4":
            data=phi4_dataset(total,seed+2000,d["condition"]=="random",cfg["nx"],cfg["steps"],cfg["dt"],cfg["sigma"],
                               cfg["random_initial_amplitude"],cfg["random_initial_decay"])
            assert digest(data["dw"][first:])==d["test_dw_sha256"]
            count=cfg["batch_size"]
            inputs={"dw":data["dw"][first:first+count],"u0":data["u0"][first:first+count],"qstd":data["qstd"]}
            comparison_device=d["environment"]["device"]
            if comparison_device.startswith("cuda") and not torch.cuda.is_available():comparison_device="cpu"
            regenerated=predict(p.parent/"model.pt",inputs,comparison_device)[:,1:]
            field="test"
        else:
            rng=np.random.default_rng(seed+1000)
            shape=(total,cfg["steps"])+(() if d["experiment"]=="ou" else (2,))
            dw=rng.normal(size=shape)*np.sqrt(cfg["dt"])
            assert digest(dw[first:])==d["test_dw_sha256"]
            count=3
            comparison_device="cpu"
            regenerated=predict(p.parent/"model.pt",{"dw":dw[first:first+count]})
            field="prefix_full_noise"
        stored=np.load(p.parent/"predictions.npz")
        delta=float(np.max(np.abs(regenerated-stored["prediction"][:count])))
        tolerance=1e-6 if comparison_device==d["environment"]["device"] else 5e-3*d["normalizer_train_only"]
        assert delta<tolerance,(p,delta,tolerance)
        if field=="test":
            actual=metrics(stored["truth"],stored["prediction"])
        else:
            end=cfg["train_steps"]+1
            actual=metrics(stored["truth"][:,:end],stored["prediction"][:,:end])
            future=slice(end,None) if end<stored["truth"].shape[1] else slice(1,None)
            future_rmse=float(np.sqrt(np.mean((stored["prediction"][:,future].astype(float)-stored["truth"][:,future])**2)))
            assert abs(future_rmse-d["future_full_noise"]["rmse"])<1e-10
            unknown_rmse=float(np.sqrt(np.mean((stored["prediction_unknown_noise"][:,future].astype(float)-stored["truth"][:,future])**2)))
            assert abs(unknown_rmse-d["future_unknown_noise"]["rmse"])<1e-10
        assert abs(actual["rmse"]-d[field]["rmse"])<1e-10
        checks.append({"run":p.parent.name,"test_noise_hash_matches":True,
                       "checkpoint_prediction_max_absolute_difference":delta,"saved_metric_recomputed":True,
                       "comparison_device":comparison_device,"comparison_tolerance":tolerance})
        print("Verified",p.parent.name,flush=True)
    # Bind delivered code, config, checkpoint, metrics and plotted arrays to hashes.
    files={}
    paths=list(ROOT.glob("chaos/*.py"))+list(ROOT.glob("scripts/*.py"))+list(ROOT.glob("configs/*.json"))
    paths+=list((ROOT/"results").glob("*/model.pt"))+list((ROOT/"results").glob("*/metrics.json"))+list((ROOT/"results").glob("*/predictions.npz"))
    paths+=list((ROOT/"report").glob("*summary.json"))+list((ROOT/"report").glob("*summary.csv"))
    for p in sorted(paths):files[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    write_json(ROOT/"verification.json",{"status":"passed","checks":checks,"sha256":files,
                                         "comparison_policy":"same device and SPDE batch size where available; CPU fallback permits 0.5% of training RMS for CUDA convolution precision differences"})


if __name__=="__main__":main()
