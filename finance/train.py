from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.linear_model import Ridge

from arboreal.model import ArborealNetwork
from .data import ROOT, build, digest, month_id, month_label


def mean_ic(prediction, y, months):
    """Mean monthly rank correlation; months, not stock rows, get equal weight."""
    frame = pd.DataFrame({"p": prediction, "y": y, "m": months})
    frame = frame[np.isfinite(frame.p) & np.isfinite(frame.y)]
    frame["p"] = frame.groupby("m").p.rank(method="average")
    # y is already a monotonically transformed rank of the realized return.
    frame["y"] = frame.groupby("m").y.rank(method="average")
    centered = frame[["p", "y"]] - frame.groupby("m")[["p", "y"]].transform("mean")
    frame["cross"] = centered.p * centered.y
    frame["p2"] = centered.p**2
    frame["y2"] = centered.y**2
    sums = frame.groupby("m")[["cross", "p2", "y2"]].sum()
    correlation = sums.cross / np.sqrt(sums.p2 * sums.y2)
    return float(correlation.dropna().mean())


def split_masks(panel, start_year, end_year, config):
    test_start = month_id(f"{start_year}-01-01")
    test_end = month_id(f"{end_year}-12-01")
    val_end = test_start - config["label_embargo_months"] - 1
    val_start = val_end - config["validation_months"] + 1
    train_end = val_start - 1
    train_start = train_end - config["train_months"] + 1
    known = panel.y_rank.notna()
    masks = {"train": panel.month.between(train_start, train_end) & known,
             "val": panel.month.between(val_start, val_end) & known,
             "test": panel.month.between(test_start, test_end)}
    boundaries = {"train_start": train_start, "train_end": train_end,
                  "val_start": val_start, "val_end": val_end,
                  "test_start": test_start, "test_end": test_end}
    assert train_end < val_start <= val_end < test_start - 1
    assert panel.loc[masks["train"], "month"].nunique() == config["train_months"]
    assert panel.loc[masks["val"], "month"].nunique() == config["validation_months"]
    assert panel.loc[masks["test"], "month"].nunique() == 12 * (end_year - start_year + 1)
    return masks, boundaries


def example_weights(months):
    _, ids, counts = np.unique(months, return_inverse=True, return_counts=True)
    return (len(months) / (len(counts) * counts[ids])).astype(np.float32)


def fit_booster(X, y, months, hp, seed, threads):
    params = {"objective": "reg:squarederror", "tree_method": "hist", "max_bin": 128,
              "max_depth": hp["max_depth"], "eta": hp["learning_rate"],
              "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0,
              "base_score": 0.0, "seed": seed, "nthread": threads}
    return xgb.train(params, xgb.DMatrix(X, label=y, weight=example_weights(months)),
                     num_boost_round=hp["n_estimators"])


def prepare_fold(panel, names, start, end, config):
    dest = ROOT / "results" / f"fold_{start}_{end}"
    dest.mkdir(parents=True, exist_ok=True)
    masks, boundaries = split_masks(panel, start, end, config)
    if (dest / "tuning.json").exists():
        return json.loads((dest / "tuning.json").read_text())["selected"]
    train, val, test = [panel.loc[masks[s]] for s in ("train", "val", "test")]
    X = train[names].to_numpy(dtype=np.float32, copy=True)
    y = train.y_rank.to_numpy(dtype=np.float32)
    Xv = val[names].to_numpy(dtype=np.float32, copy=True)
    rows = []
    for hp in config["xgb_candidates"]:
        booster = fit_booster(X, y, train.month.to_numpy(), hp, 0, config["threads_per_job"])
        prediction = booster.predict(xgb.DMatrix(Xv), output_margin=True)
        rows.append({"parameters": hp, "validation_ic": mean_ic(prediction, val.y_rank.to_numpy(), val.month.to_numpy())})
    hp = max(rows, key=lambda r: r["validation_ic"])["parameters"]
    ridge = Ridge(alpha=100.0).fit(X, y, sample_weight=example_weights(train.month.to_numpy()))
    np.savez_compressed(dest / "ridge.npz", row_ids=test.index.to_numpy(),
                        prediction=ridge.predict(test[names].to_numpy(dtype=np.float32)),
                        coefficients=ridge.coef_, intercept=ridge.intercept_)
    payload = {"selected": hp, "candidates": rows, "boundaries": boundaries,
               "boundary_labels": {k: month_label(v) for k, v in boundaries.items()},
               "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
               "selection": "validation mean monthly Spearman IC; seed 0; no test returns accessed",
               "ridge_alpha": 100.0}
    (dest / "tuning.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("TUNED", dest.name, hp, flush=True)
    return hp


def train_one(job):
    start, end, seed, hp = job
    config = json.loads((ROOT / "config.json").read_text())
    dest = ROOT / "results" / f"fold_{start}_{end}" / f"seed{seed}"
    if (dest / "complete.json").exists():
        return str(dest)
    dest.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(ROOT / "data/panel.parquet")
    names = json.loads((ROOT / "data/metadata.json").read_text())["features"]
    masks, boundaries = split_masks(panel, start, end, config)
    train, val, test = [panel.loc[masks[s]].copy() for s in ("train", "val", "test")]
    del panel
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["threads_per_job"])
    torch.use_deterministic_algorithms(True)
    X = train[names].to_numpy(dtype=np.float32, copy=True)
    y = train.y_rank.to_numpy(dtype=np.float32)
    weights = example_weights(train.month.to_numpy())
    Xv = val[names].to_numpy(dtype=np.float32, copy=True)
    Xt = test[names].to_numpy(dtype=np.float32, copy=True)
    started = time.perf_counter()
    booster = fit_booster(X, y, train.month.to_numpy(), hp, seed, config["threads_per_job"])
    booster.save_model(dest / "xgboost.json")
    model = ArborealNetwork.from_xgboost(booster, X[:512], tau1=config["tau1"], tau2=config["tau2"])
    initial_state = copy.deepcopy(model.state_dict())
    initial_ic = mean_ic(model.predict_numpy(Xv), val.y_rank.to_numpy(), val.month.to_numpy())
    best_ic, best_epoch, stale = initial_ic, 0, 0
    best_state = copy.deepcopy(initial_state)
    optimizer = torch.optim.AdamW([model.f, model.v], lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    tensor_X, tensor_y, tensor_w = map(torch.from_numpy, (X, y, weights))
    batches = int(np.ceil(len(X) / config["batch_size"]))
    total_steps, warmup_steps = config["epochs"] * batches, 2 * batches
    generator = torch.Generator().manual_seed(seed)
    history = [{"epoch": 0, "validation_ic": initial_ic, "train_loss": None}]
    step = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(X), generator=generator)
        loss_sum = 0.0
        for ids in order.split(config["batch_size"]):
            multiplier = ((step + 1) / warmup_steps if step < warmup_steps else
                          max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * multiplier
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(tensor_X[ids]) - tensor_y[ids])**2 * tensor_w[ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(ids)
            step += 1
        ic = mean_ic(model.predict_numpy(Xv), val.y_rank.to_numpy(), val.month.to_numpy())
        if ic > best_ic + 1e-10:
            best_ic, best_epoch, stale = ic, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        history.append({"epoch": epoch, "validation_ic": ic, "train_loss": loss_sum / len(X),
                        "elapsed_seconds": time.perf_counter() - started})
        (dest / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"fold={start} seed={seed} epoch={epoch} val_IC={ic:.5f} best={best_epoch} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    model.save(dest / "arbnn.pt")
    # No test labels are used anywhere above. Only predictions are exported here.
    pred_xgb = booster.predict(xgb.DMatrix(Xt), output_margin=True)
    pred_arbnn = model.predict_numpy(Xt)
    pred_hard = model.predict_numpy(Xt, hard=True)
    model.load_state_dict(initial_state)
    initial_hard = model.predict_numpy(Xt[:512], hard=True)
    error = float(np.max(np.abs(initial_hard - pred_xgb[:512])))
    assert error < 2e-4
    model.load_state_dict(best_state)
    np.savez_compressed(dest / "predictions.npz", row_ids=test.index.to_numpy(),
                        xgboost=pred_xgb, arbnn=pred_arbnn, decompiled=pred_hard)
    output = {"start_year": start, "end_year": end, "seed": seed, "parameters": hp,
              "best_validation_ic": best_ic, "initial_soft_validation_ic": initial_ic,
              "best_epoch": best_epoch, "epochs_run": len(history) - 1,
              "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
              "boundaries": boundaries, "hard_compilation_max_error_sample": error,
              "elapsed_seconds": time.perf_counter() - started,
              "config_sha256": digest(ROOT / "config.json"),
              "panel_sha256": json.loads((ROOT / "data/metadata.json").read_text())["panel_sha256"]}
    (dest / "complete.json").write_text(json.dumps(output, indent=2) + "\n")
    return str(dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    build()
    config = json.loads((ROOT / "config.json").read_text())
    panel = pd.read_parquet(ROOT / "data/panel.parquet")
    names = json.loads((ROOT / "data/metadata.json").read_text())["features"]
    jobs = []
    for start, end in config["test_blocks"]:
        hp = prepare_fold(panel, names, start, end, config)
        jobs.extend((start, end, seed, hp) for seed in config["seeds"])
    del panel
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        for result in as_completed([pool.submit(train_one, job) for job in jobs]):
            print("COMPLETED", result.result(), flush=True)


if __name__ == "__main__":
    main()
