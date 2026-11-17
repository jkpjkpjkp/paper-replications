from __future__ import annotations

import copy
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_squared_error, roc_auc_score, roc_curve)

from .data import file_sha256, load
from .model import ArborealNetwork


def metrics(y, margin, task, y_std=1.0):
    if task == "regression":
        rmse = float(np.sqrt(mean_squared_error(y, margin)))
        return {"rmse": rmse, "rmse_native": rmse * y_std}
    p = expit(margin.astype(np.float64))
    fpr, tpr, _ = roc_curve(y, p)
    ece = 0.0
    # Fixed, documented 15 equal-width probability bins.
    bins = np.minimum((p * 15).astype(int), 14)
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {"auc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "ks": float(np.max(tpr - fpr)), "ece_15": float(ece)}


def primary(result, task):
    return result["auc"] if task == "classification" else -result["rmse"]


def tune_xgboost(data, output, threads=2):
    """Small explicit validation search; exact authors' search is unreleased."""
    output = Path(output)
    if output.exists():
        return json.loads(output.read_text())["selected"]
    task = data["metadata"]["task"]
    candidates = [
        {"max_depth": 3, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1},
        {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05},
    ]
    rows = []
    for config in candidates:
        booster = fit_xgboost(data, config, seed=0, threads=threads)
        pred = booster.predict(xgb.DMatrix(data["X_val"]), output_margin=True)
        result = metrics(data["y_val"], pred, task)
        rows.append({"config": config, "validation": result})
    selected = max(rows, key=lambda row: primary(row["validation"], task))["config"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"candidates": rows, "selected": selected,
                                  "selection": "validation primary metric; model seed 0; no test access"},
                                 indent=2) + "\n")
    return selected


def fit_xgboost(data, config, seed, threads):
    task = data["metadata"]["task"]
    params = {"objective": "binary:logistic" if task == "classification" else "reg:squarederror",
              "tree_method": "hist", "max_depth": config["max_depth"],
              "eta": config["learning_rate"], "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 1.0, "seed": seed,
              "nthread": threads, "base_score": 0.5}
    return xgb.train(params, xgb.DMatrix(data["X_train"], label=data["y_train"]),
                     num_boost_round=config["n_estimators"])


def run(data_path, output, config, seed=0, epochs=30, patience=8, batch_size=2048,
        learning_rate=1e-4, weight_decay=0.02, tau1=100.0, tau2=10.0,
        threads=2, device="cpu"):
    output = Path(output)
    if (output / "result.json").exists():
        raise FileExistsError(f"Completed run exists: {output}; use a new output path")
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    data = load(data_path)
    meta = data["metadata"]
    task = meta["task"]
    all_config = {"xgboost": config, "seed": seed, "epochs": epochs, "patience": patience,
                  "batch_size": batch_size, "learning_rate": learning_rate,
                  "weight_decay": weight_decay, "tau1": tau1, "tau2": tau2,
                  "threads": threads, "device": device,
                  "routing": "algorithm1", "checkpoint_selection": "validation AUC/RMSE including epoch 0",
                  "data_sha256": file_sha256(data_path),
                  "data_metadata": meta}
    (output / "config.json").write_text(json.dumps(all_config, indent=2) + "\n")
    started = time.perf_counter()
    booster = fit_xgboost(data, config, seed, threads)
    baseline_seconds = time.perf_counter() - started
    booster.save_model(output / "xgboost.json")
    model = ArborealNetwork.from_xgboost(booster, data["X_train"][:512],
                                        tau1=tau1, tau2=tau2).to(device)
    compiled_state = copy.deepcopy(model.state_dict())
    initial_val = model.predict_numpy(data["X_val"], batch_size)
    best_val = metrics(data["y_val"], initial_val, task)
    best_score = primary(best_val, task)
    best_state, best_epoch, stale = copy.deepcopy(compiled_state), 0, 0
    optimizer = torch.optim.AdamW([model.f, model.v], lr=learning_rate, weight_decay=weight_decay)
    X_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"])
    steps_per_epoch = math.ceil(len(X_train) / batch_size)
    total_steps, warmup_steps = epochs * steps_per_epoch, 2 * steps_per_epoch
    generator = torch.Generator().manual_seed(seed)
    history = [{"epoch": 0, "train_loss": None, "validation": best_val, "seconds": 0.0}]
    step = 0
    training_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X_train), generator=generator)
        loss_sum = 0.0
        for rows in order.split(batch_size):
            Xb, yb = X_train[rows].to(device), y_train[rows].to(device)
            factor = ((step + 1) / warmup_steps if step < warmup_steps else
                      max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps)))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            prediction = model(Xb)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(prediction, yb)
                    if task == "classification" else torch.nn.functional.mse_loss(prediction, yb))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(rows)
            step += 1
        val_prediction = model.predict_numpy(data["X_val"], batch_size)
        val_result = metrics(data["y_val"], val_prediction, task)
        score = primary(val_result, task)
        improved = score > best_score + 1e-10
        if improved:
            best_score, best_epoch, best_val = score, epoch, val_result
            best_state, stale = copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": loss_sum / len(X_train),
               "validation": val_result, "learning_rate_last_batch": optimizer.param_groups[0]["lr"],
               "seconds": time.perf_counter() - training_start}
        history.append(row)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if epoch == 1 or epoch % 5 == 0 or stale >= patience:
            print(f"{meta['dataset']} seed={seed} epoch={epoch} loss={row['train_loss']:.6f} "
                  f"val={score:.6f} best_epoch={best_epoch} elapsed={row['seconds']:.1f}s", flush=True)
        if stale >= patience:
            break
    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    model.save(output / "arbnn.pt")
    model.export_rules(output / "rules.json")
    test_X, test_y = data["X_test"], data["y_test"]
    pred_xgb = booster.predict(xgb.DMatrix(test_X), output_margin=True)
    pred_soft = model.predict_numpy(test_X, batch_size)
    pred_hard = model.predict_numpy(test_X, batch_size, hard=True)
    model.load_state_dict(compiled_state)
    pred_initial = model.predict_numpy(test_X, batch_size)
    pred_compiled_hard = model.predict_numpy(test_X, batch_size, hard=True)
    compile_error = float(np.max(np.abs(pred_xgb - pred_compiled_hard)))
    if compile_error > 2e-4:
        raise AssertionError(f"Hard compilation disagrees with XGBoost: {compile_error}")
    model.load_state_dict(best_state)
    # Diagnostics on training rows only, for saturation and parameter refinement.
    with torch.no_grad():
        sample = X_train[:512].to(device)
        d = torch.tanh(tau1 * (sample[:, model.features] - model.f))
        saturation = ((d.abs() > 0.9999) & model.node_mask).sum().item()
        saturation /= len(sample) * model.node_mask.sum().item()
        f_delta = (model.f - compiled_state["f"])[model.node_mask].abs()
        v_delta = (model.v - compiled_state["v"])[model.leaf_mask].abs()
    predictions = {"xgboost": pred_xgb, "compiled_soft": pred_initial,
                   "arbnn_soft": pred_soft, "decompiled_hard": pred_hard}
    results = {name: metrics(test_y, pred, task, meta["target_std_train"])
               for name, pred in predictions.items()}
    result = {"dataset": meta["dataset"], "task": task, "seed": seed,
              "best_epoch": best_epoch, "epochs_run": len(history) - 1,
              "best_validation": best_val, "test": results,
              "compile_hard_max_abs_error": compile_error,
              "soft_hard_mean_abs_margin_gap": float(np.mean(np.abs(pred_soft - pred_hard))),
              "soft_hard_max_abs_margin_gap": float(np.max(np.abs(pred_soft - pred_hard))),
              "saturated_train_node_fraction": saturation,
              "threshold_mean_abs_change": float(f_delta.mean()) if f_delta.numel() else 0,
              "leaf_mean_abs_change": float(v_delta.mean()),
              "n_trees": len(model.trees), "internal_nodes": int(model.node_mask.sum()),
              "leaf_nodes": int(model.leaf_mask.sum()),
              "baseline_seconds": baseline_seconds, "training_seconds": training_seconds,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "sklearn": sklearn.__version__,
                              "xgboost": xgb.__version__, "platform": platform.platform()}}
    np.savez_compressed(output / "predictions.npz", y=test_y,
                        row_ids=data["row_ids_test"], **predictions)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
