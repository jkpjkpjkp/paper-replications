#!/usr/bin/env python3
"""Train real models, retain checkpoints, metrics, arrays, and data hashes."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from chaos.basis import cosine_basis, coordinates, discrete_haar, haar_values, multi_indices, wick_features
from chaos.data import ou, heston, ou_coefficients, phi4_dataset, phi4_solve
from chaos.models import SDENO, FSPDENO, real_parameter_count


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def setup(seed, device):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    return torch.device(device)


def metrics(y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    delta = pred - y
    axes = tuple(range(1, y.ndim))
    relative = np.sqrt(np.sum(delta**2, axis=axes)) / np.maximum(np.sqrt(np.sum(y**2, axis=axes)), 1e-12)
    return {"rmse": float(np.sqrt(np.mean(delta**2))), "mse": float(np.mean(delta**2)),
            "relative_l2_mean": float(relative.mean()), "relative_l2_std": float(relative.std(ddof=1)),
            "relative_l2_se": float(relative.std(ddof=1) / np.sqrt(len(y))),
            "relative_l2_global": float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(y.ravel()), 1e-12))}


def environment(device):
    return {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "numpy": np.__version__, "device": str(device), "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_threads": torch.get_num_threads()}


def run_brownian(seed):
    directory = ROOT / "results" / f"brownian_seed{seed}"
    rng = np.random.default_rng(seed)
    n, steps, dt = 1000, 256, 1 / 256
    dw = rng.normal(size=(n, steps)) * np.sqrt(dt)
    truth = np.cumsum(dw, axis=1)
    rows = []
    examples = {}
    for k in [4, 8, 16, 32, 64, 128, 256]:
        b = cosine_basis(steps, k)
        z = coordinates(dw, dt, b)
        pred = np.cumsum(np.sqrt(dt) * z @ b, axis=1)
        rows.append({"modes": k, **metrics(truth, pred)})
        if k in [4, 16, 64, 256]:
            examples[f"prediction_{k}"] = pred[:3]
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "predictions.npz", truth=truth[:3], time=np.arange(1, steps + 1) * dt, **examples)
    write_json(directory / "metrics.json", {"experiment": "brownian", "seed": seed, "paths": n,
                                           "steps": steps, "results": rows, "dw_sha256": digest(dw)})
    print("Brownian reconstruction completed", flush=True)


def sde_features(dw, basis, dt, order, train_steps=None):
    # Fixed [0,1] basis. Never rescale the observed interval when adding a suffix.
    dw = dw.copy()
    if train_steps is not None:
        dw[:, train_steps:] = 0
    if dw.ndim == 2:
        z = coordinates(dw, dt, basis)
    else:
        z = np.concatenate([coordinates(dw[:, :, i], dt, basis) for i in range(dw.shape[-1])], axis=1)
    return wick_features(z, multi_indices(z.shape[-1], order))


def run_sde(kind, cfg, seed, device, force=False):
    label = cfg.get("label", kind)
    directory = ROOT / "results" / f"{label}_seed{seed}"
    if (directory / "metrics.json").exists() and not force:
        print(f"Skipping completed {directory.name}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    setup(seed, str(device))
    ntr, nval, ntest = [cfg[k] for k in ["train_paths", "validation_paths", "test_paths"]]
    rng = np.random.default_rng(seed + 1000)
    shape = (ntr + nval + ntest, cfg["steps"]) + (() if kind == "ou" else (2,))
    dw = rng.normal(size=shape) * np.sqrt(cfg["dt"])
    y = (ou(dw, cfg["dt"], **cfg["ou"])[:, :, None] if kind == "ou"
         else heston(dw, cfg["dt"], **cfg["heston"]))
    modes = cfg["ou_modes"] if kind == "ou" else cfg["heston_modes_per_driver"]
    order = cfg[f"{kind}_order"]
    basis = cosine_basis(cfg["steps"], modes)
    ft = sde_features(dw[:ntr], basis, cfg["dt"], order, cfg["train_steps"])
    fv = sde_features(dw[ntr:ntr+nval], basis, cfg["dt"], order, cfg["train_steps"])
    # Full future noise is supplied only for evaluation. Test paths are independent.
    ftest = sde_features(dw[ntr+nval:], basis, cfg["dt"], order)
    ftest_prefix = sde_features(dw[ntr+nval:], basis, cfg["dt"], order, cfg["train_steps"])
    state_dim = y.shape[-1]
    model = SDENO(ft.shape[-1], state_dim, cfg["width"], encoded=(kind == "heston")).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    times = torch.linspace(0, cfg["steps"] * cfg["dt"], cfg["steps"] + 1, device=device)
    end = cfg["train_steps"] + 1
    # State scales fitted only to the supervised prefix of training trajectories.
    scale_np = np.sqrt(np.mean(y[:ntr, :end]**2, axis=(0, 1))).clip(1e-5)
    scale = torch.tensor(scale_np, dtype=torch.float32, device=device)
    yt = torch.tensor(y[:ntr, :end], dtype=torch.float32, device=device) / scale
    yv = torch.tensor(y[ntr:ntr+nval, :end], dtype=torch.float32, device=device) / scale
    ft, fv = [torch.tensor(a, dtype=torch.float32, device=device) for a in [ft, fv]]
    history, best, best_state, best_epoch = [], float("inf"), None, 0
    start = time.perf_counter()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(ft, times[:end])
        loss = torch.mean((pred - yt)**2)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = torch.mean((model(fv, times[:end]) - yv)**2).item()
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 100 == 0:
            record = {"epoch": epoch, "train_mse_normalized": float(loss.detach()), "val_mse_normalized": val,
                      "elapsed_seconds": time.perf_counter() - start}
            history.append(record)
            write_json(directory / "progress.json", record)
            print(f"{directory.name}: {record}", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        full = model(torch.tensor(ftest, device=device), times).cpu().numpy() * scale_np
        unknown = model(torch.tensor(ftest_prefix, device=device), times).cpu().numpy() * scale_np
        train_full_features = sde_features(dw[:ntr], basis, cfg["dt"], order)
        train_full = model(torch.tensor(train_full_features, device=device), times).cpu().numpy() * scale_np
    truth = y[ntr+nval:]
    future = slice(end, None) if end < len(times) else slice(1, None)
    result = {"experiment": kind, "label": label, "seed": seed, "config": cfg, "environment": environment(device),
              "best_epoch": best_epoch, "selection": f"minimum independent validation MSE on t<={cfg['train_steps']*cfg['dt']:.2f} only",
              "training_seconds": time.perf_counter() - start, "real_parameters": real_parameter_count(model),
              "train_prefix_dw_sha256": digest(dw[:ntr, :cfg["train_steps"]]),
              "test_dw_sha256": digest(dw[ntr+nval:]), "future_noise_used_during_training": cfg["train_steps"] == cfg["steps"],
              "prefix_full_noise": metrics(truth[:, :end], full[:, :end]),
              "future_full_noise": metrics(truth[:, future], full[:, future]),
              "future_unknown_noise": metrics(truth[:, future], unknown[:, future]),
              "same_training_paths_full_noise_prefix": metrics(y[:ntr, :end], train_full[:, :end]),
              "same_training_paths_full_noise_future": metrics(y[:ntr, future], train_full[:, future]),
              "per_state_future_rmse": np.sqrt(np.mean((truth[:, future] - full[:, future])**2, axis=(0,1))).tolist(),
              "prefix_coordinate_gram_eigenvalues": np.linalg.eigvalsh(basis[:,:cfg["train_steps"]] @ basis[:,:cfg["train_steps"]].T).tolist(),
              "history": history}
    arrays = {"truth": truth, "prediction": full, "prediction_unknown_noise": unknown, "time": times.cpu().numpy(),
              "future_rmse_by_time": np.sqrt(np.mean((full - truth)**2, axis=(0,2)))}
    if kind == "ou":
        oracle = wick_features(coordinates(dw[ntr+nval:], cfg["dt"], basis), multi_indices(modes, 1))
        oracle = oracle @ ou_coefficients(basis, cfg["dt"], **cfg["ou"]).T
        oracle = oracle[:, :, None]
        arrays["oracle"] = oracle
        result["oracle_full"] = metrics(truth, oracle)
        result["oracle_future"] = metrics(truth[:, future], oracle[:, future])
        result["oracle_future_noise_absent_rmse_lower_bound"] = float(np.sqrt(np.mean([
            cfg["ou"]["sigma"]**2 * cfg["dt"] * sum((1-cfg["ou"]["theta"]*cfg["dt"])**(2*j) for j in range(k))
            for k in range(1, cfg["steps"] - cfg["train_steps"] + 1)]))) if cfg["train_steps"] < cfg["steps"] else None
    np.savez_compressed(directory / "predictions.npz", **arrays)
    torch.save({"state_dict": best_state, "config": cfg, "kind": kind, "feature_dim": ft.shape[-1],
                "state_dim": state_dim, "scale": scale_np, "basis": basis, "seed": seed}, directory / "model.pt")
    write_json(directory / "metrics.json", result)


def phi_features(data, cfg, target_scale):
    dw, u0, qstd = data["dw"], data["u0"], data["qstd"]
    b = discrete_haar(cfg["steps"])[:cfg["noise_modes"]]
    index = multi_indices(len(b), cfg["hermite_order"])
    n, _, nx = dw.shape
    features = np.empty((n, len(index) + 3, nx), dtype=np.float32)
    # Chunking keeps peak memory bounded even for complete mixed order-2 features.
    for start in range(0, n, 32):
        stop = min(start + 32, n)
        d = dw[start:stop].transpose(0, 2, 1) / np.maximum(qstd[None, :, None], 1e-10)
        z = coordinates(d, cfg["dt"], b)
        features[start:stop, :len(index)] = wick_features(z, index).transpose(0, 2, 1)
    features[:, -3] = u0 / target_scale
    features[:, -2] = np.arange(nx)[None] / nx
    features[:, -1] = qstd[None] / max(float(qstd.max()), 1e-10)
    return features


def run_phi4(cfg, seed, device, random_initial=False, force=False):
    condition = "random" if random_initial else "fixed"
    directory = ROOT / "results" / f"phi4_{condition}_M{cfg['hermite_order']}_seed{seed}"
    if (directory / "metrics.json").exists() and not force:
        print(f"Skipping completed {directory.name}", flush=True)
        return
    directory.mkdir(parents=True, exist_ok=True)
    setup(seed, str(device))
    start = time.perf_counter()
    ntr, nval, ntest = [cfg[k] for k in ["train_paths", "validation_paths", "test_paths"]]
    data = phi4_dataset(ntr+nval+ntest, seed+2000, random_initial, cfg["nx"], cfg["steps"], cfg["dt"], cfg["sigma"],
                        cfg["random_initial_amplitude"],cfg["random_initial_decay"])
    scale = float(np.sqrt(np.mean(data["u"][:ntr, 1:]**2)))
    x = phi_features(data, cfg, scale)
    y = data["u"][:, 1:] / scale
    model = FSPDENO(x.shape[1], haar_values(cfg["steps"], cfg["temporal_basis_count"]),
                    cfg["width"], cfg["fourier_modes"], cfg["layers"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg["scheduler_factor"],
                                                         patience=cfg["scheduler_patience"])
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(tx[:ntr], ty[:ntr]), batch_size=cfg["batch_size"], shuffle=True,
                        generator=generator, num_workers=0)
    best, best_state, best_epoch, history = float("inf"), None, 0, []

    def predict_range(left, right):
        preds = []
        with torch.no_grad():
            for i in range(left, right, cfg["batch_size"]):
                preds.append(model(tx[i:min(i+cfg["batch_size"],right)].to(device)).cpu().numpy())
        return np.concatenate(preds)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        loss_sum = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xb) - yb)**2)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(xb)
        model.eval()
        valpred = predict_range(ntr, ntr+nval)
        val = float(np.mean((valpred - y[ntr:ntr+nval])**2))
        scheduler.step(val)
        if val < best:
            best, best_epoch = val, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = {"epoch": epoch, "train_mse_normalized": loss_sum / ntr, "val_mse_normalized": val,
                  "lr": optimizer.param_groups[0]["lr"], "elapsed_seconds": time.perf_counter() - start}
        history.append(record)
        if epoch == 1 or epoch % 10 == 0:
            write_json(directory / "progress.json", record)
            print(f"{directory.name}: {record}", flush=True)
        if epoch % 25 == 0:
            torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history,
                        "best_state": best_state, "best_epoch": best_epoch, "config": cfg,
                        "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state()}, directory / "latest.pt")
    model.load_state_dict(best_state)
    model.eval()
    pred = predict_range(ntr+nval, len(x)) * scale
    truth = data["u"][ntr+nval:, 1:]
    base = phi4_solve(data["u0"][ntr+nval:], data["dw"][ntr+nval:], cfg["dt"], nonlinear=False)[:, 1:]
    # Timing excludes feature construction; warmup and GPU synchronization explicit.
    sample = tx[ntr+nval:ntr+nval+1].to(device)
    with torch.no_grad():
        for _ in range(5): model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
        timer = time.perf_counter()
        for _ in range(20): model(sample)
        if device.type == "cuda": torch.cuda.synchronize()
    latency = (time.perf_counter() - timer) / 20
    result = {"experiment": "phi4", "condition": condition, "seed": seed, "config": cfg,
              "environment": environment(device), "real_parameters": real_parameter_count(model),
              "best_epoch": best_epoch, "selection": "minimum independent validation trajectory MSE",
              "normalizer_train_only": scale, "training_seconds": time.perf_counter() - start,
              "test": metrics(truth, pred), "linearized_solver_reference": metrics(truth, base),
              "inference_seconds_per_path_excluding_features": latency,
              "train_dw_sha256": digest(data["dw"][:ntr]), "test_dw_sha256": digest(data["dw"][ntr+nval:]),
              "history": history}
    np.savez_compressed(directory / "predictions.npz", truth=truth, prediction=pred,
                        linear_reference=base, u0=data["u0"][ntr+nval:],
                        time=np.arange(1,cfg["steps"]+1)*cfg["dt"], space=np.arange(cfg["nx"])/cfg["nx"])
    torch.save({"state_dict": best_state, "config": cfg, "scale": scale,
                "in_channels": x.shape[1], "random_initial": random_initial, "seed": seed}, directory / "model.pt")
    write_json(directory / "metrics.json", result)
    print(f"FINISHED {directory.name}: {result['test']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=["brownian", "ou", "heston", "phi4"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--phi-condition", choices=["fixed", "random", "both"], default="both")
    parser.add_argument("--phi-order", type=int)
    parser.add_argument("--sde-full-horizon", action="store_true", help="supervised control; not extrapolation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    for seed in args.seeds:
        if "brownian" in args.experiments: run_brownian(seed)
        for kind in ["ou", "heston"]:
            if kind in args.experiments:
                cfg = json.loads((ROOT / "configs/sde.json").read_text())
                if args.sde_full_horizon:
                    cfg["train_steps"] = cfg["steps"]
                    cfg["label"] = kind + "_full_horizon"
                if args.epochs: cfg["epochs"] = args.epochs
                run_sde(kind, cfg, seed, device, args.force)
        if "phi4" in args.experiments:
            cfg = json.loads((ROOT / "configs/phi4.json").read_text())
            if args.epochs: cfg["epochs"] = args.epochs
            if args.phi_order is not None: cfg["hermite_order"] = args.phi_order
            for condition in ["fixed", "random"]:
                if args.phi_condition in [condition, "both"]:
                    run_phi4(cfg, seed, device, condition == "random", args.force)


if __name__ == "__main__":
    main()
