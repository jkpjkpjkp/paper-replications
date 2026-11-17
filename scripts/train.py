"""Train on daily panels; checkpoint by validation IC; score each seed separately."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transalpha.data import ROOT, DailyPanel, save_json, sha
from transalpha.model import TransAlpha, LinearControl
from transalpha.loss import hybrid_loss
from transalpha.evaluation import evaluate, correlation


def predict(model, panel, indices, device, batch_size):
    out = np.full((len(indices), len(panel.symbols)), np.nan, np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start + batch_size]
            x, _, mask, stocks = panel.batch(ids, device)
            p = model(x, mask).cpu().numpy()
            for b, ss in enumerate(stocks):
                out[start + b, ss] = p[b, :len(ss)]
    return out


def training_ic(pred, target):
    vals = []
    for p, y in zip(pred, target):
        mask = np.isfinite(p) & np.isfinite(y)
        vals.append(correlation(p[mask], y[mask]))
    return float(np.mean(vals))


def fit(universe, variant, seed, config, panel, device):
    out = ROOT / "results" / universe / f"{variant}-seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps({"config": config, "variant": variant, "seed": seed,
        "data": panel.manifest["array_hashes"],
        "execution": json.loads((ROOT / "data" / universe / "execution_manifest.json").read_text()),
        "code": {name: sha(ROOT / name) for name in ["transalpha/model.py", "transalpha/loss.py", "scripts/train.py", "transalpha/evaluation.py"]}}, sort_keys=True).encode()).hexdigest()
    if (out / "metrics.json").exists():
        prior = json.loads((out / "metrics.json").read_text())
        if prior.get("fingerprint") == fingerprint:
            print(json.dumps({"skipped_verified_run": str(out)}), flush=True)
            return
        raise RuntimeError("Existing run differs from current protocol; choose a new output or archive it: " + str(out))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    model = (LinearControl(panel.x.shape[-1]) if variant == "linear" else
             TransAlpha(panel.x.shape[-1], config["hidden"], config["dropout"],
                        config["threshold"], variant)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_ids = panel.manifest["indices"]["train"]
    val_ids = panel.manifest["indices"]["validation"]
    test_ids = panel.manifest["indices"]["test"]
    rng = np.random.default_rng(seed)
    best, bad, best_epoch = -float("inf"), 0, None
    history = []
    started = time.monotonic()
    batch_size = config["batch_cross_sections"]
    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        lr = config["learning_rate"] / np.sqrt(epoch)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = []
        shuffled = rng.permutation(train_ids).tolist()
        for start in range(0, len(shuffled), batch_size):
            x, y, mask, _ = panel.batch(shuffled[start:start + batch_size], device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x, mask)
            loss, _ = hybrid_loss(prediction, y, model.parameters(), mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_predictions = predict(model, panel, val_ids, device, batch_size)
        val_ic = training_ic(val_predictions, panel.y[val_ids])
        row = {"universe": universe, "variant": variant, "seed": seed, "epoch": epoch,
               "loss": float(np.mean(losses)), "validation_ic": val_ic,
               "lr": float(lr), "seconds": time.monotonic() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_ic > best + 1e-6:
            best, bad, best_epoch = val_ic, 0, epoch
            torch.save({"state_dict": model.state_dict(), "config": config, "variant": variant,
                        "seed": seed, "features": panel.manifest["features"], "fingerprint": fingerprint}, out / "model.pt")
        else:
            bad += 1
        save_json(out / "history.json", history)
        if bad >= config["patience"]:
            break
    checkpoint = torch.load(out / "model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    validation = predict(model, panel, val_ids, device, batch_size)
    predictions = predict(model, panel, test_ids, device, batch_size)
    np.savez_compressed(out / "validation_predictions.npz", prediction=validation,
                        target=panel.y[val_ids], dates=panel.dates[val_ids], symbols=panel.symbols)
    np.savez_compressed(out / "test_predictions.npz", prediction=predictions,
                        target=panel.y[test_ids], dates=panel.dates[test_ids],
                        entry_dates=panel.dates[np.array(test_ids) + 1],
                        exit_dates=panel.dates[np.array(test_ids) + 2], symbols=panel.symbols)
    metrics, daily = evaluate(predictions, panel.y[test_ids], panel.dates[test_ids], panel.symbols,
                              annualization=config["annualization"],
                              portfolio_returns=np.load(ROOT / "data" / universe / "portfolio_returns.npy")[test_ids],
                              execution_tradable=np.load(ROOT / "data" / universe / "execution_tradable.npy")[test_ids])
    daily["entry_date"] = panel.dates[np.array(test_ids) + 1]
    daily["exit_date"] = panel.dates[np.array(test_ids) + 2]
    daily.to_csv(out / "daily.csv", index=False)
    metrics.update({"scope": config["scope"], "universe": universe, "variant": variant, "seed": seed,
                    "fingerprint": fingerprint, "best_epoch": best_epoch, "epochs": len(history),
                    "validation_ic": best, "seconds": time.monotonic() - started,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "checkpoint_sha256": sha(out / "model.pt"),
                    "predictions_sha256": sha(out / "test_predictions.npz")})
    save_json(out / "metrics.json", metrics)
    print(json.dumps({"completed": str(out), "ic": metrics["ic"], "net_sharpe": metrics["net"]["sharpe"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--universes", default="csi300,csi500,csi1000")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--variants", default="transalpha,linear,no_csd,no_tag,no_sp")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((ROOT / "config.json").read_text())
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    save_json(ROOT / "sources" / "environment.json", {"torch": torch.__version__, "cuda": torch.version.cuda,
              "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None})
    for universe in args.universes.split(","):
        panel = DailyPanel(universe, config)
        for variant in args.variants.split(","):
            if variant not in {"transalpha", "linear"} and universe not in config["ablation_universes"]:
                continue
            for seed in map(int, args.seeds.split(",")):
                fit(universe, variant, seed, config, panel, device)
