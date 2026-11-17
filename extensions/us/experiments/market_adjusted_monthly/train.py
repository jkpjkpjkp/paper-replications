from pathlib import Path
import argparse
import fcntl
import json
import random
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from usdata import config, sha, write_json
from experiments.market_adjusted_monthly.common import HERE, INDEX, TASKS, folder, input_hashes, now, protocol
from experiments.market_adjusted_monthly.data import Panel
from experiments.market_adjusted_monthly.model import Model
from experiments.joint_price_volatility.common import correlation


def save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(protocol()["training"]["cuda_memory_fraction"])
    return torch.device("cuda")


def build_model(task):
    settings = config()["model"]
    return Model(task, **{name: settings[name] for name in ["d_model", "cross_heads", "temporal_heads", "experts", "dropout"]})


def loss_fn(prediction, target, weights):
    valid = torch.isfinite(target)
    if not bool(valid.any(0).all()):
        raise ValueError("No observed labels for an active task")
    residual = prediction - torch.where(valid, target, torch.zeros_like(target))
    mse = (residual.square() * valid).sum(0) / valid.sum(0)
    return (mse * weights).sum()


@torch.no_grad()
def evaluate(model, panel, days, device, record=False):
    model.eval()
    rows, diagnostics = [], []
    for t in days:
        ids, x, normalized, raw = panel.day(t)
        standardized = model(torch.from_numpy(x).to(device)).cpu().numpy()
        assert np.isfinite(standardized).all()
        row = dict(date=panel.dates[t], realized_date=panel.dates[t + 2], symbol=panel.symbols[ids],
                   volatility_month=str(panel.dates[panel.label_end[t]])[:7], label_end_date=panel.dates[panel.label_end[t]])
        for j, task in enumerate(model.tasks):
            k = INDEX[task]
            prediction = standardized[:, j] * panel.scale[k] + panel.mean[k]
            valid = np.isfinite(raw[:, k])
            ic = correlation(prediction[valid], raw[valid, k])
            diagnostics.append(dict(date=panel.dates[t], month=row["volatility_month"], target=task,
                ic=ic, mse=float(np.mean((standardized[valid, j] - normalized[valid, k]) ** 2)), observed=int(valid.sum())))
            if record:
                row[f"pred_{task}"] = prediction
                row[f"label_{task}"] = raw[:, k]
        if record:
            rows.append(pd.DataFrame(row))
    daily = pd.DataFrame(diagnostics)
    metrics = {}
    for task in model.tasks:
        part = daily[daily.target == task]
        if task == "volatility":
            units = part.groupby("month").agg(ic=("ic", "mean"), mse=("mse", "mean"))
        else:
            units = part
        metrics[task] = dict(IC=float(units.ic.mean()), normalized_MSE=float(units.mse.mean()),
                             units=len(units), sampling_unit="target_month" if task == "volatility" else "trading_day")
    return metrics, pd.concat(rows, ignore_index=True) if record else None


def run(regime, task, seed):
    settings = protocol()["training"]
    out = folder(regime, task, seed)
    out.mkdir(parents=True, exist_ok=True)
    writer = (out / "writer.lock").open("a")
    fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"]
    assert sha(HERE / "data/manifest.json") == frozen["data_manifest_sha256"]
    lock = dict(inputs_sha256=frozen["inputs_sha256"], data_manifest_sha256=frozen["data_manifest_sha256"],
                regime=regime, task=task, seed=seed)
    if (out / "lock.json").exists():
        assert json.loads((out / "lock.json").read_text()) == lock
    write_json(out / "lock.json", lock)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text())["phase"] == "complete":
        return
    device = setup(seed)
    model = build_model(task).to(device)
    torch.manual_seed(seed + 100000)
    torch.cuda.manual_seed_all(seed + 100000)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    weights = torch.tensor(settings["weights"] if task == "joint" else [1.], device=device)
    indices = [INDEX[name] for name in model.tasks]
    panel = Panel(regime)
    days = {split: panel.split(split) for split in ["train", "valid", "test"]}
    start, best, stale, best_epoch, history = 0, -np.inf, 0, 0, []
    if (out / "latest.pt").exists():
        checkpoint = torch.load(out / "latest.pt", map_location=device, weights_only=False)
        assert checkpoint["lock"] == lock
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale, best_epoch, history = [checkpoint[k] for k in ["epoch", "best", "stale", "best_epoch", "history"]]
        torch.set_rng_state(checkpoint["rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu())
    write_json(out / "status.json", dict(phase="training", regime=regime, task=task, seed=seed,
                                          epoch=start, best_epoch=best_epoch, updated_utc=now()))
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        begin = time.perf_counter()
        model.train()
        losses = []
        for t in np.random.default_rng(seed + epoch).permutation(days["train"]):
            _, x, normalized, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(x).to(device))
            loss = loss_fn(prediction, torch.from_numpy(normalized[:, indices]).to(device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), settings["gradient_value_clip"])
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, panel, days["valid"], device)
        score = metrics["return" if task == "joint" else task]["IC"]
        if score > best:
            best, stale, best_epoch = score, 0, epoch + 1
            save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation=metrics,
                      best_epoch=best_epoch, stale=stale, seconds=time.perf_counter() - begin)
        history.append(record)
        save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lock=lock, epoch=epoch + 1,
                  best=best, stale=stale, best_epoch=best_epoch, history=history, rng=torch.get_rng_state(),
                  cuda_rng=torch.cuda.get_rng_state()), out / "latest.pt")
        write_json(out / "history.json", history)
        write_json(out / "status.json", dict(phase="training", regime=regime, task=task, seed=seed,
                                              updated_utc=now(), **record))
        print(json.dumps(dict(regime=regime, task=task, seed=seed, **record)), flush=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    write_json(out / "status.json", dict(phase="evaluating", regime=regime, task=task, seed=seed, updated_utc=now()))
    result = {}
    for split in ["valid", "test"]:
        metrics, predictions = evaluate(model, panel, days[split], device, record=True)
        predictions.to_parquet(out / f"{split}_predictions.parquet", index=False)
        result[split] = metrics
        fresh = build_model(task).to(device)
        fresh.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
        fresh.eval()
        for t in days[split][[0, len(days[split]) // 2, -1]]:
            ids, x, _, _ = panel.day(t)
            with torch.no_grad():
                actual = fresh(torch.from_numpy(x).to(device)).cpu().numpy()
            expected = predictions[predictions.date == panel.dates[t]].set_index("symbol").loc[panel.symbols[ids]]
            for j, name in enumerate(model.tasks):
                k = INDEX[name]
                np.testing.assert_allclose(actual[:, j] * panel.scale[k] + panel.mean[k], expected[f"pred_{name}"], rtol=1e-5, atol=1e-7)
        del fresh
    write_json(out / "metrics.json", result)
    paths = [out / name for name in ["best.pt", "history.json", "metrics.json", "valid_predictions.parquet", "test_predictions.parquet"]]
    write_json(out / "provenance.json", dict(**lock, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        parameters=sum(p.numel() for p in model.parameters()), artifacts_sha256={p.name: sha(p) for p in paths},
        replay="first/middle/last validation and test dates passed"))
    write_json(out / "status.json", dict(phase="complete", regime=regime, task=task, seed=seed, epochs=len(history),
        best_epoch=best_epoch, training_seconds=sum(r["seconds"] for r in history), updated_utc=now()))


def smoke():
    panel = Panel("market_adjusted")
    device = setup(0)
    results = {}
    for task in ["joint", "return", "volume", "volatility"]:
        setup(0)
        model = build_model(task).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        indices = [INDEX[name] for name in model.tasks]
        weights = torch.tensor(protocol()["training"]["weights"] if task == "joint" else [1.], device=device)
        for t in panel.split("train")[:3]:
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(x).to(device)), torch.from_numpy(y[:, indices]).to(device), weights)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
        results[task], _ = evaluate(model, panel, panel.split("valid")[:3], device)
        del optimizer, model
        torch.cuda.empty_cache()
    write_json(HERE / "preflight.json", dict(status="passed", metrics=results, test_data_used=False, updated_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["raw_monthly", "market_adjusted"])
    parser.add_argument("--task", choices=["joint", *TASKS])
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.regime is None or args.task is None or args.seed is None:
        parser.error("--regime, --task and --seed are required")
    else:
        run(args.regime, args.task, args.seed)
