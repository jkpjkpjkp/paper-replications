"""Resumable fits for the frozen joint-versus-specialist experiment."""
from pathlib import Path
import argparse
import fcntl
import json
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from experiments.joint_price_volatility.model import PriceVolatilityModel, TASK_INDEX, VARIANTS, task_loss
from experiments.joint_price_volatility.common import HERE, correlation, input_hashes, now, protocol
from usdata import Panel, config, sha, write_json

def atomic_save(value, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def setup(seed, device="cuda"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("This full-size experiment requires CUDA")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_per_process_memory_fraction(protocol()["training"]["cuda_memory_fraction"])
    return torch.device(device)


def build_model(variant):
    m = config()["model"]
    return PriceVolatilityModel(variant, d_model=m["d_model"], cross_heads=m["cross_heads"],
                               temporal_heads=m["temporal_heads"], experts=m["experts"], dropout=m["dropout"])


@torch.no_grad()
def evaluate(model, panel, days, device, save=False):
    model.eval()
    values = {task: {"ic": [], "mse": []} for task in model.tasks}
    rows = []
    for t in days:
        ids, x, normalized, raw = panel.day(t)
        prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite forecast")
        record = dict(date=panel.dates[t], realized_date=panel.dates[t + 2], symbol=panel.symbols[ids])
        for j, task in enumerate(model.tasks):
            k = TASK_INDEX[task]
            target = normalized[:, k]
            observed = np.isfinite(target)
            values[task]["ic"].append(correlation(prediction[:, j], raw[:, k]))
            values[task]["mse"].append(float(np.mean((prediction[observed, j] - target[observed]) ** 2)))
            if save:
                record.update({f"pred_{task}": prediction[:, j], f"label_{task}": raw[:, k],
                               f"normalized_{task}": target})
        if save:
            rows.append(pd.DataFrame(record))
    metrics = {task: {"IC": float(np.mean(v["ic"])), "normalized_MSE": float(np.mean(v["mse"])),
                      "days": len(days)} for task, v in values.items()}
    return metrics, pd.concat(rows, ignore_index=True) if save else None


def selection_score(metrics, tasks):
    return float(np.mean([metrics[task]["IC"] for task in tasks]))


def replay(model, panel, days, device, predictions, tasks):
    model.eval()
    for t in days[[0, len(days) // 2, -1]]:
        ids, x, _, _ = panel.day(t)
        with torch.no_grad():
            actual = model(torch.from_numpy(x).to(device)).cpu().numpy()
        expected = predictions[predictions.date == panel.dates[t]].set_index("symbol").loc[panel.symbols[ids]]
        for task in tasks:
            j = model.tasks.index(task)
            np.testing.assert_allclose(actual[:, j], expected[f"pred_{task}"], rtol=1e-5, atol=1e-6)


def smoke():
    """Real 500-stock CUDA updates and validation forecasts, no test access."""
    device = setup(0)
    panel = Panel()
    days = {name: panel.split(name) for name in ["train", "valid"]}
    results = {}
    for variant, tasks in VARIANTS.items():
        setup(0)
        model = build_model(variant).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=protocol()["training"]["learning_rate"])
        start = time.perf_counter()
        losses = []
        model.train()
        for t in days["train"][:3]:
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(x).to(device))
            loss = task_loss(prediction, torch.from_numpy(y[:, [TASK_INDEX[k] for k in tasks]]).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), protocol()["training"]["gradient_value_clip"])
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError("Nonfinite smoke-test gradient")
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, panel, days["valid"][:3], device)
        results[variant] = dict(parameters=sum(p.numel() for p in model.parameters()),
                                training_losses=losses, validation=metrics, seconds=time.perf_counter() - start)
        del optimizer, model
        torch.cuda.empty_cache()
    write_json(HERE / "preflight.json", dict(status="passed", updated_utc=now(), variants=results,
                                           torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                                           test_data_used=False))
    print(json.dumps(results, indent=2), flush=True)


def run(variant, seed):
    settings = protocol()["training"]
    out = HERE / "results" / variant / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    writer_lock = (out / "writer.lock").open("a")
    fcntl.flock(writer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"], "Frozen experiment inputs changed"
    lock = dict(inputs_sha256=frozen["inputs_sha256"], variant=variant, seed=seed)
    if (out / "lock.json").exists():
        assert json.loads((out / "lock.json").read_text()) == lock, "Refusing incompatible checkpoint reuse"
    write_json(out / "lock.json", lock)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("phase") == "complete":
        print("ALREADY COMPLETE", variant, seed, flush=True)
        return
    device = setup(seed)
    model = build_model(variant).to(device)
    # Construction differs only by deterministic extra head extraction. Give all
    # variants identical subsequent dropout RNG streams at the beginning.
    torch.manual_seed(seed + 100000)
    torch.cuda.manual_seed_all(seed + 100000)
    tasks = model.tasks
    indices = [TASK_INDEX[task] for task in tasks]
    panel = Panel()
    days = {name: panel.split(name) for name in ["train", "valid", "test"]}
    summary = {name: dict(days=len(ix), first_signal=str(panel.dates[ix[0]]),
                         last_signal=str(panel.dates[ix[-1]]), last_realization=str(panel.dates[ix[-1] + 2]))
               for name, ix in days.items()}
    assert summary["train"]["last_realization"] <= config()["train"][1] < summary["valid"]["first_signal"]
    assert summary["valid"]["last_realization"] <= config()["valid"][1] < summary["test"]["first_signal"]
    write_json(out / "data_summary.json", summary)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    start, best, stale, best_epoch, history = 0, -np.inf, 0, 0, []
    task_best = {task: -np.inf for task in tasks}
    task_best_epochs = {task: 0 for task in tasks}
    latest = out / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        assert checkpoint["lock"] == lock
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best, stale, best_epoch, history, task_best, task_best_epochs = [checkpoint[key] for key in
            ["epoch", "best", "stale", "best_epoch", "history", "task_best", "task_best_epochs"]]
        torch.set_rng_state(checkpoint["rng"].cpu())
        torch.cuda.set_rng_state(checkpoint["cuda_rng"].cpu(), device)
    write_json(out / "status.json", dict(phase="training", variant=variant, seed=seed,
                                          epoch=start, best_epoch=best_epoch, updated_utc=now()))
    for epoch in range(start, settings["epochs"]):
        if stale >= settings["patience"]:
            break
        begin = time.perf_counter()
        model.train()
        losses = []
        for t in np.random.default_rng(seed + epoch).permutation(days["train"]):
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(torch.from_numpy(x).to(device))
            loss = task_loss(prediction, torch.from_numpy(y[:, indices]).to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), settings["gradient_value_clip"])
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, panel, days["valid"], device)
        score = selection_score(metrics, tasks)
        if score > best:
            best, stale, best_epoch = score, 0, epoch + 1
            atomic_save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        for task in tasks:
            if metrics[task]["IC"] > task_best[task]:
                task_best[task], task_best_epochs[task] = metrics[task]["IC"], epoch + 1
                if variant == "joint":
                    atomic_save(model.state_dict(), out / f"best_{task}.pt")
        record = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation=metrics,
                      selection_score=score, seconds=time.perf_counter() - begin,
                      best_epoch=best_epoch, stale=stale, task_best_epochs=task_best_epochs.copy())
        history.append(record)
        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), lock=lock,
                         epoch=epoch + 1, best=best, stale=stale, best_epoch=best_epoch, history=history,
                         task_best=task_best, task_best_epochs=task_best_epochs,
                         rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), latest)
        write_json(out / "history.json", history)
        write_json(out / "status.json", dict(phase="training", variant=variant, seed=seed,
                                              updated_utc=now(), **record))
        print(json.dumps(dict(variant=variant, seed=seed, **record)), flush=True)
    write_json(out / "status.json", dict(phase="evaluating", variant=variant, seed=seed,
                                          epochs=len(history), updated_utc=now()))
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    results = {}
    for split in ["valid", "test"]:
        metrics, predictions = evaluate(model, panel, days[split], device, save=True)
        predictions.to_parquet(out / f"{split}_predictions.parquet", index=False)
        results[split] = metrics
        # Reconstruct separately and reload the on-disk checkpoint for replay.
        fresh = build_model(variant).to(device)
        fresh.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
        replay(fresh, panel, days[split], device, pd.read_parquet(out / f"{split}_predictions.parquet"), tasks)
        del fresh
    if variant == "joint":
        combined = None
        for task in tasks:
            model.load_state_dict(torch.load(out / f"best_{task}.pt", map_location=device, weights_only=True))
            _, predictions = evaluate(model, panel, days["test"], device, save=True)
            replay(model, panel, days["test"], device, predictions, [task])
            if combined is None:
                combined = predictions
            else:
                combined[f"pred_{task}"] = predictions[f"pred_{task}"]
        combined.to_parquet(out / "test_task_selected_predictions.parquet", index=False)
    write_json(out / "metrics.json", results)
    artifacts = [out / name for name in ["best.pt", "metrics.json", "history.json",
                                       "valid_predictions.parquet", "test_predictions.parquet"]]
    if variant == "joint":
        artifacts += [out / name for name in ["best_return.pt", "best_volatility.pt", "test_task_selected_predictions.parquet"]]
    write_json(out / "provenance.json", dict(**lock, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               parameters=sum(p.numel() for p in model.parameters()),
               artifacts_sha256={p.name: sha(p) for p in artifacts}, replay="passed: first/middle/last dates in each split"))
    write_json(out / "status.json", dict(phase="complete", variant=variant, seed=seed, epochs=len(history),
               best_epoch=best_epoch, task_best_epochs=task_best_epochs, best_validation_score=best,
               training_seconds=sum(row["seconds"] for row in history),
               stop_reason="early_stopping" if stale >= settings["patience"] else "epoch_limit", updated_utc=now()))
    print("COMPLETE", variant, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke()
    elif args.variant is None or args.seed is None:
        parser.error("--variant and --seed are required unless --smoke is set")
    else:
        run(args.variant, args.seed)
