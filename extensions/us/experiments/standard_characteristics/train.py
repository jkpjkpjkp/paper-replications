"""Resumable three-seed LiMT training; selection never reads test outcomes."""
from pathlib import Path
import sys
import argparse
import json
import random
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from usdata import sha, write_json
from train import evaluate, loss_fn
from experiments.standard_characteristics.common import HERE, DATA, config, input_hashes, now
from experiments.standard_characteristics.data import Panel
from experiments.standard_characteristics.model import Model


def run(seed, smoke=False):
    p = config()
    panel = Panel()
    days = {s: panel.split(s) for s in ["train", "valid", "test"]}
    summary = {s: dict(days=len(ix), first_signal=str(panel.dates[ix[0]]),
        last_signal=str(panel.dates[ix[-1]]), last_realization=str(panel.dates[ix[-1] + 2]),
        min_stocks=int(panel.member[:, ix].sum(0).min()), max_stocks=int(panel.member[:, ix].sum(0).max()))
        for s, ix in days.items()}
    assert summary["train"]["last_realization"] <= p["train"][1] < summary["valid"]["first_signal"]
    assert summary["valid"]["last_realization"] <= p["valid"][1] < summary["test"]["first_signal"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(.35)
    device = torch.device("cuda")
    model = Model(panel.month_features.shape[-1], **p["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["training"]["learning_rate"])
    weights = torch.tensor(p["training"]["weights"], device=device)
    if smoke:
        started = time.perf_counter()
        for t in days["train"][:3]:
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(torch.from_numpy(x).to(device)), torch.from_numpy(y).to(device), weights)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), p["training"]["gradient_value_clip"])
            optimizer.step()
        metrics, _ = evaluate(model, panel, days["valid"][:3], device)
        write_json(HERE / "preflight.json", dict(status="passed", seconds=time.perf_counter() - started,
            features=panel.month_features.shape[-1], parameters=sum(a.numel() for a in model.parameters()),
            split_summary=summary, validation_smoke_metrics=metrics, no_test_predictions_inspected=True))
        return
    out = HERE / "results" / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    locked = json.loads((HERE / "locked_before_training.json").read_text())
    assert locked["inputs_sha256"] == input_hashes()
    assert locked["data_manifest_sha256"] == sha(DATA / "manifest.json")
    lock = dict(frozen_sha256=sha(HERE / "locked_before_training.json"), seed=seed)
    if (out / "lock.json").exists():
        assert json.loads((out / "lock.json").read_text()) == lock
    write_json(out / "lock.json", lock)
    if (out / "status.json").exists() and json.loads((out / "status.json").read_text()).get("phase") == "complete":
        print("Already complete", seed, flush=True)
        return
    write_json(out / "data_summary.json", summary)
    history, best, stale, start, best_epoch = [], -np.inf, 0, 0, 0
    latest = out / "latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best, stale, start, history, best_epoch = [ck[k] for k in ["best", "stale", "epoch", "history", "best_epoch"]]
        torch.set_rng_state(ck["rng"].cpu())
        torch.cuda.set_rng_state(ck["cuda_rng"].cpu(), device)
    for epoch in range(start, p["training"]["epochs"]):
        if stale >= p["training"]["patience"]:
            break
        begin = time.perf_counter()
        model.train()
        losses = []
        for t in np.random.default_rng(seed + epoch).permutation(days["train"]):
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(x).to(device))
            loss = loss_fn(pred, torch.from_numpy(y).to(device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), p["training"]["gradient_value_clip"])
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, panel, days["valid"], device)
        valid = metrics["return"]["IC"]
        assert np.isfinite(valid)
        if valid > best:
            best, stale, best_epoch = valid, 0, epoch + 1
            torch.save(model.state_dict(), out / "best.pt")
        else:
            stale += 1
        rec = dict(epoch=epoch + 1, train_loss=float(np.mean(losses)), validation=metrics,
            seconds=time.perf_counter() - begin, best_epoch=best_epoch, stale=stale)
        history.append(rec)
        write_json(out / "history.json", history)
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best=best, stale=stale,
            epoch=epoch + 1, history=history, best_epoch=best_epoch, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state()), latest)
        write_json(out / "status.json", dict(phase="training", seed=seed, updated_utc=now(), **rec))
        print(json.dumps(dict(seed=seed, **rec)), flush=True)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=True))
    results = {}
    for split in ["valid", "test"]:
        metrics, pred = evaluate(model, panel, days[split], device, save=True)
        ric = pred.groupby("date", sort=True).apply(
            lambda f: f.pred_return.corr(f.label_return, method="spearman"), include_groups=False)
        metrics["return"].update(RankIC=float(ric.mean()), RankICIR=float(ric.mean() / ric.std(ddof=1)))
        pred.to_parquet(out / f"{split}_predictions.parquet", index=False)
        results[split] = metrics
    write_json(out / "metrics.json", results)
    write_json(out / "provenance.json", dict(**lock, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
        best_sha256=sha(out / "best.pt"), prediction_sha256=sha(out / "test_predictions.parquet")))
    write_json(out / "status.json", dict(phase="complete", seed=seed, epochs=len(history), best_epoch=best_epoch,
        best_validation_ic=best, updated_utc=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.seed, args.smoke)
