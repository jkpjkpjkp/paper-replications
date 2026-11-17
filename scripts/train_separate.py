"""Train one independent task using its paired joint run's fixed settings."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.model import SingleTaskLiMT
from scripts.build_data import sha, write_json
from scripts.train import corr, masked_loss

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("return", "volume", "volatility")


def task_metrics(frame, task, rank=False):
    grouped = frame.groupby("date", sort=True)
    daily = grouped.apply(lambda f: corr(f[f"pred_{task}"].to_numpy(),
        f[f"label_{task}"].to_numpy()), include_groups=False)
    metrics = dict(IC=float(daily.mean()), ICIR=float(daily.mean()/daily.std(ddof=1)),
                   days=int(daily.notna().sum()))
    if rank:
        ranked = grouped.apply(lambda f: f[f"pred_{task}"].corr(
            f[f"label_{task}"], method="spearman"), include_groups=False)
        metrics.update(RankIC=float(ranked.mean()), RankICIR=float(ranked.mean()/ranked.std(ddof=1)))
    return metrics


@torch.no_grad()
def evaluate_task(model, panel, days, device, task, rank=False):
    model.eval()
    rows = []
    k = TASKS.index(task)
    for t in days:
        ids, x, _, y = panel.day(t)
        pred = model(torch.from_numpy(x).to(device)).cpu().numpy()
        if not np.isfinite(pred).all():
            raise FloatingPointError("Non-finite predictions")
        rows.append(pd.DataFrame(dict(date=panel.dates[t], realized_date=panel.dates[t+2],
            symbol=panel.symbols[ids], **{f"pred_{task}": pred, f"label_{task}": y[:, k]})))
    frame = pd.concat(rows, ignore_index=True)
    return task_metrics(frame, task, rank=rank), frame


def load_config(args):
    path = ROOT/"results"/args.reference_run/args.universe/f"seed{args.seed}"/"config.json"
    reference = json.loads(path.read_text())
    if reference["variant"] != "full" or reference["smoke"]:
        raise ValueError("Reference must be a full-date joint model")
    config = dict(reference, run_name=args.run_name, task=args.task,
                  reference_run=args.reference_run, reference_config_sha256=sha(path),
                  selection_metric=f"validation.{args.task}.IC",
                  loss_weight=reference["weights"][TASKS.index(args.task)],
                  model_code_sha256=sha(ROOT/"limt/model.py"), trainer_sha256=sha(__file__))
    if args.smoke:
        config.update(smoke=args.smoke, epochs=2, patience=2)
    return config


def run(args):
    config = load_config(args)
    out = ROOT/"results"/args.run_name/args.universe/f"seed{args.seed}"/args.task
    out.mkdir(parents=True, exist_ok=True)
    existing = out/"config.json"
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise ValueError("Refusing to change the configuration of an existing run")
    write_json(existing, config)
    if (out/"status.json").exists() and json.loads((out/"status.json").read_text())["phase"] == "complete":
        return
    panel = Panel(ROOT/"data"/args.universe)
    days = {s: panel.split(s) for s in ("train", "valid", "test")}
    if config["smoke"]:
        days = {s: d[:config["smoke"]] for s, d in days.items()}
    write_json(out/"data_summary.json", {s: dict(days=len(d), first=str(panel.dates[d[0]]),
        last=str(panel.dates[d[-1]]), last_realization=str(panel.dates[d[-1]+2]),
        observations=int(panel.eligible[:, d].sum())) for s, d in days.items()})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high" if config["tf32"] else "highest")
    torch.backends.cudnn.benchmark = False
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = SingleTaskLiMT(args.task, dropout=config["dropout"], experts=config["experts"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    weights = torch.tensor([config["loss_weight"]], device=device)
    best, stale, start, history, best_epoch = -np.inf, 0, 0, [], 0
    latest = out/"latest.pt"
    if latest.exists():
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best, stale, start, history, best_epoch = [ck[k] for k in ("best", "stale", "epoch", "history", "best_epoch")]
        torch.set_rng_state(ck["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(ck["cuda_rng"], device)
    write_json(out/"status.json", dict(phase="training", epoch=start, best_epoch=best_epoch, stale=stale))
    k = TASKS.index(args.task)
    for epoch in range(start, config["epochs"]):
        if stale >= config["patience"]:
            break
        clock = time.perf_counter()
        model.train()
        losses = []
        for t in np.random.default_rng(args.seed+epoch).permutation(days["train"]):
            _, x, y, _ = panel.day(t)
            optimizer.zero_grad(set_to_none=True)
            pred = model(torch.from_numpy(x).to(device))
            loss = masked_loss(pred[:, None], torch.from_numpy(y[:, k:k+1]).to(device), weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
            optimizer.step()
            losses.append(loss.item())
        metric, _ = evaluate_task(model, panel, days["valid"], device, args.task)
        valid = metric["IC"]
        if not np.isfinite(valid):
            raise FloatingPointError("Non-finite validation IC")
        if valid > best:
            best, stale, best_epoch = valid, 0, epoch+1
            torch.save(model.state_dict(), out/"best.pt")
        else:
            stale += 1
        record = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation=metric,
                      seconds=time.perf_counter()-clock, best_epoch=best_epoch, stale=stale)
        history.append(record)
        write_json(out/"history.json", history)
        # Replace the checkpoint only after it is fully written.
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), best=best,
            stale=stale, epoch=epoch+1, history=history, best_epoch=best_epoch, rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device) if device.type == "cuda" else None), out/"latest.tmp")
        (out/"latest.tmp").replace(latest)
        write_json(out/"status.json", dict(phase="training", **record))
        print(json.dumps(dict(universe=args.universe, seed=args.seed, task=args.task, **record)), flush=True)
    model.load_state_dict(torch.load(out/"best.pt", map_location=device, weights_only=True))
    results = {}
    for split in ("valid", "test"):
        metric, pred = evaluate_task(model, panel, days[split], device, args.task, rank=True)
        results[split] = {args.task: metric}
        pred.to_parquet(out/f"{split}_predictions.parquet", index=False)
    write_json(out/"metrics.json", results)
    write_json(out/"provenance.json", dict(data_manifest_sha256=sha(panel.folder/"manifest.json"),
        model_code_sha256=sha(ROOT/"limt/model.py"), trainer_sha256=sha(__file__),
        author_commit="c0232623d50db26b02275b6f77b682f25effeec5", torch=torch.__version__,
        best_sha256=sha(out/"best.pt"), parameters=sum(p.numel() for p in model.parameters()),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None))
    write_json(out/"status.json", dict(phase="complete", epochs=len(history), best_epoch=best_epoch,
        stop_reason="early_stopping" if stale >= config["patience"] else "epoch_limit",
        best_validation_ic=best, test=results["test"], updated_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(dict(universe=args.universe, seed=args.seed, task=args.task, phase="complete",
                          test=results["test"])), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--universe", choices=["csi300", "csi500"], default="csi300")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--run-name", default="separate")
    p.add_argument("--reference-run", default="main")
    p.add_argument("--smoke", type=int, default=0)
    run(p.parse_args())
