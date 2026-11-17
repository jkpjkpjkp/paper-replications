"""Audit and compare joint LiMT with three independently trained task models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limt.data import Panel
from limt.model import SingleTaskLiMT
from limt.neutral import neutral_backtest
from limt.portfolio import backtest
from scripts.build_data import sha, write_json
from scripts.report import paired_interval
from scripts.train_separate import TASKS, task_metrics

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["date", "realized_date", "symbol"]
LABELS = dict(return_="Return", volume="Liquidity (volume shock)", volatility="Volatility (range/VWAP)")
LABELS["return"] = LABELS.pop("return_")


def audit_pair(joint, separate, panel):
    reference = json.loads((joint/"config.json").read_text())
    if reference["smoke"] or reference["variant"] != "full":
        raise ValueError("Cannot compare a smoke run or an ablated joint reference")
    if json.loads((joint/"status.json").read_text())["phase"] != "complete":
        raise ValueError(f"Incomplete joint reference: {joint}")
    baseline_provenance = json.loads((joint/"provenance.json").read_text())
    manifest_hash = sha(panel.folder/"manifest.json")
    if baseline_provenance["data_manifest_sha256"] != manifest_hash:
        raise ValueError("Joint model data manifest changed")
    if baseline_provenance["best_sha256"] != sha(joint/"best.pt"):
        raise ValueError("Joint checkpoint changed")
    joint_state = json.loads((joint/"status.json").read_text())
    joint_history = json.loads((joint/"history.json").read_text())
    joint_best = max(joint_history, key=lambda row: row["validation"]["return"]["IC"])
    if joint_best["epoch"] != joint_state["best_epoch"] or len(joint_history) != joint_state["epochs"]:
        raise ValueError("Joint checkpoint was not selected by validation return IC")
    reference_summary = json.loads((joint/"data_summary.json").read_text())
    checks = []
    for task in TASKS:
        folder = separate/task
        config = json.loads((folder/"config.json").read_text())
        state = json.loads((folder/"status.json").read_text())
        if config["smoke"]:
            raise ValueError("Smoke task predictions cannot enter a full comparison")
        if state["phase"] != "complete":
            raise ValueError(f"Incomplete task: {folder}")
        for key, value in reference.items():
            if key != "run_name" and config[key] != value:
                raise ValueError(f"Paired configuration mismatch: {task} {key}")
        if config["reference_config_sha256"] != sha(joint/"config.json"):
            raise ValueError("Reference configuration changed")
        if json.loads((folder/"data_summary.json").read_text()) != reference_summary:
            raise ValueError("Data summaries differ")
        provenance = json.loads((folder/"provenance.json").read_text())
        if provenance["data_manifest_sha256"] != manifest_hash or provenance["best_sha256"] != sha(folder/"best.pt"):
            raise ValueError("Task manifest or checkpoint hash changed")
        if provenance["trainer_sha256"] != sha(ROOT/"scripts/train_separate.py") or provenance["model_code_sha256"] != sha(ROOT/"limt/model.py"):
            raise ValueError("Single-task source code changed since training")
        history = json.loads((folder/"history.json").read_text())
        best = max(history, key=lambda row: row["validation"]["IC"])
        if best["epoch"] != state["best_epoch"] or len(history) != state["epochs"]:
            raise ValueError("Task checkpoint was not selected by its own validation IC")
        np.testing.assert_allclose(best["validation"]["IC"], state["best_validation_ic"], atol=1e-12)
        # Verify one complete held-out cross-section against the saved checkpoint.
        day = panel.split("test")[len(panel.split("test"))//2]
        ids, x, _, _ = panel.day(day)
        model = SingleTaskLiMT(task, dropout=config["dropout"], experts=config["experts"])
        model.load_state_dict(torch.load(folder/"best.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            actual = model(torch.from_numpy(x)).numpy()
        saved = pd.read_parquet(folder/"test_predictions.parquet")
        saved = saved[saved.date == panel.dates[day]].set_index("symbol").loc[panel.symbols[ids], f"pred_{task}"]
        np.testing.assert_allclose(actual, saved, atol=2e-4, rtol=2e-4)
        checks.append(dict(task=task, best_epoch=state["best_epoch"], epochs=state["epochs"],
            checkpoint_reconstruction_max_error=float(np.max(np.abs(actual-saved.to_numpy())))))
    for split in ("valid", "test"):
        a, b = [pd.read_parquet(path/f"{split}_predictions.parquet").sort_values(KEYS).reset_index(drop=True)
                for path in (joint, separate)]
        if a.duplicated(["date", "symbol"]).any() or b.duplicated(["date", "symbol"]).any() or not a[KEYS].equals(b[KEYS]):
            raise ValueError("Joint and independent predictions have different or duplicate keys")
        expected_keys = pd.concat([pd.DataFrame(dict(date=panel.dates[t], realized_date=panel.dates[t+2],
            symbol=panel.symbols[np.flatnonzero(panel.eligible[:, t])])) for t in panel.split(split)], ignore_index=True)
        if not b[KEYS].equals(expected_keys.sort_values(KEYS).reset_index(drop=True)):
            raise ValueError("Predictions do not cover the full eligible split")
        for k, task in enumerate(TASKS):
            np.testing.assert_array_equal(a[f"label_{task}"], b[f"label_{task}"])
            expected_labels = np.concatenate([np.array(panel.labels[np.flatnonzero(panel.eligible[:, t]), t, k])
                                              for t in panel.split(split)])
            expected_keys[f"label_{task}"] = expected_labels
            expected = expected_keys.sort_values(KEYS)[f"label_{task}"].to_numpy()
            np.testing.assert_array_equal(b[f"label_{task}"], expected)
            if not np.isfinite(b[f"pred_{task}"]).all():
                raise ValueError("Non-finite independent prediction")
    return checks


def mean_sd(values, percent=False):
    values = np.asarray(values, dtype=float)*(100 if percent else 1)
    return f"{values.mean():.4f} ± {values.std(ddof=1):.4f}"+("%" if percent else "")


def generate(run_name="separate", reference_run="main"):
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    root, reference = [ROOT/"results"/name for name in (run_name, reference_run)]
    out = root/"comparison"
    daily_out = out/"portfolio_daily"
    daily_out.mkdir(parents=True, exist_ok=True)
    forecast, portfolios, neutral, paired, audits, resources = [], [], [], [], [], []
    daily_ic, nav = {}, {}
    for universe in ("csi300", "csi500"):
        panel = Panel(ROOT/"data"/universe)
        for name, digest in panel.manifest.get("output_hashes", {}).items():
            if sha(panel.folder/name) != digest:
                raise ValueError(f"Panel array hash changed: {universe}/{name}")
        benchmark = pd.read_csv(ROOT/"sources/benchmarks"/f"{universe}.csv").set_index("date")["return"].reindex(panel.dates).to_numpy()
        if not np.isfinite(benchmark[panel.split("test")+2]).all():
            raise ValueError("Missing benchmark returns")
        for seed in range(3):
            joint, separate = [path/universe/f"seed{seed}" for path in (reference, root)]
            checks = audit_pair(joint, separate, panel)
            audits.append(dict(universe=universe, seed=seed, tasks=checks,
                joint_prediction_sha256=sha(joint/"test_predictions.parquet"),
                separate_prediction_sha256=sha(separate/"test_predictions.parquet"),
                data_manifest_sha256=sha(panel.folder/"manifest.json")))
            for method, folder in (("Joint LiMT", joint), ("Separate models", separate)):
                pred = pd.read_parquet(folder/"test_predictions.parquet")
                tag = "joint" if method == "Joint LiMT" else "separate"
                for task in TASKS:
                    metrics = task_metrics(pred, task, rank=True)
                    stored = json.loads((folder/"metrics.json").read_text())["test"][task]
                    for key in ("IC", "ICIR", "days"):
                        np.testing.assert_allclose(metrics[key], stored[key], atol=1e-10)
                    forecast.append(dict(universe=universe, seed=seed, method=method, task=task, **metrics))
                    daily_ic[universe, method, seed, task] = pred.groupby("date").apply(
                        lambda f: f[f"pred_{task}"].corr(f[f"label_{task}"]), include_groups=False)
                for strategy in ("equal", "risk_parity", "rank", "apo", "top50"):
                    daily, metrics = backtest(panel, pred, strategy=strategy, r=.1, benchmark=benchmark)
                    daily.to_csv(daily_out/f"{tag}_{universe}_seed{seed}_{strategy}.csv", index=False)
                    portfolios.append(dict(universe=universe, seed=seed, method=method, strategy=strategy, **metrics))
                    if strategy == "apo":
                        nav[universe, method, seed] = pd.Series((1+daily.return_net).cumprod().to_numpy(),
                                                               index=pd.to_datetime(daily.date))
                for sell_bp in (5., 10., 15.):
                    daily, metrics = neutral_backtest(panel, pred, buy_fee=.0005, sell_fee=sell_bp/10000)
                    daily.to_csv(daily_out/f"{tag}_{universe}_seed{seed}_neutral_sell{sell_bp:g}.csv", index=False)
                    neutral.append(dict(universe=universe, seed=seed, method=method, buy_bp=5., sell_bp=sell_bp, **metrics))
                paths = [joint] if method == "Joint LiMT" else [separate/task for task in TASKS]
                seconds = sum(sum(row["seconds"] for row in json.loads((path/"history.json").read_text())) for path in paths)
                params = sum(json.loads((path/"provenance.json").read_text())["parameters"] for path in paths)
                resources.append(dict(universe=universe, seed=seed, method=method,
                    parameters=params, training_seconds=seconds))
            print(json.dumps(dict(universe=universe, seed=seed, phase="audited_and_backtested")), flush=True)
        for task in TASKS:
            means = {method: pd.concat([daily_ic[universe, method, s, task] for s in range(3)], axis=1).mean(axis=1)
                     for method in ("Joint LiMT", "Separate models")}
            diff = (means["Separate models"]-means["Joint LiMT"]).dropna()
            lo, hi = paired_interval(diff)
            paired.append(dict(universe=universe, task=task, separate_minus_joint_IC=float(diff.mean()),
                bootstrap_95_low=lo, bootstrap_95_high=hi, days=len(diff), block=20, repetitions=2000))
    forecast, portfolios, neutral, resources = [pd.DataFrame(rows) for rows in (forecast, portfolios, neutral, resources)]
    forecast.to_csv(out/"forecast.csv", index=False)
    portfolios.to_csv(out/"portfolios.csv", index=False)
    neutral.to_csv(out/"market_neutral.csv", index=False)
    resources.to_csv(out/"resources.csv", index=False)
    pd.DataFrame(paired).to_csv(out/"paired_ic_bootstrap.csv", index=False)
    write_json(out/"verification.json", dict(phase="passed", pairs=audits,
        engine_sha256=sha(ROOT/"limt/portfolio.py"), neutral_engine_sha256=sha(ROOT/"limt/neutral.py"),
        report_code_sha256=sha(__file__), benchmark_hashes={u: sha(ROOT/"sources/benchmarks"/f"{u}.csv") for u in ("csi300", "csi500")}))
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    navfig, navaxes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ui, universe in enumerate(("csi300", "csi500")):
        for method in ("Joint LiMT", "Separate models"):
            for ti, task in enumerate(TASKS):
                series = pd.concat([daily_ic[universe, method, s, task] for s in range(3)], axis=1).mean(axis=1)
                axes[ui, ti].plot(pd.to_datetime(series.index), series.rolling(60, min_periods=60).mean(), label=method)
                axes[ui, ti].set(title=f"{universe.upper()}: {LABELS[task]}", ylabel="60-day mean IC")
                axes[ui, ti].tick_params(axis="x", rotation=25)
            curves = pd.concat([nav[universe, method, s] for s in range(3)], axis=1).mean(axis=1)
            navaxes[ui].plot(curves.index, (curves-1)*100, label=method)
        navaxes[ui].set(title=universe.upper(), ylabel="Mean APO cumulative return (%)")
        navaxes[ui].legend()
        navaxes[ui].tick_params(axis="x", rotation=25)
    axes[0, 0].legend()
    fig.savefig(out/"rolling_ic.png", dpi=160)
    navfig.savefig(out/"apo_nav.png", dpi=160)
    plt.close("all")
    text = ["# Joint versus separately trained LiMT models", "",
        "Completed: six joint reference runs versus eighteen independent task runs, on the same CSI300/CSI500 data and seeds 0, 1, 2.", ""]
    for universe in ("csi300", "csi500"):
        joint_return = forecast[(forecast.universe == universe) & (forecast.task == "return") & (forecast.method == "Joint LiMT")].IC.mean()
        separate_return = forecast[(forecast.universe == universe) & (forecast.task == "return") & (forecast.method == "Separate models")].IC.mean()
        joint_apo = portfolios[(portfolios.universe == universe) & (portfolios.strategy == "apo") & (portfolios.method == "Joint LiMT")].annualized_return.mean()
        separate_apo = portfolios[(portfolios.universe == universe) & (portfolios.strategy == "apo") & (portfolios.method == "Separate models")].annualized_return.mean()
        text += [f"On {universe.upper()}, mean return IC changes from {joint_return:.4f} (joint) to {separate_return:.4f} (separate), a difference of {separate_return-joint_return:+.4f}. Mean APO annualized return changes from {joint_apo*100:.2f}% to {separate_apo*100:.2f}%, a difference of {(separate_apo-joint_apo)*100:+.2f} percentage points.", ""]
    text += [
        "Liquidity means the existing volume-shock target; volatility means range/VWAP. Each independent model has the same 256-wide cross-stock/temporal attention backbone (4/2 heads), eight private experts, one gate and one task projection. Cross-task fusion is removed. Corresponding modules start from the same seed-matched weights as the joint model. Total capacity across the three models is larger than one joint model; this is a separation comparison rather than a parameter-budget-matched comparison.", "",
        "Data, features, normalization, eligibility, purged train/validation/test dates, 21-day windows, Adam learning rate 1e-5, dropout 0.5, FP32, gradient-value clipping at 3, date shuffles, 100-epoch limit and 21 non-improving epoch patience are held fixed. Each task retains its original loss multiplier (return 0.65, volume 0.30, volatility 0.05). The joint checkpoint uses validation return IC; each independent checkpoint uses its own task's validation IC. This necessary selection change is part of the comparison. Only validation-selected checkpoints are evaluated on test.", "",
        "Numbers are mean ± sample standard deviation across three seeds. Paired IC intervals use seed-averaged daily separate-minus-joint IC and a circular 20-day block bootstrap with 2000 draws. They describe this held-out sample; they do not establish future performance.", "",
        "| Universe | Target | Model | IC | ICIR | RankIC |", "| --- | --- | --- | --- | --- | --- |"]
    for universe in ("csi300", "csi500"):
        for task in TASKS:
            for method in ("Joint LiMT", "Separate models"):
                f = forecast[(forecast.universe == universe) & (forecast.task == task) & (forecast.method == method)]
                text.append(f"| {universe.upper()} | {LABELS[task]} | {method} | {mean_sd(f.IC)} | {mean_sd(f.ICIR)} | {mean_sd(f.RankIC)} |")
    text += ["", "| Universe | Target | Separate minus joint IC | 95% block-bootstrap interval |", "| --- | --- | --- | --- |"]
    for row in paired:
        text.append(f"| {row['universe'].upper()} | {LABELS[row['task']]} | {row['separate_minus_joint_IC']:+.4f} | [{row['bootstrap_95_low']:+.4f}, {row['bootstrap_95_high']:+.4f}] |")
    text += ["", "![Held-out rolling IC](comparison/rolling_ic.png)", "",
        "The independently selected predictions are combined by exact signal/realization-date and symbol keys for the existing portfolio engines. APO r=0.1, CNY 1 billion per side, 2% position and execution caps, 10 bp per traded value, t+1 close execution and t+2 realization are unchanged. Backtests assume short availability and omit borrow fees, board-lot rules and price limits, as documented in the project protocol.", "",
        "| Universe | Sizing | Model | Annualized return | Sharpe | Max drawdown | Gross / initial capital |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for universe in ("csi300", "csi500"):
        for strategy in ("equal", "risk_parity", "rank", "apo", "top50"):
            for method in ("Joint LiMT", "Separate models"):
                f = portfolios[(portfolios.universe == universe) & (portfolios.strategy == strategy) & (portfolios.method == method)]
                text.append(f"| {universe.upper()} | {strategy} | {method} | {mean_sd(f.annualized_return, True)} | {mean_sd(f.sharpe)} | {mean_sd(f.max_drawdown, True)} | {mean_sd(f.mean_gross_exposure, True)} |")
    text += ["", "![APO NAV](comparison/apo_nav.png)", "",
        "The dollar-neutral comparison keeps the existing equal executed long/short notional constraint, 5 bp buys and the same three sell-cost scenarios. These targets depend on return forecasts; the volume and volatility predictions affect APO sizing above.", "",
        "| Universe | Buy / sell fee | Model | Annualized return | Sharpe | Max drawdown |", "| --- | --- | --- | --- | --- | --- |"]
    for universe in ("csi300", "csi500"):
        for sell_bp in (5., 10., 15.):
            for method in ("Joint LiMT", "Separate models"):
                f = neutral[(neutral.universe == universe) & (neutral.sell_bp == sell_bp) & (neutral.method == method)]
                text.append(f"| {universe.upper()} | 5 / {sell_bp:g} bp | {method} | {mean_sd(f.annualized_return, True)} | {mean_sd(f.sharpe)} | {mean_sd(f.max_drawdown, True)} |")
    text += ["", "| Universe | Model | Total trainable parameters | Training hours (mean ± SD) |", "| --- | --- | --- | --- |"]
    for universe in ("csi300", "csi500"):
        for method in ("Joint LiMT", "Separate models"):
            f = resources[(resources.universe == universe) & (resources.method == method)]
            text.append(f"| {universe.upper()} | {method} | {int(f.parameters.iloc[0]):,} | {mean_sd(f.training_seconds/3600)} |")
    text += ["", "Training time sums completed epoch durations, including validation, and is descriptive because hardware contention can differ. Independent models have separate stopping epochs, recorded in the verification file.", "",
        "[Verification and checkpoint reconstruction](comparison/verification.json), [seed-level forecasts](comparison/forecast.csv), [paired IC intervals](comparison/paired_ic_bootstrap.csv), [portfolio metrics](comparison/portfolios.csv), [neutral metrics](comparison/market_neutral.csv), and [resource use](comparison/resources.csv). Per-task configurations, checkpoints, histories and predictions remain in each seed directory. All paired configurations, data manifests, full prediction keys, labels, checkpoint selection and reconstructed predictions were verified before writing this report.", ""]
    (root/"comparison.md").write_text("\n".join(text))
    print(f"Wrote {root/'comparison.md'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", default="separate")
    p.add_argument("--reference-run", default="main")
    args = p.parse_args()
    generate(args.run_name, args.reference_run)
