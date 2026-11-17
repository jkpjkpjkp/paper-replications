"""Audit/reuse PFL artifacts and fill gaps using the untouched author entry point.

Published architecture/batch size override release defaults. Everything else
follows protocol.json. This reproduces released PFL, not the unavailable author
data snapshot or the unreleased SD-relaxation cardinality-matching experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)

from compare_published import ROOT, atomic_text, collect, compare, digest, panel_info
from compare_published import REPORTING_CONVENTION, REPORTING_NOTE, comparison_display

sys.path.insert(0, str(ROOT / "upstream/src"))
import main as author
from artifacts import make_exp_name, pfl_model_dir_name

ARTIFACTS = ROOT / "artifacts_released"
SELECTORS = ["oscar_hard", "ssrm"]


def write_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def arguments(market, seed, save_dir=ARTIFACTS):
    return ["--mode", "pfl", "--dataset", market, "--seed", str(seed),
            "--hidden_dims", "512", "256", "--batchsize", "64", "--epochs", "300",
            "--lr", ".001", "--weight_decay", ".0001", "--activation", "tanh",
            "--dropout", ".1", "--cuda", "-1", "--splits", "test",
            "--save_dir", str(save_dir), "--selector", *SELECTORS, "--print_every", "100"]


def run_folders(market, seed, n_assets):
    return [ARTIFACTS / make_exp_name("pfl", market, rho, round(n_assets * rho / 100),
                                    0., seed, selector)
            for rho in (10, 15, 20) for selector in SELECTORS]


def model_error(left, right):
    if left.keys() != right.keys():
        raise ValueError("Checkpoint parameter names differ")
    errors = []
    for key in left:
        if left[key].shape != right[key].shape:
            raise ValueError(f"Checkpoint shape differs: {key}")
        if not torch.isfinite(left[key]).all() or not torch.isfinite(right[key]).all():
            raise ValueError(f"Nonfinite checkpoint parameter: {key}")
        errors.append(float((left[key] - right[key]).abs().max()))
    return max(errors)


def validate_receipt(receipt, model, data_hash, protocol_hash, source_hashes):
    if (receipt["model_sha256"] != digest(model)
            or receipt["data_sha256"] != data_hash
            or receipt["protocol_sha256"] != protocol_hash
            or receipt["source_sha256"] != source_hashes):
        raise ValueError(f"PFL model provenance mismatch: {model}")


def certify_existing_model(market, seed, panel, protocol_hash, source_hashes, jobs):
    folder = ARTIFACTS / pfl_model_dir_name(market, seed)
    model = folder / "model.pt"
    receipt_path = folder / "provenance.json"
    if not model.exists():
        return None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, model, panel["sha256"], protocol_hash, source_hashes)
        return receipt
    job = next(j for j in jobs if j["mode"] == "pfl" and j["market"] == market
               and j["seed"] == seed and j["phase"] == 1)
    command = job["command"]
    args = author.build_argparser().parse_args(command[command.index("--mode"):])
    expected = author.build_argparser().parse_args(arguments(market, seed))
    for key, value in vars(expected).items():
        if key not in {"save_dir", "selector"} and getattr(args, key) != value:
            raise ValueError(f"Cached training configuration differs: {key}")
    identity = {"arguments": vars(args), "log_prefix": f"PFL {market}|seed={seed}",
                "data_sha256": panel["sha256"], "protocol_sha256": protocol_hash}
    import hashlib
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    checkpoint = ROOT / "runtime/checkpoints" / f"{fingerprint}.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["identity"] != identity or saved["next_epoch"] != 300:
        raise ValueError(f"Incomplete or mismatched training checkpoint: {checkpoint}")
    error = model_error(saved["model"], torch.load(model, map_location="cpu", weights_only=True))
    if error != 0:
        raise ValueError(f"Saved model differs from completed training: {model}")
    receipt = {"origin": "completed memory-adapter checkpoint; direct source equivalence checked separately",
               "model_sha256": digest(model), "data_sha256": panel["sha256"],
               "protocol_sha256": protocol_hash, "source_sha256": source_hashes,
               "identity": identity, "completed_updates": 300, "model_max_abs_error": error,
               "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest(checkpoint),
               "training_history": saved["history"]}
    write_json(receipt_path, receipt)
    return receipt


def execute(market, seed, output, destination=ARTIFACTS):
    command = [sys.executable, "-u", str(ROOT / "upstream/src/main.py"),
               *arguments(market, seed, destination)]
    log = output / f"{market}_seed{seed}.log"
    record = {"command": command, "started_utc": datetime.now(timezone.utc).isoformat()}
    with log.open("a") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    record.update(exit_code=result.returncode, finished_utc=datetime.now(timezone.utc).isoformat())
    record_path = output / f"{market}_seed{seed}.execution.json"
    history = json.loads(record_path.read_text()) if record_path.exists() else []
    if isinstance(history, dict):
        history = [history]
    write_json(record_path, [*history, record])
    if result.returncode:
        raise RuntimeError(f"Author PFL failed; see {log}")
    return record


def report(output, protocol, panels, source_hashes, model_records):
    hashes = dict(source_hashes)
    roots = [folder for market in protocol["markets"] for seed in protocol["seeds"]
             for folder in run_folders(market, seed, panels[market]["n_assets"])]
    runs, issues = collect(roots, panels, hashes, protocol)
    if issues or len(runs) != 120:
        raise ValueError(f"PFL evaluation incomplete/invalid: {len(runs)}/120 runs; {issues}")
    for row in runs.itertuples():
        trace = pd.read_csv(row.trace_path)
        weights = trace[["weight_sum", "weight_abs_sum", "weight_nnz_1e-8"]].to_numpy()
        if not np.isfinite(weights).all():
            raise ValueError(f"Nonfinite portfolio diagnostics: {row.trace_path}")
        runs.loc[row.Index, "max_budget_error"] = float((trace.weight_sum - 1).abs().max())
        runs.loc[row.Index, "support_violations"] = int((trace["weight_nnz_1e-8"] > row.actual_k).sum())
    for record in model_records:
        path = ROOT / record["provenance_path"]
        hashes[record["provenance_path"]] = digest(path)
        hashes[str((path.parent / "model.pt").relative_to(ROOT))] = record["model_sha256"]
    for path in [Path(__file__), ROOT / "reproduction/compare_published.py", ROOT / "reproduction/protocol.json",
                 ROOT / "sources/paper_table1_reference.csv", ROOT / "sources/environment.lock.txt"]:
        hashes[str(path.relative_to(ROOT))] = digest(path)
    reference = pd.read_csv(ROOT / "sources/paper_table1_reference.csv")
    comparison = compare(reference[reference.method == "pfl"], runs, panels,
                         protocol["seeds"], "oscar_hard")
    atomic_text(output / "comparison.csv", comparison.to_csv(index=False))
    atomic_text(output / "runs.csv", runs.to_csv(index=False))
    completed = comparison[comparison.all_required_repeats_complete]
    check_path = output / "direct_checks.json"
    checks = json.loads(check_path.read_text()) if check_path.exists() else []
    for check in checks:
        if "data_sha256" in check:
            model = ARTIFACTS / pfl_model_dir_name(check["market"], check["seed"]) / "model.pt"
            if (check["data_sha256"] != panels[check["market"]]["sha256"]
                    or check["reference_model_sha256"] != digest(model)
                    or check["source_sha256"] != source_hashes):
                raise ValueError("Independent check refers to different inputs/model/source")
    checked_markets = sorted({c["market"] for c in checks if c["model_max_abs_error"] == 0
                              and len(c["traces"]) == 6
                              and all(t["return_max_abs_error"] == 0 for t in c["traces"])})
    if check_path.exists():
        hashes[str(check_path.relative_to(ROOT)) if check_path.is_relative_to(ROOT) else str(check_path)] = digest(check_path)
    verification = {"scope": "Released PFL with paper architecture/batch size on frozen contemporary inputs",
        "models": model_records, "verified_runs": len(runs), "complete_pfl_cells": len(completed),
        "total_paper_pfl_cells": len(comparison), "support_violations": int(runs.support_violations.sum()),
        "independently_checked_markets": checked_markets,
        "max_budget_error": float(runs.max_budget_error.max()),
        "mean_absolute_daily_sharpe_gap": float(completed.daily_sr_minus_paper.abs().mean()),
        "mean_absolute_annualized_sharpe_gap": float(completed.annualized_sr_minus_paper_display.abs().mean()),
        "daily_mean_and_sd_matches": int(completed.daily_mean_and_sd_match.sum()),
        "annualized_mean_and_sd_matches": int(completed.annualized_mean_and_sd_match.sum()),
        "annualized_mean_matches": int(completed.annualized_mean_match.sum()),
        "annualized_sd_matches": int(completed.annualized_sd_match.sum()),
        "reporting_convention": REPORTING_CONVENTION,
        "exact_published_result_replicated": False, "sha256": hashes}
    write_json(output / "verification.json", verification)
    display = comparison_display(comparison, ["market", "rho_percent", "paper_optimizer", "paper_k", "current_k"])
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>PFL source replay</title>
<style>body{{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}table{{border-collapse:collapse;font-size:13px}}.scroll{{overflow:auto}}</style>
<h1>PFL baseline: annualized Sharpe comparison</h1>
<p><strong>{REPORTING_NOTE}</strong></p>
<p>20 MSE predictors; 120 complete daily evaluations: 4 markets × 5 seeds × 3 cardinalities ×
2 optimizers. The same predictor is shared across cardinalities and optimizers. All test dates
are evaluated; no interpolation, test tuning or seed removal. Training and daily return traces
are unchanged; annualization is applied only to reported Sharpe statistics.</p>
<p>The untouched author entry point fills missing runs. Independent training from initialization
and all six evaluation traces match exactly for seed 16 in {len(checked_markets)} of 4 markets.
Existing models are checked against completed training checkpoints and frozen-data hashes
before reuse. Use --check-training to create independent equivalence checks.</p>
<p>Published settings: 512/256 hidden layers, batch 64, lr .001, 100-session inputs,
chronological 80/20 split, rolling covariance and .10 shrinkage. Release defaults fill the
unreported choices: Adam, tanh, dropout .1, weight decay 1e-4, 300 sampled-batch updates
(not 300 full epochs), and seeds 16/37/47/214/1947. Training uses prediction MSE only.</p>
<p>Conventional OSCAR uses hard selection by |Lᵀw| and re-optimization on the selected
support. Soft OSCAR uses fractional sigmoid gates on predicted returns before all-asset
optimization; it can hold every asset and is not used as the conventional PFL comparator.</p>
<p>Faithfulness is to the released implementation where the paper is incomplete or conflicts:
the release standardizes the full panel before splitting and trains PFL with per-asset mean
squared error. The paper's DFL algorithm prints a squared vector norm; its PFL description
does not fully specify the training loss reduction. The released QCQP imposes ||Ly||≤1
with t≥1. The paper prints t≥0, and covariance risk for Σ=LLᵀ
is ||Lᵀy||. Released mSSRM uses a quadratic gradient and signed-value top-k, whereas
Appendix B.3 describes Sharpe-gradient ascent and magnitude top-k. These are material
method differences, not recovered paper settings.</p>
<p>SD-relaxation remains explicitly unavailable as a faithful paper comparison: Appendix B.2
requires realized-cardinality matching but supplies no numerical threshold, adjustment rule,
search bounds, or stopping rule; the release only implements fixed nominal k. Its 12 cells
are left incomplete rather than relabeled as reproduced.</p>
<p>The downloaded inputs are not the author's original matrices. Asset counts are
47/92/206/159 versus 47/93/208/162 (Euro/FTSE/Nikkei/KOSPI). Stable historical membership
is unverified, FTSE contains a verified US ticker collision, and KOSPI uses a documented
constituent fallback. Exact published-number reproduction is not established.</p>
<p>Complete PFL cells: {len(completed)}/36. Mean absolute gap under the annualized interpretation:
<strong>{verification['mean_absolute_annualized_sharpe_gap']:.6f}</strong>, versus
{verification['mean_absolute_daily_sharpe_gap']:.6f} under the stated daily definition.
Annualized mean-and-SD matches: {verification['annualized_mean_and_sd_matches']}/{len(completed)};
daily mean-and-SD matches: {verification['daily_mean_and_sd_matches']}/{len(completed)}.
Both mean and SD must be within 0.0005 of the displayed paper values. Support violations:
{verification['support_violations']}; maximum budget error: {verification['max_budget_error']:.3g}.</p>
<div class="scroll">{display.to_html(index=False, na_rep='unavailable', float_format=lambda x: f'{x:.6f}')}</div>
<p><a href="comparison.csv">Comparison CSV</a> · <a href="runs.csv">Individual runs</a> ·
<a href="verification.json">Provenance</a> ·
{'<a href="direct_checks.json">Direct checks</a> ·' if check_path.exists() else ''}
<a href="https://proceedings.mlr.press/v306/jeon26a.html">Paper</a> ·
<a href="https://github.com/feuerwerksh/Diffble-card-SR">Author code</a></p></html>'''
    atomic_text(output / "report.html", doc)
    print(f"Verified {len(runs)} PFL runs; {len(completed)}/36 published PFL cells", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pfl_replay")
    parser.add_argument("--check-training", action="store_true",
                        help="Independently train seed16 from initialization in all four markets")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "reproduction/protocol.json").read_text())
    protocol_hash = digest(ROOT / "reproduction/protocol.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "upstream", text=True).strip()
    if commit != protocol["official_commit"]:
        raise ValueError("Author source commit differs from the pinned release")
    subprocess.run(["git", "diff", "--exit-code", protocol["official_commit"], "--", "src"],
                   cwd=ROOT / "upstream", check=True, stdout=subprocess.DEVNULL)
    source_hashes = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "upstream/src").glob("*.py")}
    panels = {market: panel_info(market, {}) for market in protocol["markets"]}
    jobs = json.loads((ROOT / "runtime/jobs.json").read_text())
    records = []
    # Validate every existing artifact before author.main may skip it.
    _, issues = collect([ARTIFACTS], panels, {}, protocol)
    if any("_PFL_" in issue["path"] for issue in issues):
        raise ValueError(issues)
    for market in protocol["markets"]:
        for seed in protocol["seeds"]:
            receipt = certify_existing_model(market, seed, panels[market], protocol_hash, source_hashes, jobs)
            print(f"PFL {market} seed {seed}: {'verified reuse' if receipt else 'train with original source'}", flush=True)
            folders = run_folders(market, seed, panels[market]["n_assets"])
            if receipt is None or any(not (folder / "summary_metrics.csv").exists() for folder in folders):
                execution = execute(market, seed, output)
            folder = ARTIFACTS / pfl_model_dir_name(market, seed)
            if receipt is None:
                receipt = {"origin": "unmodified author entry point", "execution": execution,
                           "model_sha256": digest(folder / "model.pt"),
                           "data_sha256": panels[market]["sha256"], "protocol_sha256": protocol_hash,
                           "source_sha256": source_hashes, "completed_updates": 300}
                write_json(folder / "provenance.json", receipt)
            records.append({"market": market, "seed": seed, "model_sha256": receipt["model_sha256"],
                            "provenance_path": str((folder / "provenance.json").relative_to(ROOT))})
    if args.check_training:
        checks = []
        check_logs = output / "direct"
        check_logs.mkdir(exist_ok=True)
        for market in protocol["markets"]:
            print(f"Direct source training/evaluation check: {market}, seed 16", flush=True)
            with tempfile.TemporaryDirectory(prefix="pfl-direct-", dir=output) as temp:
                destination = Path(temp)
                execution = execute(market, 16, check_logs, destination)
                name = pfl_model_dir_name(market, 16)
                error = model_error(torch.load(destination / name / "model.pt", weights_only=True),
                                    torch.load(ARTIFACTS / name / "model.pt", weights_only=True))
                check = {"market": market, "seed": 16, "model_max_abs_error": error,
                         "data_sha256": panels[market]["sha256"],
                         "reference_model_sha256": digest(ARTIFACTS / name / "model.pt"),
                         "source_sha256": source_hashes,
                         "execution": execution, "traces": []}
                for percent in (10, 15, 20):
                    k = round(panels[market]["n_assets"] * percent / 100)
                    for selector in SELECTORS:
                        name = make_exp_name("pfl", market, percent, k, 0., 16, selector)
                        direct = pd.read_csv(destination / name / "test_trace.csv")
                        reused = pd.read_csv(ARTIFACTS / name / "test_trace.csv")
                        if not direct.date.equals(reused.date):
                            raise ValueError("Direct evaluation dates differ")
                        delta = float((direct.ret_true_raw - reused.ret_true_raw).abs().max())
                        check["traces"].append({"selector": selector, "rho": percent,
                                                "days": len(direct), "return_max_abs_error": delta})
                        if error != 0 or delta != 0:
                            raise ValueError(f"Direct source equivalence failed: {check}")
                checks.append(check)
                write_json(output / "direct_checks.json", checks)
    report(output, protocol, panels, source_hashes, records)


if __name__ == "__main__":
    main()
