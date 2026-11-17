#!/usr/bin/env python3
"""Run unchanged author CASH code on a tiny synthetic engineering fixture.

This verifies software, gradients, and forecast timing. It is NOT an empirical
experiment, a paper-data replacement, or reproduction of published scores.
Use a new --output directory for each run; existing artifacts are not replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream"
PIN = "dc03215a699ff08073172098b3f1033a7c27f8c6"
sys.path.insert(0, str(UPSTREAM))


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "engineering")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if revision != PIN:
        raise RuntimeError(f"Unexpected author revision {revision}; expected {PIN}")
    dirty = subprocess.check_output(["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty.strip():
        raise RuntimeError("The author code has local changes")
    from config import load_config
    from cash_model import grad_reverse
    from train import StockDataset, load_data
    from predict import compute_test_predictions
    from itransformer_encoder import TemporalCrossSectionEncoder
    from factor_network import VAE_Network
    from beta_network import BetaNetwork
    from evaluate import compute_r2, compute_ic, compute_rank_ic

    torch.set_num_threads(2)
    rng = np.random.default_rng(20261110)
    time_count, asset_count, feature_count = 80, 12, 6
    features = rng.normal(size=(time_count, asset_count, feature_count)).astype(np.float32)
    realized = (0.01 * rng.normal(size=(time_count, asset_count))).astype(np.float32)
    excess = realized - np.float32(0.0001)
    dates = pd.bdate_range("2000-01-03", periods=time_count).strftime("%Y-%m-%d").to_numpy(dtype="U10")
    data = output / "synthetic_fixture"
    data.mkdir()
    for name, array in {"features": features, "returns": excess, "realized_returns": realized, "timestamps": dates}.items():
        np.save(data / f"{name}.npy", array, allow_pickle=False)
    write_json(data / "PROVENANCE.json", {
        "kind": "synthetic_engineering_fixture", "not_paper_data": True,
        "seed": 20261110, "shape_TNP": [time_count, asset_count, feature_count],
        "purpose": "tiny bounded end-to-end software verification only",
    })
    cfg = load_config()
    cfg["output_root"] = str(output / "pipeline")
    cfg["data"] = {"layout": "TNP", "return_type": "excess", **{
        key + "_path": str(data / (name + ".npy")) for key, name in {
            "features": "features", "returns": "returns", "realized_returns": "realized_returns", "timestamps": "timestamps"
        }.items()}}
    # Retain the released architecture/loss/optimizer defaults. Reduce only the
    # run length and portfolio breadth to fit this 12-asset engineering fixture.
    cfg["train"].update(epochs=2, num_workers=0, amp_mode="off")
    cfg["evaluation"]["frequency"] = "daily"
    cfg["portfolio"].update(portfolio_size=4, max_turnover=1)
    config_path = output / "config.json"
    write_json(config_path, cfg)
    env = dict(os.environ, FORCE_CPU="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    command = [sys.executable, str(UPSTREAM / "run_cash_pipeline.py"), "--config", str(config_path)]
    with (output / "pipeline.log").open("w") as log:
        subprocess.run(command, cwd=UPSTREAM, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)

    checks = {}
    pipeline = output / "pipeline"
    model_config = json.loads((pipeline / "model/config.json").read_text())
    summary = json.loads((pipeline / "model/train_summary.json").read_text())
    checkpoint = torch.load(pipeline / "model/model.pt", map_location="cpu", weights_only=True)
    expected_indices = np.arange(64, 80)
    predictions = np.load(pipeline / "predictions/mu_pred.npy")
    assert predictions.shape == (16, 12) and np.isfinite(predictions).all()
    assert summary["completed_epochs"] == 2 and np.isfinite(summary["best_val_mse"])
    assert all(torch.isfinite(tensor).all() for state in checkpoint.values() for tensor in state.values())
    checks["full_cli_train_predict_evaluate_portfolio"] = True
    for folder in ["predictions", "portfolio"]:
        np.testing.assert_array_equal(np.load(pipeline / folder / "prediction_indices.npy"), expected_indices)
    np.testing.assert_allclose(np.load(pipeline / "portfolio/mu_pred.npy"), predictions, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.load(pipeline / "portfolio/returns_true.npy"), realized[64:], rtol=0, atol=0)
    np.testing.assert_allclose(np.load(pipeline / "predictions/returns_true.npy"), excess[64:], rtol=0, atol=0)
    checks["chronological_test_indices_and_raw_portfolio_returns"] = True

    encoder = TemporalCrossSectionEncoder(feature_count, 32, window=3, nhead=4, num_layers_time=2, dropout=0.1)
    vae = VAE_Network(asset_count, 8, asset_count, hidden_dim=32)
    beta = BetaNetwork(32, 8, hidden_dim=32)
    for name, module in [("encoder", encoder), ("vae", vae), ("beta_network", beta)]:
        module.load_state_dict(checkpoint[name])
    ds = StockDataset(features, excess, window=3)
    np.testing.assert_array_equal(ds[0]["hist_features"].numpy(), features[:3])
    np.testing.assert_array_equal(ds[0]["returns"].numpy(), excess[3])
    checks["dataset_uses_past_features_for_next_return"] = True
    baseline = compute_test_predictions(encoder, vae, beta, ds, expected_indices.tolist(), torch.device("cpu"), 3)
    np.testing.assert_allclose(baseline[1], predictions, rtol=1e-6, atol=1e-7)
    changed_features, changed_returns = features.copy(), excess.copy()
    cutoff = 70
    changed_features[cutoff:] += 10
    changed_returns[cutoff:] += 0.05
    changed = compute_test_predictions(encoder, vae, beta, StockDataset(changed_features, changed_returns, 3),
                                       expected_indices.tolist(), torch.device("cpu"), 3)
    observable = expected_indices <= cutoff
    np.testing.assert_allclose(baseline[1][observable], changed[1][observable], rtol=0, atol=0)
    assert not np.allclose(baseline[0][expected_indices == cutoff], changed[0][expected_indices == cutoff])
    checks["forecasts_ignore_contemporaneous_and_future_inputs"] = True
    checks["total_fit_is_contemporaneous_and_separate_from_forecast"] = True

    base_x = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    grad_reverse(base_x, 0.7).square().sum().backward()
    torch.testing.assert_close(base_x.grad, -1.4 * base_x.detach())
    checks["adversarial_gradient_sign_and_magnitude"] = True
    prediction_fixture = np.array([0.0, 0.02, -0.01])
    return_fixture = np.array([0.01, 0.01, -0.02])
    expected_r2 = 100 * (1 - np.sum((prediction_fixture - return_fixture) ** 2) / np.sum(return_fixture ** 2))
    np.testing.assert_allclose(compute_r2(prediction_fixture, return_fixture), expected_r2)
    np.testing.assert_allclose(compute_ic(prediction_fixture, return_fixture), np.corrcoef(prediction_fixture, return_fixture)[0, 1])
    rank_a = pd.Series(prediction_fixture).rank().to_numpy()
    rank_b = pd.Series(return_fixture).rank().to_numpy()
    np.testing.assert_allclose(compute_rank_ic(prediction_fixture, return_fixture), np.corrcoef(rank_a, rank_b)[0, 1])
    checks["uncentered_R2_percent_and_cross_sectional_correlations"] = True
    np.save(data / "features_ntp.npy", features.transpose(1, 0, 2))
    np.save(data / "returns_ntp.npy", excess.T)
    a, b = load_data(str(data / "features_ntp.npy"), str(data / "returns_ntp.npy"), "NTP", "excess")
    np.testing.assert_array_equal(a, features)
    np.testing.assert_array_equal(b, excess)
    checks["NTP_TNP_layout_agreement"] = True
    versions = {name: importlib.metadata.version(name) for name in ["torch", "numpy", "pandas"]}
    write_json(output / "validation.json", {
        "status": "passed", "kind": "synthetic_engineering_only", "paper_results_replicated": False,
        "real_market_experiments": 0, "upstream_commit": revision, "upstream_modified": False,
        "architecture": "released main preset", "training_epochs": 2, "device": "cpu", "versions": versions,
        "checks": checks, "wall_seconds": time.monotonic() - started,
        "files": [{"path": str(path.relative_to(output)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in sorted(pipeline.rglob("*")) if path.is_file()],
    })
    print(json.dumps({"status": "passed", "checks": len(checks), "paper_results_replicated": False, "output": str(output)}))


if __name__ == "__main__":
    main()
