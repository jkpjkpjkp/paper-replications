from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import t as student_t

from arboreal.model import ArborealNetwork
from .data import ROOT, digest, month_id, source_path
from .statistics import factor_alpha, hac_mean
from .train import split_masks


def manual_hac(y, X, lag):
    y, X = np.asarray(y, float), np.asarray(X, float)
    coefficient = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ coefficient
    score = X * residual[:, None]
    meat = score.T @ score
    for k in range(1, lag + 1):
        cross = score[k:].T @ score[:-k]
        meat += (1 - k / (lag + 1)) * (cross + cross.T)
    bread = np.linalg.inv(X.T @ X)
    covariance = bread @ meat @ bread * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(covariance[0, 0])
    t = coefficient[0] / se
    p = 2 * student_t.sf(abs(t), len(y) - X.shape[1])
    return coefficient[0], se, t, p


def main():
    config = json.loads((ROOT / "config.json").read_text())
    metadata = json.loads((ROOT / "data/metadata.json").read_text())
    assert digest(ROOT / "data/panel.parquet") == metadata["panel_sha256"]
    source_manifest = json.loads((source_path(config).parents[1] / "manifest.json").read_text())
    assert digest(source_path(config).parents[1] / "manifest.json") == metadata["source_manifest_sha256"]
    for entry in source_manifest["files"]:
        assert digest(source_path(config).parent / entry["name"]) == entry["sha256"]
    for file, expected in metadata["factor_archives_sha256"].items():
        assert digest(ROOT / "sources" / file) == expected
    for line in (ROOT / "protocol_initial.sha256").read_text().splitlines():
        checksum, file = line.split(maxsplit=1)
        assert digest(file.strip()) == checksum
    panel = pd.read_parquet(ROOT / "data/panel.parquet")
    predictions = pd.read_parquet(ROOT / "results/predictions.parquet")
    holdings = pd.read_parquet(ROOT / "results/holdings.parquet")
    returns = pd.read_parquet(ROOT / "results/returns.parquet")
    assert not panel[["month", "permno"]].duplicated().any()
    assert not panel[["month", "permco"]].duplicated().any()
    assert (panel.feature_month == panel.month - 2).all()
    assert (panel.formation_month == panel.month - 1).all()
    test_mask = panel.month.between(month_id("2005-01-01"), month_id("2025-12-01"))
    np.testing.assert_array_equal(np.sort(predictions.row_id), panel.index[test_mask])
    assert len(predictions) == 417706 and predictions.month.nunique() == 252
    names = metadata["features"]
    assert all(name.startswith("x_") for name in names)
    assert np.isfinite(panel[names].to_numpy()).all()
    torch.set_num_threads(2)
    for start, end in config["test_blocks"]:
        masks, boundaries = split_masks(panel, start, end, config)
        ids = panel.index[masks["test"]].to_numpy()
        X = panel.loc[ids[:128], names].to_numpy(dtype=np.float32, copy=True)
        for seed in config["seeds"]:
            directory = ROOT / "results" / f"fold_{start}_{end}" / f"seed{seed}"
            done = json.loads((directory / "complete.json").read_text())
            assert done["boundaries"] == boundaries
            assert done["config_sha256"] == digest(ROOT / "config.json")
            with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(ids, saved["row_ids"])
                model = ArborealNetwork.load(directory / "arbnn.pt")
                assert set(dict(model.named_parameters())) == {"f", "v"}
                np.testing.assert_allclose(model.predict_numpy(X), saved["arbnn"][:128], atol=2e-6)
                booster = xgb.Booster(model_file=directory / "xgboost.json")
                booster.set_param({"nthread": 2})
                np.testing.assert_allclose(booster.predict(xgb.DMatrix(X)), saved["xgboost"][:128], atol=2e-6)
        subset = predictions[predictions.fold.eq(start)]
        original = [np.load(ROOT / "results" / f"fold_{start}_{end}" / f"seed{s}/predictions.npz")
                    for s in config["seeds"]]
        try:
            for method in ("arbnn", "xgboost", "decompiled"):
                np.testing.assert_allclose(subset[method], np.mean([s[method] for s in original], axis=0), atol=1e-12)
        finally:
            for source in original:
                source.close()
    balance = holdings.groupby(["method", "fraction", "month"]).weight.agg(["sum", lambda w: w.abs().sum()])
    np.testing.assert_allclose(balance.iloc[:, 0], 0, atol=1e-12)
    np.testing.assert_allclose(balance.iloc[:, 1], 2, atol=1e-12)
    assert not holdings[["method", "fraction", "month", "permno"]].duplicated().any()
    assert not holdings[["method", "fraction", "month", "permco"]].duplicated().any()
    # Compare all unique held returns directly with unfiltered CRSP outcomes.
    unique_held = holdings[["permno", "month", "realized_return"]].drop_duplicates(["permno", "month"])
    connection = duckdb.connect(config={"threads": 2, "memory_limit": "1GB"})
    connection.register("held", unique_held)
    checked = connection.execute("""SELECT h.realized_return stored, CAST(c.mthret AS DOUBLE) raw
          FROM held h LEFT JOIN read_parquet(?) c ON c.permno=h.permno
          AND c.yyyymm = (h.month//12)*100 + h.month%12 + 1""", [str(source_path(config))]).fetchdf()
    assert len(checked) == len(unique_held)
    np.testing.assert_allclose(checked.stored, checked.raw, rtol=0, atol=0, equal_nan=True)
    # Independently recompute the primary P&L with dense security-by-month matrices.
    held = holdings[holdings.method.eq("arbnn") & holdings.fraction.eq(.1)]
    W = held.pivot(index="month", columns="permno", values="weight").fillna(0).to_numpy()
    R = held.pivot(index="month", columns="permno", values="realized_return").fillna(0).to_numpy()
    terminal = held.pivot(index="month", columns="permno", values="terminal").fillna(False).to_numpy(bool)
    prev = np.zeros(W.shape[1])
    calculated, turns = [], []
    for t in range(len(W)):
        turnover = abs(W[t] - prev).sum()
        pnl = W[t] @ R[t] - .001 * turnover - .01 / 12
        calculated.append(pnl)
        turns.append(turnover)
        prev = W[t] * (1 + R[t]) / (1 + pnl)
        prev[terminal[t]] = 0
    primary = returns[returns.method.eq("arbnn") & returns.fraction.eq(.1) & returns.scenario.eq("primary")].sort_values("month")
    np.testing.assert_allclose(calculated, primary["return"], atol=1e-12)
    np.testing.assert_allclose(turns, primary.turnover, atol=1e-12)
    ff = pd.read_parquet(ROOT / "data/factors.parquet")
    primary = primary.merge(ff, on="month", validate="one_to_one")
    y = primary["return"].to_numpy()
    manual = manual_hac(y, np.ones((len(y), 1)), 6)
    reported = hac_mean(y, 6)
    np.testing.assert_allclose(manual[0] * 12, reported["mean_annual"], rtol=1e-10)
    np.testing.assert_allclose(manual[1] * 12, reported["mean_se_annual"], rtol=1e-10)
    X = np.column_stack([np.ones(len(y)), primary[["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]]])
    manual_alpha = manual_hac(y, X, 6)
    reported_alpha = factor_alpha(y, X[:, 1:], 6)
    np.testing.assert_allclose(manual_alpha[0] * 12, reported_alpha["alpha_annual"], rtol=1e-10)
    np.testing.assert_allclose(manual_alpha[2:], [reported_alpha["alpha_t"], reported_alpha["alpha_p"]], rtol=1e-9)
    payload = {"status": "passed", "trained_models": 15, "oos_months": 252,
               "oos_stock_months": len(predictions), "unique_held_return_rows_checked": len(checked),
               "checkpoint_rows_replayed_per_model": 128,
               "checks": ["original protocol hashes", "panel hash", "all six CRSP source-file SHA-256 hashes",
                          "French archive hashes", "consecutive date splits and label embargo",
                          "all OOS row IDs retained", "feature dates precede holding months by two months",
                          "three-seed ensembles replayed", "XGBoost and ArbNN checkpoints replayed",
                          "zero net and two gross exposure every month", "one share class per issuer",
                          "all unique held returns equal unfiltered CRSP source", "independent dense portfolio accounting",
                          "independent Bartlett HAC mean and FF6 alpha covariance"]}
    (ROOT / "results/verification.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
