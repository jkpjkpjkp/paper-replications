from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import ROOT, digest, month_id, month_label
from .portfolio import equal_long_short, portfolio_path
from .train import mean_ic


def merge_predictions(panel, config):
    tables = []
    for start, end in config["test_blocks"]:
        fold = ROOT / "results" / f"fold_{start}_{end}"
        stored = []
        for seed in config["seeds"]:
            directory = fold / f"seed{seed}"
            completion = json.loads((directory / "complete.json").read_text())
            assert completion["config_sha256"] == digest(ROOT / "config.json")
            with np.load(directory / "predictions.npz", allow_pickle=False) as source:
                record = {k: source[k] for k in source.files}
            stored.append(record)
        assert all(np.array_equal(stored[0]["row_ids"], s["row_ids"]) for s in stored)
        rows = stored[0]["row_ids"]
        frame = panel.loc[rows, ["permno", "permco", "month", "formation_month", "feature_month",
                                 "realized_return", "outcome_delist_flag", "y_rank", "beta_signal",
                                 "momentum_signal", "reversal_signal"]].copy()
        assert frame.month.between(month_id(f"{start}-01-01"), month_id(f"{end}-12-01")).all()
        for method in ("arbnn", "xgboost", "decompiled"):
            frame[method] = np.mean([s[method] for s in stored], axis=0)
        for seed, s in zip(config["seeds"], stored):
            frame[f"arbnn_seed{seed}"] = s["arbnn"]
        with np.load(fold / "ridge.npz", allow_pickle=False) as ridge:
            assert np.array_equal(rows, ridge["row_ids"])
            frame["ridge"] = ridge["prediction"]
        frame["momentum"] = frame.momentum_signal
        frame["reversal"] = frame.reversal_signal
        frame["fold"] = start
        frame["row_id"] = rows
        tables.append(frame)
    frame = pd.concat(tables).sort_values(["month", "permno"]).reset_index(drop=True)
    assert not frame[["month", "permno"]].duplicated().any()
    frame.to_parquet(ROOT / "results/predictions.parquet", index=False)
    return frame


def main():
    config = json.loads((ROOT / "config.json").read_text())
    panel = pd.read_parquet(ROOT / "data/panel.parquet")
    predictions = merge_predictions(panel, config)
    del panel
    methods = ["arbnn", "xgboost", "decompiled", "ridge", "momentum", "reversal"]
    methods += [f"arbnn_seed{seed}" for seed in config["seeds"]]
    assert predictions.month.nunique() == 252
    returns, held, ic_rows = [], [], []
    scenarios = dict(config["cost_scenarios"])
    scenarios["primary_adverse_missing"] = dict(scenarios["primary"])
    for method in methods:
        ic_rows.append({"method": method, "mean_monthly_ic": mean_ic(predictions[method], predictions.y_rank, predictions.month)})
        for fraction in config["quantiles"]:
            selected = []
            for month, sample in predictions.groupby("month", sort=True):
                weights = equal_long_short(sample.permno.to_numpy(), sample[method].to_numpy(), fraction)
                positions = sample.loc[weights != 0, ["month", "permno", "permco", "realized_return",
                                                      "outcome_delist_flag", "beta_signal"]].copy()
                positions["weight"] = weights[weights != 0]
                positions["terminal"] = positions.outcome_delist_flag.notna() & positions.outcome_delist_flag.ne("N")
                positions["unresolved_payoff"] = positions.outcome_delist_flag.isin(["G", "M", "V"])
                positions["score"] = sample.loc[weights != 0, method]
                positions["method"] = method
                positions["fraction"] = fraction
                selected.append(positions)
            holdings = pd.concat(selected, ignore_index=True)
            held.append(holdings)
            for scenario, costs in scenarios.items():
                path = portfolio_path(holdings, **costs, adverse_missing=scenario.endswith("adverse_missing"))
                path["method"], path["fraction"], path["scenario"] = method, fraction, scenario
                path["date"] = [month_label(m) for m in path.month]
                returns.append(path)
            print("BACKTESTED", method, fraction, flush=True)
    all_returns = pd.concat(returns, ignore_index=True)
    all_returns.to_parquet(ROOT / "results/returns.parquet", index=False)
    all_returns.to_csv(ROOT / "results/returns.csv", index=False)
    pd.concat(held, ignore_index=True).to_parquet(ROOT / "results/holdings.parquet", compression="zstd", index=False)
    pd.DataFrame(ic_rows).to_csv(ROOT / "results/information_coefficients.csv", index=False)
    print("All backtest paths saved", flush=True)


if __name__ == "__main__":
    main()
