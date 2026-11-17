"""Exercise all 21 synthetic artifacts through full portfolio reporting."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from usdata import sha
from experiments.market_adjusted_monthly import analyze
from experiments.market_adjusted_monthly.common import TASKS, INDEX, REGIMES
from experiments.market_adjusted_monthly.data import month_targets


def main():
    rng = np.random.default_rng(91)
    dates = np.array(pd.bdate_range("2020-01-01", "2020-06-30").strftime("%Y-%m-%d"))
    n, length = 20, len(dates)
    market_return = rng.normal(scale=.004, size=length)
    returns = rng.normal(scale=.01, size=(n, length))
    vol, market_vol, end = month_targets(returns, market_return, dates)
    raw = np.full((n, length, 3), np.nan, dtype=np.float32)
    raw[:, :-2, 0] = returns[:, 2:]
    raw[:, :, 1] = rng.normal(scale=.1, size=(n, length))
    raw[:, :, 2] = vol
    adjusted = raw.copy()
    adjusted[:, :-2, 0] -= market_return[None, 2:]
    adjusted[:, :, 2] -= market_vol[None, :]
    days = np.arange(5, 45)
    symbols = np.array([f"P{i}" for i in range(n)])
    market = np.ones((n, length, 7)) * 100
    market[:, :, 1] = rng.uniform(1e9, 3e9, size=(n, length))
    labels = dict(raw_monthly=raw, market_adjusted=adjusted)
    panels = {regime: SimpleNamespace(dates=dates, symbols=symbols, labels=labels[regime], label_end=end,
        mean=np.zeros(3), scale=np.array([.02, .1, .01]), eligible=np.ones((n, length), bool),
        member=np.ones((n, length), bool), market=market, total_return=returns, price_return=returns,
        execution_tradable=np.ones((n, length), bool), terminal=np.zeros((n, length), bool), split=lambda _: days)
        for regime in REGIMES}
    with tempfile.TemporaryDirectory(prefix="limt-monthly-integration-") as name:
        root = Path(name)
        here = root / "experiment"
        for data in [root / "data", here / "data"]:
            data.mkdir(parents=True)
            np.save(data / "fixture.npy", returns)
            (data / "manifest.json").write_text(json.dumps({"output_sha256": {"fixture.npy": sha(data / "fixture.npy")}}))
        hashes = {"fixture": "fixed"}
        (here / "locked_before_training.json").write_text(json.dumps(dict(inputs_sha256=hashes,
            data_manifest_sha256=sha(here / "data/manifest.json"))))

        def folder(regime, task, seed):
            if task == "volume":
                regime = "raw_monthly"
            return here / "results" / regime / task / f"seed{seed}"

        frames = {}
        for regime in REGIMES:
            parts = []
            for t in days:
                record = dict(date=dates[t], realized_date=dates[t + 2], symbol=symbols,
                              volatility_month=str(dates[end[t]])[:7], label_end_date=dates[end[t]])
                for task in TASKS:
                    k = INDEX[task]
                    record[f"label_{task}"] = labels[regime][:, t, k]
                    record[f"pred_{task}"] = labels[regime][:, t, k].astype(float) * .8
                parts.append(pd.DataFrame(record))
            frames[regime] = pd.concat(parts, ignore_index=True)
        for regime in REGIMES:
            for task in ["joint", *TASKS]:
                if task == "volume" and regime == "market_adjusted":
                    continue
                columns = analyze.KEYS + [f"{prefix}_{name}" for name in (TASKS if task == "joint" else [task]) for prefix in ["label", "pred"]]
                for seed in [0, 1, 2]:
                    out = folder(regime, task, seed)
                    out.mkdir(parents=True)
                    frames[regime][columns].to_parquet(out / "test_predictions.parquet", index=False)
                    (out / "status.json").write_text(json.dumps({"phase": "complete"}))
                    (out / "provenance.json").write_text(json.dumps(dict(inputs_sha256=hashes,
                        artifacts_sha256={"test_predictions.parquet": sha(out / "test_predictions.parquet")})))
        with patch.multiple(analyze, ROOT=root, HERE=here, OUT=here / "analysis",
                            Panel=lambda regime: panels[regime], folder=folder, input_hashes=lambda: hashes):
            analyze.main()
            verification = json.loads((here / "analysis/verification.json").read_text())
            assert verification["status"] == "passed"
            paired = pd.read_csv(here / "analysis/paired_profitability.csv")
            assert len(paired) == 12 and (paired.available).all()
            assert len(pd.read_csv(here / "analysis/portfolio_summary.csv")) == 12
            assert sha(here / "REPORT.txt") == verification["report_sha256"]
    print("PASS: all21 synthetic artifacts through exact labels, shared-volume reuse, physical APO, fees, paired tests and final report.", flush=True)


if __name__ == "__main__":
    main()
