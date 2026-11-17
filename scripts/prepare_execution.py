"""Derive last-observed marks and next-session execution masks from raw bars."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transalpha.data import ROOT, sha, save_json


def build_execution(universe, config):
    folder = ROOT / "data" / universe
    dates = np.load(folder / "dates.npy")
    symbols = np.load(folder / "symbols.npy")
    close = np.load(folder / "adjusted_close.npy")
    source = Path(config["source"])
    source_calendar = np.array((source / "calendars/day.txt").read_text().splitlines())
    offsets = np.searchsorted(source_calendar, dates)
    volume = np.full(close.shape, np.nan)
    for j, symbol in enumerate(symbols):
        data = np.fromfile(source / f"features/{symbol.lower()}/volume.day.bin", dtype="<f4")
        index = offsets - int(data[0])
        valid = (index >= 0) & (index < len(data) - 1)
        volume[valid, j] = data[index[valid] + 1]
    tradable = np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
    marked_close = pd.DataFrame(close).ffill().to_numpy()
    returns = np.full(close.shape, np.nan, np.float32)
    returns[:-2] = marked_close[2:] / marked_close[1:-1] - 1
    entry_mask = np.zeros(close.shape, bool)
    entry_mask[:-1] = tradable[1:]
    np.save(folder / "portfolio_returns.npy", returns)
    np.save(folder / "execution_tradable.npy", entry_mask)
    receipt = {"builder_sha256": sha(__file__), "portfolio_returns_sha256": sha(folder / "portfolio_returns.npy"),
               "execution_tradable_sha256": sha(folder / "execution_tradable.npy"),
               "marks": "Forward-fill last observed adjusted close for P&L only; raw prediction labels retain NaNs",
               "execution": "At t+1 close, block stocks with missing close or nonpositive/missing volume; freeze existing untradable holdings",
               "limitations": "Idealized close execution; A-share short availability, T+1 constraints, price limits and fills not modeled"}
    save_json(folder / "execution_manifest.json", receipt)


if __name__ == "__main__":
    config = json.loads((ROOT / "config.json").read_text())
    for universe in config["universes"]:
        build_execution(universe, config)
