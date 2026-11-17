"""Daily input construction from raw Qlib data, without future-label filtering."""
import hashlib
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ("open", "high", "low", "close", "vwap", "volume", "factor")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def memberships(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        symbol, start, stop = line.split()
        result.setdefault(symbol, []).append((start, stop))
    return result


def feature_spec():
    from qlib.contrib.data.loader import Alpha158DL
    exprs, names = Alpha158DL.get_feature_config()
    exprs = [re.sub(r"\$(open|high|low|close|vwap)\b", r"($\1*$factor)", e) for e in exprs]
    return exprs, names


def init_worker(source, dates):
    global _DATES, _EXPRS
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=source, region=REG_CN, kernels=1, logging_level=40)
    _DATES = pd.DatetimeIndex(dates)
    _EXPRS = feature_spec()[0] + ["$" + f for f in FIELDS]


def query_stock(symbol):
    from qlib.data import D
    from qlib.data.cache import H
    frame = D.features([symbol], _EXPRS, start_time=str(_DATES[0].date()),
                       end_time=str(_DATES[-1].date()), freq="day")
    if len(frame):
        values = frame.xs(symbol, level="instrument").reindex(_DATES).to_numpy(np.float32)
    else:
        values = np.full((len(_DATES), len(_EXPRS)), np.nan, np.float32)
    values[~np.isfinite(values)] = np.nan
    factor = values[:, 158 + FIELDS.index("factor")]
    values[~(np.isfinite(factor) & (factor > 0)), :158] = np.nan
    H.clear()
    return values


def window_eligibility(x, current, lookback):
    good = np.isfinite(x).all(-1).astype(np.int32)
    cumulative = np.concatenate([np.zeros_like(good[:1]), good.cumsum(0)], axis=0)
    out = np.zeros_like(current, dtype=bool)
    out[lookback - 1:] = (cumulative[lookback:] - cumulative[:-lookback] == lookback)
    return out & current


def split_indices(dates, config):
    indices = {}
    for split in ("train", "validation", "test"):
        start, stop = config[split]
        ids = [i for i, d in enumerate(dates[:-2]) if start <= d <= stop and
               (split == "test" or dates[i + 2] <= stop)]
        indices[split] = ids
    return indices


def build_universe(universe, config, workers=2):
    folder = ROOT / "data" / universe
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(config["source"])
    all_dates = np.array((source / "calendars/day.txt").read_text().splitlines())
    dates = all_dates[(all_dates >= "2020-11-01") & (all_dates <= "2025-01-06")]
    member = memberships(source / "instruments" / (universe + ".txt"))
    symbols = sorted(s for s, intervals in member.items()
                     if any(a <= config["test"][1] and b >= config["train"][0] for a, b in intervals))
    source_files = ["calendars/day.txt", f"instruments/{universe}.txt"]
    source_files += [f"features/{s.lower()}/{f}.day.bin" for s in symbols for f in FIELDS]
    transfer = json.loads((source.parents[1] / "download_manifest.json").read_text())
    if transfer["revision"] != config["source_revision"]:
        raise ValueError("Source revision mismatch")
    verified = {}
    for relative in source_files:
        actual = sha(source / relative)
        if actual != transfer["files"][relative]["sha256"]:
            raise ValueError("Source hash mismatch: " + relative)
        verified[relative] = actual
    save_json(folder / "source_hashes.json", verified)
    save_json(folder / "status.json", {"phase": "extracting", "stocks": len(symbols)})
    raw_x = np.lib.format.open_memmap(folder / "raw_features.npy", mode="w+", dtype=np.float32,
                                     shape=(len(dates), len(symbols), 158))
    raw = np.empty((len(dates), len(symbols), len(FIELDS)), np.float32)
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                             initargs=(str(source), dates.tolist())) as executor:
        for j, values in enumerate(executor.map(query_stock, symbols, chunksize=1)):
            raw_x[:, j] = values[:, :158]
            raw[:, j] = values[:, 158:]
            if (j + 1) % 100 == 0 or j + 1 == len(symbols):
                print(json.dumps({"universe": universe, "extracted": j + 1, "stocks": len(symbols)}), flush=True)
    current = np.zeros((len(dates), len(symbols)), dtype=bool)
    for j, s in enumerate(symbols):
        for a, b in member[s]:
            current[:, j] |= (dates >= a) & (dates <= b)
    close = raw[:, :, FIELDS.index("close")].astype(np.float64)
    factor = raw[:, :, FIELDS.index("factor")].astype(np.float64)
    positive = (close > 0) & (factor > 0) & np.isfinite(close * factor)
    current &= positive & (raw[:, :, FIELDS.index("volume")] > 0)
    adjusted = np.where(positive, close * factor, np.nan)
    y = np.full(current.shape, np.nan, np.float32)
    y[:-2] = adjusted[2:] / adjusted[1:-1] - 1
    indices = split_indices(dates, config)
    training_mask = np.zeros_like(current)
    training_mask[indices["train"]] = current[indices["train"]]
    # Fit using only training decisions; no labels enter feature normalization.
    sums = np.zeros(158, np.float64)
    squares = np.zeros(158, np.float64)
    count = np.zeros(158, np.int64)
    for t in indices["train"]:
        values = np.asarray(raw_x[t, training_mask[t]], dtype=np.float64)
        finite = np.isfinite(values)
        values = np.where(finite, values, 0)
        count += finite.sum(0)
        sums += values.sum(0)
        squares += np.square(values).sum(0)
    mean = sums / count.clip(1)
    std = np.sqrt(np.maximum(squares / count.clip(1) - mean ** 2, 1e-12))
    x = np.lib.format.open_memmap(folder / "features.npy", mode="w+", dtype=np.float32, shape=raw_x.shape)
    for t in range(len(dates)):
        x[t] = np.clip((raw_x[t] - mean) / std, -5, 5)
    eligible = window_eligibility(x, current, config["lookback"])
    counts = {s: [int(eligible[i].sum()) for i in ids] for s, ids in indices.items()}
    if any(min(v) < 100 for v in counts.values()):
        raise ValueError("Fewer than 100 available stocks on an experiment date")
    arrays = {"target": y, "eligible": eligible, "dates": dates, "symbols": np.array(symbols),
              "adjusted_close": adjusted, "current_member_and_trading": current}
    for key, value in arrays.items():
        np.save(folder / (key + ".npy"), value)
    np.savez(folder / "normalization.npz", mean=mean, std=std, count=count)
    x.flush()
    raw_x.flush()
    exprs, names = feature_spec()
    manifest = {"complete": True, "scope": config["scope"], "universe": universe,
                "builder_sha256": sha(__file__), "source_revision": transfer["revision"],
                "source_hashes_sha256": sha(folder / "source_hashes.json"),
                "features": names, "expressions": exprs, "indices": indices,
                "dates": {s: [str(dates[ids[0]]), str(dates[ids[-1]])] for s, ids in indices.items()},
                "days": {s: len(ids) for s, ids in indices.items()}, "stocks_in_union": len(symbols),
                "eligible_stocks": {s: {"min": min(v), "mean": float(np.mean(v)), "max": max(v)} for s, v in counts.items()},
                "unknown_eligible_targets": {s: int(sum((eligible[i] & ~np.isfinite(y[i])).sum() for i in ids)) for s, ids in indices.items()},
                "array_hashes": {key: sha(folder / (key + ".npy")) for key in ["features", *arrays]}}
    save_json(folder / "manifest.json", manifest)
    save_json(folder / "status.json", {"phase": "complete", "days": manifest["days"]})
    del raw_x
    (folder / "raw_features.npy").unlink()
    print(json.dumps({"universe": universe, "complete": True, "days": manifest["days"], "eligible": manifest["eligible_stocks"]}), flush=True)


class DailyPanel:
    def __init__(self, universe, config):
        folder = ROOT / "data" / universe
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.x = np.load(folder / "features.npy", mmap_mode="r")
        self.y = np.load(folder / "target.npy")
        self.eligible = np.load(folder / "eligible.npy")
        self.dates = np.load(folder / "dates.npy")
        self.symbols = np.load(folder / "symbols.npy")
        self.lookback = config["lookback"]

    def batch(self, indices, device):
        import torch
        stocks = [np.flatnonzero(self.eligible[i]) for i in indices]
        n = max(map(len, stocks))
        x = np.zeros((len(indices), n, self.lookback, self.x.shape[-1]), np.float32)
        y = np.full((len(indices), n), np.nan, np.float32)
        mask = np.zeros((len(indices), n), bool)
        for b, (t, ids) in enumerate(zip(indices, stocks)):
            x[b, :len(ids)] = self.x[t - self.lookback + 1:t + 1, ids].transpose(1, 0, 2)
            y[b, :len(ids)] = self.y[t, ids]
            mask[b, :len(ids)] = True
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
                torch.from_numpy(mask).to(device), stocks)
