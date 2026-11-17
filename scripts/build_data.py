"""Build Alpha158 from the pinned local Qlib snapshot; retain raw missingness."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from limt.data import labels_from_market

ROOT = Path(__file__).resolve().parents[1]
PROVIDER = ROOT.parent / 'csi/qlib/cn_data'


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def init_worker(provider, dates):
    global DATES, EXPR
    import qlib
    from qlib.contrib.data.loader import Alpha158DL
    qlib.init(provider_uri=provider, region="cn", kernels=1, logging_level=40,
              expression_cache=None, dataset_cache=None)
    DATES = pd.DatetimeIndex(dates)
    expr, _ = Alpha158DL.get_feature_config()
    # This community provider stores raw OHLC with a separate adjustment factor.
    EXPR = [re.sub(r"\$(open|high|low|close|vwap)\b", r"($\1*$factor)", e) for e in expr]


def extract(task):
    symbol, cache = task
    from qlib.data import D
    from qlib.data.cache import H
    # Layout: adjusted Alpha158, adjusted close, raw volume, factor,
    # raw close, raw high, raw low, raw vwap.
    expr = EXPR + ["$close*$factor", "$volume", "$factor", "$close", "$high", "$low", "$vwap"]
    frame = D.features([symbol], expr, start_time=str(DATES[0].date()), end_time=str(DATES[-1].date()))
    if len(frame):
        a = frame.xs(symbol, level="instrument").reindex(DATES).to_numpy(dtype=np.float32)
    else:
        a = np.full((len(DATES), len(expr)), np.nan, dtype=np.float32)
    a[~np.isfinite(a)] = np.nan
    np.save(cache, a)
    H.clear()
    return symbol


def build(universe, workers):
    out = ROOT / "data" / universe
    out.mkdir(parents=True, exist_ok=True)
    calendar = pd.DatetimeIndex(pd.read_csv(PROVIDER / "calendars/day.txt", header=None)[0])
    dates = calendar[(calendar >= "2010-01-01") & (calendar <= "2020-08-05")]
    spans = {}
    for line in (PROVIDER / f"instruments/{universe}.txt").read_text().splitlines():
        s, a, b = line.split()
        if a <= "2020-08-01" and b >= "2010-01-01":
            spans.setdefault(s.upper(), []).append((a, b))
    symbols = sorted(spans)
    member = np.zeros((len(symbols), len(dates)), dtype=bool)
    for i, s in enumerate(symbols):
        for a, b in spans[s]:
            member[i] |= (dates >= a) & (dates <= b)
    cache = ROOT / "data/instrument_cache"
    cache.mkdir(exist_ok=True)
    tasks = [(s, cache / f"{s}.npy") for s in symbols if not (cache / f"{s}.npy").exists()]
    write_json(out / "build_status.json", dict(phase="extracting", symbols=len(symbols), pending=len(tasks)))
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
        initargs=(str(PROVIDER), dates.tolist())) as pool:
        for n, _ in enumerate(pool.map(extract, tasks), 1):
            if n % 50 == 0:
                print(json.dumps(dict(universe=universe, extracted=n, total=len(tasks))), flush=True)
    raw = np.stack([np.load(cache / f"{s}.npy") for s in symbols])
    x = raw[:, :, :158]
    # Purge t+2 across the train boundary before fitting feature statistics.
    train = member & (np.asarray(dates.strftime("%Y-%m-%d"))[None, :] <= "2014-12-29")
    med, scale = [], []
    for j in range(158):
        values = x[:, :, j][train]
        m = float(np.nanmedian(values))
        sc = float((np.nanmedian(np.abs(values - m)) + 1e-12) * 1.4826)
        med.append(m); scale.append(sc)
        x[:, :, j] = np.clip((x[:, :, j] - m) / sc, -3, 3)
    # market channels: adjusted close, raw volume, factor, raw close, high, low, vwap
    market = raw[:, :, 158:].copy()
    close, volume, factor, rawclose, high, low, vwap = np.moveaxis(market, -1, 0)
    labels = labels_from_market(close, volume, high, low, vwap)
    # Preserve observed volume shock and range for strictly as-of-t APO moments.
    for name, a in dict(features=x, market=market, labels=labels, member=member,
                        dates=np.asarray(dates.strftime("%Y-%m-%d"), dtype="U10"), symbols=np.array(symbols)).items():
        np.save(out / f"{name}.npy", a)
    np.savez(out / "normalization.npz", median=np.array(med), scale=np.array(scale))
    from qlib.contrib.data.loader import Alpha158DL
    source_files = [PROVIDER / "calendars/day.txt", PROVIDER / f"instruments/{universe}.txt"]
    source_files += [PROVIDER / f"features/{s.lower()}/{f}.day.bin" for s in symbols
                     for f in ["open", "high", "low", "close", "vwap", "volume", "factor"]]
    write_json(out / "source_hashes.json", {str(p):sha(p) for p in source_files})
    manifest = dict(universe=universe, shape=list(x.shape), date_start=str(dates[0].date()),
        date_end=str(dates[-1].date()), training_start_missing="2008-2009 unavailable",
        created_utc=datetime.now(timezone.utc).isoformat(), provider=str(PROVIDER),
        source_archive_sha256="50d6fbb35634eb858f0e665e4811cbcbdcaccfba1d1e0084e8b7227deae17757",
        provider_source="https://huggingface.co/datasets/joshuaxql/qlib_data/tree/ee0fa6b2299cb5e927d2df491888c84f6c9c027a",
        feature_names=Alpha158DL.get_feature_config()[1],
        price_adjustment="All Alpha158 price references and return labels use raw price * factor. Volume stays in provider units.",
        normalization="Training dated members only: median, (MAD+1e-12)*1.4826, clip[-3,3]. Window-local ffill+mean; all-missing -> 0.",
        train_feature_fit_end="2014-12-29", lookback=21, target_offset=2,
        split="train through2014; valid2015-2016; test2017-2020-08-01; t+2 purged at boundaries",
        market_channels=["adjusted_close", "raw_volume", "factor", "raw_close", "raw_high", "raw_low", "raw_vwap"],
        builder_sha256=sha(__file__), output_hashes={p.name:sha(p) for p in sorted(out.glob("*.npy"))})
    write_json(out / "manifest.json", manifest)
    write_json(out / "build_status.json", dict(phase="complete", shape=list(x.shape)))
    print(json.dumps(dict(universe=universe, phase="complete", shape=list(x.shape))), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--universe", choices=["csi300", "csi500"], default="csi300")
    p.add_argument("--workers", type=int, default=2)
    args = p.parse_args()
    build(args.universe, args.workers)
