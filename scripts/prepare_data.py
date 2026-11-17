"""Download author-provided raw data/forecasts and generate Appendix N scores."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://raw.githubusercontent.com/AdaMLLab/up-for-ocp/3575f1f94a14620e089422f52cf9e1a6766ec7b1/"


def acquire(relative):
    target = ROOT / "data" / relative.removeprefix("tests/datasets/")
    target.parent.mkdir(parents=True, exist_ok=True)
    author_copy = ROOT / "upstream" / relative
    if not target.exists():
        if author_copy.exists():
            shutil.copyfile(author_copy, target)
        else:
            print(f"Downloading {relative}", flush=True)
            urllib.request.urlretrieve(BASE + relative, target)
    return target


def main():
    output = ROOT / "data" / "scores"
    output.mkdir(parents=True, exist_ok=True)
    files, datasets = [], []
    raw = acquire("tests/datasets/raw/djia.csv")
    files.append(raw)
    stock = pd.read_csv(raw)
    for name in ("AXP", "AAPL", "AMZN", "GOOGL"):
        forecast_file = acquire(f"tests/datasets/processed/{name}/prophet.npz")
        files.append(forecast_file)
        series = stock.pivot(index="Date", columns="Name", values="Open").sort_index().interpolate()[name]
        forecast = np.load(forecast_file)["forecasts"]
        assert len(forecast) == len(series)
        scores = np.abs(series.to_numpy() - forecast)
        assert np.isfinite(scores).all()
        np.savez_compressed(output / f"{name}.npz", scores=scores[None, :],
                            y=series.to_numpy(), forecasts=forecast,
                            dates=series.index.to_numpy(dtype=str), burnin=100,
                            seeds=np.array([-1]))
        datasets.append(dict(name=name, length=len(scores), burnin=100, repeats=1,
                             source="author raw prices + cached Prophet forecasts",
                             date_start=str(series.index[0]), date_end=str(series.index[-1])))
    raw = acquire("tests/datasets/raw/elec2.csv")
    cache = acquire("tests/datasets/processed/elec2/ar.npz")
    files += [raw, cache]
    y = pd.read_csv(raw)["nswdemand"].iloc[:2000].interpolate().to_numpy()
    forecast = np.load(cache)["forecasts"]
    assert len(y) == len(forecast)
    np.savez_compressed(output / "elec2.npz", scores=np.abs(y - forecast)[None, :],
                        y=y, forecasts=forecast, burnin=300, seeds=np.array([-1]))
    datasets.append(dict(name="elec2", length=len(y), burnin=300, repeats=1,
                         source="author raw NSW demand + cached AR(3) forecasts"))
    for name in ("sinusoid", "stationary", "quadratic"):
        scores = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            t = np.arange(1, 3001)
            if name == "sinusoid":
                s = np.maximum(0, (np.sin(2 * np.pi * t / 200) + 0.5) * 10 + 2
                               + rng.normal(0, 0.3, len(t)))
            else:
                trend = np.full(len(t), 10.0) if name == "stationary" else 20 * (t / 3000) ** 2
                mask = rng.binomial(1, 0.1, len(t))
                magnitudes = rng.exponential(10, len(t))
                s = maximum_filter1d(trend * (1 + mask * magnitudes), size=25, mode="nearest")
            scores.append(s)
        np.savez_compressed(output / f"{name}.npz", scores=np.array(scores), burnin=300,
                            seeds=np.arange(10))
        datasets.append(dict(name=name, length=3000, burnin=300, repeats=10,
                             source="Appendix N equations; NumPy default_rng seeds 0..9",
                             edge_policy="centered max filter with nearest padding (random waves only)"))
    for p in sorted(output.glob("*.npz")):
        files.append(p)
    manifest = dict(raw_source=BASE, datasets=datasets,
                    files=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(datasets, indent=2))


if __name__ == "__main__":
    main()
