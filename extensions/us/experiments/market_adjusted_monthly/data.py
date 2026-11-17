"""Future-month labels, explicit realization endpoints and shared train-only scales."""
from pathlib import Path
import json
import math
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from usdata import Panel as BasePanel, ROOT, config, sha, write_json
from experiments.market_adjusted_monthly.common import HERE, FACTORS, INDEX, now


def month_targets(returns, market, dates):
    months = np.asarray(dates).astype("U7")
    unique = np.unique(months)
    sigma = np.full(returns.shape, np.nan, dtype=np.float32)
    market_sigma = np.full(len(dates), np.nan, dtype=np.float64)
    end = np.full(len(dates), -1, dtype=np.int64)
    for first, second in zip(unique[:-1], unique[1:]):
        if pd.Period(second, "M") != pd.Period(first, "M") + 1:
            continue
        source = np.flatnonzero(months == first)
        target = np.flatnonzero(months == second)
        # The retained dataset ends at 2025-12-31; require a complete calendar month.
        if pd.Timestamp(str(dates[target[-1]])).to_period("M") != pd.Period(second, "M"):
            raise AssertionError("Bad monthly calendar")
        minimum = max(15, math.ceil(.8 * len(target)))
        observed = np.isfinite(returns[:, target]).sum(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.nanstd(returns[:, target], axis=1, ddof=1)
        values[observed < minimum] = np.nan
        sigma[:, source] = values[:, None]
        market_sigma[source] = np.std(market[target], ddof=1)
        end[source] = target[-1]
    return sigma, market_sigma, end


def build():
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    provenance = dict(base_manifest=sha(ROOT / "data/manifest.json"), factors=sha(FACTORS / "ff5_daily.csv"),
                      builder=sha(__file__), protocol=sha(HERE / "protocol.json"))
    if (out / "manifest.json").exists():
        old = json.loads((out / "manifest.json").read_text())
        assert old["provenance"] == provenance
        for name, expected in old["output_sha256"].items():
            assert sha(out / name) == expected
        return
    source_audit = json.loads((FACTORS / "verification.json").read_text())
    assert source_audit["complete"]
    assert sha(FACTORS / "ff5_daily.csv") == source_audit["outputs"]["ff5_daily"]["files"]["csv"]["sha256"]
    base = BasePanel()
    source_factors = pd.read_csv(FACTORS / "ff5_daily.csv").set_index("date")
    calendar = source_factors.loc[(source_factors.index >= base.dates[0]) & (source_factors.index <= base.dates[-1])].index
    np.testing.assert_array_equal(calendar, base.dates)
    factors = source_factors.reindex(base.dates)
    market = (factors.mkt_rf + factors.rf).to_numpy(dtype=float)
    assert np.isfinite(market).all() and len(market) == len(base.dates)
    np.save(out / "market_return.npy", market)
    sigma, market_sigma, end = month_targets(base.total_return, market, base.dates)
    raw = np.array(base.labels, dtype=np.float32)
    raw[:, :-2, 0] = np.asarray(base.total_return[:, 2:])
    raw[:, :, 2] = sigma
    adjusted = raw.copy()
    adjusted[:, :-2, 0] -= market[None, 2:]
    adjusted[:, :, 2] -= market_sigma[None, :]
    raw[~np.isfinite(raw)] = np.nan
    adjusted[~np.isfinite(adjusted)] = np.nan
    for name, array in [("raw_monthly.npy", raw), ("market_adjusted.npy", adjusted),
                        ("monthly_market_sigma.npy", market_sigma), ("label_end.npy", end)]:
        np.save(out / name, array)
    ix = np.arange(base.lookback - 1, len(base.dates) - 2)
    valid_end = end[ix] >= 0
    ix = ix[valid_end]
    a, b = config()["train"]
    training = ix[(base.dates[ix] >= a) & (base.dates[np.maximum(ix + 2, end[ix])] <= b)]
    means, scales, counts = [], [], []
    for k in range(3):
        values = raw[:, training, k][base.member[:, training]].astype(float)
        values = values[np.isfinite(values)]
        means.append(float(values.mean()))
        scales.append(float(values.std(ddof=1)))
        counts.append(len(values))
    assert np.isfinite(scales).all() and min(scales) > 0
    np.savez(out / "scaling.npz", mean=means, scale=scales)
    result = dict(provenance=provenance, created_utc=now(), mean=means, scale=scales, training_counts=counts,
                  market_definition="Fama-French Mkt-RF + RF, decimal total returns", volatility="Next calendar-month sample daily total-return standard deviation",
                  output_sha256={path.name: sha(path) for path in sorted(out.glob("*.npy")) + sorted(out.glob("*.npz"))})
    write_json(out / "manifest.json", result)
    summary = {}
    for split in ["train", "valid", "test"]:
        panel = Panel("raw_monthly")
        days = panel.split(split)
        summary[split] = dict(days=len(days), first_signal=str(base.dates[days[0]]), last_signal=str(base.dates[days[-1]]),
                              last_label_date=str(base.dates[end[days[-1]]]), target_months=len(np.unique(base.dates[end[days]].astype("U7"))))
    write_json(out / "split_summary.json", summary)
    print("DATA BUILT", json.dumps(summary), flush=True)


class Panel(BasePanel):
    def __init__(self, regime):
        super().__init__()
        self.regime = regime
        self.labels = np.load(HERE / "data" / f"{regime}.npy", mmap_mode="r")
        self.label_end = np.load(HERE / "data/label_end.npy")
        scale = np.load(HERE / "data/scaling.npz")
        self.mean, self.scale = scale["mean"], scale["scale"]

    def split(self, split):
        a, b = config()[split]
        ix = np.arange(self.lookback - 1, len(self.dates) - 2)
        ix = ix[self.label_end[ix] >= 0]
        end = np.maximum(ix + 2, self.label_end[ix])
        return ix[(self.dates[ix] >= a) & (self.dates[end] <= b) & (self.eligible[:, ix].sum(0) >= 50)]

    def day(self, t):
        ids, x, _, raw = super().day(t)
        normalized = ((raw - self.mean) / self.scale).astype(np.float32)
        return ids, x, normalized, raw


if __name__ == "__main__":
    build()
