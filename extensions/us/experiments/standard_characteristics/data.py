"""Align month-end CZ signals to daily information sets without future filling."""
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from usdata import Panel as OriginalPanel, arrow_read, sha, write_json
from experiments.standard_characteristics.common import HERE, DATA, CZ, config
from experiments.standard_characteristics.download import download


def prior_month(dates):
    period = pd.DatetimeIndex(dates).to_period("M") - 1
    return np.array(period.year * 100 + period.month, dtype=int)


def common_shares(frame):
    return (frame.securitytype.eq("EQTY") & frame.securitysubtype.eq("COM") &
        frame.sharetype.eq("NS") & frame.issuertype.eq("CORP") & frame.usincflg.eq("Y") &
        frame.primaryexch.isin(["N", "A", "Q"]) & frame.conditionaltype.eq("RW") &
        frame.tradingstatusflg.eq("A") & frame.mthdelflg.eq("N") & frame.mthcap.gt(0))


def nonmicro_flags(monthly):
    population = monthly.loc[common_shares(monthly)].copy()
    thresholds = population.loc[population.primaryexch.eq("N")].groupby("yyyymm").mthcap.quantile(.2)
    population["nyse20"] = population.yyyymm.map(thresholds)
    population["nonmicro"] = population.mthcap.ge(population.nyse20)
    return population, thresholds


def rank_signals(frame):
    """Ranks of observed values: endpoints -1,+1; ties average; missing neutral."""
    count = frame.notna().sum()
    values = 2 * (frame.rank(method="average") - 1) / (count - 1).replace(0, np.nan) - 1
    return values.fillna(0).to_numpy(np.float32)


class Panel(OriginalPanel):
    def __init__(self):
        super().__init__()
        self.p = config()
        self.folder = DATA
        self.member = np.load(DATA / "member.npy", mmap_mode="r")
        self.eligible = self.member
        self.month_features = np.load(DATA / "monthly_features.npy", mmap_mode="r")
        self.month_index = np.load(DATA / "month_index.npy")
        self.capitalization = np.load(DATA / "lagged_cap.npy", mmap_mode="r")
        # Do not keep an unused daily Alpha158 mapping on this object.
        del self.features

    def day(self, t):
        ids = np.flatnonzero(self.eligible[:, t])
        months = self.month_index[t - self.lookback + 1:t + 1]
        x = np.array(self.month_features[ids[:, None], months[None, :], :])
        raw = np.array(self.labels[ids, t])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            target = (raw - np.nanmean(raw, axis=0)) / np.maximum(np.nanstd(raw, axis=0, ddof=1), 1e-8)
        return ids, x, target.astype(np.float32), raw


def build():
    DATA.mkdir(exist_ok=True)
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text())
        assert manifest["protocol_sha256"] == sha(HERE / "protocol.json")
        for name, expected in manifest["output_sha256"].items():
            assert sha(DATA / name) == expected, name
        print("Using verified standard-characteristic data", flush=True)
        return
    path = download()
    base = OriginalPanel()
    symbols = np.array([int(s[1:]) for s in base.symbols])
    months = np.unique(prior_month(base.dates))
    columns = ["permno", "yyyymm", "mthcap", "primaryexch", "securitytype", "securitysubtype",
        "sharetype", "issuertype", "usincflg", "conditionaltype", "tradingstatusflg", "mthdelflg"]
    monthly = arrow_read(ROOT.parent / "crsp-monthly/parquet", columns,
        [("yyyymm", ">=", int(months[0])), ("yyyymm", "<=", int(months[-1]))])
    assert not monthly.duplicated(["permno", "yyyymm"]).any()
    population, thresholds = nonmicro_flags(monthly)
    thresholds.rename("nyse20").to_csv(DATA / "nyse20_breakpoints.csv")
    key = pd.MultiIndex.from_product([symbols, months], names=["permno", "yyyymm"])
    m = monthly.set_index(["permno", "yyyymm"]).reindex(key)
    cap = m.mthcap.to_numpy().reshape(len(symbols), len(months))
    screen = population.set_index(["permno", "yyyymm"]).reindex(key).nonmicro
    screen = screen.fillna(False).to_numpy(bool).reshape(cap.shape)
    source_names = [n for n in pq.read_schema(path).names if n not in ["permno", "yyyymm"]]
    docs = pd.read_csv(CZ / "SignalDoc.csv")
    predictors = set(docs.loc[docs["Cat.Signal"].eq("Predictor"), "Acronym"])
    assert set(source_names) <= predictors and len(source_names) == 209
    # A year at a time limits memory, retaining only target symbols after ranking.
    all_features = np.zeros((*cap.shape, len(source_names)), np.float32)
    observed = np.zeros(cap.shape, bool)
    coverage = []
    for year in sorted(set(months // 100)):
        frame = pd.read_parquet(path, filters=[("yyyymm", ">=", int(year * 100 + 1)),
                                                ("yyyymm", "<=", int(year * 100 + 12))])
        assert not frame.duplicated(["permno", "yyyymm"]).any()
        eligible = population.loc[population.nonmicro & (population.yyyymm // 100 == year), ["permno", "yyyymm"]]
        frame = frame.merge(eligible, on=["permno", "yyyymm"], how="inner", validate="one_to_one")
        frame[source_names] = frame[source_names].replace([np.inf, -np.inf], np.nan)
        for month, group in frame.groupby("yyyymm", sort=True):
            if month not in months:
                continue
            rank = pd.DataFrame(rank_signals(group[source_names]), index=group.permno, columns=source_names)
            j = int(np.searchsorted(months, month))
            all_features[:, j] = rank.reindex(symbols).fillna(0).to_numpy(np.float32)
            observed[:, j] = np.isin(symbols, group.permno.to_numpy())
            coverage.append(dict(yyyymm=int(month), rank_population=len(group), target_records=int(observed[:, j].sum()),
                observed_fraction=float(group[source_names].notna().to_numpy().mean())))
        print("CZ rank/alignment", year, flush=True)
    month_index = np.searchsorted(months, prior_month(base.dates))
    lag_cap = cap[:, month_index]
    nonmicro = screen[:, month_index]
    records = observed[:, month_index]
    member = np.array(base.member) & nonmicro & records
    train_days = (base.dates >= config()["train"][0]) & (base.dates <= config()["train"][1])
    train_months = np.unique(month_index[train_days])
    train_values = all_features[:, train_months]
    train_mask = np.stack([member[:, np.flatnonzero(month_index == j)[0]] for j in train_months], axis=1)
    values = train_values[train_mask]
    keep = np.ptp(values, axis=0) > 1e-12
    assert keep.sum() > 100
    np.save(DATA / "monthly_features.npy", all_features[:, :, keep])
    np.save(DATA / "month_index.npy", month_index)
    np.save(DATA / "source_months.npy", months)
    np.save(DATA / "member.npy", member)
    np.save(DATA / "lagged_cap.npy", lag_cap)
    pd.DataFrame(coverage).to_csv(DATA / "coverage.csv", index=False)
    audit = []
    for label, mask in [("all", np.ones(len(base.dates), bool))] + [(s, (base.dates >= config()[s][0]) &
            (base.dates <= config()[s][1])) for s in ["train", "valid", "test"]]:
        before = np.asarray(base.member[:, mask])
        known = np.isfinite(lag_cap[:, mask]) & (lag_cap[:, mask] > 0)
        below = known & (lag_cap[:, mask] < np.array(thresholds.reindex(months))[month_index[mask]][None, :])
        audit.append(dict(split=label, original_stock_days=int(before.sum()),
            microcap_removed=int((before & below).sum()), unknown_lag_cap_removed=int((before & ~known).sum()),
            other_screen_removed=int((before & known & ~below & ~nonmicro[:, mask]).sum()),
            missing_cz_removed=int((before & nonmicro[:, mask] & ~records[:, mask]).sum()),
            retained_stock_days=int(member[:, mask].sum()),
            min_stocks=int(member[:, mask].sum(0).min()), max_stocks=int(member[:, mask].sum(0).max())))
    pd.DataFrame(audit).to_csv(DATA / "universe_audit.csv", index=False)
    source_paths = [path, CZ / "SignalDoc.csv", ROOT / "data/manifest.json",
        ROOT.parent / "crsp-monthly/manifest.json", ROOT.parent / "crsp-monthly/validation.json"]
    outputs = list(DATA.glob("*.npy")) + list(DATA.glob("*.csv"))
    manifest = dict(protocol_sha256=sha(HERE / "protocol.json"), builder_sha256=sha(__file__),
        feature_names=np.array(source_names)[keep].tolist(), dropped_training_constant=np.array(source_names)[~keep].tolist(),
        omitted_from_public_source=["Price", "Size", "STreversal"], shape=list(all_features[:, :, keep].shape),
        source_sha256={str(f): sha(f) for f in source_paths},
        output_sha256={f.name: sha(f) for f in outputs}, universe_audit=audit)
    write_json(manifest_path, manifest)
    print("Data complete", manifest["shape"], audit, flush=True)


if __name__ == "__main__":
    build()
