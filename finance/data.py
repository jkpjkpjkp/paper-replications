from __future__ import annotations

import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def source_path(config):
    """Resolve the frozen source specification against the shared input root."""
    root = Path(os.environ.get("REPLICATION_DATA_ROOT", ROOT.parents[1]))
    original = Path(config["source"])
    if "crsp-monthly" in original.parts:
        return root.joinpath(*original.parts[original.parts.index("crsp-monthly"):])
    return original if original.is_absolute() else root / original


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def month_id(date):
    d = pd.Timestamp(date)
    return d.year * 12 + d.month - 1


def month_label(value):
    return f"{int(value) // 12:04d}-{int(value) % 12 + 1:02d}"


def read_french(path, names):
    with zipfile.ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8-sig")
    lines = [line for line in text.splitlines() if re.match(r"^\d{6},", line)]
    frame = pd.read_csv(io.StringIO("\n".join(lines)), header=None, names=["yyyymm"] + names)
    frame["month"] = frame.yyyymm // 100 * 12 + frame.yyyymm % 100 - 1
    for name in names:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        frame.loc[frame[name].isin([-99.99, -999]), name] = np.nan
        frame[name] /= 100.0
    assert not frame.month.duplicated().any()
    return frame.drop(columns="yyyymm")


def factors():
    ff = read_french(ROOT / "sources/ff5_monthly.zip", ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    mom = read_french(ROOT / "sources/momentum_monthly.zip", ["mom"])
    frame = ff.merge(mom, on="month", validate="one_to_one")
    frame.to_parquet(ROOT / "data/factors.parquet", index=False)
    return frame


def stock_features(raw, ff):
    """Every value on row s depends only on that PERMNO's data through month s."""
    df = raw.sort_values(["permno", "month"]).reset_index(drop=True)
    assert not df[["permno", "month"]].duplicated().any()
    key = df.permno
    group = df.groupby("permno", sort=False)
    feat = df[["permno", "month"]].copy()
    r = df.ret.astype(float)

    def roll(series, window, operation):
        obj = series.groupby(key, sort=False).rolling(window, min_periods=window)
        return getattr(obj, operation)().reset_index(level=0, drop=True).to_numpy()

    for lag in range(12):
        feat[f"ret_lag{lag}"] = group.ret.shift(lag).to_numpy()
    # Flooring affects log-return predictors only, never realized portfolio P&L.
    logr = np.log1p(r.clip(lower=-0.999999))
    for window in (3, 6, 11, 12, 24):
        feat[f"momentum_{window}"] = np.expm1(roll(logr, window, "sum"))
    for window in (3, 6, 12, 24):
        feat[f"volatility_{window}"] = roll(r, window, "std")
    feat["skewness_12"] = roll(r, 12, "skew")
    feat["downside_12"] = np.sqrt(roll(r.clip(upper=0).pow(2), 12, "mean"))
    feat["positive_fraction_12"] = roll(r.gt(0).where(r.notna()).astype(float), 12, "mean")
    feat["max_return_12"] = roll(r, 12, "max")
    feat["min_return_12"] = roll(r, 12, "min")
    turnover = df.volume / (df.shrout * 1000.0)
    feat["turnover"] = turnover
    feat["turnover_12"] = roll(turnover, 12, "mean")
    feat["turnover_ratio"] = turnover / feat["turnover_12"]
    feat["log_price"] = np.log(df.price.abs().where(df.price.abs() > 0))
    feat["log_cap"] = np.log(df.cap.where(df.cap > 0))
    feat["log_dollar_volume"] = np.log1p(df.dollar_volume.clip(lower=0))
    feat["dollar_volume_change_12"] = (feat.log_dollar_volume -
                                      feat.log_dollar_volume.groupby(key).shift(12))
    feat["volatility_ratio"] = feat.volatility_3 / feat.volatility_12
    factor_map = ff.set_index("month")
    market = df.month.map(factor_map.mkt_rf)
    excess = r - df.month.map(factor_map.rf)
    em = pd.Series(roll(excess, 24, "mean"))
    mm = pd.Series(roll(market, 24, "mean"))
    cov = roll(excess * market, 24, "mean") - em * mm
    variance = roll(market * market, 24, "mean") - mm * mm
    feat["beta_24"] = cov / variance
    feat["residual_return_12"] = roll(excess - feat.beta_24 * market, 12, "sum")
    # A 24-month beta plus a 12-month residual lookback needs 35 source months;
    # omit residuals here to retain the protocol's fixed 24-month minimum.
    feat.drop(columns="residual_return_12", inplace=True)
    complete = group.month.shift(23).eq(df.month - 23)
    columns = [c for c in feat if c not in {"permno", "month"}]
    complete &= np.isfinite(feat[columns].to_numpy()).all(axis=1)
    return feat.loc[complete].copy(), columns


def build():
    config = json.loads((ROOT / "config.json").read_text())
    dest = ROOT / "data/panel.parquet"
    if dest.exists():
        print("Using prepared", dest, flush=True)
        return dest
    ff = factors()
    con = duckdb.connect(config={"threads": 2, "memory_limit": "2GB"})
    query = """
      SELECT permno, permco, siccd, yyyymm/100*12 + yyyymm%100 - 1 AS month_expression,
        yyyymm, primaryexch, tradingstatusflg, conditionaltype, mthdelflg,
        CAST(mthprc AS DOUBLE) price, CAST(mthcap AS DOUBLE) cap,
        CAST(mthret AS DOUBLE) ret, CAST(mthvol AS DOUBLE) volume,
        CAST(mthprcvol AS DOUBLE) dollar_volume, CAST(shrout AS DOUBLE) shrout
      FROM read_parquet(?)
      WHERE mthcaldt >= ? AND securitytype='EQTY' AND securitysubtype='COM'
        AND sharetype='NS' AND usincflg='Y' AND issuertype IN ('CORP','ACOR')
      ORDER BY permno, mthcaldt
    """
    raw = con.execute(query, [str(source_path(config)), config["raw_start"]]).fetchdf()
    raw["month"] = (raw.yyyymm // 100 * 12 + raw.yyyymm % 100 - 1).astype(np.int32)
    raw.drop(columns=["yyyymm", "month_expression"], inplace=True)
    for column in ("permno", "permco"):
        raw[column] = raw[column].astype(np.int32)
    print("Raw corporate common-stock rows:", len(raw), flush=True)
    feat, feature_names = stock_features(raw, ff)
    print("Rows with complete trailing features:", len(feat), flush=True)
    u = config["universe"]
    active = (raw.tradingstatusflg.eq("A") & raw.conditionaltype.eq("RW") &
              raw.mthdelflg.eq("N") & raw.primaryexch.isin(u["primaryexch"]) & raw.cap.gt(0))
    nyse = raw[active & raw.primaryexch.eq("N")].groupby("month").cap.quantile(u["nyse_market_cap_quantile"])
    eligible = (active & raw.price.abs().ge(u["minimum_price"]) &
                (raw.dollar_volume / 21).ge(u["minimum_average_daily_dollar_volume"]) &
                raw.cap.ge(raw.month.map(nyse)))
    formation = raw.loc[eligible, ["permno", "permco", "month", "cap", "price", "dollar_volume"]].copy()
    formation = (formation.sort_values(["month", "permco", "cap", "permno"],
                                      ascending=[True, True, False, True])
                 .drop_duplicates(["month", "permco"]))
    formation.rename(columns={"month": "formation_month", "cap": "formation_cap",
                              "price": "formation_price", "dollar_volume": "formation_dollar_volume"}, inplace=True)
    formation["month"] = formation.formation_month + 1
    feat.rename(columns={"month": "feature_month"}, inplace=True)
    feat["month"] = feat.feature_month + config["feature_lag_months"]
    panel = formation.merge(feat, on=["permno", "month"], validate="one_to_one")
    # Labels join ALL source securities, without filtering on future status/type.
    outcome = con.execute("""SELECT permno, yyyymm, CAST(mthret AS DOUBLE) realized_return,
                           mthdelflg AS outcome_delist_flag FROM read_parquet(?)
                           WHERE mthcaldt >= ?""", [str(source_path(config)), config["raw_start"]]).fetchdf()
    outcome["month"] = (outcome.yyyymm // 100 * 12 + outcome.yyyymm % 100 - 1).astype(np.int32)
    outcome.drop(columns="yyyymm", inplace=True)
    panel = panel.merge(outcome, on=["permno", "month"], how="left", validate="one_to_one")
    panel = panel[panel.month.between(month_id("1992-01-01"), month_id("2025-12-01"))].copy()
    panel.sort_values(["month", "permno"], inplace=True)
    panel.reset_index(drop=True, inplace=True)
    # Ranks depend only on known predictors and known formation-universe membership.
    raw_columns = panel[feature_names].copy()
    ranked = panel.groupby("month", sort=False)[feature_names].rank(method="average", pct=True)
    names = ["x_" + c for c in feature_names]
    panel[names] = (ranked * 2 - 1).to_numpy(dtype=np.float32)
    panel["momentum_signal"] = raw_columns.momentum_11.to_numpy(dtype=np.float32)
    panel["reversal_signal"] = -raw_columns.ret_lag0.to_numpy(dtype=np.float32)
    panel["beta_signal"] = raw_columns.beta_24.to_numpy(dtype=np.float32)
    panel.drop(columns=feature_names, inplace=True)
    panel["y_rank"] = (panel.groupby("month").realized_return.rank(method="average", pct=True) - 0.5).astype(np.float32)
    assert (panel.feature_month == panel.month - 2).all()
    assert (panel.formation_month == panel.month - 1).all()
    assert not panel[["permno", "month"]].duplicated().any()
    assert not panel[["permco", "month"]].duplicated().any()
    assert np.isfinite(panel[names].to_numpy()).all()
    assert panel.realized_return.dropna().ge(-1.000001).all()
    panel.to_parquet(dest, index=False, compression="zstd")
    counts = panel.groupby("month").agg(n=("permno", "size"), missing_returns=("realized_return", lambda s: s.isna().sum()))
    counts["date"] = [month_label(m) for m in counts.index]
    counts.to_csv(ROOT / "data/universe_by_month.csv")
    metadata = {"rows": len(panel), "stocks": int(panel.permno.nunique()), "features": names,
                "first_month": month_label(panel.month.min()), "last_month": month_label(panel.month.max()),
                "config_sha256": digest(ROOT / "config.json"), "panel_sha256": digest(dest),
                "minimum_monthly_universe": int(counts.n.min()), "maximum_monthly_universe": int(counts.n.max()),
                "missing_realized_returns": int(panel.realized_return.isna().sum()),
                "source_manifest_sha256": digest(source_path(config).parents[1] / "manifest.json"),
                "source_validation": json.loads((source_path(config).parents[1] / "validation.json").read_text()),
                "factor_archives_sha256": {p.name: digest(p) for p in (ROOT / "sources").glob("*.zip")}}
    (ROOT / "data/metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in {"features", "source_validation"}}, indent=2))
    return dest


if __name__ == "__main__":
    build()
