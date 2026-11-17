"""Table 1 estimands and validation, independent of download/output paths."""

from io import BytesIO
from importlib.resources import files

import numpy as np
import pandas as pd
import statsmodels.api as sm

MONTHS = pd.period_range("2001-01", "2024-12", freq="M")
SIZES = {1: "Micro", 2: "Small", 3: "Big"}
SIGNALS = {1: "Low", 2: "Neutral", 3: "High"}
KEYS = ["model", "size", "signal"]


def read_portfolios(content: bytes) -> pd.DataFrame:
    """Require exactly one finite observation per month/size/signal cell."""
    frame = pd.read_csv(BytesIO(content))
    required = {"date", "size_decile_3", "signal_decile_3", "Cap_VW_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame["month"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise").dt.to_period("M")
    for column in ["size_decile_3", "signal_decile_3"]:
        if not frame[column].isin([1, 2, 3]).all():
            raise ValueError(f"{column} must contain only 1, 2, 3")
    index_columns = ["month", "size_decile_3", "signal_decile_3"]
    if frame.duplicated(index_columns).any():
        raise ValueError("Duplicate month/size/signal observations")
    expected = pd.MultiIndex.from_product([MONTHS, [1, 2, 3], [1, 2, 3]], names=index_columns)
    actual = pd.MultiIndex.from_frame(frame[index_columns])
    if len(actual) != len(expected) or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError("Expected a complete 288-month x 3-size x 3-signal panel for 2001-01 through 2024-12")
    values = pd.to_numeric(frame["Cap_VW_return"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Portfolio returns must be finite; no missing-value imputation is permitted")
    if (values < -1).any():
        raise ValueError("A long-only portfolio return cannot be below -1 in decimal units")
    frame["Cap_VW_return"] = values
    return frame.sort_values(index_columns).reset_index(drop=True)


def mean_hac(values) -> dict:
    """Monthly mean (%) and constant-only OLS t-statistic, Bartlett HAC(6)."""
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) <= 7 or not np.isfinite(y).all():
        raise ValueError("HAC(6) requires a finite one-dimensional series with more than seven observations")
    fit = sm.OLS(y, np.ones((len(y), 1)), missing="raise").fit(
        cov_type="HAC",
        cov_kwds={"maxlags": 6, "kernel": "bartlett", "use_correction": False},
        use_t=False,
    )
    if not np.isfinite(fit.tvalues[0]):
        raise ValueError("Undefined t-statistic (degenerate return series)")
    return {"mean_pct": float(y.mean() * 100), "t_stat": float(fit.tvalues[0]), "n_months": len(y)}


def compute(inputs: dict[str, bytes]) -> pd.DataFrame:
    if set(inputs) != {"vit", "cnn"}:
        raise ValueError("Both ViT and CNN inputs are required")
    rows = []
    for model in ["vit", "cnn"]:
        frame = read_portfolios(inputs[model])
        for size, size_label in SIZES.items():
            block = frame.loc[frame.size_decile_3 == size].pivot(
                index="month", columns="signal_decile_3", values="Cap_VW_return"
            ).reindex(MONTHS)
            series = [(label, block[signal]) for signal, label in SIGNALS.items()]
            series.append(("H-L", block[3] - block[1]))
            for label, values in series:
                rows.append({"model": model, "size": size_label, "signal": label, **mean_hac(values)})
    return pd.DataFrame(rows)


def compare(table: pd.DataFrame) -> pd.DataFrame:
    """Compare every mean and t-statistic with the paper's printed precision."""
    with files("vision_to_value").joinpath("table1_expected.csv").open("rb") as stream:
        expected = pd.read_csv(stream)
    merged = expected.merge(table, on=KEYS, suffixes=("_published", "_computed"), how="outer", validate="1:1", indicator=True)
    if len(merged) != 24 or not merged["_merge"].eq("both").all():
        raise ValueError("Comparison requires all 24 portfolio series exactly once")
    rows = []
    for _, row in merged.iterrows():
        for metric in ["mean_pct", "t_stat"]:
            target, value = float(row[f"{metric}_published"]), float(row[f"{metric}_computed"])
            rows.append({**{key: row[key] for key in KEYS}, "metric": metric,
                         "published": target, "computed": value, "difference": value - target,
                         "matches_3dp": bool(np.isfinite(value) and round(value, 3) == target)})
    return pd.DataFrame(rows)
