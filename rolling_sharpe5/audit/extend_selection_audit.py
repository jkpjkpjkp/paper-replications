"""Annual HAC1 and pooled-60-month selection audits, never use target-year returns."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10"
OUT = Path(__file__).resolve().parent
docs = pd.read_csv(LIBRARY / "SignalDoc.csv")
docs = docs.loc[docs["Cat.Signal"].eq("Predictor")].set_index("Acronym")
returns = duckdb.read_parquet(
    str(LIBRARY / "Portfolios/Full Sets OP/PredictorLSretWide.parquet")
).df().set_index("date")
returns.index = returns.index.to_period("M")
assert returns.index.is_unique
assert returns.index.equals(pd.period_range(returns.index.min(), returns.index.max(), freq="M"))
factors = returns.columns


def stats_complete(frame, expected, lag):
    """Finite-sample corrected Bartlett HAC intercept t, with t(n-1) p-values."""
    nobs = frame.notna().sum()
    mean = frame.mean()
    sd = frame.std(ddof=1)
    complete = nobs.eq(expected) & (len(frame) == expected)
    x = frame.to_numpy()
    u = x - np.mean(x, axis=0)
    meat = np.sum(u * u, axis=0)
    for j in range(1, lag + 1):
        meat += 2 * (1 - j / (lag + 1)) * np.sum(u[j:] * u[:-j], axis=0)
    se = np.sqrt(meat / expected**2 * expected / (expected - 1))
    hac_t = mean / se
    iid_t = mean / (sd / np.sqrt(expected))
    result = pd.DataFrame({
        "factor": factors,
        "n": nobs.to_numpy(),
        "complete": complete.to_numpy(),
        "mean_pct": mean.to_numpy(),
        "sharpe_annualized": (mean / sd * np.sqrt(12)).to_numpy(),
        "t_iid": iid_t.to_numpy(),
        "p2_iid": 2 * t.sf(np.abs(iid_t), expected - 1),
        "t_hac": hac_t.to_numpy(),
        "p2_hac": 2 * t.sf(np.abs(hac_t), expected - 1),
    }).set_index("factor")
    result.loc[~complete, ["t_iid", "p2_iid", "t_hac", "p2_hac"]] = np.nan
    return result


annual = {}
for year in range(1985, 2026):
    expected = pd.period_range(f"{year}-01", f"{year}-12", freq="M")
    annual[year] = stats_complete(returns.reindex(expected), 12, 1)
pd.concat(annual, names=["year"]).to_csv(OUT / "annual_hac1_audit.csv")

count_rows, name_rows, pooled_rows = [], [], []
for year in range(1990, 2027):
    expected = pd.period_range(f"{year-5}-01", f"{year-1}-12", freq="M")
    frame = returns.reindex(expected)
    pooled = stats_complete(frame, 60, 3)
    pooled_rows.append(pooled.assign(target_year=year).reset_index())
    yearly = pd.concat([annual[y] for y in range(year - 5, year)], keys=range(year - 5, year), names=["year"])
    for method, underlying in [("each_year_iid", "iid"), ("each_year_hac1", "hac"), ("pooled60_iid", "iid"), ("pooled60_hac3", "hac")]:
        for tail, p2_cutoff in [("two_sided", .05), ("one_sided", .10)]:
            if method.startswith("each_year"):
                passed = ((yearly[f"p2_{underlying}"] < p2_cutoff) & yearly["mean_pct"].gt(0) & yearly.complete).groupby("factor").all()
            else:
                passed = (pooled[f"p2_{underlying}"] < p2_cutoff) & pooled.mean_pct.gt(0) & pooled.complete
            for availability in ["published_before_year", "unrestricted"]:
                selected = passed.copy()
                if availability == "published_before_year":
                    selected &= docs.Year.reindex(selected.index).lt(year)
                names = sorted(selected.index[selected].tolist())
                key = dict(target_year=year, method=method, test=tail, availability=availability)
                count_rows.append(dict(**key, factor_count=len(names), window_start=str(expected.min()), window_end=str(expected.max())))
                name_rows.extend(dict(**key, factor=f, publication_year=int(docs.loc[f,"Year"])) for f in names)

counts = pd.DataFrame(count_rows)
counts.to_csv(OUT / "extended_selection_counts.csv", index=False)
pd.DataFrame(name_rows).to_csv(OUT / "extended_selection_members.csv", index=False)
pd.concat(pooled_rows, ignore_index=True).to_csv(OUT / "pooled60_stats.csv", index=False)

# Independent check of HAC implementation against statsmodels on full windows.
for year, factor, lag, periods in [(2025,"STreversal",1,12),(2024,"BM",1,12),(2025,"Mom12m",3,60)]:
    end = pd.Period(f"{year}-12",freq="M")
    index = pd.period_range(end=end,periods=periods,freq="M")
    frame = returns.reindex(index)
    mine = stats_complete(frame,periods,lag).loc[factor]
    vector = frame[factor].to_numpy()
    check = sm.OLS(vector,np.ones((periods,1)),missing="raise").fit(cov_type="HAC",cov_kwds={"maxlags":lag,"use_correction":True},use_t=True)
    assert np.isclose(mine.t_hac,check.tvalues[0],rtol=1e-10)
    assert np.isclose(mine.p2_hac,check.pvalues[0],rtol=1e-10)
print(counts[(counts.target_year.eq(2026)) & counts.test.eq("two_sided")].to_string(index=False))
print("HAC vectorized implementation checked against statsmodels for 3 windows")
print(counts[counts.test.eq("two_sided")].groupby(["method","availability"]).factor_count.agg(["min","max","mean",lambda s:int(s.gt(0).sum())]).to_string())
