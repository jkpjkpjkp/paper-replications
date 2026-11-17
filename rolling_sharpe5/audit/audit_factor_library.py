"""Independent feasibility audit; run with ../../.venv/bin/python from this folder."""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10"
OUT = Path(__file__).resolve().parent

docs = pd.read_csv(LIBRARY / "SignalDoc.csv")
docs = docs.loc[docs["Cat.Signal"].eq("Predictor")].set_index("Acronym")
returns = duckdb.read_parquet(
    str(LIBRARY / "Portfolios/Full Sets OP/PredictorLSretWide.parquet")
).df().set_index("date")
assert set(returns.columns) == set(docs.index)
assert returns.index.is_unique
assert not np.isinf(returns.to_numpy()).any()
assert returns.index.to_period("M").equals(
    pd.period_range(returns.index.min(), returns.index.max(), freq="M")
)
group = returns.groupby(returns.index.year)
n = group.count()
mean = group.mean()
sd = group.std()
tstat = mean / (sd / np.sqrt(n))
p1 = pd.DataFrame(t.sf(tstat, n - 1), index=n.index, columns=n.columns)
annual = pd.DataFrame(
    {
        "n": n.stack(),
        "mean_pct": mean.stack(),
        "std_pct": sd.stack(),
        "annualized_sharpe": (mean / sd * np.sqrt(12)).stack(),
        "t_iid": tstat.stack(),
        "p_one_sided": p1.stack(),
    }
).reset_index(names=["year", "factor"])
annual.to_csv(OUT / "annual_iid_stats.csv", index=False)
rows = []
valid = n.eq(12) & mean.gt(0)
for year in range(int(n.index.min()) + 5, int(n.index.max()) + 2):
    prior = list(range(year - 5, year))
    for test, alpha in [("one_sided", 0.05), ("two_sided", 0.025)]:
        passed = ((p1.loc[prior] < alpha) & valid.loc[prior]).all(axis=0)
        names = passed.index[passed].tolist()
        published = [name for name in names if docs.loc[name, "Year"] < year]
        rows.append(
            {
                "year": year,
                "test": test,
                "count": len(names),
                "factors": ";".join(names),
                "published_factors": ";".join(published),
                "published_count": len(published),
            }
        )
pool = pd.DataFrame(rows)
pool.to_csv(OUT / "strict_annual_iid_pool_counts.csv", index=False)
print(pool.tail(12).to_string(index=False))
print(
    pool.groupby("test").agg(
        years=("year", "size"),
        nonempty=("count", lambda x: int((x > 0).sum())),
        total_factor_years=("count", "sum"),
        published_nonempty=("published_count", lambda x: int((x > 0).sum())),
        published_total=("published_count", "sum"),
    ).to_string()
)
