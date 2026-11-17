"""Independent NumPy/SciPy reconstruction, without statsmodels, of dynamic alpha."""
from pathlib import Path
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
LIBRARY = HERE.parents[2] / "Chen_Zimmermann_Factor_Zoo_2026_10"
factors = duckdb.read_parquet(str(LIBRARY / "Portfolios/Full Sets OP/PredictorLSretWide.parquet")).df().set_index("date") / 100
factors.index = factors.index.to_period("M")
paper = pd.read_csv(RESULTS / "paper_monthly.csv", index_col="month")
paper.index = pd.PeriodIndex(paper.index, freq="M")
members = pd.read_csv(HERE / "extended_selection_members.csv")
selected = members[members.method.eq("each_year_iid") & members.availability.eq("published_before_year") & members.test.eq("two_sided")]
reported = pd.read_csv(RESULTS / "dynamic_pool_alpha.csv")
checks = []
for start, end in [(1990, 2025), (2005, 2025)]:
    idx = pd.period_range(f"{start}-01", f"{end}-12", freq="M")
    columns = [np.ones(len(idx))]
    names = ["constant"]
    for row in selected[selected.target_year.between(start, end)].sort_values(["target_year", "factor"]).itertuples():
        col = np.zeros(len(idx))
        active = idx.year == row.target_year
        col[active] = factors.loc[idx[active], row.factor]
        columns.append(col)
        names.append(f"{row.target_year}:{row.factor}")
    X = np.column_stack(columns)
    y = paper.loc[idx, "5bps"].to_numpy()
    n, p = X.shape
    assert np.linalg.matrix_rank(X) == p
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y - X @ coefficients
    scores = X * residuals[:, None]
    bread = np.linalg.inv(X.T @ X)
    for lag in [3, 6, 12]:
        meat = scores.T @ scores
        for ell in range(1, lag + 1):
            lagcov = scores[ell:].T @ scores[:-ell]
            meat += (1 - ell / (lag + 1)) * (lagcov + lagcov.T)
        covariance = bread @ meat @ bread * n / (n - p)
        se = np.sqrt(covariance[0, 0])
        tstat = coefficients[0] / se
        pval = 2 * student_t.sf(abs(tstat), n - p)
        critical = student_t.ppf(.975, n - p)
        derived = {
            "annual_alpha": coefficients[0] * 12,
            "t": tstat,
            "p": pval,
            "ci_low": (coefficients[0] - critical * se) * 12,
            "ci_high": (coefficients[0] + critical * se) * 12,
        }
        baseline = reported[reported.method.eq("each_year_iid") & reported.availability.eq("published_before_year") & reported.scenario.eq("5bps") & reported.period.eq(f"{start}-{end}") & reported.hac_lags.eq(lag)].iloc[0]
        for field, value in derived.items():
            np.testing.assert_allclose(value, baseline[field], atol=1e-12, rtol=1e-10)
        assert baseline.slope_count == p - 1
        checks.append({"period": f"{start}-{end}", "hac_lags": lag, "n": n, "slope_count": p - 1, "intercept_columns": 1, "annual_se": se * 12, **derived})

inference = pd.read_csv(RESULTS / "inference.csv")
costed = inference[inference.method.str.startswith("each_year") & inference.scenario.isin(["5bps", "5bps_1pct_borrow"]) & inference.series.isin(["increment", "oos_spanning_residual"]) & inference.hac_lags.isin([3, 6, 12])]
positive_significant = costed[costed.annual_mean.gt(0) & costed.p.lt(.05)]
dynamic_costed = reported[reported.scenario.isin(["5bps", "5bps_1pct_borrow"])]
dynamic_positive_significant = dynamic_costed[dynamic_costed.annual_alpha.gt(0) & dynamic_costed.p.lt(.05)]
assert positive_significant.empty
assert dynamic_positive_significant.empty

report = {
    "independent_manual_hac_checks": checks,
    "costed_increment_or_oos_residual_rows_checked": len(costed),
    "positive_significant_costed_increment_or_oos_residual_rows": len(positive_significant),
    "costed_dynamic_alpha_rows_checked": len(dynamic_costed),
    "positive_significant_costed_dynamic_alpha_rows": len(dynamic_positive_significant),
    "interpretation": "Common-alpha ex-post conditional regression, one separate selected-factor slope per target year, no year intercepts; not out-of-sample tradable hedge weights. Incumbent controls remain gross.",
    "qualification": "Gross scenarios do have significant positive findings. Absence of significance applies to the specified costed cases; not proof the true effect is zero or discovery-adjusted inference.",
    "source_sha256": hashlib.sha256((HERE.parent / "analyze_increment.py").read_bytes()).hexdigest(),
}
(HERE / "dynamic_alpha_validation.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
