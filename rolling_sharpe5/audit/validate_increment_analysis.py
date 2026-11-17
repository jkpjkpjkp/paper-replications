"""Independent checks of chronological selection and actual construction code."""
from pathlib import Path
import hashlib
import importlib.util
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve().parent
MODULE = HERE.parent / "analyze_increment.py"
spec = importlib.util.spec_from_file_location("increment_under_review", MODULE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
analysis.OUT = HERE / "increment_validation_artifacts"
analysis.OUT.mkdir(exist_ok=True)
report = {}

# Use identifiable independent factors over 60 training months plus one year.
rng = np.random.default_rng(43812)
idx = pd.period_range("1985-01", "1990-12", freq="M")
factors = pd.DataFrame(rng.normal(0.01, 0.03, (len(idx), 2)), index=idx, columns=["A", "B"])
beta_true = np.array([0.7, -0.3])
alpha_true = .006
paper = pd.DataFrame({"toy": alpha_true + factors.dot(beta_true)}, index=idx)
counts = pd.DataFrame([
    dict(target_year=1990, method="each_year_iid", test="two_sided", availability="published_before_year", factor_count=0),
    dict(target_year=1990, method="each_year_hac1", test="two_sided", availability="published_before_year", factor_count=2),
])
members = pd.DataFrame([
    dict(target_year=1990, method="each_year_hac1", test="two_sided", availability="published_before_year", factor=f, publication_year=1980)
    for f in factors
])
result = analysis.make_pools(factors, counts, members, paper)
betas = pd.read_csv(analysis.OUT / "oos_betas.csv").set_index("factor")
nonempty = result[result.method.eq("each_year_hac1")]
empty = result[result.method.eq("each_year_iid")]
expected_base = factors.loc["1990"].mean(axis=1).to_numpy()
expected_y = paper.loc["1990", "toy"].to_numpy()
np.testing.assert_allclose(nonempty.pool_gross, expected_base)
np.testing.assert_allclose(nonempty.augmented_mixed_cost, (2 * expected_base + expected_y) / 3)
np.testing.assert_allclose(nonempty.increment, (expected_y - expected_base) / 3)
np.testing.assert_allclose(empty.pool_gross, 0)
np.testing.assert_allclose(empty.augmented_mixed_cost, expected_y)
np.testing.assert_allclose(empty.increment, expected_y)
np.testing.assert_allclose(empty.oos_spanning_residual, expected_y)
np.testing.assert_allclose(betas.loc[["A", "B"], "beta"], beta_true, atol=1e-12)
np.testing.assert_allclose(betas.train_intercept, alpha_true, atol=1e-12)
np.testing.assert_allclose(nonempty.oos_spanning_residual, alpha_true, atol=1e-12)
report["known_beta_recovered"] = True
report["intercept_retained_in_oos_residual"] = True
report["k0_and_k2_equal_sleeve_accounting"] = True

# Alter target-year factor returns AND paper returns drastically. Estimated
# coefficients must be unchanged because estimation ends in prior December.
changed_factors = factors.copy()
changed_paper = paper.copy()
changed_factors.loc["1990"] += np.array([10., -20.])
changed_paper.loc["1990", "toy"] += 30.
analysis.make_pools(changed_factors, counts, members, changed_paper)
changed_betas = pd.read_csv(analysis.OUT / "oos_betas.csv").set_index("factor")
np.testing.assert_allclose(changed_betas.beta, betas.beta, atol=1e-12)
np.testing.assert_allclose(changed_betas.train_intercept, betas.train_intercept, atol=1e-12)
report["target_year_changes_do_not_change_betas"] = True

# Verify real-library annual selection independently and perturb future rows.
library = HERE.parents[2] / "Chen_Zimmermann_Factor_Zoo_2026_10"
real = duckdb.read_parquet(str(library / "Portfolios/Full Sets OP/PredictorLSretWide.parquet")).df().set_index("date")
real.index = real.index.to_period("M")
docs = pd.read_csv(library / "SignalDoc.csv").set_index("Acronym")


def select_annual_iid(panel, target_year):
    prior = panel.reindex(pd.period_range(f"{target_year-5}-01", f"{target_year-1}-12", freq="M"))
    grouped = prior.groupby(prior.index.year)
    n = grouped.count()
    mean = grouped.mean()
    sd = grouped.std(ddof=1)
    tstat = mean / (sd / np.sqrt(n))
    p = pd.DataFrame(2 * t.sf(np.abs(tstat), n - 1), index=n.index, columns=n.columns)
    passed = ((p < .05) & mean.gt(0) & n.eq(12)).all(axis=0)
    passed &= docs.Year.reindex(passed.index).lt(target_year)
    return sorted(passed.index[passed])


all_members = pd.read_csv(HERE / "extended_selection_members.csv")
for target in [1997, 1998, 2008, 2026]:
    expected = sorted(all_members[(all_members.target_year.eq(target)) & all_members.method.eq("each_year_iid") & all_members.test.eq("two_sided") & all_members.availability.eq("published_before_year")].factor.tolist())
    assert select_annual_iid(real, target) == expected
    altered = real.copy()
    mask = altered.index.year >= target
    altered.loc[mask] = rng.normal(100., 100., (mask.sum(), len(real.columns)))
    assert select_annual_iid(altered, target) == expected
report["independent_real_selection_matches_4_years"] = True
report["target_future_return_changes_do_not_change_selection"] = True

# Missing one prior month must exclude that factor, never skip across a gap.
assert "AnnouncementReturn" in select_annual_iid(real, 1998)
missing = real.copy()
missing.loc[pd.Period("1997-07", freq="M"), "AnnouncementReturn"] = np.nan
assert "AnnouncementReturn" not in select_annual_iid(missing, 1998)
report["incomplete_prior_year_excluded"] = True

# Confirm actual monthly compounding numerical convention on a tiny known path.
toy_daily = pd.DataFrame({"p": [.1, -.05, .03]}, index=pd.to_datetime(["2001-01-02", "2001-01-03", "2001-02-01"]))
compounded = analysis.monthly_compound(toy_daily)
np.testing.assert_allclose(compounded.p, [1.1 * .95 - 1, .03], atol=1e-12)
report["monthly_compounding_known_path"] = True
report["test_source"] = str(MODULE)
report["test_source_sha256"] = hashlib.sha256(MODULE.read_bytes()).hexdigest()
(HERE / "increment_validation_results.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
