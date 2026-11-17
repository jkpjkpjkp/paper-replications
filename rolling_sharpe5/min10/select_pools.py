"""Keep strict five-year passers and top up to ten using past-only rankings.

The primary ranking maximizes the weakest signed annual IID t-statistic over
the preceding five complete calendar years. Current-year returns are read only
after selection for a missing-return audit; they never alter eligibility/rank.
"""
from pathlib import Path
import argparse
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
LIBRARY = PROJECT.parent / "Chen_Zimmermann_Factor_Zoo_2026_10"
OUT = HERE / "results"
METHODS = {
    "min10_annual_iid": dict(annual="iid", pooled="iid", rank="annual_min_t"),
    "min10_annual_hac1": dict(annual="hac", pooled="hac", rank="annual_min_t"),
    "min10_pooled_iid": dict(annual="iid", pooled="iid", rank="pooled_t"),
    "min10_pooled_hac3": dict(annual="hac", pooled="hac", rank="pooled_t"),
}
for _name, _spec in list(METHODS.items()):
    METHODS[_name] = dict(_spec, universe="all")
    METHODS[f"{_name}_no_options"] = dict(_spec, universe="no_options")
UNIVERSE_DEFINITIONS = {
    "all": "All212Predictor signals in the original-paper factor library",
    "no_options": "Exclude every signal with SignalDoc.csv Cat.Data == Options, independently of holding-period returns",
}


def complete_window_stats(frame, required_n, hac_lag):
    """Signed IID/HAC intercept t; Bartlett HAC correction n/(n-1)."""
    n = frame.notna().sum()
    complete = n.eq(required_n) & (len(frame) == required_n)
    x = frame.to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.mean(x, axis=0)
        sd = np.std(x, axis=0, ddof=1)
        residual = x - mean
        meat = np.sum(residual**2, axis=0)
        for lag in range(1, hac_lag + 1):
            meat += 2 * (1 - lag / (hac_lag + 1)) * np.sum(residual[lag:] * residual[:-lag], axis=0)
        hac_se = np.sqrt(meat / required_n**2 * required_n / (required_n - 1))
        iid_t = mean / (sd / np.sqrt(required_n))
        hac_t = mean / hac_se
    result = pd.DataFrame({
        "n": n, "complete": complete, "mean_pct": mean,
        "t_iid": iid_t, "t_hac": hac_t,
        "p2_iid": 2 * student_t.sf(np.abs(iid_t), required_n - 1),
        "p2_hac": 2 * student_t.sf(np.abs(hac_t), required_n - 1),
    }, index=frame.columns)
    result.index.name = "factor"
    result.loc[~complete, ["mean_pct", "t_iid", "t_hac", "p2_iid", "p2_hac"]] = np.nan
    return result


def score_candidates(returns, docs, target_year, method, availability, annual_cache=None):
    """Pure selection inputs: slice only target_year-5 through target_year-1."""
    spec = METHODS[method]
    index = pd.period_range(f"{target_year-5}-01", f"{target_year-1}-12", freq="M")
    window = returns.reindex(index)
    pooled = complete_window_stats(window, 60, 3)
    annual = []
    for year in range(target_year - 5, target_year):
        a = (annual_cache[year].copy() if annual_cache is not None else
             complete_window_stats(window.loc[window.index.year == year], 12, 1))
        annual.append(a.assign(year=year))
    annual = pd.concat(annual).reset_index()
    grouped = annual.groupby("factor", sort=False)
    candidate = pd.DataFrame(index=returns.columns)
    candidate.index.name = "factor"
    candidate["publication_year"] = docs.Year.reindex(candidate.index).astype(int)
    candidate["option_derived"] = docs["Cat.Data"].reindex(candidate.index).str.strip().str.casefold().eq("options")
    candidate["universe"] = spec["universe"]
    candidate["universe_definition"] = UNIVERSE_DEFINITIONS[spec["universe"]]
    candidate["training_months"] = pooled.n
    candidate["complete_years"] = grouped.complete.sum()
    candidate["complete_60_months"] = pooled.complete & candidate.complete_years.eq(5)
    candidate["publication_eligible"] = candidate.publication_year.lt(target_year)
    candidate["annual_min_t"] = grouped[f"t_{spec['annual']}"] .min()
    candidate["annual_max_p2"] = grouped[f"p2_{spec['annual']}"] .max()
    candidate["annual_min_mean_pct"] = grouped.mean_pct.min()
    candidate["pooled_t"] = pooled[f"t_{spec['pooled']}"]
    candidate["pooled_p2"] = pooled[f"p2_{spec['pooled']}"]
    candidate.loc[~candidate.complete_60_months, ["annual_min_t", "annual_max_p2", "annual_min_mean_pct", "pooled_t", "pooled_p2"]] = np.nan
    candidate["strict_pass"] = candidate.complete_60_months & candidate.annual_max_p2.lt(.05) & candidate.annual_min_mean_pct.gt(0)
    candidate["score"] = candidate[spec["rank"]]
    candidate["tie_score"] = candidate["pooled_t" if spec["rank"] == "annual_min_t" else "annual_min_t"]
    candidate["eligible"] = candidate.complete_60_months & np.isfinite(candidate.score) & np.isfinite(candidate.tie_score)
    if availability == "published_before_year":
        candidate["eligible"] &= candidate.publication_eligible
    elif availability != "unrestricted":
        raise ValueError(availability)
    if spec["universe"] == "no_options":
        candidate["eligible"] &= ~candidate.option_derived
    candidate["ineligibility_reason"] = ""
    candidate.loc[~candidate.complete_60_months, "ineligibility_reason"] = "incomplete_prior_60_months"
    candidate.loc[candidate.complete_60_months & (~np.isfinite(candidate.score) | ~np.isfinite(candidate.tie_score)), "ineligibility_reason"] = "nonfinite_score"
    if availability == "published_before_year":
        candidate.loc[~candidate.publication_eligible, "ineligibility_reason"] += ";not_published_before_target_year"
    if spec["universe"] == "no_options":
        candidate.loc[candidate.option_derived, "ineligibility_reason"] += ";excluded_options_category"
    candidate["ineligibility_reason"] = candidate.ineligibility_reason.str.lstrip(";")
    candidate["target_year"] = target_year
    candidate["method"] = method
    candidate["availability"] = availability
    candidate["window_start"] = str(index.min())
    candidate["window_end"] = str(index.max())
    return candidate.reset_index()


def admit_candidates(candidate, minimum=10):
    """Retain all strict eligible factors, then top up; deterministic ties."""
    c = candidate.copy()
    c["rank"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    c["selection_rank"] = pd.Series(pd.NA, index=c.index, dtype="Int64")
    c["selected"] = False
    c["admission_reason"] = np.where(c.eligible, "eligible_not_selected", "ineligible")
    ranked = c[c.eligible].sort_values(["score", "tie_score", "factor"], ascending=[False, False, True], kind="stable")
    c.loc[ranked.index, "rank"] = np.arange(1, len(ranked) + 1)
    strict_index = ranked.index[ranked.strict_pass]
    extra_needed = max(0, minimum - len(strict_index))
    topup_index = ranked.index[~ranked.strict_pass][:extra_needed]
    c.loc[strict_index, "selected"] = True
    c.loc[topup_index, "selected"] = True
    c.loc[strict_index, "admission_reason"] = "strict_five_annual_tests"
    c.loc[topup_index, "admission_reason"] = "minimum_pool_topup"
    selected = c[c.selected].sort_values("rank")
    c.loc[selected.index, "selection_rank"] = np.arange(1, len(selected) + 1)
    assert c.loc[c.eligible & c.strict_pass, "selected"].all()
    assert c.selected.sum() == max(len(strict_index), min(minimum, len(ranked)))
    return c


def audit_holding_returns(returns, names, year):
    """Diagnostic only. Never feeds back into eligibility or admission."""
    if year > returns.index.max().year:
        return dict(oos_status="not_yet_in_source", oos_observed_months=0,
                    missing_oos_factor_observations=None, missing_oos_pool_months=None)
    idx = pd.period_range(f"{year}-01", f"{year}-12", freq="M")
    held = returns.reindex(index=idx, columns=names)
    return dict(oos_status="observed", oos_observed_months=int(idx.isin(returns.index).sum()),
                missing_oos_factor_observations=int(held.isna().sum().sum()),
                missing_oos_pool_months=int(held.isna().any(axis=1).sum()))


def write_csv_atomic(frame, filename):
    target = OUT / filename
    temporary = target.with_name(f".{target.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum", type=int, default=10)
    args = parser.parse_args()
    if args.minimum < 1:
        raise ValueError("minimum must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    return_path = LIBRARY / "Portfolios/Full Sets OP/PredictorLSretWide.parquet"
    doc_path = LIBRARY / "SignalDoc.csv"
    returns = duckdb.read_parquet(str(return_path)).df().set_index("date")
    returns.index = returns.index.to_period("M")
    docs = pd.read_csv(doc_path)
    docs = docs[docs["Cat.Signal"].eq("Predictor")].set_index("Acronym")
    excluded_options = sorted(docs.index[docs["Cat.Data"].str.strip().str.casefold().eq("options")].tolist())
    assert set(returns.columns) == set(docs.index)
    assert returns.index.is_unique and returns.index.is_monotonic_increasing
    assert returns.index.equals(pd.period_range(returns.index.min(), returns.index.max(), freq="M"))
    assert not np.isinf(returns.to_numpy()).any()
    annual_cache = {year: complete_window_stats(returns.reindex(pd.period_range(f"{year}-01", f"{year}-12", freq="M")), 12, 1)
                    for year in range(1985, 2026)}
    candidates, counts = [], []
    for year in range(1990, 2027):
        for method in METHODS:
            for availability in ["published_before_year", "unrestricted"]:
                c = admit_candidates(score_candidates(returns, docs, year, method, availability, annual_cache), args.minimum)
                selected = c[c.selected].sort_values("selection_rank")
                names = selected.factor.tolist()
                candidates.append(c)
                counts.append(dict(target_year=year, method=method, availability=availability,
                    universe=METHODS[method]["universe"], universe_definition=UNIVERSE_DEFINITIONS[METHODS[method]["universe"]],
                    factor_count=len(names), strict_count=int((c.eligible & c.strict_pass).sum()),
                    topup_count=int(selected.admission_reason.eq("minimum_pool_topup").sum()),
                    eligible_count=int(c.eligible.sum()), minimum_required=args.minimum,
                    minimum_satisfied=len(names) >= args.minimum,
                    strict_inclusion_verified=bool(c.loc[c.eligible & c.strict_pass, "selected"].all()),
                    window_start=f"{year-5}-01", window_end=f"{year-1}-12",
                    **audit_holding_returns(returns, names, year)))
    candidate_table = pd.concat(candidates, ignore_index=True)
    count_table = pd.DataFrame(counts)
    member_table = candidate_table[candidate_table.selected].copy()
    key = ["target_year", "method", "availability"]
    member_table = member_table.sort_values(key + ["selection_rank"])
    assert not member_table.duplicated(key + ["factor"]).any()
    assert count_table.strict_inclusion_verified.all()
    assert member_table[member_table.availability.eq("published_before_year")].publication_year.lt(member_table[member_table.availability.eq("published_before_year")].target_year).all()
    assert not member_table.loc[member_table.universe.eq("no_options"), "option_derived"].any()
    existing_all_universe_unchanged = None
    if (OUT / "selection_members.csv").exists():
        old_output = pd.read_csv(OUT / "selection_members.csv")
        old_output = old_output[~old_output.method.str.endswith("_no_options")]
        old_keys = old_output.sort_values(key + ["selection_rank"])[key + ["factor", "rank", "selection_rank", "admission_reason"]].reset_index(drop=True)
        new_keys = member_table[member_table.universe.eq("all")].sort_values(key + ["selection_rank"])[key + ["factor", "rank", "selection_rank", "admission_reason"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(old_keys, new_keys, check_dtype=False)
        existing_all_universe_unchanged = True
    write_csv_atomic(candidate_table, "selection_candidates.csv")
    write_csv_atomic(member_table, "selection_members.csv")
    write_csv_atomic(count_table, "selection_counts.csv")
    missing_rows = []
    for (year, method, availability), selected in member_table.groupby(key):
        if year > returns.index.max().year:
            continue
        for factor in selected.factor:
            holding = returns.reindex(index=pd.period_range(f"{year}-01", f"{year}-12", freq="M"), columns=[factor])[factor]
            missing_rows.extend(dict(target_year=int(year), method=method, availability=availability,
                                     factor=factor, month=str(month)) for month in holding.index[holding.isna()])
    write_csv_atomic(pd.DataFrame(missing_rows, columns=key + ["factor", "month"]), "selection_missing_oos.csv")
    current = member_table[member_table.target_year.eq(2026)]
    write_csv_atomic(current, "selection_pool_2026.csv")
    current_json = {f"{method}/{availability}": json.loads(g.to_json(orient="records"))
                    for (method, availability), g in current.groupby(["method", "availability"])}
    (OUT / "selection_pool_2026.json").write_text(json.dumps(current_json, ensure_ascii=False, indent=2))

    # Independent baseline agreement: cached prior audit is read only, never
    # used for ranking; retained strict names must exactly match its members.
    old_path = PROJECT / "rolling_sharpe5/audit/extended_selection_members.csv"
    old = pd.read_csv(old_path)
    for row in count_table.itertuples():
        annual_method = "each_year_iid" if METHODS[row.method]["annual"] == "iid" else "each_year_hac1"
        prior_names = set(old[old.target_year.eq(row.target_year) & old.method.eq(annual_method) & old.availability.eq(row.availability) & old.test.eq("two_sided")].factor)
        if METHODS[row.method]["universe"] == "no_options":
            prior_names -= set(excluded_options)
        retained = member_table[member_table.target_year.eq(row.target_year) & member_table.method.eq(row.method) & member_table.availability.eq(row.availability) & member_table.strict_pass]
        assert prior_names == set(retained.factor), (row.target_year, row.method, row.availability)

    # Metamorphic check: replacing all current/future returns leaves rankings
    # and selections unchanged; altering a prior observation can matter.
    rng = np.random.default_rng(38071)
    invariance_checks = []
    for year in [1990, 1998, 2023, 2026]:
        altered = returns.copy()
        future = altered.index.year >= year
        altered.loc[future] = rng.normal(100, 100, (int(future.sum()), len(altered.columns)))
        for method in METHODS:
            a = admit_candidates(score_candidates(returns, docs, year, method, "published_before_year"), args.minimum)
            b = admit_candidates(score_candidates(altered, docs, year, method, "published_before_year"), args.minimum)
            pd.testing.assert_frame_equal(a, b)
            invariance_checks.append(dict(target_year=year, method=method, passed=True))
    source_paths = [return_path, doc_path, Path(__file__), old_path]
    report = {
        "primary": "min10_annual_iid/published_before_year",
        "minimum": args.minimum,
        "selection_dates": "1990-2026, each January; use only previous five complete calendar years",
        "definitions": METHODS,
        "universe_definitions": UNIVERSE_DEFINITIONS,
        "no_options_exclusion_field": "SignalDoc.csv Cat.Data == Options (case-insensitive exact category match)",
        "no_options_excluded_count": len(excluded_options),
        "no_options_excluded_factors": excluded_options,
        "existing_all_universe_selection_unchanged": existing_all_universe_unchanged,
        "strict_pass_rule": "positive mean and two-sided p<0.05 in every prior year; keep all eligible strict passers even if count exceeds minimum",
        "rank_rule": "rank descending signed score, descending tie_score, then acronym ascending; never use absolute t or flip factor sign",
        "eligibility": "all60prior monthly returns valid plus finite ranking statistics; published rule also requires publication_year<target_year",
        "costs": "ranking uses original-paper gross factor returns; no local holding/turnover data for factor-library costs",
        "candidate_rows": len(candidate_table), "member_rows": len(member_table), "configuration_years": len(count_table),
        "minimum_satisfied_all": bool(count_table.minimum_satisfied.all()),
        "minimum_eligible_count": int(count_table.eligible_count.min()),
        "below_minimum_eligible_configuration_years": int(count_table.eligible_count.lt(args.minimum).sum()),
        "strict_inclusion_matches_old_audit": True,
        "no_options_complete_oos_1990_2025": bool(count_table[count_table.universe.eq("no_options") & count_table.target_year.le(2025)].missing_oos_pool_months.eq(0).all()),
        "future_invariance_checks": invariance_checks,
        "missing_holding_return_rows": json.loads(count_table[count_table.missing_oos_pool_months.gt(0)].to_json(orient="records")),
        "sources": {str(path): {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in source_paths},
    }
    (OUT / "selection_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(count_table[count_table.target_year.eq(2026)].to_string(index=False))
    print("All configuration-years satisfy minimum:", report["minimum_satisfied_all"], "min eligible:", report["minimum_eligible_count"])
    print("Missing OOS rows:", len(report["missing_holding_return_rows"]))
    print(current[current.method.eq("min10_annual_iid") & current.availability.eq("published_before_year")][["factor", "rank", "annual_min_t", "pooled_t", "strict_pass", "admission_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
