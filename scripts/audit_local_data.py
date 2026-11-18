"""Read-only audit of local data, plus explicitly provisional equity cohorts.

No neural model training, performance claims, or alteration of source files.
Cohort assumptions: EQTY/COM securities, latest trading day of prior year
for capitalization, complete returns on every exchange calendar day.
These are NOT confirmed author choices. The paper's test-period completeness
filter uses future information; a training-only comparison quantifies its effect.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit(data_root, out):
    out.mkdir(parents=True, exist_ok=True)
    db = duckdb.connect(config={"threads": 4, "memory_limit": "2GB"})
    annual = []
    calendar = []
    provenance = []
    for year in range(1994, 2022):
        path = data_root / f"crsp-daily/parquet/{year}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        provenance.append({"path": str(path), "bytes": path.stat().st_size,
                           "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()})
        db.read_parquet(str(path)).create_view("daily", replace=True)
        cal = db.execute("""SELECT count(DISTINCT dlycaldt) n_days,
            min(dlycaldt) first_date, max(dlycaldt) last_date,
            count(*) n_rows, count(DISTINCT permno) n_securities,
            count(*) FILTER (WHERE dlyret IS NULL) missing_returns FROM daily""").df()
        cal.insert(0, "year", year)
        calendar.append(cal)
        yearly = db.execute("""SELECT permno, count(*) n_rows,
            count(DISTINCT dlycaldt) n_dates,
            count(dlyret) n_valid_returns, max(dlycaldt) last_date,
            arg_max(dlycap, dlycaldt) latest_nonnull_cap,
            first(dlycap ORDER BY dlycaldt DESC) year_end_cap,
            bool_and(securitytype = 'EQTY' AND securitysubtype = 'COM') all_common
            FROM daily GROUP BY permno""").df()
        yearly.insert(0, "year", year)
        annual.append(yearly)
    calendar = pd.concat(calendar, ignore_index=True)
    annual = pd.concat(annual, ignore_index=True)
    calendar.to_csv(out / "crsp_calendar_audit.csv", index=False)
    annual.to_parquet(out / "crsp_security_year_audit.parquet", index=False)
    cohorts, summary = [], []
    for year in range(2002, 2022):
        train = annual[annual.year.between(year - 8, year - 1)]
        train_days = int(calendar.loc[calendar.year.between(year - 8, year - 1), "n_days"].sum())
        groups = train.groupby("permno").agg(
            train_dates=("n_dates", "sum"), train_valid=("n_valid_returns", "sum"),
            all_common=("all_common", "all"))
        eligible = groups[(groups.train_dates == train_days) &
                          (groups.train_valid == train_days) & groups.all_common]
        caps = annual[annual.year == year - 1].set_index("permno")["year_end_cap"]
        eligible = eligible.join(caps).dropna(subset=["year_end_cap"])
        eligible = eligible[eligible.year_end_cap > 0]
        test = annual[annual.year == year].set_index("permno")
        days = int(calendar.loc[calendar.year == year, "n_days"].iloc[0])
        full_test = test[(test.n_dates == days) & (test.n_valid_returns == days) & test.all_common]
        ex_post = eligible.loc[eligible.index.isin(full_test.index)]
        ranked_train = eligible.reset_index().sort_values(["year_end_cap", "permno"], ascending=[False, True]).head(500)
        ranked_paper = ex_post.reset_index().sort_values(["year_end_cap", "permno"], ascending=[False, True]).head(500)
        for label, frame in [("train_only", ranked_train), ("paper_test_complete", ranked_paper)]:
            frame = frame.copy()
            frame.insert(0, "test_year", year)
            frame.insert(1, "cohort", label)
            frame.insert(2, "rank", range(1, len(frame) + 1))
            cohorts.append(frame)
        summary.append(dict(test_year=year, train_days=train_days, test_days=days,
            eligible_train_only=len(eligible), eligible_test_complete=len(ex_post),
            selected_paper=len(ranked_paper),
            selected_train_only=len(ranked_train),
            train_top500_removed_by_test_filter=len(set(ranked_train.permno) - set(ranked_paper.permno))))
    pd.concat(cohorts, ignore_index=True).to_csv(out / "candidate_equity_cohorts.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "candidate_cohort_summary.csv", index=False)
    factors = data_root / "factors-daily/ff3_daily.parquet"
    db.read_parquet(str(factors)).create_view("ff3", replace=True)
    factor_audit = db.execute("""SELECT count(*) AS n_rows, count(DISTINCT date) unique_dates,
        min(date) first_date,max(date) last_date,
        count(*) FILTER (WHERE rf IS NULL) missing_rf FROM ff3
        WHERE date BETWEEN DATE '1994-01-01' AND DATE '2021-12-31'""").df().to_dict("records")[0]
    output = {
        "status": "data_audit_only_no_model_results",
        "paper_version": "2505.11243v2",
        "crsp_source": "CRSP CIZ dsf_v2, local 2026 download; paper vintage unspecified",
        "crsp_rows_1994_2021": int(calendar.n_rows.sum()),
        "calendar_days_1994_2021": int(calendar.n_days.sum()),
        "distinct_permnos_1994_2021": int(annual.permno.nunique()),
        "ff3": factor_audit,
        "all_years_have_500_candidate_stocks": all(x["selected_paper"] == 500 for x in summary),
        "candidate_assumptions": ["EQTY/COM at all observed dates", "Prior year final date capitalization", "Each calendar day must have a non-null return", "Ties broken by PERMNO"],
        "author_matched_universe": False,
        "exact_79_feature_panel_found": False,
        "corelogic_found_in_filename_search": False,
        "warning": "Cohorts are diagnostic only; no claim of exact paper preprocessing or historical point-in-time availability.",
        "raw_sources": provenance,
    }
    (out / "local_data_audit.json").write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(json.dumps({k:v for k,v in output.items() if k != "raw_sources"}, indent=2, default=str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "results/data_audit")
    args = parser.parse_args()
    audit(args.data_root, args.out)
