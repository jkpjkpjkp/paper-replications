#!/usr/bin/env python3
"""Audit the complete Song et al. US download without filling source gaps.

Every completed daily file is hashed and reconciled to its WRDS receipt. Missing
trades and missing price-eligible prints are recorded separately from incomplete
transfers. Use --partial for a progress report while the downloader is running.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import duckdb
import exchange_calendars as xcals
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
FIRST = dt.date(2019, 1, 2)
LAST = dt.date(2025, 8, 1)
EXPECTED_SESSIONS = 1655
META_NAMES = ["sp500_membership", "sp500_security_names", "taq_trade_partitions",
              "taq_master_2025"] + [f"taq_crsp_link_{y}" for y in range(2019, 2026)]
BAR_COLUMNS = {
    "date", "permno", "sym_root", "sym_suffix", "taq_symbol_count",
    "minute_index", "minute_start", "open", "high", "low", "close",
    "amount", "volume", "trade_count", "raw_open", "raw_high", "raw_low",
    "raw_close", "eligible_amount", "eligible_volume", "eligible_trade_count",
    "first_sip_time_us", "last_sip_time_us",
}
GAP_SCHEMA = pa.schema([
    ("session_date", pa.date32()), ("permno", pa.int32()),
    ("membership_active", pa.bool_()), ("mapped", pa.bool_()),
    ("expected_minutes", pa.int32()), ("observed_minutes", pa.int32()),
    ("eligible_price_minutes", pa.int32()), ("absent_trade_minutes", pa.int32()),
    ("missing_eligible_price_minutes", pa.int32()),
])


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(db, statement, parameters=None):
    cursor = db.execute(statement, parameters)
    keys = [column[0] for column in cursor.description]
    return [dict(zip(keys, row)) for row in cursor.fetchall()]


def scalar(db, statement, parameters=None):
    return db.execute(statement, parameters).fetchone()[0]


def columns(db, view):
    return {row[0] for row in db.execute("DESCRIBE " + view).fetchall()}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")
    temporary.replace(path)


def equal_number(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, float) or isinstance(expected, float):
        return math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=0.01)
    return actual == expected


def audit(root: Path) -> dict:
    report = {
        "complete": False,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": {"first_session": FIRST.isoformat(), "last_session": LAST.isoformat(),
                  "expected_sessions": EXPECTED_SESSIONS,
                  "session_timezone": "America/New_York",
                  "last_session_purpose": "next-session labels for 2025-07-31",
                  "completeness_definition": "all planned sessions transferred and verified; source gaps retained",
                  "amount_comparison_tolerance": {"relative": 1e-10, "absolute_dollars": 0.01}},
        "errors": [], "pending": [], "metadata": [], "source_hashes": [],
        "days": [],
    }
    errors = report["errors"]

    def error(check, path=None, **details):
        errors.append({"check": check, **({"path": str(path)} if path else {}), **details})

    def fingerprint(path):
        name = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        record = {"path": name, "bytes": path.stat().st_size,
                  "sha256": digest(path)}
        report["source_hashes"].append(record)
        return record

    def check_receipt(path, receipt_path, expected_rows_key="rows"):
        receipt = json.loads(receipt_path.read_text())
        file_info = {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                     "sha256": digest(path), "rows": pq.ParquetFile(path).metadata.num_rows}
        for name in ("bytes", "sha256"):
            if receipt.get(name) != file_info[name]:
                error("receipt_" + name, path, actual=file_info[name], expected=receipt.get(name))
        if expected_rows_key == "rows":
            expected_rows = receipt.get("rows")
        else:
            expected_rows = receipt.get("totals", {}).get("rows")
        if file_info["rows"] != expected_rows:
            error("receipt_rows", path, actual=file_info["rows"], expected=expected_rows)
        return receipt, file_info

    for name in META_NAMES:
        path = root / "metadata" / (name + ".parquet")
        receipt_path = path.with_suffix(".receipt.json")
        if not path.is_file() or not receipt_path.is_file():
            report["pending"].append({"kind": "metadata", "name": name})
            continue
        try:
            _, info = check_receipt(path, receipt_path)
            report["metadata"].append(info)
            report["source_hashes"].append(info)
        except Exception as exc:
            error("metadata_read", path, message=str(exc))
    known_metadata = {name + ".parquet" for name in META_NAMES}
    for path in sorted((root / "metadata").glob("*.parquet")):
        if path.name not in known_metadata:
            try:
                _, info = check_receipt(path, path.with_suffix(".receipt.json"))
                report["metadata"].append(info)
                report["source_hashes"].append(info)
            except Exception as exc:
                error("additional_metadata_read", path, message=str(exc))

    required_support = ["trading_calendar", "sp500_daily_membership", "taq_daily_mappings"]
    for name in required_support:
        path = root / "support" / (name + ".parquet")
        if not path.is_file():
            report["pending"].append({"kind": "support", "name": name})
    for path in sorted((root / "support").glob("*.parquet")):
        try:
            fingerprint(path)
        except Exception as exc:
            error("support_read", path, message=str(exc))
    plan = None
    plan_path = root / "download-plan.json"
    worker_hash = None
    if plan_path.is_file():
        try:
            plan = json.loads(plan_path.read_text())
            fingerprint(plan_path)
            worker_source = plan.get("worker_source", plan.get("worker_source_path", "taq_worker.py"))
            worker_path = Path(worker_source)
            if not worker_path.is_absolute():
                worker_path = root / worker_path
            worker_hash = fingerprint(worker_path)["sha256"]
            if plan.get("worker_sha256") != worker_hash:
                error("plan_worker_hash", plan_path, actual=plan.get("worker_sha256"), expected=worker_hash)
        except Exception as exc:
            error("download_plan_read", plan_path, message=str(exc))
    else:
        report["pending"].append({"kind": "download_plan", "path": str(plan_path)})
    support_audit = root / "support" / "validation.json"
    if support_audit.is_file():
        try:
            support_result = json.loads(support_audit.read_text())
            fingerprint(support_audit)
            if support_result.get("complete") is not True:
                error("support_validation_incomplete", support_audit)
        except Exception as exc:
            error("support_validation_read", support_audit, message=str(exc))
    if any(item["kind"] == "support" for item in report["pending"]):
        report["summary"] = {"expected_sessions": EXPECTED_SESSIONS, "verified_sessions": 0}
        return report

    gap_rows = []
    with duckdb.connect() as db:
        db.execute("SET threads=2")
        db.read_parquet(str(root / "support" / "trading_calendar.parquet")).create_view("calendar")
        db.read_parquet(str(root / "support" / "sp500_daily_membership.parquet")).create_view("membership")
        db.read_parquet(str(root / "support" / "taq_daily_mappings.parquet")).create_view("mapping_raw")
        map_columns = columns(db, "mapping_raw")
        date_column = next((c for c in ("session_date", "date") if c in map_columns), None)
        if not date_column or not {"permno", "sym_root", "sym_suffix"} <= map_columns:
            error("mapping_schema", actual=sorted(map_columns))
            return report
        db.execute(f"""CREATE TEMP VIEW mappings AS SELECT CAST({date_column} AS DATE) AS session_date,
            CAST(permno AS INTEGER) AS permno, trim(sym_root) AS sym_root,
            trim(coalesce(sym_suffix,'')) AS sym_suffix FROM mapping_raw""")
        for name, query in {
            "duplicate_calendar_dates": "SELECT count(*)-count(DISTINCT session_date) FROM calendar",
            "null_mapping_keys": "SELECT count(*) FROM mappings WHERE session_date IS NULL OR permno IS NULL OR sym_root IS NULL OR sym_root=''",
            "duplicate_mapping_symbols": "SELECT count(*) FROM (SELECT session_date,sym_root,sym_suffix FROM mappings GROUP BY ALL HAVING count(*)>1)",
            "duplicate_membership_keys": "SELECT count(*) FROM (SELECT session_date,permno FROM membership GROUP BY ALL HAVING count(*)>1)",
            "mapping_outside_calendar": "SELECT count(*) FROM mappings m ANTI JOIN calendar c USING(session_date)",
            "mapping_outside_universe": "SELECT count(*) FROM mappings m ANTI JOIN membership u USING(session_date,permno)",
        }.items():
            count = scalar(db, query)
            if count:
                error(name, actual=count)
        # Fetch local clock strings instead of TIMESTAMPTZ Python objects;
        # DuckDB's optional pytz conversion is unnecessary for this audit.
        sessions = rows(db, """SELECT session_date,expected_minutes,early_close,
            market_open_new_york,market_close_new_york,session_split
            FROM calendar ORDER BY session_date""")
        calendar = xcals.get_calendar("XNYS", start=FIRST.isoformat(), end=LAST.isoformat())
        official_dates = list(calendar.sessions.date)
        actual_dates = [day["session_date"] for day in sessions]
        if actual_dates != official_dates or len(actual_dates) != EXPECTED_SESSIONS:
            error("calendar_sessions", actual_count=len(actual_dates), official_count=len(official_dates),
                  extra_dates=sorted(set(actual_dates)-set(official_dates)),
                  missing_dates=sorted(set(official_dates)-set(actual_dates)))
        if plan is not None and (plan.get("sessions") != EXPECTED_SESSIONS
                or sorted(plan.get("dates", [])) != [str(date) for date in actual_dates]):
            error("download_plan_sessions", plan_path, actual_count=plan.get("sessions"))
        official_schedule = calendar.schedule
        open_col = "open" if "open" in official_schedule else "market_open"
        close_col = "close" if "close" in official_schedule else "market_close"
        for day in sessions:
            date = day["session_date"]
            official = official_schedule.loc[str(date)]
            expected_minutes = int((official[close_col]-official[open_col]).total_seconds()/60)
            if day["expected_minutes"] != expected_minutes:
                error("calendar_session_minutes", date=str(date), actual=day["expected_minutes"], expected=expected_minutes)

        all_bar_files = set((root / "bars").glob("*/*.parquet"))
        expected_files = {root / "bars" / str(day["session_date"].year) /
                          (str(day["session_date"]) + ".parquet") for day in sessions}
        for extra in sorted(all_bar_files - expected_files):
            error("unplanned_bar_file", extra)
        for day in sessions:
            date = day["session_date"]
            path = root / "bars" / str(date.year) / (str(date) + ".parquet")
            receipt_path = path.with_suffix(".receipt.json")
            if not path.is_file() or not receipt_path.is_file():
                report["pending"].append({"kind": "bars", "date": str(date),
                                          "file_present": path.is_file(), "receipt_present": receipt_path.is_file()})
                continue
            before_errors = len(errors)
            result = {"date": str(date), "year": date.year, "split": day["session_split"],
                      "expected_minutes": day["expected_minutes"], "early_close": day["early_close"]}
            try:
                receipt, file_info = check_receipt(path, receipt_path, "totals.rows")
                result.update(file_info)
                if receipt.get("state") != "complete":
                    error("day_receipt_state", path, actual=receipt.get("state"))
                if receipt.get("date") != str(date):
                    error("day_receipt_date", path, actual=receipt.get("date"), expected=str(date))
                if worker_hash is not None and receipt.get("worker_sha256") != worker_hash:
                    error("day_worker_hash", path, actual=receipt.get("worker_sha256"), expected=worker_hash)
                if plan is not None:
                    for key in ("price_condition_sql", "volume_condition_sql"):
                        if receipt.get(key) != plan.get(key):
                            error("plan_" + key, path, actual=receipt.get(key), expected=plan.get(key))
                if receipt.get("close_time_exclusive") != day["market_close_new_york"]:
                    error("day_receipt_close", path, actual=receipt.get("close_time_exclusive"), expected=day["market_close_new_york"])
                db.read_parquet(str(path)).create_view("bars", replace=True)
                if not BAR_COLUMNS <= columns(db, "bars"):
                    raise ValueError("Missing bar columns: " + repr(sorted(BAR_COLUMNS-columns(db,"bars"))))
                metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
                expected_metadata = {
                    "timezone": "America/New_York", "session_open": "09:30:00",
                    "session_close_exclusive": day["market_close_new_york"],
                    "source": f"taqm_{date.year}.ctm_{date:%Y%m%d}",
                    "price_condition_sql": receipt.get("price_condition_sql"),
                    "volume_condition_sql": receipt.get("volume_condition_sql"),
                    "base_filter": "price>0 AND size>0 AND tr_corr IN (00,01)",
                }
                for key, expected in expected_metadata.items():
                    actual = metadata.get(key.encode(), b"").decode()
                    if actual != expected:
                        error("parquet_metadata_" + key, path, actual=actual, expected=expected)
                mapped = rows(db, "SELECT DISTINCT permno,sym_root,sym_suffix FROM mappings WHERE session_date=?", [date])
                mapped_permnos = {item["permno"] for item in mapped}
                if plan is not None:
                    canonical = rows(db, f"""SELECT permno,sym_root,sym_suffix FROM mapping_raw
                        WHERE {date_column}=? ORDER BY permno,sym_root,coalesce(sym_suffix,'')""", [date])
                    mapping_hash = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
                    if receipt.get("mappings_sha256") != mapping_hash:
                        error("mapping_hash", path, actual=receipt.get("mappings_sha256"), expected=mapping_hash)
                totals = rows(db, """SELECT count(*) AS rows, count(DISTINCT permno) AS securities,
                    sum(volume)::HUGEINT AS volume, sum(trade_count)::HUGEINT AS trades,
                    sum(amount)::DOUBLE AS amount,
                    sum(eligible_trade_count)::HUGEINT AS price_eligible_trades,
                    count(*) FILTER(WHERE close IS NULL) AS minutes_without_eligible_price,
                    count(*) FILTER(WHERE taq_symbol_count>1) AS minutes_with_multiple_symbols
                    FROM bars""")[0]
                result["totals"] = totals
                for key, actual in totals.items():
                    if not equal_number(actual, receipt.get("totals", {}).get(key)):
                        error("remote_total_" + key, path, actual=actual, expected=receipt.get("totals", {}).get(key))
                if receipt.get("copy_rows") != totals["trades"]:
                    error("copy_rows", path, actual=totals["trades"], expected=receipt.get("copy_rows"))
                for key, expected in (("mapped_securities", len(mapped_permnos)), ("mapped_symbols", len(mapped))):
                    if receipt.get(key) != expected:
                        error("receipt_" + key, path, actual=receipt.get(key), expected=expected)
                bounds = {"date": date, "minutes": day["expected_minutes"]}
                checks = rows(db, """SELECT
                    count(*)-count(DISTINCT struct_pack(p:=permno,m:=minute_index)) AS duplicate_keys,
                    count(*) FILTER(WHERE permno IS NULL OR minute_index IS NULL OR date IS NULL OR minute_start IS NULL) AS null_keys,
                    count(*) FILTER(WHERE date != $date OR minute_index<0 OR minute_index >= $minutes
                        OR minute_start != CAST($date AS TIMESTAMP)+(570+minute_index)*INTERVAL 1 MINUTE) AS time_or_date_errors,
                    count(*) FILTER(WHERE amount IS NULL OR NOT isfinite(amount) OR amount<=0
                        OR volume IS NULL OR volume<=0 OR trade_count IS NULL OR trade_count<=0
                        OR volume<trade_count OR taq_symbol_count IS NULL OR taq_symbol_count<1) AS invalid_broad_totals,
                    count(*) FILTER(WHERE eligible_trade_count IS NULL OR eligible_trade_count<0 OR eligible_trade_count>trade_count
                        OR (eligible_trade_count=0 AND (eligible_amount IS NOT NULL OR eligible_volume IS NOT NULL))
                        OR (eligible_trade_count>0 AND (eligible_amount IS NULL OR eligible_volume IS NULL
                            OR NOT isfinite(eligible_amount) OR eligible_amount<=0 OR eligible_volume<eligible_trade_count
                            OR eligible_volume>volume OR eligible_amount>amount*(1+1e-12)+0.000001))) AS invalid_eligible_totals,
                    count(*) FILTER(WHERE (eligible_trade_count=0 AND (open IS NOT NULL OR high IS NOT NULL OR low IS NOT NULL OR close IS NOT NULL))
                        OR (eligible_trade_count>0 AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                            OR NOT isfinite(open) OR NOT isfinite(high) OR NOT isfinite(low) OR NOT isfinite(close)
                            OR low<=0 OR low>open OR low>close OR high<open OR high<close OR low>high))) AS invalid_ohlc,
                    count(*) FILTER(WHERE raw_open IS NULL OR raw_high IS NULL OR raw_low IS NULL OR raw_close IS NULL
                        OR NOT isfinite(raw_open) OR NOT isfinite(raw_high) OR NOT isfinite(raw_low) OR NOT isfinite(raw_close)
                        OR raw_low<=0 OR raw_low>raw_open OR raw_low>raw_close OR raw_high<raw_open OR raw_high<raw_close
                        OR raw_low>raw_high OR (low IS NOT NULL AND (low<raw_low OR high>raw_high))) AS invalid_raw_ohlc,
                    count(*) FILTER(WHERE amount/volume<raw_low*(1-1e-10) OR amount/volume>raw_high*(1+1e-10)
                        OR (eligible_volume>0 AND (eligible_amount/eligible_volume<low*(1-1e-10)
                            OR eligible_amount/eligible_volume>high*(1+1e-10)))) AS amount_outside_price_bounds,
                    count(*) FILTER(WHERE first_sip_time_us IS NULL OR last_sip_time_us IS NULL
                        OR first_sip_time_us>last_sip_time_us OR first_sip_time_us<(570+minute_index)::BIGINT*60000000
                        OR last_sip_time_us>=(571+minute_index)::BIGINT*60000000) AS invalid_sip_time_bounds
                    FROM bars""", bounds)[0]
                checks["unplanned_permnos"] = scalar(db, """SELECT count(*) FROM bars b ANTI JOIN mappings m
                    ON m.session_date=? AND b.permno=m.permno""", [date])
                checks["unplanned_symbols"] = scalar(db, """SELECT count(*) FROM bars b ANTI JOIN mappings m
                    ON m.session_date=? AND b.permno=m.permno AND b.sym_root=m.sym_root
                    AND coalesce(b.sym_suffix,'')=m.sym_suffix""", [date])
                result["checks"] = checks
                for key, count in checks.items():
                    if count:
                        error(key, path, actual=count)
                observed = {record["permno"]: record for record in rows(db, """SELECT permno,
                    count(*)::INTEGER AS observed_minutes,
                    count(*) FILTER(WHERE close IS NOT NULL)::INTEGER AS eligible_price_minutes
                    FROM bars GROUP BY permno""")}
                missing = sorted(mapped_permnos-set(observed))
                if sorted(receipt.get("missing_permnos", [])) != missing:
                    error("missing_permnos_receipt", path, actual=missing, expected=receipt.get("missing_permnos"))
                universe = rows(db, "SELECT permno,membership_active FROM membership WHERE session_date=?", [date])
                result["source_gaps"] = {"mapped_securities": len(mapped_permnos),
                    "universe_securities": len(universe), "active_securities": sum(bool(u["membership_active"]) for u in universe),
                    "unmapped_permnos": sorted({u["permno"] for u in universe}-mapped_permnos),
                    "missing_mapped_permnos": missing,
                    "absent_trade_minutes": len(mapped_permnos)*day["expected_minutes"]-totals["rows"],
                    "minutes_without_eligible_price": totals["minutes_without_eligible_price"]}
                for member in universe:
                    count = observed.get(member["permno"], {})
                    got = count.get("observed_minutes", 0)
                    eligible = count.get("eligible_price_minutes", 0)
                    if got < day["expected_minutes"] or eligible < day["expected_minutes"]:
                        gap_rows.append({"session_date": date, "permno": member["permno"],
                            "membership_active": member["membership_active"], "mapped": member["permno"] in mapped_permnos,
                            "expected_minutes": day["expected_minutes"], "observed_minutes": got,
                            "eligible_price_minutes": eligible, "absent_trade_minutes": day["expected_minutes"]-got,
                            "missing_eligible_price_minutes": day["expected_minutes"]-eligible})
            except Exception as exc:
                error("day_read_or_query", path, message=str(exc))
            result["valid"] = len(errors) == before_errors
            report["days"].append(result)

    gap_path = root / "validation_gaps.parquet"
    temporary_gap = gap_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(gap_rows, schema=GAP_SCHEMA), temporary_gap, compression="zstd")
    temporary_gap.replace(gap_path)
    group_totals = {}
    for key in ("year", "split"):
        group = defaultdict(lambda: {"sessions": 0, "rows": 0, "securities_days": 0,
                                     "volume": 0, "trades": 0, "amount": 0.0,
                                     "price_eligible_trades": 0, "absent_trade_minutes": 0,
                                     "minutes_without_eligible_price": 0})
        for day in report["days"]:
            if not day.get("valid"):
                continue
            record = group[str(day[key])]
            record["sessions"] += 1
            for name in ("rows", "volume", "trades", "amount", "price_eligible_trades", "minutes_without_eligible_price"):
                record[name] += day["totals"][name] or 0
            record["securities_days"] += day["totals"]["securities"]
            record["absent_trade_minutes"] += day["source_gaps"]["absent_trade_minutes"]
        group_totals["by_" + key] = dict(sorted(group.items()))
    total = {key: sum(value[key] for value in group_totals["by_year"].values())
             for key in next(iter(group_totals["by_year"].values()), {})}
    report["summary"] = {"expected_sessions": EXPECTED_SESSIONS,
        "present_receipted_sessions": len(report["days"]),
        "verified_sessions": sum(day.get("valid", False) for day in report["days"]),
        "pending_sessions": sum(item["kind"] == "bars" for item in report["pending"]),
        "integrity_error_count": len(errors), "totals": total, **group_totals}
    report["source_gaps"] = {"policy": "Preserved absent rows and null eligible prices; never filled with zero or synthetic prices",
        "security_days_with_gaps": len(gap_rows),
        "unmapped_security_days": sum(not item["mapped"] for item in gap_rows),
        "mapped_security_days_without_trades": sum(item["mapped"] and item["observed_minutes"] == 0 for item in gap_rows),
        "active_security_days_with_gaps": sum(bool(item["membership_active"]) for item in gap_rows),
        "gap_file": {"path": gap_path.name, "rows": len(gap_rows),
                     "bytes": gap_path.stat().st_size, "sha256": digest(gap_path)}}
    report["complete"] = (not errors and not report["pending"]
                          and len(report["days"]) == EXPECTED_SESSIONS
                          and report["summary"]["verified_sessions"] == EXPECTED_SESSIONS)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--partial", action="store_true",
                        help="Write an honest incomplete progress report but return success")
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root)
    save_json(root / "validation.json", report)
    print(json.dumps({"complete": report["complete"], "summary": report.get("summary"),
                      "errors": report["errors"][:10], "pending_count": len(report["pending"]),
                      "report": str(root / "validation.json")}, default=str), flush=True)
    return 0 if report["complete"] or args.partial else 1


if __name__ == "__main__":
    sys.exit(main())
