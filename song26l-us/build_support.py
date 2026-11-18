#!/usr/bin/env python3
"""Build dated S&P 500 membership, CRSP auxiliaries and an XNYS session calendar.

This is an offline step. It preserves raw CRSP values and never fills prices,
returns, adjustment factors or missing security/day records. The calendar uses
observed CRSP market dates, checked against exchange_calendars' XNYS schedule.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path

import duckdb
import exchange_calendars as xcals
import pandas as pd


ROOT = Path(__file__).resolve().parent
FIRST_DATE = dt.date(2019, 1, 1)
LAST_DATE = dt.date(2025, 8, 1)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def scalar(db: duckdb.DuckDBPyConnection, query: str):
    return db.execute(query).fetchone()[0]


def records(db: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    result = db.execute(query)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def session_frame() -> pd.DataFrame:
    # Extra sessions establish the previous/next session at both output bounds.
    calendar = xcals.get_calendar("XNYS", start="2018-12-20", end="2025-08-06")
    schedule = calendar.schedule.copy()
    open_column = "open" if "open" in schedule else "market_open"
    close_column = "close" if "close" in schedule else "market_close"
    frame = pd.DataFrame({"session_date": schedule.index.date})
    for target, source in (("market_open_utc", open_column),
                           ("market_close_utc", close_column)):
        frame[target] = pd.to_datetime(schedule[source].to_numpy(), utc=True)
    frame["previous_session"] = frame["session_date"].shift(1)
    frame["next_session"] = frame["session_date"].shift(-1)
    frame["expected_minutes"] = (
        (frame["market_close_utc"] - frame["market_open_utc"])
        .dt.total_seconds().div(60).astype("int32")
    )
    frame["early_close"] = frame["expected_minutes"] < 390
    frame["market_open_new_york"] = (
        frame["market_open_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M:%S")
    )
    frame["market_close_new_york"] = (
        frame["market_close_utc"].dt.tz_convert("America/New_York").dt.strftime("%H:%M:%S")
    )
    frame["session_split"] = frame["session_date"].map(
        lambda date: "train" if date < dt.date(2023, 7, 1)
        else "validation" if date < dt.date(2024, 7, 1)
        else "test" if date <= dt.date(2025, 7, 31)
        else "label_extension"
    )
    return frame.loc[frame["session_date"].between(FIRST_DATE, LAST_DATE)].copy()


def verify_daily_sources(db: duckdb.DuckDBPyConnection, paths: list[Path]) -> list[dict]:
    sources = []
    for path in paths:
        receipt_path = path.with_suffix(".done.json")
        if not receipt_path.is_file():
            raise ValueError(f"Missing source receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        digest = sha256(path)
        summary = records(db, f"""
            SELECT count(*) AS rows, min(dlycaldt) AS first_date,
                   max(dlycaldt) AS last_date
            FROM read_parquet({sql_string(path)})
        """)[0]
        if (receipt.get("state") != "complete"
                or receipt.get("sha256") != digest
                or receipt.get("bytes") != path.stat().st_size
                or receipt.get("rows") != summary["rows"]):
            raise ValueError(f"Source receipt/hash/size/row check failed: {path}")
        summary.update(path=str(path), receipt_path=str(receipt_path),
                       bytes=path.stat().st_size, sha256=digest)
        sources.append(summary)
    return sources


def copy_parquet(db: duckdb.DuckDBPyConnection, query: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        db.execute(f"COPY ({query}) TO {sql_string(temporary)} "
                   "(FORMAT PARQUET, COMPRESSION ZSTD)")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict:
    if not args.membership.is_file():
        raise ValueError(f"Download metadata first; missing membership file: {args.membership}")
    paths = [args.daily_root / f"{year}.parquet" for year in range(2019, 2026)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError("Missing daily source files: " + ", ".join(missing))
    with duckdb.connect() as db:
        sources = verify_daily_sources(db, paths)
        db.read_parquet([str(path) for path in paths]).create_view("crsp_raw")
        db.read_parquet(str(args.membership)).create_view("membership_raw")
        columns = {row[0] for row in db.execute("DESCRIBE membership_raw").fetchall()}
        required = {"permno", "mbrstartdt", "mbrenddt"}
        if not required <= columns:
            raise ValueError(f"Membership columns missing: {sorted(required - columns)}")
        invalid_intervals = scalar(db, """
            SELECT count(*) FROM membership_raw
            WHERE permno IS NULL OR mbrstartdt IS NULL OR mbrenddt < mbrstartdt
        """)
        if invalid_intervals:
            raise ValueError(f"Membership has {invalid_intervals} invalid intervals/identifiers")
        db.execute("""
            CREATE TEMP TABLE membership AS
            SELECT DISTINCT CAST(permno AS INTEGER) AS permno,
                   CAST(mbrstartdt AS DATE) AS start_date,
                   COALESCE(CAST(mbrenddt AS DATE), DATE '9999-12-31') AS end_date
            FROM membership_raw
            WHERE mbrstartdt <= DATE '2025-08-01'
              AND COALESCE(mbrenddt, DATE '9999-12-31') >= DATE '2018-12-31'
        """)
        db.register("xnys_schedule", session_frame())
        db.execute("""
            CREATE TEMP TABLE market_dates AS
            SELECT DISTINCT dlycaldt AS session_date FROM crsp_raw
            WHERE dlycaldt BETWEEN DATE '2019-01-01' AND DATE '2025-08-01'
        """)
        calendar_differences = records(db, """
            SELECT COALESCE(m.session_date, x.session_date) AS session_date,
                   CASE WHEN m.session_date IS NULL THEN 'missing_from_CRSP'
                        ELSE 'missing_from_XNYS' END AS difference
            FROM market_dates m FULL JOIN xnys_schedule x USING (session_date)
            WHERE m.session_date IS NULL OR x.session_date IS NULL
            ORDER BY session_date
        """)
        if calendar_differences:
            raise ValueError("CRSP dates disagree with XNYS schedule: "
                             + json.dumps(calendar_differences, default=str))
        db.execute("""
            CREATE TEMP TABLE trading_calendar AS
            SELECT x.* FROM market_dates m JOIN xnys_schedule x USING (session_date)
        """)
        # Membership dates are inclusive, as represented by CRSP's interval data.
        # Prior-session members retain the next day's raw records after removal.
        db.execute("""
            CREATE TEMP TABLE universe AS
            SELECT c.session_date, m.permno,
                   bool_or(c.session_date BETWEEN m.start_date AND m.end_date)
                       AS membership_active,
                   bool_or(c.previous_session BETWEEN m.start_date AND m.end_date)
                       AS membership_previous_session
            FROM trading_calendar c JOIN membership m
              ON c.session_date BETWEEN m.start_date AND m.end_date
              OR c.previous_session BETWEEN m.start_date AND m.end_date
            GROUP BY c.session_date, m.permno
        """)
        db.execute("""
            CREATE TEMP TABLE auxiliary AS
            SELECT r.*, u.membership_active, u.membership_previous_session,
                   c.previous_session, c.next_session, c.session_split,
                   c.expected_minutes, c.early_close
            FROM crsp_raw r JOIN universe u
              ON r.permno = u.permno AND r.dlycaldt = u.session_date
            JOIN trading_calendar c USING (session_date)
        """)
        duplicate_keys = scalar(db, """
            SELECT count(*) FROM (
                SELECT permno, dlycaldt FROM auxiliary
                GROUP BY permno, dlycaldt HAVING count(*) > 1
            )
        """)
        null_keys = scalar(db, """
            SELECT count(*) FROM auxiliary WHERE permno IS NULL OR dlycaldt IS NULL
        """)
        if duplicate_keys or null_keys:
            raise ValueError(f"Auxiliary key check failed: {duplicate_keys=}, {null_keys=}")
        missing_active = records(db, """
            SELECT u.session_date, u.permno
            FROM universe u LEFT JOIN auxiliary a
              ON u.permno = a.permno AND u.session_date = a.dlycaldt
            WHERE u.membership_active AND a.permno IS NULL
            ORDER BY 1,2
        """)
        summary = {
            "complete": True,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": {"first_date": "2019-01-01", "last_date": "2025-08-01",
                      "membership_endpoints": "inclusive",
                      "auxiliary_selection": "active member OR prior trading session member",
                      "missing_values": "preserved; no synthetic prices, returns or rows",
                      "splits": {"train": "2019-01-01 <= date < 2023-07-01",
                                 "validation": "2023-07-01 <= date < 2024-07-01",
                                 "test": "2024-07-01 <= date <= 2025-07-31",
                                 "label_extension": "2025-08-01"}},
            "source_files": sources,
            "membership_source": {"path": str(args.membership),
                                  "sha256": sha256(args.membership),
                                  "raw_rows": scalar(db, "SELECT count(*) FROM membership_raw"),
                                  "relevant_distinct_intervals": scalar(db, "SELECT count(*) FROM membership")},
            "calendar": {"schedule": "XNYS", "timezone": "America/New_York",
                         "exchange_calendars_version": importlib.metadata.version("exchange_calendars"),
                         "observed_CRSP_dates_equal_XNYS_sessions": True,
                         "session_counts": records(db, """
                             SELECT session_split, count(*) AS sessions,
                                    count(*) FILTER (WHERE early_close) AS early_closes
                             FROM trading_calendar GROUP BY session_split ORDER BY min(session_date)
                         """)},
            "universe": {"rows": scalar(db, "SELECT count(*) FROM universe"),
                         "securities": scalar(db, "SELECT count(DISTINCT permno) FROM universe"),
                         "active_rows": scalar(db, "SELECT count(*) FROM universe WHERE membership_active"),
                         "previous_only_rows": scalar(db, "SELECT count(*) FROM universe WHERE NOT membership_active"),
                         "active_rows_without_CRSP_record": len(missing_active),
                         "missing_active_records": missing_active},
            "auxiliary": {"rows": scalar(db, "SELECT count(*) FROM auxiliary"),
                          "columns": len(db.execute("DESCRIBE auxiliary").fetchall()),
                          "duplicate_keys": duplicate_keys, "null_keys": null_keys,
                          "adjustment_factor_quality": records(db, """
                              SELECT count(*) FILTER (WHERE dlycumfacpr IS NULL) AS null_price_factor,
                                     count(*) FILTER (WHERE dlycumfacpr <= 0) AS nonpositive_price_factor,
                                     count(*) FILTER (WHERE dlycumfacshr IS NULL) AS null_share_factor,
                                     count(*) FILTER (WHERE dlycumfacshr <= 0) AS nonpositive_share_factor
                              FROM auxiliary
                          """)[0]},
            "label_boundary_note": "Session splits annotate input dates. Purge training/validation examples whose future labels cross a split boundary when constructing model tensors.",
        }
        args.out.mkdir(parents=True, exist_ok=True)
        outputs = {
            "trading_calendar.parquet": "SELECT * FROM trading_calendar ORDER BY session_date",
            "sp500_daily_membership.parquet": """
                SELECT u.*, c.session_split, a.permno IS NOT NULL AS crsp_record_available
                FROM universe u JOIN trading_calendar c USING (session_date)
                LEFT JOIN auxiliary a ON u.permno = a.permno AND u.session_date = a.dlycaldt
                ORDER BY u.session_date, u.permno
            """,
            "crsp_daily_auxiliary.parquet": "SELECT * FROM auxiliary ORDER BY dlycaldt, permno",
        }
        summary["output_files"] = []
        for name, query in outputs.items():
            destination = args.out / name
            copy_parquet(db, query, destination)
            summary["output_files"].append({"path": str(destination),
                                            "bytes": destination.stat().st_size,
                                            "sha256": sha256(destination)})
        temporary = args.out / "validation.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        temporary.replace(args.out / "validation.json")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-root", type=Path,
                        default=ROOT.parent / "crsp-daily/data/crsp_a_stock/dsf_v2")
    parser.add_argument("--membership", type=Path,
                        default=ROOT / "metadata/sp500_membership.parquet")
    parser.add_argument("--out", type=Path, default=ROOT / "support")
    args = parser.parse_args()
    try:
        summary = build(args)
    except (ValueError, OSError, duckdb.Error) as exc:
        parser.exit(1, f"Support build failed: {exc}\n")
    print(json.dumps({"complete": summary["complete"],
                      "auxiliary_rows": summary["auxiliary"]["rows"],
                      "active_membership_rows": summary["universe"]["active_rows"],
                      "missing_active_CRSP_records": summary["universe"]["active_rows_without_CRSP_record"],
                      "validation": str(args.out / "validation.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
