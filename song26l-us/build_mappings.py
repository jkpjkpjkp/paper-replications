#!/usr/bin/env python3
"""Resolve dated S&P 500 PERMNOs to TAQ symbols using local metadata only.

Prefer the smallest available official WRDS match level for each security/day.
For missing official links, require an exact eight-character dated CRSP CUSIP
match to the first eight characters of the same day's TAQ master CUSIP. Static
header CUSIPs and ticker-name guesses are never used. Conflicting symbol/PERMNO
assignments are excluded and retained in the rejection audit.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parent


def quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def rows(db, statement: str) -> list[dict]:
    result = db.execute(statement)
    names = [column[0] for column in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def count(db, statement: str) -> int:
    return db.execute(statement).fetchall()[0][0]


def file_record(db, path: Path) -> dict:
    record = {"path": str(path), "bytes": path.stat().st_size,
              "sha256": digest(path),
              "rows": count(db, f"SELECT count(*) FROM read_parquet({quote(path)})")}
    receipt_path = path.with_suffix(".receipt.json")
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(record[key] != receipt[key] for key in ("bytes", "sha256", "rows")):
            raise ValueError(f"Source receipt verification failed: {path}")
        record.update(receipt_path=str(receipt_path), receipt_verified=True)
    return record


def write_parquet(db, statement: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        db.execute(f"COPY ({statement}) TO {quote(temporary)} "
                   "(FORMAT PARQUET, COMPRESSION ZSTD)")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args) -> dict:
    official_files = [args.metadata / f"taq_crsp_link_{year}.parquet" for year in range(2019, 2026)]
    master_files = [args.metadata / f"taq_master_{year}.parquet" for year in range(2019, 2026)]
    support_files = [args.support / name for name in
                     ("sp500_daily_membership.parquet", "crsp_daily_auxiliary.parquet",
                      "trading_calendar.parquet")]
    missing = [str(path) for path in official_files + master_files + support_files if not path.is_file()]
    if missing:
        raise ValueError("Required inputs have not arrived: " + ", ".join(missing))

    with duckdb.connect() as db:
        source_files = [file_record(db, path) for path in official_files + master_files + support_files]
        db.read_parquet([str(path) for path in official_files], filename=True).create_view("official_raw")
        db.read_parquet([str(path) for path in master_files], filename=True).create_view("master_raw")
        db.read_parquet(str(support_files[0])).create_view("universe")
        db.read_parquet(str(support_files[1])).create_view("auxiliary")
        db.read_parquet(str(support_files[2])).create_view("calendar")

        db.execute("""
            CREATE TEMP TABLE official_candidates AS
            SELECT DISTINCT l.date AS session_date, l.permno,
                   trim(l.sym_root) AS sym_root,
                   nullif(trim(l.sym_suffix), '') AS sym_suffix,
                   'wrds_taq_crsp_link' AS mapping_method, l.match_lvl,
                   a.cusip AS dated_crsp_cusip8,
                   CAST(NULL AS VARCHAR) AS taq_cusip9,
                   CAST(NULL AS VARCHAR) AS taq_symbol_15,
                   l.cusip AS wrds_cusip, l.ncusip AS wrds_ncusip,
                   l.ticker AS wrds_ticker, l.filename AS mapping_source_file,
                   l.sym_root IS NOT NULL AND trim(l.sym_root) <> '' AS symbol_valid
            FROM official_raw l JOIN universe u
              ON l.date = u.session_date AND l.permno = u.permno
            LEFT JOIN auxiliary a ON l.date = a.dlycaldt AND l.permno = a.permno
        """)
        db.execute("""
            CREATE TEMP TABLE official_ranked AS
            SELECT *, min(coalesce(match_lvl, 1e100)) OVER
                (PARTITION BY session_date, permno) AS best_match_level
            FROM official_candidates
        """)
        db.execute("""
            CREATE TEMP TABLE official_best AS
            SELECT * EXCLUDE (best_match_level) FROM official_ranked
            WHERE coalesce(match_lvl, 1e100) = best_match_level
        """)
        # TAQ uses 'ROOT SUFFIX' in symbol_15. Whitespace is normalized only to
        # separate the two feed components; punctuation is left unchanged.
        db.execute(r"""
            CREATE TEMP TABLE master_normalized AS
            SELECT date, symbol_15, cusip, filename,
                   regexp_replace(trim(symbol_15), '\s+', ' ', 'g') AS symbol_normalized,
                   upper(trim(cusip)) AS cusip_normalized
            FROM master_raw
        """)
        db.execute(r"""
            CREATE TEMP TABLE fallback_candidates AS
            SELECT DISTINCT a.dlycaldt AS session_date, a.permno,
                   split_part(m.symbol_normalized, ' ', 1) AS sym_root,
                   nullif(regexp_extract(m.symbol_normalized, '^[^ ]+ (.+)$', 1), '') AS sym_suffix,
                   'exact_dated_cusip8' AS mapping_method,
                   CAST(NULL AS DOUBLE) AS match_lvl,
                   a.cusip AS dated_crsp_cusip8, m.cusip AS taq_cusip9,
                   m.symbol_15 AS taq_symbol_15,
                   CAST(NULL AS VARCHAR) AS wrds_cusip,
                   CAST(NULL AS VARCHAR) AS wrds_ncusip,
                   CAST(NULL AS VARCHAR) AS wrds_ticker,
                   m.filename AS mapping_source_file,
                   m.symbol_normalized IS NOT NULL AND m.symbol_normalized <> ''
                     AND len(string_split(m.symbol_normalized, ' ')) <= 2 AS symbol_valid
            FROM auxiliary a JOIN master_normalized m
              ON a.dlycaldt = m.date
             AND upper(trim(a.cusip)) = substr(m.cusip_normalized, 1, 8)
            WHERE len(trim(a.cusip)) = 8 AND len(m.cusip_normalized) = 9
              AND NOT EXISTS (
                  SELECT 1 FROM official_best o
                  WHERE o.session_date = a.dlycaldt AND o.permno = a.permno AND o.symbol_valid
              )
        """)
        db.execute("""
            CREATE TEMP TABLE candidates AS
            SELECT * FROM official_best UNION ALL SELECT * FROM fallback_candidates
        """)
        db.execute("""
            CREATE TEMP TABLE ambiguous_symbols AS
            SELECT session_date, sym_root, coalesce(sym_suffix, '') AS suffix_key,
                   count(DISTINCT permno) AS permnos
            FROM candidates WHERE symbol_valid
            GROUP BY 1,2,3 HAVING count(DISTINCT permno) > 1
        """)
        db.execute("""
            CREATE TEMP TABLE rejected AS
            SELECT c.*, 'symbol_maps_to_multiple_permnos' AS rejection_reason
            FROM candidates c JOIN ambiguous_symbols x
              ON c.session_date = x.session_date AND c.sym_root = x.sym_root
             AND coalesce(c.sym_suffix, '') = x.suffix_key
            UNION ALL
            SELECT *, 'invalid_or_multicomponent_TAQ_symbol' AS rejection_reason
            FROM candidates WHERE NOT coalesce(symbol_valid, false)
            UNION ALL
            SELECT * EXCLUDE (best_match_level), 'not_best_official_match_level' AS rejection_reason
            FROM official_ranked WHERE coalesce(match_lvl, 1e100) > best_match_level
        """)
        db.execute("""
            CREATE TEMP TABLE accepted AS
            SELECT c.* EXCLUDE (symbol_valid) FROM candidates c
            WHERE symbol_valid AND NOT EXISTS (
                SELECT 1 FROM ambiguous_symbols x
                WHERE c.session_date = x.session_date AND c.sym_root = x.sym_root
                  AND coalesce(c.sym_suffix, '') = x.suffix_key
            )
        """)
        duplicate_keys = count(db, """
            SELECT count(*) FROM (
                SELECT session_date, sym_root, coalesce(sym_suffix, '')
                FROM accepted GROUP BY 1,2,3 HAVING count(*) > 1
            )
        """)
        if duplicate_keys:
            raise ValueError(f"Accepted candidate provenance has {duplicate_keys} duplicate symbol/date keys; inspect metadata")
        db.execute("""
            CREATE TEMP TABLE mapped AS
            SELECT a.*, u.membership_active, u.membership_previous_session,
                   u.crsp_record_available, c.session_split, c.expected_minutes,
                   c.early_close, c.market_open_utc, c.market_close_utc,
                   c.market_open_new_york, c.market_close_new_york,
                   c.previous_session, c.next_session,
                   count(*) OVER (PARTITION BY a.session_date, a.permno) AS symbols_for_security_day,
                   count(*) OVER (PARTITION BY a.session_date, a.permno) > 1
                       AS multiple_symbols_for_security_day
            FROM accepted a JOIN universe u USING (session_date, permno)
            JOIN calendar c USING (session_date)
        """)
        db.execute("""
            CREATE TEMP TABLE unresolved AS
            SELECT u.*, a.cusip AS dated_crsp_cusip8,
                   CASE WHEN EXISTS (
                       SELECT 1 FROM rejected r
                       WHERE r.session_date = u.session_date AND r.permno = u.permno
                         AND r.rejection_reason = 'symbol_maps_to_multiple_permnos'
                   ) THEN 'ambiguous_symbol_maps_to_multiple_permnos'
                   WHEN EXISTS (
                       SELECT 1 FROM rejected r
                       WHERE r.session_date = u.session_date AND r.permno = u.permno
                         AND r.rejection_reason = 'invalid_or_multicomponent_TAQ_symbol'
                   ) THEN 'invalid_or_multicomponent_TAQ_symbol'
                   WHEN NOT u.crsp_record_available THEN 'no_dated_CRSP_observation_for_CUSIP_fallback'
                   WHEN a.cusip IS NULL OR len(trim(a.cusip)) <> 8 THEN 'no_valid_dated_CRSP_CUSIP8'
                   ELSE 'no_official_link_or_exact_same_day_CUSIP_master_match'
                   END AS unresolved_reason
            FROM universe u LEFT JOIN auxiliary a
              ON u.session_date = a.dlycaldt AND u.permno = a.permno
            WHERE NOT EXISTS (
                SELECT 1 FROM mapped m WHERE m.session_date = u.session_date AND m.permno = u.permno
            )
        """)

        universe_days = count(db, "SELECT count(*) FROM universe")
        mapped_days = count(db, "SELECT count(*) FROM (SELECT DISTINCT session_date,permno FROM mapped)")
        unresolved_days = count(db, "SELECT count(*) FROM unresolved")
        if universe_days != mapped_days + unresolved_days:
            raise ValueError("Mapped and unresolved security/day counts do not reconcile to the universe")
        summary = {
            "complete": True,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "policy": {"official_match_level": "smallest available per security/day; equally ranked distinct symbols retained",
                       "fallback": "exact same-day dated CRSP cusip[8] = TAQ master cusip[1:8]",
                       "header_CUSIP_or_ticker_fallback": False,
                       "symbol_ambiguity": "all symbol/date candidates mapping to multiple PERMNOs excluded",
                       "multiple_symbols_one_security_day": "retained and explicitly flagged"},
            "sources": source_files,
            "universe_security_days": universe_days,
            "mapped_security_days": mapped_days,
            "unresolved_security_days": unresolved_days,
            "mapping_rows": count(db, "SELECT count(*) FROM mapped"),
            "duplicate_symbol_date_keys": duplicate_keys,
            "ambiguous_symbol_date_keys": count(db, "SELECT count(*) FROM ambiguous_symbols"),
            "multiple_symbol_security_days": count(db, """
                SELECT count(*) FROM (
                    SELECT session_date,permno FROM mapped
                    WHERE multiple_symbols_for_security_day GROUP BY 1,2
                )
            """),
            "mapping_counts_by_year_and_method": rows(db, """
                SELECT year(session_date) AS year, mapping_method, count(*) AS mapping_rows,
                       count(DISTINCT (session_date,permno)) AS security_days
                FROM mapped GROUP BY 1,2 ORDER BY 1,2
            """),
            "unresolved_by_year_and_reason": rows(db, """
                SELECT year(session_date) AS year, unresolved_reason, count(*) AS security_days
                FROM unresolved GROUP BY 1,2 ORDER BY 1,2
            """),
            "rejected_by_reason": rows(db, """
                SELECT rejection_reason, count(*) AS candidate_rows FROM rejected GROUP BY 1 ORDER BY 1
            """),
            "official_match_levels": rows(db, """
                SELECT match_lvl, count(*) AS candidate_rows FROM official_candidates GROUP BY 1 ORDER BY 1
            """),
        }
        args.support.mkdir(parents=True, exist_ok=True)
        summary["outputs"] = []
        for name, query in {
            "taq_daily_mappings.parquet": "SELECT * FROM mapped ORDER BY session_date,permno,sym_root,sym_suffix",
            "taq_mapping_unresolved.parquet": "SELECT * FROM unresolved ORDER BY session_date,permno",
            "taq_mapping_rejected.parquet": "SELECT * FROM rejected ORDER BY session_date,permno,sym_root,sym_suffix",
        }.items():
            path = args.support / name
            write_parquet(db, query, path)
            summary["outputs"].append(file_record(db, path))
        temporary = args.support / "mapping_validation.json.tmp"
        temporary.write_text(json.dumps(summary, indent=2, default=str) + "\n")
        temporary.replace(args.support / "mapping_validation.json")
        return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=ROOT / "metadata")
    parser.add_argument("--support", type=Path, default=ROOT / "support")
    args = parser.parse_args()
    try:
        result = build(args)
    except (OSError, ValueError, duckdb.Error) as exc:
        parser.exit(1, f"Mapping build failed: {exc}\n")
    print(json.dumps({key: result[key] for key in
                      ("complete", "mapped_security_days", "unresolved_security_days",
                       "mapping_rows", "multiple_symbol_security_days")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
