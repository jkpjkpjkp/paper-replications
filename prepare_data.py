"""Retrieve bounded observed inputs; preserve provider bytes and every failure.

Current provider vintage is an explicit adaptation: the author does not release
its raw master panel. Source adjusted closes are never replaced by generated
prices. Existing French factor files are validated and reused read-only.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import io
import json
import os
import urllib.request
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "provider_bytes"
RAW.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
MANIFEST = ROOT / "data" / "manifest.json"
TICKERS = "EEM GLD HYG IEF IWM LQD QQQ SPY TLT UUP XLB XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split()
manifest = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "scope": "current provider vintage, same paper universes and periods; not author raw-file identity", "expected_download_bytes_upper_bound": 50000000, "peak_additional_disk_gib": 0.3, "sources": [], "errors": []}


def persist():
    temp = MANIFEST.with_suffix(".json.tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temp, MANIFEST)


def get(url, filename, max_bytes=30000000):
    dest = CACHE / filename
    if not dest.exists():
        print("Fetching", filename, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as response:
            body = response.read(max_bytes + 1)
            if len(body) > max_bytes:
                raise ValueError("download exceeds declared per-file bound")
        dest.write_bytes(body)
    body = dest.read_bytes()
    manifest["sources"].append({"url": url, "path": str(dest), "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    persist()
    return body


def attempt(label, function):
    try:
        function()
    except Exception as exc:
        manifest["errors"].append({"component": label, "error": repr(exc)})
        print(label, "FAILED", repr(exc), flush=True)
        persist()


def industry():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
    body = get(url, "49_Industry_Portfolios_daily_CSV.zip")
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        assert archive.getinfo(name).file_size < 100000000
        text = archive.read(name).decode("utf-8-sig")
    # Preserve header/sections and truncate all dated rows to the paper data end.
    filtered = []
    for line in text.splitlines():
        first = line.split(",", 1)[0].strip()
        if len(first) == 8 and first.isdigit() and first > "20260331":
            continue
        filtered.append(line)
    (RAW / "49_Industry_Portfolios_Daily.csv").write_text("\n".join(filtered) + "\n")


def factors():
    base = Path(os.environ.get("REPLICATION_DATA_ROOT", ROOT.parent)) / "factors-daily"
    receipt = json.loads((base / "download-receipt.json").read_text())
    for key in ["ff3_daily", "ff5_daily", "momentum_daily"]:
        item = receipt["sources"][key]
        source = base / item["source_csv"]
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        assert checksum == item["source_sha256"]
        dest = RAW / source.name
        if not dest.exists():
            dest.symlink_to(source)
        manifest["sources"].append({"url": item["url"], "path": str(source), "reuse": "read-only symlink", "sha256": checksum, "original_retrieval": receipt["downloaded_utc"]})


def fred(series):
    url = f"https://fred.stlouisfed.org/graph/?g=1&id={series}&cosd=1990-01-01&coed=2026-03-31"
    # FRED's graph CSV route supplies the same source series as its series page.
    url = f"https://fred.stlouisfed.org/graph/graph.csv?id={series}&cosd=1990-01-01&coed=2026-03-31"
    body = get(url, series + ".csv", max_bytes=2000000)
    frame = pd.read_csv(io.BytesIO(body))
    assert series in frame
    frame.columns = ["date" if c != series else series for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["date"])
    frame[series] = pd.to_numeric(frame[series], errors="coerce")
    frame = frame.loc[frame.date <= "2026-03-31"]
    assert len(frame) > 4000
    frame.to_parquet(RAW / (series + ".parquet"), index=False)


def etf():
    begin = int(datetime(2015, 10, 8, tzinfo=timezone.utc).timestamp())
    end = int(datetime(2026, 3, 17, tzinfo=timezone.utc).timestamp())
    panels = []
    for ticker in TICKERS:
        def one():
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={begin}&period2={end}&interval=1d&events=div%2Csplits&includeAdjustedClose=true"
            body = get(url, ticker + ".json", max_bytes=2000000)
            obj = json.loads(body)
            result = obj["chart"]["result"][0]
            adj = result["indicators"]["adjclose"][0]["adjclose"]
            dates = pd.to_datetime(result["timestamp"], unit="s", utc=True).tz_convert("America/New_York").tz_localize(None).normalize()
            frame = pd.Series(adj, index=dates, name=ticker).sort_index()
            assert frame.index.is_unique and len(frame) > 2500
            assert (frame.dropna() > 0).all()
            panels.append(frame)
        attempt(ticker, one)
    if len(panels) != 20:
        raise ValueError(f"Only {len(panels)}/20 original ETF inputs; no partial universe output is generated")
    panel = pd.concat(panels, axis=1).sort_index()
    assert not panel.isna().any().any(), panel.isna().sum().to_dict()
    panel.index.name = "date"
    panel.to_csv(RAW / "master_daily_features_macro_dailyonly_raw_only.csv")
    manifest["ETF_panel"] = {"rows": len(panel), "columns": list(panel), "start": str(panel.index.min()), "end": str(panel.index.max()), "convention": "Yahoo adjusted close includes distributions and splits; current vintage; no fabricated macro inputs; extra master-panel macro columns unused by official model"}


attempt("French49", industry)
attempt("shared_factors", factors)
attempt("VIXCLS", lambda: fred("VIXCLS"))
attempt("VXVCLS", lambda: fred("VXVCLS"))
attempt("20ETF", etf)
manifest["ready"] = all((RAW / n).exists() for n in ["49_Industry_Portfolios_Daily.csv", "F-F_Research_Data_Factors_daily.csv", "VIXCLS.parquet", "master_daily_features_macro_dailyonly_raw_only.csv"])
persist()
print(json.dumps({"ready": manifest["ready"], "errors": manifest["errors"]}, indent=2), flush=True)
