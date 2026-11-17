"""Fresh downloads using the public release, isolated from frozen experiments.

The paper's stable-membership restriction cannot be established by these
generators. Receipts deliberately distinguish a successful download from an
author-exact data reconstruction. No replacement KOSPI constituent list is used.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PAPER_N = {"euro50": 47, "ftse100": 93, "nikkei225": 208, "kospi200": 162}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markets", nargs="+", choices=PAPER_N, default=list(PAPER_N))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "receipts.json").exists():
        raise SystemExit("Refusing to overwrite an existing download audit; choose a new output directory")
    records = []
    for market in args.markets:
        record = dict(market=market, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                      start="2016-01-01", end="2025-12-31", paper_n=PAPER_N[market],
                      author_snapshot=False, stable_membership_verified=False)
        try:
            if market == "kospi200":
                # The release requires a KRX export; try its provider, without
                # silently substituting Wikipedia when KRX is unavailable.
                import requests
                import FinanceDataReader as fdr
                original_request = requests.Session.request

                def krx_https(self, method, url, **kwargs):
                    if url.startswith("http://data.krx.co.kr/"):
                        url = "https://" + url[len("http://"):]
                    kwargs.setdefault("timeout", 30)
                    return original_request(self, method, url, **kwargs)

                requests.Session.request = krx_https
                try:
                    listing = fdr.SnapDataReader("KRX/INDEX/STOCK/1028")
                finally:
                    requests.Session.request = original_request
                codes = listing.Code.astype(str).str.zfill(6)
                if not (180 <= len(codes) <= 220 and codes.str.fullmatch(r"\d{6}").all()):
                    raise ValueError("Unexpected KRX constituent response")
                raw = out / "data/raw"
                raw.mkdir(parents=True, exist_ok=True)
                codes.rename("종목코드").to_csv(raw / "kospi200.csv", index=False, encoding="cp949")
                record["constituent_source"] = "KRX/INDEX/STOCK/1028 via FinanceDataReader, HTTPS transport"
            command = [sys.executable, str(ROOT / "upstream/src/data_generator.py"),
                       "--data", market, "--start", record["start"], "--end", record["end"]]
            record["command"] = command
            with (out / f"{market}.log").open("w") as log:
                subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT,
                               check=True, timeout=900)
            import numpy as np
            import pandas as pd
            path = out / "data/daily" / f"{market}_160101_251231.csv"
            data = pd.read_csv(path, index_col=0)
            if not (len(data) > 2000 and len(data.columns) > 10 and np.isfinite(data.to_numpy()).all()):
                raise ValueError("Downloaded panel is empty, incomplete, or nonfinite")
            frozen = pd.read_csv(ROOT / "data/daily" / path.name, index_col=0)
            shared_dates = data.index.intersection(frozen.index)
            shared_assets = data.columns.intersection(frozen.columns)
            error = (data.loc[shared_dates, shared_assets] - frozen.loc[shared_dates, shared_assets]).abs()
            record.update(status="downloaded", rows=len(data), n_assets=len(data.columns),
                          first=str(data.index[0]), last=str(data.index[-1]),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          same_values_as_frozen=data.equals(frozen),
                          shared_values_max_abs_change=float(error.to_numpy().max()),
                          added_columns=sorted(set(data.columns)-set(frozen.columns)),
                          removed_columns=sorted(set(frozen.columns)-set(data.columns)))
        except Exception as error:
            record.update(status="failed", error_type=type(error).__name__, error=str(error))
        records.append(record)
        (out / "receipts.json").write_text(json.dumps(records, indent=2, allow_nan=False))
        print(json.dumps(record), flush=True)
    return int(any(r["status"] != "downloaded" for r in records))


if __name__ == "__main__":
    raise SystemExit(main())
