"""Reconcile all held source payoffs, including the newly broad rank books."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from usdata import Panel, arrow_read, sha, write_json


def main():
    panel = Panel()
    plan = json.loads((HERE / "protocol.json").read_text())
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    source_hashes = {}
    held, missing = [], []
    for case in plan["cases"]:
        path = HERE / f"results/{case}_weights.npz"
        z = np.load(path)
        weights, dates = z["weights"], z["dates"]
        ids = np.array([int(s[1:]) for s in z["symbols"]])
        lookup = {str(d): i for i, d in enumerate(panel.dates)}
        ix = np.array([lookup[str(d)] for d in dates])
        values = np.asarray(panel.total_return[:, ix]).T
        row, col = np.nonzero(weights)
        held.append(pd.DataFrame(dict(permno=ids[col], date=dates[row],
                                      prepared_return=values[row, col])))
        bad = ~np.isfinite(values[row, col])
        if bad.any():
            missing.append(pd.DataFrame(dict(case=case, permno=ids[col[bad]], date=dates[row[bad]],
                                             weight=weights[row[bad], col[bad]])))
    keys = pd.concat(held, ignore_index=True).drop_duplicates(["permno", "date"])
    audited, unobserved, bad_returns = 0, 0, []
    for year in range(2017, 2026):
        path = ROOT.parent / f"crsp-daily/parquet/{year}.parquet"
        assert sha(path) == manifest["source_sha256"][str(path)], ("Source changed", path)
        source_hashes[str(path)] = manifest["source_sha256"][str(path)]
        subset = keys[keys.date.str[:4] == str(year)]
        raw = arrow_read(path, ["permno", "dlycaldt", "dlyret"],
                         filters=[("permno", "in", subset.permno.unique().tolist())])
        raw["date"] = pd.to_datetime(raw.dlycaldt).dt.strftime("%Y-%m-%d")
        joined = subset.merge(raw[["permno", "date", "dlyret"]], on=["permno", "date"],
                              how="left", validate="one_to_one", indicator=True)
        np.testing.assert_allclose(joined.prepared_return, joined.dlyret, equal_nan=True, atol=1e-12)
        audited += len(joined)
        unobserved += int((joined._merge == "left_only").sum())
        bad_returns.append(joined[joined.prepared_return.isna()])
        print("SOURCE", year, len(joined), flush=True)
    assert audited == len(keys) and unobserved == 0
    pd.concat(bad_returns, ignore_index=True).to_csv(HERE / "results/missing_source_payoffs.csv", index=False)
    if missing:
        pd.concat(missing, ignore_index=True).to_csv(HERE / "results/missing_held_positions.csv", index=False)
    write_json(HERE / "results/source_audit.json", dict(status="passed", distinct_held_stock_dates=audited,
               missing_source_payoffs=sum(len(x) for x in bad_returns), unobserved_source_rows=unobserved,
               source_sha256=source_hashes, code_sha256=sha(__file__),
               created_utc=datetime.now(timezone.utc).isoformat()))
    print("ALL HELD SOURCE PAYOFFS VERIFIED", audited, flush=True)


if __name__ == "__main__":
    main()
