"""Execute the country transfer protocol without per-country selection/tuning."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from ocp import run, controller, metrics


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    expected = (HERE / "protocol.sha256").read_text().split()[0]
    assert sha256(protocol_path) == expected, "Registered protocol changed"
    sha = (HERE / "source_commit.txt").read_text().split()[0]
    assert len(sha) == 40
    url = f"https://raw.githubusercontent.com/CSSEGISandData/COVID-19/{sha}/{protocol['source_artifact']}"
    raw_path = HERE / "confirmed_global.csv"
    if not raw_path.exists():
        urllib.request.urlretrieve(url, raw_path)
    downloaded_at = datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat()
    df = pd.read_csv(raw_path)
    date_columns = [c for c in df.columns if c not in ["Province/State", "Country/Region", "Lat", "Long"]]
    dates = pd.to_datetime(date_columns, format="%m/%d/%y")
    assert np.all(np.diff(dates.to_numpy()) == np.timedelta64(1, "D")), "Non-daily input"
    b = int(protocol["burnin_scores"])
    offset = int(protocol["evaluation_start_score_index_zero_based"])
    rows, country_info = [], {}
    trace_dir = HERE / "traces"
    trace_dir.mkdir(exist_ok=True)
    for country in protocol["countries"]:
        selected = df[df["Country/Region"] == country]
        assert len(selected), country
        cumulative = selected[date_columns].sum(axis=0).to_numpy(dtype=float)
        increments = np.diff(cumulative, prepend=cumulative[0])
        cases = np.maximum(0, increments)
        forecasts = cases[:-7]
        observed = cases[7:]
        raw_score = np.abs(observed - forecasts)
        scale = float(np.median(raw_score[:b]))
        if scale <= 0:
            scale = float(np.mean(raw_score[:b]))
        if scale <= 0:
            scale = 1.0
        score = raw_score / scale
        assert np.isfinite(score).all() and np.all(score >= 0)
        country_info[country] = dict(rows_aggregated=len(selected), raw_length=len(cases), score_length=len(score),
                                     raw_date_start=dates[0].strftime("%Y-%m-%d"),
                                     raw_date_end=dates[-1].strftime("%Y-%m-%d"),
                                     evaluation_date_start=dates[7 + offset].strftime("%Y-%m-%d"),
                                     n_evaluation=len(score)-offset, normalization_scale=scale,
                                     negative_daily_revisions=int(np.sum(increments < 0)),
                                     negative_revision_dates=[d.strftime("%Y-%m-%d") for d in dates[increments < 0]])
        traces = dict(cumulative_cases=cumulative, daily_cases=cases, observed=observed, forecasts=forecasts,
                      raw_scores=raw_score, normalized_scores=score, scale=scale,
                      score_dates=dates[7:].to_numpy(dtype="datetime64[D]"), evaluation_start=offset)
        for init, start in [("cold", b), ("warm", 0)]:
            for method, cfg in protocol["frozen_methods"].items():
                alpha = float(cfg["alpha"])
                if method == "P":
                    q = controller(score[start:], alpha=alpha, lr=float(cfg["lr"]), window=int(cfg["window"]))
                else:
                    q = run(score[start:], method=method, alpha=alpha, lr=float(cfg.get("lr", 0.0)))
                q_eval = q[offset-start:]
                target_score = score[offset:]
                m = metrics(target_score, q_eval)
                for key in ["mean_width", "median_width", "q75_width", "q90_width", "q95_width"]:
                    m[key] *= scale
                rows.append(dict(country=country, initialization=init, method=method, alpha=alpha,
                                 lr=cfg.get("lr", 0.0), actual_coverage_ge_95=m["coverage"]>=0.95,
                                 **m))
                traces[init + "__" + method + "__radii"] = q * scale
                traces[init + "__" + method + "__start_index"] = start
        slug = country.lower().replace(" ", "_")
        np.savez_compressed(trace_dir / (slug + ".npz"), **traces)
    result = pd.DataFrame(rows)
    assert len(result) == 48
    result.to_csv(HERE / "confirmation_results.csv", index=False)
    provenance = dict(protocol_registered_at=protocol["preregistered_at_utc"], protocol_sha256=expected,
                      source_repository=protocol["source_repository"], source_commit=sha, download_url=url,
                      raw_file=str(raw_path.name), raw_sha256=sha256(raw_path), downloaded_at=downloaded_at,
                      executed_at=datetime.now(timezone.utc).isoformat(),
                      ocp_code_sha256=sha256(ROOT / "ocp.py"), confirmation_script_sha256=sha256(Path(__file__)),
                      countries=country_info,
                      interpretation="Fixed-setting transfer on all prespecified countries; countries are not asserted independent; retrospective source snapshots are index-causal, not an as-of-vintage real-time evaluation.")
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2))
    lines = ["# Preregistered country transfer confirmation", "",
             "Registered before country artifact download/outcome inspection: " + protocol["preregistered_at_utc"], "",
             "All six methods, all four countries, both common initialization protocols are retained. Settings are frozen from the US screen. No per-country tuning or independence/confidence-interval claim.", "",
             "The target is nonnegative reported daily case increments. Forecasts are the observed counts seven days earlier. Each series is divided by its first-115-score median (mean/1 fallback), fixed before evaluation. Widths below are in original case-count units.", "",
             "| Country | Initialization | Method | Actual coverage | Mean width | Median width | Infinite fraction | Longest error streak |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        width = "inf" if np.isinf(r["mean_width"]) else f"{r['mean_width']:.6g}"
        median = "inf" if np.isinf(r["median_width"]) else f"{r['median_width']:.6g}"
        lines.append(f"| {r['country']} | {r['initialization']} | {r['method']} | {r['coverage']:.4%} | {width} | {median} | {r['infinite_fraction']:.4%} | {r['longest_error_streak']} |")
    lines += ["", "These results confirm transfer of fixed configurations, not matched achieved coverage in each new country. A width advantage without comparable coverage is not an efficiency win. The held-out countries have overlapping calendar shocks and are not treated as independent replications for inference.", "",
              "Final JHU snapshots can revise historical counts. The forecast construction uses only past indices, but this is not proven to reproduce the information available to a forecaster in real time.", "",
              "Sources and exact hashes: [provenance.json](provenance.json). Frozen protocol: [protocol.json](protocol.json). Numeric results: [confirmation_results.csv](confirmation_results.csv). Per-step data and intervals: [traces/](traces/).", ""]
    (HERE / "README.md").write_text("\n".join(lines))
    print(result[["country", "initialization", "method", "coverage", "mean_width", "infinite_fraction", "longest_error_streak"]].to_string(index=False))


if __name__ == "__main__":
    main()
