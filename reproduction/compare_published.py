"""Compare completed original-method runs with all 84 published Table 1 cells.

Reads author summary_metrics.csv/test_trace.csv artifacts, never trains or selects.
Unfinished and failed cells remain visible. No contemporary panel is called the
unreleased author snapshot, even if its rounded statistic agrees with the paper.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER_N = {"euro50": 47, "ftse100": 93, "kospi200": 162, "nikkei225": 208}
PAPER_K = {"euro50": [5, 7, 9], "ftse100": [9, 14, 19],
           "kospi200": [16, 24, 32], "nikkei225": [21, 31, 42]}
SHARPE_FACTOR = float(np.sqrt(252))
REPORTING_CONVENTION = {
    "primary_comparison": "annualized_sharpe_interpretation",
    "paper_stated_metric": "daily_sharpe",
    "paper_table_annualization_confirmed": False,
    "trading_days_per_year": 252,
    "annualization_factor": SHARPE_FACTOR,
    "seed_aggregation": "arithmetic mean and sample standard deviation (ddof=1)",
    "paper_values": "reported mean and SD copied unchanged",
    "legacy_fields": {"released_seed_sd": "daily Sharpe seed SD in Table 1; metric-specific in appendix tables",
                      "rounded_display_numbers_agree": "daily mean and SD agreement only"},
    "drawdown": "unchanged; never annualized",
}
REPORTING_NOTE = (
    "Primary comparison: interpret the paper's table values as annualized Sharpe. "
    "Each reproduced daily Sharpe is multiplied by sqrt(252); the mean AND the sample "
    "standard deviation across seeds therefore receive the same factor. Published means "
    "and standard deviations are copied unchanged. This is the requested reporting "
    "interpretation, not a confirmed author convention: Section 5.3 explicitly defines daily "
    "Sharpe. Daily results remain visible. No factor is fitted per market or method, and "
    "numerical agreement alone does not establish exact reproduction."
)


def sharpe_fields(mean, sd, paper_mean, paper_sd, complete):
    """Keep daily and annualized comparisons explicit, including uncertainty."""
    annual_mean, annual_sd = mean * SHARPE_FACTOR, sd * SHARPE_FACTOR
    def matches(actual, reported):
        return bool(complete and np.isfinite(actual) and np.isclose(actual, reported, atol=.0005, rtol=0.))
    am, ass = matches(annual_mean, paper_mean), matches(annual_sd, paper_sd)
    daily_match = matches(mean, paper_mean) and matches(sd, paper_sd)
    return dict(paper_reported_sharpe_mean=paper_mean, paper_seed_sd=paper_sd,
                released_daily_gross_sr_mean=mean, released_daily_seed_sd=sd,
                released_annualized_gross_sr_mean=annual_mean, released_annualized_seed_sd=annual_sd,
                daily_sr_minus_paper=mean-paper_mean,
                annualized_sr_minus_paper_display=annual_mean-paper_mean,
                daily_seed_sd_minus_paper=sd-paper_sd,
                annualized_seed_sd_minus_paper=annual_sd-paper_sd,
                daily_mean_and_sd_match=daily_match, annualized_mean_match=am,
                annualized_sd_match=ass, annualized_mean_and_sd_match=am and ass,
                primary_comparison="annualized_sharpe_interpretation",
                rounded_display_numbers_agree=daily_match)


def comparison_summary(frame):
    rows = []
    for method in ["all", "historic", "pfl", "dfl"]:
        d = frame[frame.all_required_repeats_complete]
        if method != "all":
            d = d[d.method == method]
        rows.append(dict(method=method, completed_cells=len(d),
            annualized_sharpe_mae=float(d.annualized_sr_minus_paper_display.abs().mean()),
            daily_sharpe_mae=float(d.daily_sr_minus_paper.abs().mean()),
            annualized_mean_matches=int(d.annualized_mean_match.sum()),
            annualized_sd_matches=int(d.annualized_sd_match.sum()),
            annualized_mean_and_sd_matches=int(d.annualized_mean_and_sd_match.sum()),
            daily_mean_and_sd_matches=int(d.daily_mean_and_sd_match.sum())))
    return pd.DataFrame(rows)


def format_mean_sd(mean, sd):
    return f"{mean:.3f} ± {sd:.3f}" if pd.notna(mean) and pd.notna(sd) and np.isfinite(mean) and np.isfinite(sd) else "unavailable"


def comparison_display(frame, identity_columns=None):
    identity_columns = identity_columns or ["market", "rho_percent", "method", "paper_optimizer", "paper_k", "current_k"]
    display = frame[identity_columns].copy()
    display["Paper reported mean ± SD"] = [format_mean_sd(m, s) for m, s in zip(frame.paper_reported_sharpe_mean, frame.paper_seed_sd)]
    display["Reproduced annualized mean ± SD"] = [format_mean_sd(m, s) for m, s in zip(frame.released_annualized_gross_sr_mean, frame.released_annualized_seed_sd)]
    display["Annualized mean − paper"] = frame.annualized_sr_minus_paper_display
    display["Reproduced daily mean ± SD"] = [format_mean_sd(m, s) for m, s in zip(frame.released_daily_gross_sr_mean, frame.released_daily_seed_sd)]
    display["Completed / required seeds"] = frame.completed_seed_count.astype(str) + "/" + frame.required_seed_count.astype(str)
    display["status"] = frame.status
    return display.rename(columns={"market": "Market", "rho_percent": "Cardinality (%)", "method": "Method",
                                   "paper_optimizer": "Optimizer", "paper_k": "Paper k", "current_k": "Actual k",
                                   "alpha": "Alpha", "status": "Status"})


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
        stream.write(text)
        temp = Path(stream.name)
    temp.replace(path)


def panel_info(market, hashes):
    path = ROOT / "data/daily" / f"{market}_160101_251231.csv"
    if not path.exists():
        return None
    data = pd.read_csv(path, index_col=0)
    assert np.isfinite(data.to_numpy(dtype=float)).all(), path
    assert data.index.is_unique and data.index.is_monotonic_increasing, path
    hashes[str(path.relative_to(ROOT))] = digest(path)
    receipt = ROOT / "data" / f"{market}_receipt.json"
    constituent_info = None
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved.get("sha256") == hashes[str(path.relative_to(ROOT))], "Downloaded panel no longer matches its receipt"
        assert saved.get("n_assets") == data.shape[1] and saved.get("rows") == len(data)
        hashes[str(receipt.relative_to(ROOT))] = digest(receipt)
    if market == "kospi200":
        listing_receipt = ROOT / "data/raw/kospi200_listing_receipt.json"
        if listing_receipt.exists():
            constituent_info = json.loads(listing_receipt.read_text())
            hashes[str(listing_receipt.relative_to(ROOT))] = digest(listing_receipt)
    split = int(.8 * len(data))
    return {"n_assets": data.shape[1], "rows": len(data), "first": str(data.index[0]),
            "last": str(data.index[-1]), "test_dates": list(map(str, data.index[split+100:])),
            "author_snapshot": False, "sha256": hashes[str(path.relative_to(ROOT))],
            "constituent_source_receipt": constituent_info}


def collect(roots, panels, hashes, protocol):
    valid, issues = [], []
    paths = sorted({path for root in roots if root.exists() for path in root.rglob("summary_metrics.csv")})
    for path in paths:
        frame = pd.read_csv(path)
        if len(frame) != 1:
            issues.append({"path": str(path), "issue": "Expected one run per summary"})
            continue
        row = frame.iloc[0].to_dict()
        row["method"] = {"HIST": "historic", "PFL": "pfl", "DFL": "dfl"}.get(row.get("method"), row.get("method"))
        if row.get("dataset") not in PAPER_N or row.get("method") not in ("historic", "pfl", "dfl"):
            continue
        meta_path = path.parent / "meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        problem = panels.get(row["dataset"])
        trace_path = path.parent / "test_trace.csv"
        if problem is None or not trace_path.exists():
            issues.append({"path": str(path), "issue": "Data panel or completed holdout trace unavailable"})
            continue
        trace = pd.read_csv(trace_path)
        try:
            assert trace.date.astype(str).tolist() == problem["test_dates"], "Test dates differ from the author's 80/20 plus 100-session rule"
            assert trace.t.tolist() == list(range(len(trace))), "Missing or reordered test observations"
            returns = trace.ret_true_raw.to_numpy(dtype=float)
            assert np.isfinite(returns).all(), "Nonfinite realized returns"
            assert not trace.interpolated.astype(bool).any(), "Interpolated SDP returns are not exact daily evaluations"
            observed = returns.mean() / (returns.std(ddof=1) + 1e-12)
            np.testing.assert_allclose(observed, float(row["test_raw_sr"]), rtol=1e-7, atol=1e-9)
            # Preserve the author's MDD convention, including its omission of
            # initial wealth from the running peak and its insolvency policy.
            if np.any(returns <= -1.):
                mdd = float("nan")
            else:
                log_wealth = np.cumsum(np.log1p(returns))
                mdd = float(np.max(1.-np.exp(log_wealth-np.maximum.accumulate(log_wealth))))
            assert (np.isnan(mdd) and pd.isna(row["test_raw_mdd"])) or np.isclose(mdd, float(row["test_raw_mdd"]), rtol=1e-7, atol=1e-9), "Drawdown differs from the source return trace"
            if row["method"] != "historic":
                assert int(row["batchsize"]) == 64, "Batch differs from the paper"
                assert list(meta.get("hidden_dims", [])) == [512, 256], "MLP differs from the paper"
                assert int(row["epochs"]) == int(protocol["epochs"]), "Update count differs from the declared replay"
                for name in ("lr", "weight_decay", "dropout"):
                    assert np.isclose(float(meta[name]), float(protocol[name])), f"{name} differs from the declared replay"
                assert meta["activation"] == protocol["activation"]
            assert int(row["window_size"]) == protocol["window_size"]
            assert row["cov_mode"] == protocol["cov_mode"]
            for name in ("cov_shrinkage", "cov_eps"):
                assert np.isclose(float(row[name]), protocol[name])
            assert int(row["cov_max_tries"]) == protocol["cov_max_tries"]
            assert int(row.get("sdr_stride", 1)) == 1, "SDP stride differs from daily evaluation"
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            issues.append({"path": str(path), "issue": str(error)})
            continue
        row.update(verified_daily_sr=float(observed), verified_annualized_sr=float(observed)*SHARPE_FACTOR,
                   verified_test_mdd=mdd, verified_test_days=len(trace),
                   summary_path=str(path), trace_path=str(trace_path),
                   observed_actual_holdings_max=int(trace["weight_nnz_1e-8"].max()),
                   observed_gross_exposure_max=float(trace.weight_abs_sum.max()))
        for p in (path, trace_path, meta_path):
            if p.exists():
                hashes[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = digest(p)
        valid.append(row)
    columns = ["dataset", "method", "k_percent", "selector", "seed", "verified_daily_sr",
               "verified_test_days", "summary_path", "trace_path", "observed_actual_holdings_max",
               "observed_gross_exposure_max"]
    return pd.DataFrame(valid) if valid else pd.DataFrame(columns=columns), issues


def compare(reference, runs, panels, seeds, baseline_oscar):
    output = []
    for ref in reference.itertuples():
        percent = int(ref.rho_percent)
        paper_k = PAPER_K[ref.market][[10, 15, 20].index(percent)]
        panel = panels.get(ref.market)
        selector = baseline_oscar if ref.optimizer == "oscar" and ref.method != "dfl" else ref.optimizer
        subset = pd.DataFrame()
        if not runs.empty:
            subset = runs[(runs.dataset == ref.market) & (runs.method == ref.method) &
                          (runs.k_percent == percent) & (runs.selector == selector)].copy()
            if ref.method == "dfl":
                subset = subset[np.isclose(subset.alpha.astype(float), .5)]
            subset = subset[subset.seed.astype(int).isin([0] if ref.method == "historic" else seeds)]
        required = 1 if ref.method == "historic" else len(seeds)
        if not subset.empty and subset.seed.duplicated().any():
            raise ValueError(f"Duplicate runs for {ref.market}/{percent}/{ref.method}/{selector}; specify one artifact root")
        actual_k = int(round(panel["n_assets"] * percent / 100)) if panel else None
        row = {"market": ref.market, "rho_percent": percent, "method": ref.method,
               "paper_optimizer": ref.optimizer, "release_selector": selector,
               "paper_n_assets": PAPER_N[ref.market], "current_n_assets": panel["n_assets"] if panel else None,
               "paper_k": paper_k, "current_k": actual_k,
               "paper_daily_gross_sharpe_mean": ref.paper_daily_gross_sharpe_mean,
               "paper_seed_sd": ref.paper_seed_sd, "completed_seed_count": len(subset),
               "required_seed_count": required,
               "completed_seeds": ",".join(map(str, sorted(subset.seed.astype(int)))) if not subset.empty else "",
               "all_required_repeats_complete": len(subset) == required,
               "author_snapshot_available": False, "exact_published_result_replicated": False}
        row.update(sharpe_fields(float("nan"), float("nan"), ref.paper_daily_gross_sharpe_mean, ref.paper_seed_sd, False))
        if panel is None:
            row["status"] = "data_unavailable"
        elif subset.empty:
            row["status"] = "not_completed"
        else:
            mean = float(subset.verified_daily_sr.mean())
            sd = 0. if ref.method == "historic" else float(subset.verified_daily_sr.std(ddof=1))
            row.update(sharpe_fields(mean, sd, ref.paper_daily_gross_sharpe_mean, ref.paper_seed_sd, len(subset) == required))
            row.update(released_seed_sd=sd,
                       observed_actual_holdings_max=int(subset.observed_actual_holdings_max.max()),
                       observed_gross_exposure_max=float(subset.observed_gross_exposure_max.max()))
            row["status"] = "completed_release_current_snapshot" if len(subset) == required else "incomplete_seed_repeats"
        output.append(row)
    return pd.DataFrame(output)


def additional_tables(runs, seeds, baseline_oscar):
    outputs = {}
    for table in (2, 3):
        reference = pd.read_csv(ROOT / f"sources/paper_table{table}_reference.csv")
        rows = []
        for ref in reference.itertuples():
            subset = pd.DataFrame()
            if not runs.empty:
                method = "dfl" if table == 2 else ref.method
                selector = "oscar" if method == "dfl" else baseline_oscar
                subset = runs[(runs.dataset == ref.market) & (runs.method == method) &
                              (runs.k_percent == ref.rho_percent) & (runs.selector == selector) &
                              runs.seed.astype(int).isin(seeds)]
                alpha = ref.alpha if table == 2 else .5
                if method == "dfl":
                    subset = subset[np.isclose(subset.alpha.astype(float), alpha)]
                assert not subset.seed.duplicated().any(), "Duplicate seeds in appendix comparison"
            column = "verified_daily_sr" if table == 2 else "verified_test_mdd"
            values = subset[column] if not subset.empty else pd.Series(dtype=float)
            defined = values[np.isfinite(values)]
            row = ref._asdict()
            row.pop("Index", None)
            row.update(completed_seed_count=len(subset), defined_metric_count=len(defined),
                       required_seed_count=len(seeds), all_required_repeats_complete=len(defined) == len(seeds),
                       exact_published_result_replicated=False,
                       released_mean=float(defined.mean()), released_seed_sd=float(defined.std(ddof=1)),
                       status="complete_release_current_snapshot" if len(defined) == len(seeds) else "incomplete")
            if len(defined) < len(values):
                row["status"] = "undefined_metric_in_some_runs"
            if table == 2:
                row.update(sharpe_fields(row["released_mean"], row["released_seed_sd"],
                    ref.paper_daily_gross_sharpe_mean, ref.paper_seed_sd, len(defined) == len(seeds)))
            else:
                row["metric"] = "maximum_drawdown_unscaled"
            rows.append(row)
        outputs[table] = pd.DataFrame(rows)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, action="append")
    parser.add_argument("--baseline-oscar", choices=["oscar", "oscar_hard"], default="oscar_hard",
                        help="Paper Sec. 5 specifies hard support selection for Historic/PFL OSCAR; DFL always uses oscar")
    parser.add_argument("--output", type=Path, default=ROOT / "results/published_comparison")
    args = parser.parse_args()
    artifact_roots = args.artifacts or [ROOT / "artifacts", ROOT / "results/runs", ROOT / "artifacts_released"]
    protocol = json.loads((ROOT / "reproduction/protocol.json").read_text())
    seeds = list(map(int, protocol["seeds"]))
    reference_path = ROOT / "sources/paper_table1_reference.csv"
    reference = pd.read_csv(reference_path)
    assert len(reference) == 84
    hashes = {str(reference_path.relative_to(ROOT)): digest(reference_path),
              "reproduction/protocol.json": digest(ROOT / "reproduction/protocol.json"),
              "reproduction/compare_published.py": digest(Path(__file__))}
    panels = {market: panel_info(market, hashes) for market in PAPER_N}
    runs, issues = collect(artifact_roots, panels, hashes, protocol)
    comparison = compare(reference, runs, panels, seeds, args.baseline_oscar)
    errors = comparison_summary(comparison)
    appendix = additional_tables(runs, seeds, args.baseline_oscar)
    for table in (2, 3):
        p = ROOT / f"sources/paper_table{table}_reference.csv"
        hashes[str(p.relative_to(ROOT))] = digest(p)
    symbol_audit = ROOT / "sources/ftse_symbol_resolution_observation.json"
    if symbol_audit.exists():
        hashes[str(symbol_audit.relative_to(ROOT))] = digest(symbol_audit)
    reconciliation_html = ""
    reconciliation = ROOT / "sources/setting_reconciliation.json"
    if reconciliation.exists():
        history = json.loads(reconciliation.read_text())
        assert history["final_publication"]["hidden_dims"] == protocol["hidden_dims"]
        reconciliation_html = '''<h2>Final publication versus original submission</h2>
<p>The archived original submission uses 256/128 hidden layers and fixed training-history covariance.
The final published paper uses 512/256 and rolling covariance, and some table values and uncertainty
labels changed. The target here remains the final PMLR Tables 1–3. Original-submission settings or
numbers are not substituted to improve a match.</p>
<p><a href="../../sources/setting_reconciliation.json">Setting reconciliation</a> ·
<a href="../../sources/openreview-discussion.txt">Archived public author/review discussion</a> ·
<a href="../../sources/original-submission.pdf">Original submission</a>.</p>'''
        for name in ["sources/setting_reconciliation.json", "sources/openreview-discussion.txt",
                     "sources/openreview-links.json", "sources/original-submission.pdf",
                     "sources/original-submission.txt", "sources/anonymous-repository-status.txt"]:
            path = ROOT / name
            if path.exists():
                hashes[name] = digest(path)
    upstream = ROOT / "upstream"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    diff = subprocess.run(["git", "diff", "--exit-code", protocol["official_commit"], "--", "src"],
                          cwd=upstream, capture_output=True, text=True)
    assert commit == protocol["official_commit"] and diff.returncode == 0, "Author source changed"
    for path in (upstream / "src").glob("*.py"):
        hashes[str(path.relative_to(ROOT))] = digest(path)
    for name in ["reproduction/native_runtime.py", "reproduction/run_native.py",
                 "reproduction/resumable_sdr.py", "reproduction/verify_runtime.py",
                 "reproduction/study.py", "runtime/equivalence.json", "sources/environment.lock.txt"]:
        path = ROOT / name
        if path.exists():
            hashes[name] = digest(path)
    complete = int(comparison.all_required_repeats_complete.sum())
    main_runs = int(comparison.completed_seed_count.sum())
    manifest = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_commit": commit, "official_src_unchanged": True,
                "expected_table_cells": 84, "complete_release_cells": complete,
                "verified_individual_runs": main_runs, "expected_strategy_runs": 276,
                "verified_total_runs_including_sensitivities": len(runs),
                "table2_complete_cells": int(appendix[2].all_required_repeats_complete.sum()),
                "table2_expected_cells": 15,
                "table3_complete_cells": int(appendix[3].all_required_repeats_complete.sum()),
                "table3_expected_cells": 24,
                "exact_published_result_replicated": False,
                "artifact_roots": list(map(str, artifact_roots)), "seeds": seeds,
                "baseline_oscar_selector": args.baseline_oscar,
                "reporting_convention": REPORTING_CONVENTION,
                "trace_verification_issues": issues,
                "panels": {k: {n: v for n, v in p.items() if n != "test_dates"} if p else None for k, p in panels.items()},
                "input_sha256": hashes,
                "scope": "Released-code contemporary-data replay; unpublished original snapshot and experiment details remain unavailable"}
    dest = args.output
    atomic_text(dest / "table1_comparison.csv", comparison.to_csv(index=False))
    atomic_text(dest / "error_summary.csv", errors.to_csv(index=False))
    for table, frame in appendix.items():
        atomic_text(dest / f"table{table}_comparison.csv", frame.to_csv(index=False))
    atomic_text(dest / "verified_runs.csv", runs.to_csv(index=False))
    atomic_text(dest / "verification.json", json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    root_href = html.escape(os.path.relpath(ROOT, dest.resolve()), quote=True)
    reconciliation_html = reconciliation_html.replace('href="../../', f'href="{root_href}/')
    table1_display = comparison_display(comparison)
    table2_display = comparison_display(appendix[2], ["market", "alpha", "rho_percent"])
    doc = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Published Table 1 replication</title>
<style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:32px auto;padding:0 24px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}.scroll{{overflow:auto}}</style>
<h1>Published comparisons: annualized Sharpe interpretation</h1>
<p><strong>{html.escape(REPORTING_NOTE)}</strong></p>
<p>Verified {html.escape(manifest['checked_at_utc'])}. Completed cells: {complete}/84;
verified main runs: {main_runs}/276. Another {len(runs)-main_runs} verified sensitivity runs are retained.
Incomplete cells remain in the table.</p>
<p>The author implementation is unchanged at <code>{commit}</code>. Published settings are
512/256 hidden layers, batch64, lr=.001, alpha=.5, 100-session inputs and rolling covariance,
2016–2025 and an 80/20 split. Undisclosed choices are recorded in the protocol.
The original author CSV snapshot is unavailable. These are contemporary Yahoo/FinanceDataReader
counterparts; matching an asset count or a rounded statistic cannot establish exact numerical replication.</p>
{reconciliation_html}
<p>All error summaries use complete seed sets only. Agreement requires both the mean and seed SD
to match the displayed paper values within 0.0005. No retraining, costs, risk-free subtraction,
support changes or leverage caps are introduced by this reporting conversion.</p>
<div class="scroll">{errors.to_html(index=False, escape=True, na_rep='unavailable', float_format=lambda x:f'{x:.6f}')}</div>
<div class="scroll">{table1_display.to_html(index=False, escape=True, na_rep='pending', float_format=lambda x:f'{x:.6f}')}</div>
<h2>Table 2: alpha sensitivity</h2>
<p>Complete cells: {manifest['table2_complete_cells']}/15. All five planned seeds are required.</p>
<div class="scroll">{table2_display.to_html(index=False, escape=True, na_rep='pending', float_format=lambda x:f'{x:.6f}')}</div>
<h2>Table 3: maximum drawdown</h2>
<p>Complete cells: {manifest['table3_complete_cells']}/24. Drawdown is not annualized. It is recomputed from each source return
trace using the author's running-peak convention, which starts at the first realized wealth level.
Source insolvency cases retain undefined MDD and cannot silently count as completed numerical cells.</p>
<div class="scroll">{appendix[3].to_html(index=False, escape=True, na_rep='pending', float_format=lambda x:f'{x:.6f}')}</div>
<h2>Observed input discrepancy: FTSE symbol resolution</h2>
<p>The released generator retains valid bare Yahoo symbols before retrying failed symbols with .L.
The current FTSE CSV contains AAL; Yahoo identifies AAL as American Airlines (USD, Nasdaq), whereas
AAL.L identifies Anglo American (GBp, London). The frozen release-generated panel is therefore not
a verified London-only FTSE universe. This is a finding about the available release and current data,
not evidence about the unpublished author matrix. The literal replay remains labeled accordingly.</p>
<p><a href="{root_href}/sources/ftse_symbol_resolution_observation.json">Observed provider identities and source URLs</a>.</p>
<p>Korean counterpart inputs use a contemporary Wikipedia constituent list after the KRX endpoint
returned LOGOUT. The constituent receipt is retained with the panel audit. This fallback is not the
author's missing KRX export or stable-membership cohort; the return-generation procedure is separately recorded.</p>
<p>Paper/source gaps include the Historic mean definition, SD-relaxation cardinality matching,
MSE reduction, training-update semantics and the conic scaling bound. Baseline OSCAR here uses
the explicitly selected released variant <code>{args.baseline_oscar}</code>. The default is oscar_hard,
because Section 5 describes selection and re-optimization over the selected support for the conventional
OSCAR baseline. Soft oscar remains available as a separately labeled release-default sensitivity.
DFL continues to use the released differentiable oscar layer in both comparisons.
The source's full-panel scaler and risk-factor orientation are retained and disclosed rather than repaired.</p>
<p><a href="table1_comparison.csv">All 84 comparison cells</a> · <a href="verified_runs.csv">Verified runs</a> ·
<a href="error_summary.csv">Annualized and daily error summary</a> ·
<a href="table2_comparison.csv">Alpha sensitivity CSV</a> · <a href="table3_comparison.csv">Drawdown CSV</a> ·
<a href="verification.json">Source, data and trace hashes</a> · <a href="{root_href}/reproduction/protocol.json">Protocol</a> ·
<a href="https://proceedings.mlr.press/v306/jeon26a.html">Published paper</a> ·
<a href="https://github.com/feuerwerksh/Diffble-card-SR">Author code</a>.</p></html>'''
    atomic_text(dest / "report.html", doc)
    print(json.dumps({k: manifest[k] for k in ["complete_release_cells", "verified_individual_runs", "expected_table_cells",
                                             "official_src_unchanged", "exact_published_result_replicated"]}, indent=2))


if __name__ == "__main__":
    main()
