"""Recompute the fidelity audit from verified comparisons and controlled reruns.

Run compare_published.py for both baseline selectors first. This script never
changes financial inputs, trains a model, or fills missing published cells.
"""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import torch
from compare_published import REPORTING_CONVENTION, REPORTING_NOTE, SHARPE_FACTOR, comparison_summary, comparison_display

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/fidelity_audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_checks():
    checks = []
    for mode, name in [
        ("pfl_soft", "euro50_PFL_oscar_k10_actualk5_alpha0p0_seed16"),
        ("pfl_hard", "euro50_PFL_oscar_hard_k10_actualk5_alpha0p0_seed16"),
        ("dfl", "euro50_k10_actualk5_alpha0p5_seed16"),
    ]:
        original = ROOT / "artifacts_released" / name
        frozen = OUT / "direct_frozen" / name
        fresh = OUT / "direct_fresh" / name
        if not (frozen / "summary_metrics.csv").exists():
            checks.append({"method": mode, "status": "direct_frozen_not_completed"})
            continue
        a = pd.read_csv(original / "test_trace.csv")
        b = pd.read_csv(frozen / "test_trace.csv")
        if not a.date.equals(b.date):
            raise ValueError(f"Mismatched dates for {name}")
        model_name = "euro50_PFL_MSE_seed16" if mode.startswith("pfl") else name
        old_model = torch.load(ROOT / "artifacts_released" / model_name / "model.pt", weights_only=True)
        new_model = torch.load(OUT / "direct_frozen" / model_name / "model.pt", weights_only=True)
        record = dict(method=mode, seed=16, k=5, updates=300, status="direct_frozen_completed",
                      model_max_abs_error=max(float((old_model[k]-new_model[k]).abs().max()) for k in old_model),
                      trace_return_max_abs_error=float((a.ret_true_raw-b.ret_true_raw).abs().max()),
                      original_daily_sharpe=float(a.ret_true_raw.mean()/(a.ret_true_raw.std(ddof=1)+1e-12)),
                      direct_frozen_daily_sharpe=float(b.ret_true_raw.mean()/(b.ret_true_raw.std(ddof=1)+1e-12)))
        if (fresh / "summary_metrics.csv").exists():
            c = pd.read_csv(fresh / "test_trace.csv")
            if not b.date.equals(c.date):
                raise ValueError(f"Fresh date mismatch for {name}")
            record.update(status="both_completed",
                          direct_fresh_daily_sharpe=float(c.ret_true_raw.mean()/(c.ret_true_raw.std(ddof=1)+1e-12)),
                          fresh_minus_frozen_daily_sharpe=float(c.ret_true_raw.mean()/(c.ret_true_raw.std(ddof=1)+1e-12))-record["direct_frozen_daily_sharpe"])
        for key in list(record):
            if key.endswith("daily_sharpe"):
                record[key.replace("daily_sharpe", "annualized_sharpe")] = record[key] * SHARPE_FACTOR
        checks.append(record)
    return pd.DataFrame(checks)


def main():
    source = ROOT / "upstream"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    expected = json.loads((ROOT / "reproduction/protocol.json").read_text())["official_commit"]
    if commit != expected or subprocess.run(["git", "diff", "--quiet", expected, "--", "src"], cwd=source).returncode:
        raise RuntimeError("The controlled source is no longer the pinned unmodified author release")
    comparisons, summary = {}, []
    for selector in ("soft", "hard"):
        frame = pd.read_csv(OUT / f"{selector}_baseline/table1_comparison.csv")
        completed = frame[frame.all_required_repeats_complete].copy()
        comparisons[selector] = completed
        summary.extend(comparison_summary(frame).assign(baseline=selector).to_dict("records"))
    totals = pd.DataFrame(summary)
    dfl = comparisons["hard"].query("method == 'dfl'").copy()
    dfl["annualized_percent_difference_from_paper"] = 100*(dfl.released_annualized_gross_sr_mean/dfl.paper_reported_sharpe_mean-1)
    dfl["daily_percent_difference_from_paper"] = 100*(dfl.released_daily_gross_sr_mean/dfl.paper_reported_sharpe_mean-1)
    overall = totals.query("method == 'all'").set_index("baseline")
    pfl = totals.query("method == 'pfl' and baseline == 'hard'").iloc[0]
    controlled = direct_checks()
    if not controlled.status.eq("both_completed").all():
        raise RuntimeError("Finish all controlled reruns before publishing the completed audit")
    if not (controlled.model_max_abs_error.eq(0).all() and controlled.trace_return_max_abs_error.eq(0).all()):
        raise RuntimeError("Direct source differs from the wrapper; investigate before claiming equivalence")
    receipts = json.loads((ROOT / "sources/fidelity_audit/fresh_download/receipts.json").read_text())
    fresh_table = pd.DataFrame(receipts).reindex(columns=["market", "status", "paper_n", "n_assets", "shared_values_max_abs_change", "error"])
    findings = [
        ("Reporting convention: annualized interpretation", "requested interpretation; author convention unconfirmed",
         REPORTING_NOTE + f" For the {int(pfl.completed_cells)} completed hard-baseline PFL cells, mean absolute error is "
         f"{pfl.annualized_sharpe_mae:.6f} under this interpretation versus {pfl.daily_sharpe_mae:.6f} under the daily definition. "
         f"There are {int(pfl.annualized_mean_and_sd_matches)} annualized mean-and-SD matches at the paper's three-decimal precision. "
         "A smaller error is evidence of a scale discrepancy, not proof that the authors annualized their table."),
        ("1. Data cohort differs from the stated experiment", "paper versus downloader; exact input unknown",
         "Section 5 specifies stable index membership throughout January 2016–December 2025, omitting entrants and exits. "
         "The public Euro/FTSE downloader instead takes today's Wikipedia constituents and filters price availability; Nikkei uses a hard-coded list. "
         "Complete prices do not establish continuous index membership. The frozen panels contain 47/92/206/159 assets for Euro/FTSE/Nikkei/KOSPI, "
         "versus 47/93/208/162 in Table 1. At 20%, this also changes FTSE k from 19 to 18 and Nikkei k from 42 to 41. "
         "Even Euro's matching count does not validate its cohort: ARGX.BR and DBK.DE are present, while the index provider records Argenx and Deutsche Bank entering "
         "on 22 September 2025. They violate the paper's literal no-entry/no-exit rule. Source: upstream/src/data_generator.py:41, :48, :142; "
         "sources/paper.txt:435; sources/fidelity_audit/stoxx_2025_changes.txt. This proves a specification mismatch, not how many Sharpe units it causes."),
        ("2. FTSE input contains incorrect instrument identities", "released downloader defect",
         "FTSE100 first downloads bare symbols and only appends .L to failed/incomplete symbols. The frozen matrix retains AAL, identified by Yahoo as "
         "American Airlines (USD/Nasdaq), rather than AAL.L, Anglo American (GBp/London). A complete US history survives the filter. "
         "There are 27 bare symbols in the frozen panel; that count is not a claim that all 27 are wrong. Fresh retrieval even switches NWG to NWG.L. "
         "Source: upstream/src/data_generator.py:48; sources/ftse_symbol_resolution_observation.json. This is more than a missing-stock count, "
         "but its performance impact has not been isolated."),
        ("3. The previous report used the wrong conventional OSCAR variant", "reproduction comparison choice; corrected",
         "Section 5 defines conventional OSCAR as selecting a hard top-k support and re-optimizing only that support. The old primary comparison mapped "
         "Historic/PFL OSCAR to the released soft differentiable selector. The report now defaults to oscar_hard for those baselines; DFL still uses oscar. "
         "The soft comparison is retained as a sensitivity using the same available runs and reporting convention. "
         f"Current completed-cell annualized MAE is {overall.loc['soft','annualized_sharpe_mae']:.6f} for soft baselines and "
         f"{overall.loc['hard','annualized_sharpe_mae']:.6f} for hard baselines. "
         "This baseline choice is dictated by the paper's selected-support description, not the size of the error. "
         "Source: sources/paper.txt:469; reproduction/compare_published.py; hard_baseline and soft_baseline comparison CSVs."),
        ("4. Soft gate mass is not portfolio cardinality", "paper claim versus actual algorithm; shared with release",
         "DiffbleOSCAR multiplies the predicted mean by a soft gate, then solves an unrestricted N-variable portfolio problem. "
         "No constraint sets weights outside a selected support to zero. In the saved seed-16 10% DFL traces, all 412 Euro days hold 47 assets "
         "instead of 5; all 414 FTSE days hold 92 instead of 9; all 389 Nikkei days hold 206 instead of 21 (threshold |w|>1e-8). "
         "Gate sums remain approximately 5, 9, and 21. These are observed weights, not just a theoretical objection. "
         "Source: upstream/src/oscar.py:46; test_trace.csv in the corresponding artifacts. Hardening DFL would change its gradient and algorithm, "
         "so it is not silently substituted as an exact replay."),
        ("5. MSE changes the relative loss weight by N", "paper/release discrepancy",
         "The paper's Algorithm 1 and Eq. 14 average squared vector norms over samples. losses.MSE instead averages squared errors across assets too. "
         "At alpha=0.5, released loss is 0.5*DFL + 0.5*SSE/N, versus 0.5*DFL + 0.5*SSE in the paper. "
         "Thus the relative decision-to-prediction weight is multiplied by N: 47, 92, 206, or 159 in these inputs. "
         "Expressed only as a relative vector-loss mixture, the Euro setting corresponds to alpha=47/48, not 0.5. "
         "This algebra does not imply exact Adam equivalence after global loss scaling. Source: upstream/src/losses.py:10; sources/paper.txt:424."),
        ("6. Training duration and iteration semantics are not recovered", "paper/release discrepancy and undisclosed protocol",
         "Algorithm 1 iterates over mini-batches inside each epoch. The release performs exactly one randomly sampled batch per so-called epoch. "
         "With 1,947 Euro training windows and batch 64, a full pass would take 31 updates, not one; 300 full passes would take 9,300 updates "
         "rather than the replay's 300. The paper does not give an epoch/update count, so 9,300 is an illustration, not a recovered setting. "
         "The replay fixes the paper's 512/256 layers and batch 64, overriding release defaults 256/128 and 32. "
         "The five seed IDs, 300 updates, tanh, dropout 0.1, weight decay 1e-4, and top-k sigmoid steepness 5 come from release choices, "
         "not a complete published experiment configuration. Source: upstream/src/train.py:32; upstream/src/main.py:30; reproduction/protocol.json."),
        ("7. Optimization has two distinct specification problems", "one paper/release discrepancy; one shared mathematical defect",
         "The release imposes t>=1 whereas Eq. 5 states t>=0, changing the feasible set. Separately, both the printed formulation and release "
         "use ||L y|| although covariance=LL^T. Correct covariance risk is ||L^T y||. With L=[[2,0],[1,1]] and y=[1,0], "
         "the covariance variance is 4 but the released squared norm is 5. The orientation defect is not evidence that the wrapper altered the author code. "
         "Source: upstream/src/oscar.py:19; sources/paper.txt:326; structural_diagnostics.json."),
        ("8. Preprocessing and evaluation dates are not fully specified", "released behavior retained; paper does not resolve it",
         "StandardScaler fits the entire return panel before the train/test split, using test distribution information. Forecasts and covariance are "
         "in standardized asset units, while portfolio weights are applied to raw returns. The release also splits first and constructs 100-day windows "
         "separately, discarding the first 100 test dates rather than borrowing preceding training history. Euro's nominal test split starts 2023-12-28, "
         "but evaluation starts 2024-05-22. This is verified release behavior; the paper does not establish which preprocessing and boundary convention "
         "produced its tables. Source: upstream/src/data.py:55, :134; upstream/src/evaluate.py:62; structural_diagnostics.json."),
        ("9. Historic and other baselines are not fully aligned", "paper/release discrepancies and missing procedure",
         "Table 1 says Historic estimates use the full training set; historic_mu uses a changing 100-day window, including test-period history when "
         "evaluating test samples. Appendix B.2 says SD-relaxation adjusts its sparsity parameter until realized cardinality matches; the release uses "
         "fixed nominal k and leading-eigenvector rounding, without the matching search or nonzero threshold. Appendix B.3 describes a Sharpe-gradient "
         "step and top-k magnitudes; the executed mSSRM_PGA uses Qw-p and ranks signed entries. Section 5's positive-entry description also differs "
         "from the appendix's magnitude rule. A second revised routine exists but is not called. Source: upstream/src/problem.py:71; "
         "upstream/src/baseline.py:6, :58; sources/paper.txt:525, :883, :904. Inventing missing choices cannot establish exact reproduction."),
        ("10. Completion and attribution must be distinguished", "execution limit; no causal decomposition claimed",
         f"At this report's rebuild, {len(comparisons['hard'])}/84 Table 1 cells have all required repeats. "
         "The comparison CSV records completion per cell; background execution can continue independently. "
         "Missing cells are not assigned zero and partial seed averages are excluded from the error summary. "
         "There is no controlled ablation attributing the full published gap to one item above. Some small data revisions may be minor; the larger "
         "algorithm and protocol differences are not numerically minor by definition. Failure to reproduce is established; the precise cause of the "
         "authors' high Sharpe values is not established."),
    ]
    now = datetime.now(timezone.utc).isoformat()
    lead = (
        "The comparison now leads with the annualized-Sharpe interpretation and retains the paper's stated daily definition as a separate view. "
        "The mean discrepancy is smaller under annualization, but mean-and-SD agreement and exact experiment recovery must be assessed separately. "
        "This audit separates errors in the comparison, "
        "differences between paper and release, defects shared by paper and release, and information that cannot be recovered from the public specification. "
        "Annualization does not repair data or methodology discrepancies. Exact published-number reproduction remains unestablished."
    )
    conclusion = (
        "Completed here: new provider downloads using the unchanged release; all saved comparison traces rechecked; conventional OSCAR mapping corrected; "
        "full unmodified-source Euro seed-16 PFL and DFL runs on the frozen panel; matching full runs on the fresh Euro panel. "
        "Both frozen PFL and DFL match the earlier wrapper bit-for-bit in model tensors and realized-return traces for these tested runs. "
        "The subsequent PFL audit also independently retrains seed16 in all four markets and matches the cached parameters and all six traces per market exactly; "
        "its evidence is retained in results/pfl_replay/direct_checks.json. These checks do not certify every seed, device or nonfinite solver path. "
        "The fresh-data comparison is one controlled seed, not a five-seed replacement for Table 1. "
        "The controlled-runs table shows both daily and annualized changes with an identical fixed factor. "
        "These single-seed observations do not establish a five-seed match or quantify the effect of an unknown original cohort."
    )
    missing = (
        "To identify the original experiment, the remaining requirements are a reproducible constituent/price snapshot (or complete membership history "
        "and adjustment conventions), actual training duration and random seeds, and reconciliation of loss scaling, Historic estimates, optimization "
        "constraints, baseline implementation and SDR matching. The paper and public release do not uniquely determine these. "
        "Downloading the specified period is possible and was done; resolving conflicting experimental definitions requires additional evidence. "
        "No missing choices were fitted to test Sharpe and no run is labeled an exact published-result reproduction."
    )
    tables = [("Completed Table 1 error summary: annualized primary, daily reference", totals),
              ("All completed five-seed DFL cells", comparison_display(dfl)),
              ("Controlled unmodified-source runs (one seed; both scales)", controlled), ("Fresh download receipts", fresh_table)]
    text_parts = ["Sparse Tangent fidelity audit", f"Checked {now}; official commit {commit}", lead, REPORTING_NOTE]
    body = [f"<h1>Sparse Tangent fidelity audit</h1><p>{html.escape(now)}</p><p>{html.escape(lead)}</p><p><strong>{html.escape(REPORTING_NOTE)}</strong></p>"]
    for title, table in tables:
        text_parts.extend([title, table.to_string(index=False)])
        body.append(f"<h2>{html.escape(title)}</h2><div class='scroll'>{table.to_html(index=False, escape=True, float_format=lambda n:f'{n:.6f}')}</div>")
    text_parts += ["What the checks establish", conclusion]
    body.append(f"<h2>What the checks establish</h2><p>{html.escape(conclusion)}</p>")
    for title, kind, explanation in findings:
        text_parts += [title, kind, explanation]
        body.append(f"<h2>{html.escape(title)}</h2><p><strong>{html.escape(kind)}</strong></p><p>{html.escape(explanation)}</p>")
    text_parts += ["Exact reproduction status", missing,
                   "Paper: https://proceedings.mlr.press/v306/jeon26a.html",
                   "Author release: https://github.com/feuerwerksh/Diffble-card-SR",
                   "Membership evidence: https://stoxx.com/stoxx-blue-chip-indices-composition-changes-sep-1-2025/",
                   "Repeat the direct check in a new directory: .venv/bin/python reproduction/run_direct_audit.py --data-root data/daily --output results/new_direct_audit",
                   "Repeat fresh downloads in a new directory: .venv/bin/python reproduction/download_specification.py --output sources/new_download",
                   "Rebuild this report: .venv/bin/python reproduction/audit_fidelity.py"]
    body.append(f"<h2>Exact reproduction status</h2><p>{html.escape(missing)}</p>")
    body.append('<p><a href="audit.txt">Detailed text</a> · <a href="dfl_comparison.csv">DFL CSV</a> · '
                '<a href="controlled_runs.csv">Controlled runs CSV</a> · <a href="hard_baseline/table1_comparison.csv">All 84 cells, hard baseline</a> · '
                '<a href="soft_baseline/table1_comparison.csv">Soft baseline sensitivity</a> · <a href="../../sources/fidelity_audit/fresh_download/receipts.json">Download receipts</a></p>'
                '<p><a href="https://proceedings.mlr.press/v306/jeon26a.html">Paper</a> · '
                '<a href="https://github.com/feuerwerksh/Diffble-card-SR">Author release</a> · '
                '<a href="https://stoxx.com/stoxx-blue-chip-indices-composition-changes-sep-1-2025/">Index-provider membership evidence</a></p>')
    (OUT / "audit.txt").write_text("\n\n".join(text_parts)+"\n")
    (OUT / "report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Sparse Tangent fidelity audit</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#172033}h2{margin-top:36px}table{border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}a{color:#1259a7}</style>'+"\n".join(body)+"</html>")
    for name, table in [("dfl_comparison", dfl), ("error_summary", totals), ("controlled_runs", controlled)]:
        table.to_csv(OUT / f"{name}.csv", index=False)
    evidence = [ROOT / "sources/paper.pdf", ROOT / "reproduction/protocol.json",
                ROOT / "sources/fidelity_audit/fresh_download/receipts.json",
                OUT / "soft_baseline/table1_comparison.csv", OUT / "hard_baseline/table1_comparison.csv"]
    evidence += list((source / "src").glob("*.py"))
    evidence += list((OUT / "direct_frozen").glob("*/test_trace.csv"))
    evidence += list((OUT / "direct_fresh").glob("*/test_trace.csv"))
    evidence += list((OUT / "direct_frozen").glob("*/model.pt"))
    evidence += list((OUT / "direct_fresh").glob("*/model.pt"))
    evidence += [ROOT / "data/daily/euro50_160101_251231.csv",
                 ROOT / "sources/fidelity_audit/fresh_download/data/daily/euro50_160101_251231.csv",
                 ROOT / "reproduction/audit_fidelity.py",
                 ROOT / "reproduction/compare_published.py",
                 ROOT / "results/pfl_replay/direct_checks.json",
                 ROOT / "reproduction/run_direct_audit.py",
                 ROOT / "reproduction/download_specification.py",
                 OUT / "direct_fresh/execution.json"]
    manifest = dict(checked_at_utc=now, official_commit=commit, official_src_unchanged=True,
                    exact_published_results_reproduced=False,
                    reporting_convention=REPORTING_CONVENTION,
                    controlled_runs_complete=bool(controlled.status.eq("both_completed").all()),
                    findings=[dict(title=a, category=b, detail=c) for a,b,c in findings],
                    hashes={str(p.relative_to(ROOT)):sha(p) for p in evidence})
    (OUT / "audit.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(totals.to_string(index=False))
    print(controlled.to_string(index=False))


if __name__ == "__main__":
    main()
