"""Report literal published-method attempts without silently repairing failures."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from compare_published import REPORTING_CONVENTION, REPORTING_NOTE, SHARPE_FACTOR, format_mean_sd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results/paper_method"


def main():
    manifest=json.loads((OUT/"study/manifest.json").read_text())
    if hashlib.sha256((ROOT/"reproduction/paper_method.py").read_bytes()).hexdigest()!=manifest["code_sha256"]:
        raise ValueError("Method implementation differs from the declared run identity")
    if hashlib.sha256((ROOT/"reproduction/paper_method_protocol.json").read_bytes()).hexdigest()!=manifest["protocol_sha256"]:
        raise ValueError("Protocol differs from the declared run identity")
    reference=pd.read_csv(ROOT/"sources/paper_table1_reference.csv")
    source=pd.read_csv(ROOT/"results/published_comparison/table1_comparison.csv")
    protocol=json.loads((ROOT/"reproduction/paper_method_protocol.json").read_text())
    status_path=OUT/"study/status.json"
    process_records=json.loads(status_path.read_text()).get("results",[]) if status_path.exists() else []
    rows=[];hashes={};panel_hashes={}
    for job in manifest["jobs"]:
        name=f'{job["market"]}_{job["method"]}_rho{job["rho"]}_seed{job["seed"]}_epochs{job["epochs"]}'
        path=OUT/"study"/name/"result.json"
        row={**job,"status":"pending","result_path":str(path.relative_to(ROOT))}
        if path.exists():
            result=json.loads(path.read_text())
            # A stale run may not quietly join a new protocol or code version.
            if result["source_sha256"]!=manifest["code_sha256"] or result["protocol_sha256"]!=manifest["protocol_sha256"]:
                raise ValueError(f"Experiment identity mismatch: {path}")
            row.update(result)
            data_path=Path(result["data_path"])
            if str(data_path) not in panel_hashes:
                panel_hashes[str(data_path)]=hashlib.sha256(data_path.read_bytes()).hexdigest()
            if panel_hashes[str(data_path)]!=result["data_sha256"]:
                raise ValueError("Financial input changed after the experiment")
            hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
            trace_path=path.parent/"test_trace.csv"
            if result["status"].startswith("completed"):
                trace=pd.read_csv(trace_path)
                panel=pd.read_csv(result["data_path"],index_col=0)
                expected_dates=panel.index[int(.8*len(panel)):].astype(str).tolist()
                if trace.date.astype(str).tolist()!=expected_dates:
                    raise ValueError(f"Missing or reordered chronological test targets: {trace_path}")
                expected_updates=0 if job["method"]=="historic" else job["epochs"]*((result["train_windows"]+63)//64)
                if result["completed_updates"]!=expected_updates:
                    raise ValueError("Completed run did not traverse all declared full epochs")
                if not np.isfinite(trace["return"]).all() or len(trace)!=result["test_days"]:
                    raise ValueError(f"Invalid or truncated trace: {trace_path}")
                sr=trace["return"].mean()/(trace["return"].std(ddof=1)+1e-12)
                np.testing.assert_allclose(sr,result["daily_sharpe_diagnostic"],rtol=1e-10,atol=1e-12)
                row["annualized_sharpe_diagnostic"] = float(sr * SHARPE_FACTOR)
                budget_valid=bool((trace.budget_error<=1e-3).all())
                if budget_valid!=result["budget_valid_all_test_days"]:
                    raise ValueError("Budget validity differs from saved trace")
                if not budget_valid and result["valid_daily_sharpe"] is not None:
                    raise ValueError("Invalid-budget run was assigned a valid Sharpe")
                negative_scale=trace.t.lt(-1e-8)
                if "initial_t" in trace:
                    negative_scale |= trace.initial_t.lt(-1e-8)
                row["negative_scale_test_days"]=int(negative_scale.sum())
                conic_invalid=bool((trace.conic_risk_excess>1e-4).any())
                gate_invalid=bool((trace.gate_mass_error>1e-4).any()) if "gate_mass_error" in trace else False
                row["numerically_valid_all_test_days"]=budget_valid and not negative_scale.any() and not conic_invalid and not gate_invalid
                row["claimed_cardinality_satisfied"]=bool((trace["holdings_1e-8"]<=result["k"]).all())
                hashes[str(trace_path.relative_to(ROOT))]=hashlib.sha256(trace_path.read_bytes()).hexdigest()
        else:
            for record in process_records:
                if (record.get("market"),record.get("method"),record.get("rho"),record.get("seed")) == (job["market"],job["method"],job["rho"],job["seed"]):
                    if record.get("status") in ["process_failed","interrupted_requires_review"]:
                        row.update(record)
        rows.append(row)
    runs=pd.DataFrame(rows)
    summary=[]
    for method in ["historic","pfl","dfl"]:
        for rho in [10,15,20]:
            selected=runs[(runs.market=="euro50")&(runs.method==method)&(runs.rho==rho)]
            required=1 if method=="historic" else 5
            if len(selected)!=required or selected.seed.duplicated().any():
                raise ValueError("Unexpected study schedule or duplicate seed")
            done=selected[selected.status.str.startswith("completed")]
            ref=reference[(reference.market=="euro50")&(reference.method==method)&(reference.optimizer=="oscar")&(reference.rho_percent==rho)].iloc[0]
            previous=source[(source.market=="euro50")&(source.method==method)&(source.paper_optimizer=="oscar")&(source.rho_percent==rho)].iloc[0]
            record=dict(method=method,rho_percent=rho,required_runs=required,completed_runs=len(done),
                        failed_runs=int(selected.status.isin(["failed","process_failed","interrupted_requires_review"]).sum()),
                        paper_mean=float(ref.paper_daily_gross_sharpe_mean),paper_seed_sd=float(ref.paper_seed_sd),
                        release_replay_mean=float(previous.released_daily_gross_sr_mean),
                        release_replay_annualized_mean=float(previous.released_annualized_gross_sr_mean),
                        release_replay_annualized_seed_sd=float(previous.released_annualized_seed_sd),
                        status="pending_or_failed",literal_mean_diagnostic=None,literal_seed_sd_diagnostic=None,
                        literal_annualized_mean_diagnostic=None,literal_annualized_seed_sd_diagnostic=None,
                        primary_comparison="annualized_sharpe_interpretation",
                        all_runs_budget_valid=False,exact_published_results_reproduced=False)
            if len(done)==required:
                values=done.daily_sharpe_diagnostic.astype(float)
                valid=bool(done.budget_valid_all_test_days.all())
                numerical_valid=bool(done.numerically_valid_all_test_days.all())
                record.update(status="completed_numerically_valid" if numerical_valid else "completed_with_numerical_violations",
                              literal_mean_diagnostic=float(values.mean()),
                              literal_seed_sd_diagnostic=0. if required==1 else float(values.std(ddof=1)),
                              literal_annualized_mean_diagnostic=float(values.mean()*SHARPE_FACTOR),
                              literal_annualized_seed_sd_diagnostic=0. if required==1 else float(values.std(ddof=1)*SHARPE_FACTOR),
                              literal_annualized_minus_published=float(values.mean()*SHARPE_FACTOR-ref.paper_daily_gross_sharpe_mean),
                              all_runs_budget_valid=valid,
                              all_runs_numerically_valid=numerical_valid,
                              numerically_invalid_runs=int((~done.numerically_valid_all_test_days.astype(bool)).sum()),
                              negative_scale_test_days=int(done.negative_scale_test_days.sum()),
                              claimed_cardinality_satisfied=bool(done.claimed_cardinality_satisfied.all()),
                              budget_invalid_runs=int((~done.budget_valid_all_test_days.astype(bool)).sum()),
                              holdings_min=int(done.test_holdings_min.min()),holdings_max=int(done.test_holdings_max.max()),
                              test_budget_error_max=float(done.test_budget_error_max.max()),
                              literal_minus_published=float(values.mean()-ref.paper_daily_gross_sharpe_mean))
            summary.append(record)
    table=pd.DataFrame(summary)
    summary_view=table[["method","rho_percent"]].rename(columns={"method":"Method","rho_percent":"Cardinality (%)"})
    for label,mean,sd in [("Paper reported mean ± SD","paper_mean","paper_seed_sd"),
                          ("Release annualized mean ± SD","release_replay_annualized_mean","release_replay_annualized_seed_sd"),
                          ("Literal annualized diagnostic ± SD","literal_annualized_mean_diagnostic","literal_annualized_seed_sd_diagnostic"),
                          ("Literal daily diagnostic ± SD","literal_mean_diagnostic","literal_seed_sd_diagnostic")]:
        summary_view[label]=[format_mean_sd(m,s) for m,s in zip(table[mean],table[sd])]
    summary_view["Status"]=table.status
    summary_view["Completed / required"]=table.completed_runs.astype(str)+"/"+table.required_runs.astype(str)
    breadth=runs[runs.market!="euro50"].reindex(columns=["market","method","seed","epochs","status","annualized_sharpe_diagnostic","daily_sharpe_diagnostic","numerically_valid_all_test_days","test_holdings_min","test_holdings_max","error"])
    completed=int(runs.status.str.startswith("completed").sum());failed=int(runs.status.isin(["failed","process_failed","interrupted_requires_review"]).sum())
    now=datetime.now(timezone.utc).isoformat()
    paragraphs=[
        ("Annualized reporting interpretation", REPORTING_NOTE + " Literal-method values remain diagnostics when budget, conic feasibility or cardinality checks fail. "
         "Annualization does not turn an invalid portfolio into a valid result. Columns literal_mean_diagnostic and literal_seed_sd_diagnostic retain daily units in the CSV."),
        ("What soft OSCAR means",
         "Predict expected returns; solve the full-universe portfolio; score each component by abs(L.T@w); form sigmoid gates "
         "g_i=sigmoid(beta*(score_i+shift)) with sum(g)=k; multiply predicted returns by these gates; re-optimize over all N assets. "
         "For example, gates [0.8,0.7,0.3,0.2] sum to 2 while all four entries remain nonzero. "
         "Even setting a forecast to zero does not force its holding to zero when covariances permit hedging. With covariance [[1,.5],[.5,1]] "
         "and masked mean [1,0], the conventional tangency weights are [2,-1]. Hard OSCAR instead restricts the final optimization variables "
         "to an actual selected subset. Algorithm 1 in this paper explicitly uses the soft pipeline for DFL."),
        ("What was implemented from the publication",
         "Algorithm 1 and Eq.14 use the same realized-return target in both terms: 0.5*(-y dot w)+0.5*sum((prediction-y)^2), "
         "averaged over the batch. Every training sample is visited once in each full shuffled epoch, batch size 64. "
         "The MLP has hidden widths 512/256 and learning rate .001. The printed right-hand Eq.5 uses norm(L@y)<=1, sum(y)=t, t>=0. "
         "Eq.12 uses sigmoid gates and 32 bisections. Rolling covariance uses the preceding 100 observations, 10% identity shrinkage, "
         "and jitter only if factorization fails. Historic uses the full training mean; conventional Historic/PFL OSCAR restricts the selected support. "
         "Saved returns and source metrics remain daily raw simple returns. Only the comparison reports annualize Sharpe, with no new training or test-driven parameter selection."),
        ("Numerically equivalent conic formulation",
         "Direct raw-unit conic coordinates gave SCS an inaccurate-unbounded status before the pilot's first update, even though "
         "the positive-definite covariance makes the problem bounded. The production implementation uses q=F*y, a=F^-T*1, b=F^-T*mu: "
         "maximize (b/||b||).q, subject to ||q||<=1 and (a/||a||).q>=0. Recover y=F^-1*q and t=sum(y). "
         "Positive normalization leaves the feasible set and optimizer unchanged. F=L reproduces printed Eq.5; F=L.T represents the actual "
         "covariance risk in Eq.1 for conventional OSCAR. Tests compare the solutions and gradients with the unpreconditioned problem and an analytic solution. "
         "This does not replace t>=0 with t>=1."),
        ("Choices that the paper does not determine",
         "The primary trial is five full epochs, not an asserted published epoch count. Seeds 16/37/47/214/1947 and tanh are inherited from the release. "
         "There is no dropout, weight decay or clipping in this literal-update trial. Plain SGD follows the displayed theta <- theta-eta*gradient line; "
         "the paper does not identify the actual experimental optimizer. No return standardization is applied, so inputs, targets, covariance and "
         "decision loss share raw-return units. Test targets use their immediately preceding 100-day history, including pre-split history. "
         "The unspecified gate steepness is set to 5 and the recovery offset to 1e-8, inherited from the release. "
         "The pinned numerical libraries are retained; solver tolerance is 1e-6 and maximum iterations 10000. All choices are recorded in paper_method_protocol.json. "
         "These assumptions prevent an honest claim that the complete original experiment has been recovered."),
        ("Conflicts that cannot be satisfied simultaneously",
         "When covariance=L L.T, the covariance risk is norm(L.T@y), whereas the printed right-hand Eq.5 uses norm(L@y). "
         "The literal DFL trial preserves the printed equation. Soft mass k does not imply exactly k holdings; no hard or straight-through DFL is inserted. "
         "The allowed t=0 boundary makes y/t singular. Adding the offset in B.1 makes division finite but can violate sum(w)=1. "
         "A two-asset unit-covariance example with mean [1,-2] has its conic optimum at t=0. The tests explicitly exercise this case. "
         "Completed traces report holdings, gate-mass errors, near-zero scales and budget errors. A run with any test budget error above 1e-3 "
         "has no valid_daily_sharpe in the raw result; its diagnostic Sharpe remains visible rather than being dropped. "
         "The report additionally flags t<-1e-8 in either recorded DFL stage or the final baseline stage, conic risk excess>1e-4, or gate-mass error>1e-4. "
         "These are declared numerical-audit thresholds, not recovered author settings. Numerically valid does not mean sparse: actual holdings are checked separately."),
        ("Scope and interpretation",
         "The Euro study includes five seeds at each of 10%, 15%, 20% for DFL and PFL, plus deterministic Historic results. "
         "The other markets receive one-seed, one-epoch breadth checks, which cannot count as five-seed Table1 reproductions. "
         "All attempts use the frozen available panels to isolate implementation changes; the original stable-membership cohort is not established. "
         "Euro evaluates 512 test dates, versus 412 in the release's split-before-window pipeline. Several methodological choices change together, "
         "so this is not a causal ablation of any one defect. SD-relaxation's unreleased matching search and the inconsistent mSSRM descriptions "
         "are not silently reconstructed or counted as completed methods."),
    ]
    lead=f"Published-method attempt: {completed} completed, {failed} failed, {len(runs)-completed-failed} pending of {len(runs)} planned runs. Exact published experiment recovered: false."
    text=["Sparse Tangent: literal published-method attempt",now,lead]
    body=[f"<h1>Sparse Tangent: published-method attempt</h1><p>{html.escape(now)}</p><p><strong>{html.escape(lead)}</strong></p>"]
    for title,paragraph in paragraphs:
        text += [title,paragraph]
        body += [f"<h2>{html.escape(title)}</h2><p>{html.escape(paragraph)}</p>"]
    for title,frame in [("Euro: diagnostic annualized Sharpe, with daily reference",summary_view),("Other markets: single-seed one-epoch diagnostics",breadth)]:
        text += [title,frame.to_string(index=False)]
        body += [f"<h2>{html.escape(title)}</h2><div class='scroll'>{frame.to_html(index=False,na_rep='pending / unavailable',float_format=lambda x:f'{x:.6f}')}</div>"]
    gates_path=OUT/"first_test_gate_diagnostics.json"
    if gates_path.exists():
        gates=pd.DataFrame(json.loads(gates_path.read_text()))
        text += ["First-test-day gate diagnostic (seed16 only)",gates.to_string(index=False)]
        body += ["<h2>First-test-day gates, seed 16 only</h2><p>These are individual-date diagnostics, not an average over the test period.</p>",
                 f"<div class='scroll'>{gates.to_html(index=False,float_format=lambda x:f'{x:.6f}')}</div>"]
    body.append('<p><a href="comparison.csv">Comparison CSV</a> · <a href="runs.csv">Every run and diagnostic</a> · '
                '<a href="../../reproduction/paper_method_protocol.json">Specification and assumptions</a> · '
                '<a href="../../reproduction/paper_method.py">Implementation</a> · <a href="study/manifest.json">Predeclared schedule</a> · '
                '<a href="../../sources/paper.pdf">Published PDF</a> · <a href="../fidelity_audit/report.html">Earlier release audit</a></p>')
    (OUT/"report.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Published-method reproduction attempt</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#182236}h2{margin-top:32px}table{font-size:13px;border-collapse:collapse}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}.scroll{overflow:auto}</style>'+"\n".join(body)+"</html>")
    (OUT/"report.txt").write_text("\n\n".join(text)+"\n")
    runs.to_csv(OUT/"runs.csv",index=False);table.to_csv(OUT/"comparison.csv",index=False)
    (OUT/"verification.json").write_text(json.dumps({"checked_utc":now,"completed":completed,"failed":failed,"expected":len(runs),
        "exact_published_experiment_recovered":False,"source_hash":manifest["code_sha256"],"protocol_hash":manifest["protocol_sha256"],
        "reporting_convention":REPORTING_CONVENTION,
        "reporter_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_reporting_sha256":hashlib.sha256((ROOT/"reproduction/compare_published.py").read_bytes()).hexdigest(),
        "panel_hashes":panel_hashes,"input_hashes":hashes},indent=2))
    print(lead);print(summary_view.to_string(index=False))


if __name__=="__main__":
    main()
