#!/usr/bin/env python3
"""Bind the local computational report to sources without claiming human review."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    definitions = [
        ("E001", "A Linearly Convergent Proximal Subgradient Algorithm for Sparse Portfolio Optimization with Transaction Cost", "sources/paper.pdf", "Sections 3-4; Appendix B, Tables 4-7",
         "https://openreview.net/forum?id=yZAo4TPqhE"),
        ("E002", "Pinned author source", "upstream/STCO-code", "All .m functions; git commit in metadata",
         "https://github.com/Ting221/STCO/tree/a2e04730cbdfb6ccb0017456d4f835a4d7446ec3"),
        ("E003", "Computed market results", "results/main/metrics.csv", "All rows", ""),
        ("E004", "Table 5/7 comparisons", "results/main/paper_comparison.csv", "24 proposed-method rows", ""),
        ("E005", "Numerical validation", "results/validation/summary.json", "All fields and linked audit CSVs", ""),
        ("E006", "Test result", "results/tests.log", "Final pytest summary", ""),
        ("E007", "Backtest implementation", "stco/market.py", "Prediction, accounting, baselines, metrics", ""),
        ("E008", "Solver implementation", "stco/solver.py", "Prox, ADMM reference, PSGA and event skipping", ""),
        ("E009", "Experiment metadata", "results/main/metadata.json", "Parameters, shape, hashes, environment", ""),
        ("E010", "Risk comparisons", "results/main/paper_risk_comparison.csv", "24 Table 6 rows", ""),
        ("E011", "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents", "", "arXiv metadata; current v2, revised 2026-09-02",
         "https://arxiv.org/abs/2609.00065"),
        ("E012", "Benchmark comparisons", "results/main/baseline_comparison.csv", "36 Table 5 benchmark cells", ""),
        ("E013", "Display generation", "scripts/plot_results.py", "All figure transformations", ""),
        ("E014", "Parameter sensitivity", "results/sensitivity/paper_comparison.csv", "Tables 2 and 3, 32 cells", ""),
    ]
    sources = []
    for eid, title, path, locator, url in definitions:
        sources.append({"evidence_id": eid, "title": title,
                        "authors": ["Xiaoting Yao", "Na Zhang"] if eid == "E001" else [],
                        "year": 2026, "source_type": "other", "confidentiality": "public",
                        "identifiers": {"url": url, "doi": "", "pmid": "", "pmcid": "", "isbn": ""},
                        "locator": (path+": " if path else "")+locator,
                        "verification": {"source_opened": True, "status": "unverified",
                                         "verified_by": "", "verified_on": ""}})
    manifest = {"schema_version": "1.0", "sources": sources}
    (ROOT/"sources/source_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    sections = {
        "OUTCOME": ["E003", "E004"],
        "COMPLETED EXPERIMENTS": ["E003", "E009", "E014"],
        "TABLE 5: REPORTED VERSUS COMPUTED FINAL WEALTH": ["E003", "E004"],
        "SOURCE AND SPECIFICATION DISCREPANCIES": ["E001", "E002", "E012"],
        "ALGORITHM AND ACCOUNTING": ["E001", "E007", "E008", "E009"],
        "NUMERICAL VERIFICATION": ["E005", "E006", "E008"],
        "SPARSITY AND CONVERGENCE LIMITS": ["E001", "E003", "E005"],
        "RISK METRICS AND BENCHMARK COVERAGE": ["E001", "E007", "E010"],
        "REPRODUCIBILITY": ["E009", "E013"],
        "PROCEDURAL SOFTWARE REFERENCE": ["E011"],
        "CONCLUSION": ["E001", "E002", "E004", "E005"],
    }
    records, annotated = [], []
    section, evidence = "Paper and status", ["E001", "E002", "E009"]
    for line_number, line in enumerate((ROOT/"REPORT.txt").read_text().splitlines(), 1):
        if line in sections:
            section, evidence = line, sections[line]
        if not line.strip() or line in sections or set(line) <= {"="}:
            annotated.append(line)
            continue
        cid = f"C{len(records)+1:03d}"
        digest = hashlib.sha256(" ".join(line.split()).encode()).hexdigest()
        records.append({"claim_id": cid, "section": section, "claim_kind": "computational_record",
                        "claim_text_sha256": digest, "evidence_ids": ",".join(evidence),
                        "verification_status": "unverified", "uncertainty": "human_review_not_recorded",
                        "analysis_intent": "descriptive", "report_line": line_number})
        annotated.append(line+f" [claim:{cid}] [evidence:{','.join(evidence)}]")
    with (ROOT/"sources/claims.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    (ROOT/"sources/REPORT.evidence.txt").write_text("\n".join(annotated)+"\n")
    registry = {
        "human_review": "not recorded", "submission_ready": False,
        "agent_check": "Sources opened/inspected by Codex; no accountable human verification is recorded.",
        "source_locations": {eid: path for eid, title, path, locator, url in definitions},
        "ai_use": "Codex implemented code, ran checks, generated plots, and drafted the local report.",
        "authorship": "No human manuscript authorship, declarations, funding, or approval inferred.",
        "reporting_guideline": {"status": "not_applicable",
            "rationale": "Deterministic optimization-paper reproduction; no clinical, animal, epidemiological, or systematic-review study."},
        "report_claim_mapping": "claims.csv; line numbers refer to REPORT.txt",
        "annotated_draft": "REPORT.evidence.txt",
        "figures": {
            "cost_sensitivity": {"data": ["results/main/metrics.csv", "sources/paper_targets.csv"],
                "evidence": ["E001", "E003", "E004", "E013"],
                "alt": "Four panels show final wealth versus cost. Independent STCO results differ from reported markers; window choice affects STCO2."},
            "wealth_paths": {"data": ["results/main/runs/*.npz"],
                "evidence": ["E003", "E007", "E013"],
                "alt": "Four market panels compare cumulative self-financing wealth at a 0.25 percent cost; trajectories use previously observed prices only."},
            "paper_agreement": {"data": ["results/main/paper_comparison.csv"],
                "evidence": ["E004", "E013"],
                "alt": "Twenty-four bars give log10 reproduced divided by reported wealth. Zero means exact agreement; both positive and negative deviations occur."},
            "convergence": {"data": ["results/validation/convergence_trace.csv"],
                "evidence": ["E005", "E008", "E013"],
                "alt": "On one direct PSGA problem the objective decreases and Euclidean distance reaches numerical zero. This is not a proof of the R-linear theorem."},
        },
        "limitations": ["Main wealth table not replicated", "Author source mismatch",
                         "STCO2 window unstated", "Only three separate benchmark methods implemented",
                         "No general rate proof performed", "No accountable human review recorded"],
    }
    (ROOT/"sources/evidence_registry.json").write_text(json.dumps(registry, indent=2)+"\n")
    # The check validates identity/binding only, not scientific truth.
    report_lines = (ROOT/"REPORT.txt").read_text().splitlines()
    for r in records:
        actual = hashlib.sha256(" ".join(report_lines[r["report_line"]-1].split()).encode()).hexdigest()
        assert actual == r["claim_text_sha256"]
    print(f"Bound {len(records)} report lines to evidence; human verification remains unrecorded.")


if __name__ == "__main__":
    main()
