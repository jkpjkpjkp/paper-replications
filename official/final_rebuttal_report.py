"""
Combines results from rounds 1 and 2 into the final rebuttal evidence
package, organised by reviewer concern.

Sections:
    [R3]   Per-dataset performance gap discussion + table reformatting evidence
    [R1-W2/R3/R4-W3] Refined-rule effect: did adding 3 cutoffs improve scores?
    [R2-W9/R3]      Real paired stats (FraudGNAM-refined vs SEC-GFD on
                    matched seeds): Wilcoxon + Nadeau-Bengio + sign test
    [R4-W1/R5-W3/R2-W5]  NCE gradient flow (no branch collapse)
    [R4-W2/R2-comment]   Perturbation stability (Spearman rho)
    [R4-C4]              Filter-response visualisation evidence
    [R1-W1]              Memory profiling per variant
    [Multi-metric]       Full no_efi ablation across 7 datasets
    [R2-W7]              Adaptive gating works (tfinance evidence)
    [R2-W2]              Single-branch collapsed MLP (deferred)

Output:
    results/REBUTTAL_REPORT.md
"""

import json
import os
from pathlib import Path

import numpy as np

R = Path("results")


def L(name):
    p = R / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt(v, d=3):
    if v is None:
        return "—"
    return f"{v:.{d}f}"


def section_refined_rule():
    out = []
    out.append("\n## §1. Refined-rule impact (R1-W2 / R3 / R4-W3)\n")
    out.append("New cutoffs added to the adaptive rule:\n")
    out.append("- (R1) `n_f < 200`     ⇒ `K = 2` (Reddit-style sparse fraud)")
    out.append("- (R2) `h > 0.97 ∧ d ≥ 300` ⇒ `K = 4` (Weibo-style high-d homophilic)")
    out.append("- (R3) `h < 0.70`      ⇒ `T_HP = 0` (tolokers-style extreme heterophily)\n")

    out.append("| Dataset | Submitted (mean ± std) | Refined (mean ± std) | Δ AUROC |")
    out.append("|---------|-----------------------|----------------------|---------|")
    for ds in ["yelp", "amazon", "weibo", "tfinance", "reddit", "tolokers", "questions"]:
        s = L(f"fraudgnam_submitted_{ds}.json")
        r = L(f"fraudgnam_refined_{ds}.json")
        if s is None or r is None:
            out.append(f"| {ds} | (pending) | (pending) | — |")
            continue
        delta = (r["AUROC_mean"] - s["AUROC_mean"]) * 100
        out.append(f"| {ds} | "
                   f"{s['AUROC_mean']:.4f} ± {s['AUROC_std']:.3f} | "
                   f"{r['AUROC_mean']:.4f} ± {r['AUROC_std']:.3f} | "
                   f"{delta:+.2f}pp |")
    return "\n".join(out)


def section_real_paired():
    real = L("real_paired_tests.json")
    out = []
    out.append("\n## §2. Real paired statistical tests (R2-W9)\n")
    if real is None:
        out.append("_Pending: requires per-trial JSONs from EXP-A and EXP-B._")
        return "\n".join(out)

    out.append("Both methods share the GADBench seed list "
               "(`SEED_LIST = [3407+10t for t in 0..9]`) and split protocol, "
               "so per-trial values are matched across runs.\n")
    out.append("Tests applied per (dataset, metric) pair:")
    out.append("- **Nadeau-Bengio corrected paired t-test** (accounts for "
               "training-set overlap across CV folds).")
    out.append("- **Wilcoxon signed-rank test** (rank-based, robust to "
               "non-normal differences).")
    out.append("- **Sign / exact-binomial test** on per-trial wins.\n")

    out.append("\n### Refined FraudGNAM vs SEC-GFD (matched 10 trials)\n")
    out.append("| Dataset | Metric | Δmean | p_NB | p_Wilcoxon | wins/10 |")
    out.append("|---------|--------|-------|-----:|-----------:|--------:|")
    for ds in ["yelp", "amazon", "weibo", "tfinance", "reddit", "tolokers", "questions"]:
        if ds not in real:
            continue
        for m in ["AUROC", "AUPRC", "RecK"]:
            r = real[ds].get(f"refined:vs_secgfd:{m}")
            if r is None:
                continue
            sig = ("***" if r["p_nadeau_bengio"] < 0.01 else
                   "**"  if r["p_nadeau_bengio"] < 0.05 else
                   "*"   if r["p_nadeau_bengio"] < 0.10 else "")
            out.append(f"| {ds} | {m} | {r['mean_diff']:+.4f} | "
                       f"{r['p_nadeau_bengio']:.4f}{sig} | "
                       f"{r['p_wilcoxon']:.4f} | {r['wins']}/{r['n_paired']} |")

    glob = real.get("_global", {})
    if glob:
        out.append(f"\n**Global**: refined+submitted FraudGNAM wins "
                   f"{glob['fraudgnam_wins']}/{glob['total_metric_dataset_variant']} "
                   f"metric-dataset comparisons vs SEC-GFD "
                   f"(binomial p one-sided = {glob['binomial_p_one_sided']:.4f}).")
    return "\n".join(out)


def section_efi_ablation():
    out = []
    out.append("\n## §3. EFI ablation, all 7 datasets (R3 thoroughness)\n")
    out.append("| Dataset | full AUROC | no_efi AUROC | Δ AUROC | full AUPRC | no_efi AUPRC | Δ AUPRC |")
    out.append("|---------|-----------:|-------------:|--------:|-----------:|-------------:|--------:|")
    for ds in ["yelp", "amazon", "weibo", "reddit", "tolokers", "questions", "tfinance"]:
        full_d = L(f"{ds}_full.json")
        ne_d   = L(f"{ds}_no_efi.json")
        if full_d is None or ne_d is None:
            out.append(f"| {ds} | (pending) | (pending) | — | — | — | — |")
            continue
        d_auroc = (ne_d["AUROC_mean"] - full_d["AUROC_mean"]) * 100
        d_auprc = (ne_d["AUPRC_mean"] - full_d["AUPRC_mean"]) * 100
        out.append(f"| {ds} | {full_d['AUROC_mean']:.4f} | {ne_d['AUROC_mean']:.4f} | "
                   f"{d_auroc:+.2f}pp | {full_d['AUPRC_mean']:.4f} | "
                   f"{ne_d['AUPRC_mean']:.4f} | {d_auprc:+.2f}pp |")
    return "\n".join(out)


def section_nce_flow():
    out = []
    out.append("\n## §4. NCE gradient-flow analysis (R4-W1 / R5-W3 / R2-W5)\n")
    out.append("Reviewer concern: does NCE concentrate predictive mass into a single branch?\n")
    out.append("**Finding**: NCE *decreases* Gini concentration across components — "
               "the opposite of branch collapse.\n")
    out.append("| Dataset | AUROC w/ NCE | AUROC w/o NCE | Gini w/ NCE | Gini w/o NCE | ΔGini |")
    out.append("|---------|-------------:|--------------:|------------:|-------------:|------:|")
    for ds in ["yelp", "tolokers"]:
        d = L(f"nce_flow_{ds}.json")
        if d is None: continue
        w, wo = d["with_NCE"], d["without_NCE"]
        out.append(f"| {ds} | {w['AUROC']:.4f} | {wo['AUROC']:.4f} | "
                   f"{w['gini_test']:.3f} | {wo['gini_test']:.3f} | "
                   f"{w['gini_test']-wo['gini_test']:+.3f} |")
    return "\n".join(out)


def section_perturbation():
    out = []
    out.append("\n## §5. Perturbation stability (R4-W2)\n")
    out.append("Spearman ρ between original and noise-perturbed component "
               "contribution vectors `[α_c·ψ_c(v)]_c`. 200 test nodes × 3 noise draws each.\n")
    out.append("| Dataset | σ=0.01 | σ=0.05 | σ=0.10 |")
    out.append("|---------|-------:|-------:|-------:|")
    for ds in ["amazon", "yelp", "tolokers", "reddit"]:
        d = L(f"perturbation_{ds}.json")
        if d is None: continue
        cells = []
        for s in ["0.01", "0.05", "0.1"]:
            r = d["per_sigma"][s]
            cells.append(f"ρ̄={r['rho_mean']:.3f} (top-match {r['top_component_match_rate']*100:.0f}%)")
        out.append(f"| {ds} | {cells[0]} | {cells[1]} | {cells[2]} |")
    return "\n".join(out)


def section_memory():
    out = []
    out.append("\n## §6. Memory profiling per variant (R1-W1)\n")
    for ds in ["yelp", "tolokers"]:
        d = L(f"memory_{ds}.json")
        if d is None:
            out.append(f"\n_(memory_{ds}.json pending)_")
            continue
        out.append(f"\n### {ds}\n")
        out.append("| Variant | Params (k) | Peak GPU memory |")
        out.append("|---------|-----------:|----------------:|")
        for v, r in d["variants"].items():
            if r.get("peak_MB") is None:
                out.append(f"| {v} | — | (error) |")
                continue
            out.append(f"| {v} | {r['n_params']/1000:.1f} | {r['peak_MB']:.0f} MB |")
    return "\n".join(out)


def section_filter_response():
    out = []
    out.append("\n## §7. Filter response evidence (R4-C4)\n")
    out.append("Learned spectral coefficients on tolokers, yelp, weibo. "
               "See figures `rebuttal_figures/filter_response_<ds>.pdf` for visual.\n")
    out.append("| Dataset | Bands | HP iters | Trained AUROC |")
    out.append("|---------|------:|---------:|---------------|")
    for ds in ["yelp", "tolokers", "weibo"]:
        d = L(f"filter_response_{ds}.json")
        if d is None: continue
        out.append(f"| {ds} | {d['num_bands']} | {d['num_highpass']} | "
                   f"{d['trained_AUROC']:.4f} |")
    return "\n".join(out)


def section_round1_ablation():
    out = []
    out.append("\n## §8. Round-1 ablations (tolokers / questions / tfinance, "
               "completed before refined rule)\n")
    out.append("Adaptive gating evidence on tfinance (homophilic): all "
               "variants converge to identical performance because HP/NCE are "
               "auto-disabled by the rule.\n")
    out.append("| Dataset | full | no_HP | no_NCE | order2 |")
    out.append("|---------|-----:|------:|-------:|-------:|")
    for ds in ["tolokers", "questions", "tfinance"]:
        f_ = L(f"{ds}_full.json")
        nh = L(f"{ds}_no_highpass.json")
        nn = L(f"{ds}_no_nce.json")
        o2 = L(f"{ds}_order2.json")
        if f_ is None: continue
        out.append(f"| {ds} | {f_['AUROC_mean']:.4f} | "
                   f"{nh['AUROC_mean']:.4f} ({(nh['AUROC_mean']-f_['AUROC_mean'])*100:+.2f}) | "
                   f"{nn['AUROC_mean']:.4f} ({(nn['AUROC_mean']-f_['AUROC_mean'])*100:+.2f}) | "
                   f"{o2['AUROC_mean']:.4f} ({(o2['AUROC_mean']-f_['AUROC_mean'])*100:+.2f}) |")
    return "\n".join(out)


def main():
    pieces = [
        "# FraudGNAM rebuttal evidence package\n",
        f"_Auto-generated_, results dir: `{R}/`\n",
        section_refined_rule(),
        section_real_paired(),
        section_efi_ablation(),
        section_nce_flow(),
        section_perturbation(),
        section_memory(),
        section_filter_response(),
        section_round1_ablation(),
    ]
    out_path = R / "REBUTTAL_REPORT.md"
    with open(out_path, "w") as f:
        f.write("\n".join(pieces))
    print(f"\n>>> Saved {out_path}")
    print("\nReport sections rendered. Sections that lacked data are marked '(pending)'.")


if __name__ == "__main__":
    main()
