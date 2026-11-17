"""
Aggregate all rebuttal experiment results into a single summary table for
inclusion in the rebuttal write-up.

Prints markdown-style tables covering:
  - Ablation results (tolokers/questions/tfinance × full/no_highpass/no_nce/order2)
  - Perturbation stability
  - NCE gradient flow (if available)
  - Paired statistical tests (headline rows)
"""

import glob
import json
import os


RESULT_DIR = "results"


def load_json(name):
    path = os.path.join(RESULT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return f"{v:.{digits}f}"


def aggregate_ablations():
    print("\n===== TABLE A: NEW ABLATION RESULTS (10 trials each) =====")
    print(f"{'Dataset':<10} {'Variant':<14} {'AUROC':>17} {'AUPRC':>17} {'RecK':>17} {'Time':>7}")
    datasets = ["tolokers", "questions", "tfinance"]
    variants = ["full", "no_highpass", "no_nce", "order2"]
    for ds in datasets:
        for v in variants:
            d = load_json(f"{ds}_{v}.json")
            if d is None:
                print(f"{ds:<10} {v:<14} {'(pending)':>17}")
                continue
            auroc = f"{d['AUROC_mean']:.4f}±{d['AUROC_std']:.3f}"
            auprc = f"{d['AUPRC_mean']:.4f}±{d['AUPRC_std']:.3f}"
            reck  = f"{d['RecK_mean']:.4f}±{d['RecK_std']:.3f}"
            t     = f"{d['time_mean']:.1f}s"
            err   = f" ({d.get('errors',0)} err)" if d.get('errors',0) else ""
            print(f"{ds:<10} {v:<14} {auroc:>17} {auprc:>17} {reck:>17} {t:>7}{err}")
        # Delta rows if full present
        full = load_json(f"{ds}_full.json")
        if full:
            for v in variants[1:]:
                d = load_json(f"{ds}_{v}.json")
                if d is None:
                    continue
                d_auroc = (d["AUROC_mean"] - full["AUROC_mean"]) * 100
                d_auprc = (d["AUPRC_mean"] - full["AUPRC_mean"]) * 100
                d_reck  = (d["RecK_mean"]  - full["RecK_mean"])  * 100
                print(f"{'':<10} {'Δ vs full ('+v+')':<26} AUROC{d_auroc:+.2f}pp  "
                      f"AUPRC{d_auprc:+.2f}pp  RecK{d_reck:+.2f}pp")
        print()


def aggregate_perturbation():
    print("\n===== TABLE B: PERTURBATION STABILITY (Spearman ρ, 200 test nodes × 3 draws) =====")
    print(f"{'Dataset':<10} {'σ=0.01':>26} {'σ=0.05':>26} {'σ=0.10':>26}")
    print(f"{'':10} {'ρ̄ (Top-match)':>26} {'ρ̄ (Top-match)':>26} {'ρ̄ (Top-match)':>26}")
    for ds in ["amazon", "yelp", "tolokers", "reddit"]:
        d = load_json(f"perturbation_{ds}.json")
        if d is None:
            continue
        row = f"{ds:<10}"
        for s in ["0.01", "0.05", "0.1"]:
            r = d["per_sigma"][s]
            row += f" {r['rho_mean']:.3f}/{r['rho_median']:.3f} ({r['top_component_match_rate']*100:.0f}%) "
        print(row)


def aggregate_nce():
    print("\n===== TABLE C: NCE GRADIENT FLOW (mean |α·ψ| per component, test nodes) =====")
    for ds in ["yelp", "tolokers"]:
        d = load_json(f"nce_flow_{ds}.json")
        if d is None:
            continue
        print(f"\n  Dataset: {ds}")
        print(f"  AUROC w/ NCE = {d['with_NCE']['AUROC']:.4f}  |  AUROC w/o NCE = {d['without_NCE']['AUROC']:.4f}")
        print(f"  Gini concentration w/ NCE = {d['with_NCE']['gini_test']:.3f}  w/o NCE = {d['without_NCE']['gini_test']:.3f}")
        print(f"  {'Component':<10} {'w/ NCE':>10} {'w/o NCE':>10} {'Δ':>10}")
        for c in ["Feature", "Pairwise", "Band", "EFI", "HP", "HP-EFI"]:
            a = d["with_NCE"]["mean_abs_contribution_test"][c]
            b = d["without_NCE"]["mean_abs_contribution_test"][c]
            print(f"  {c:<10} {a:>10.4f} {b:>10.4f} {a-b:>+10.4f}")


def aggregate_stats():
    print("\n===== TABLE D: PAIRED SIGNIFICANCE (FraudGNAM vs SEC-GFD, Nadeau-Bengio p) =====")
    d = load_json("statistical_tests.json")
    if d is None:
        return
    print(f"{'Dataset':<10} {'Metric':<6} {'Δmean':>8} {'p_NB':>10} {'wins':>6}")
    for ds in ["yelp", "amazon", "weibo", "tfinance", "tolokers", "reddit", "questions"]:
        if ds not in d:
            continue
        for m in ["AUROC", "AUPRC", "RecK"]:
            key = f"SEC-GFD:{m}"
            if key in d[ds] and d[ds][key] is not None:
                r = d[ds][key]
                sig = "***" if r["p_nadeau_bengio"] < 0.01 else \
                      "**"  if r["p_nadeau_bengio"] < 0.05 else \
                      "*"   if r["p_nadeau_bengio"] < 0.10 else ""
                print(f"{ds:<10} {m:<6} {r['mean_diff']:>+8.4f} "
                      f"{r['p_nadeau_bengio']:>8.4f}{sig:<3} {r['wins_out_of_10']:>4d}/10")
    g = d.get("_global", {})
    if g:
        print(f"\n  Global: {g['fraudgnam_wins_vs_secgfd']}/{g['total_metric_dataset']} wins, "
              f"exact binomial p (one-sided) = {g['binomial_p_one_sided']:.4f}")


if __name__ == "__main__":
    aggregate_ablations()
    aggregate_perturbation()
    aggregate_nce()
    aggregate_stats()
