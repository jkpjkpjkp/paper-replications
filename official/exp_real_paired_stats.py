"""
Replace the simulated paired tests in exp_stats.py with REAL paired tests
using the per-trial values from EXP-A (SEC-GFD) and EXP-B (FraudGNAM
refined + submitted).

Both methods are run on the same seed list (SEED_LIST[0:10]) and the same
GADBench split protocol, so per-trial outcomes are matched.

Outputs:
    results/real_paired_tests.json   <- machine-readable results
    results/real_paired_tests.txt    <- human-readable table
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats


RESULT_DIR = Path("results")


def load(name):
    p = RESULT_DIR / name
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def nadeau_bengio(diffs, train_ratio=0.4, test_ratio=0.4):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    mean = diffs.mean()
    var = diffs.var(ddof=1)
    if var <= 0:
        return mean, float("nan"), 1.0
    corrected_var = var * (1.0 / n + test_ratio / train_ratio)
    if corrected_var <= 0:
        return mean, float("nan"), 1.0
    t_stat = mean / math.sqrt(corrected_var)
    p_two = 2 * (1 - stats.t.cdf(abs(t_stat), n - 1))
    return mean, t_stat, p_two


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05, seed=1):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2:
        return float("nan"), float("nan")
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_test(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    if len(diffs) < 2:
        return None
    mean, t_corr, p_corr = nadeau_bengio(diffs)
    try:
        w_stat, p_wilcox = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except Exception:
        p_wilcox = float("nan")
    wins = int((a > b).sum())
    p_sign = stats.binomtest(wins, n=len(a), p=0.5, alternative="two-sided").pvalue
    ci_lo, ci_hi = bootstrap_ci(diffs)
    return {
        "n_paired": int(len(diffs)),
        "mean_diff": float(mean),
        "t_corr": float(t_corr),
        "p_nadeau_bengio": float(p_corr),
        "p_wilcoxon": float(p_wilcox),
        "wins": wins,
        "p_sign": float(p_sign),
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "a_mean": float(np.mean(a)), "a_std": float(np.std(a)),
        "b_mean": float(np.mean(b)), "b_std": float(np.std(b)),
    }


def main():
    metrics = ["AUROC", "AUPRC", "RecK"]
    metric_field = {"AUROC": "auc_per_trial",
                    "AUPRC": "pre_per_trial",
                    "RecK":  "rec_per_trial"}

    datasets = ["yelp", "amazon", "weibo", "tfinance",
                "reddit", "tolokers", "questions"]

    results = {}
    print(f"{'Dataset':<10} {'Variant':<10} {'Metric':<6} "
          f"{'A_mean':>8} {'B_mean':>8} {'D_mean':>8} "
          f"{'p_NB':>8} {'p_Wilc':>8} {'wins':>5}")
    for ds in datasets:
        sec = load(f"secgfd_{ds}_paired.json")
        ours_ref = load(f"fraudgnam_refined_{ds}.json")
        ours_sub = load(f"fraudgnam_submitted_{ds}.json")
        if sec is None or (ours_ref is None and ours_sub is None):
            print(f"  [{ds}] skipped (sec={sec is not None}, "
                  f"refined={ours_ref is not None}, sub={ours_sub is not None})")
            continue

        results[ds] = {}
        for tag, theirs in [("refined", ours_ref), ("submitted", ours_sub)]:
            if theirs is None:
                continue
            for m in metrics:
                a = theirs[metric_field[m]]
                b = sec[metric_field[m]]
                r = paired_test(a, b)
                if r is None:
                    continue
                results[ds][f"{tag}:vs_secgfd:{m}"] = r
                sig = ("***" if r["p_nadeau_bengio"] < 0.01 else
                       "**"  if r["p_nadeau_bengio"] < 0.05 else
                       "*"   if r["p_nadeau_bengio"] < 0.10 else "")
                print(f"{ds:<10} {tag:<10} {m:<6} "
                      f"{r['a_mean']:>8.4f} {r['b_mean']:>8.4f} "
                      f"{r['mean_diff']:>+8.4f} "
                      f"{r['p_nadeau_bengio']:>7.4f}{sig:<3} "
                      f"{r['p_wilcoxon']:>8.4f} "
                      f"{r['wins']:>2d}/{r['n_paired']}")

        # Refined-vs-Submitted comparison (does the refinement help?)
        if ours_ref and ours_sub:
            for m in metrics:
                a = ours_ref[metric_field[m]]
                b = ours_sub[metric_field[m]]
                r = paired_test(a, b)
                if r is None:
                    continue
                results[ds][f"refined:vs_submitted:{m}"] = r
                sig = ("***" if r["p_nadeau_bengio"] < 0.01 else
                       "**"  if r["p_nadeau_bengio"] < 0.05 else
                       "*"   if r["p_nadeau_bengio"] < 0.10 else "")
                print(f"{ds:<10} {'rule_diff':<10} {m:<6} "
                      f"{r['a_mean']:>8.4f} {r['b_mean']:>8.4f} "
                      f"{r['mean_diff']:>+8.4f} "
                      f"{r['p_nadeau_bengio']:>7.4f}{sig:<3} "
                      f"{r['p_wilcoxon']:>8.4f} "
                      f"{r['wins']:>2d}/{r['n_paired']}")
        print()

    # Global win counts
    all_secgfd_wins = 0
    all_secgfd_total = 0
    for ds in datasets:
        for tag in ["refined", "submitted"]:
            for m in metrics:
                key = f"{tag}:vs_secgfd:{m}"
                r = results.get(ds, {}).get(key)
                if r is None:
                    continue
                all_secgfd_total += 1
                if r["mean_diff"] > 0:
                    all_secgfd_wins += 1
    if all_secgfd_total:
        results["_global"] = {
            "total_metric_dataset_variant": all_secgfd_total,
            "fraudgnam_wins": all_secgfd_wins,
            "binomial_p_one_sided": float(
                stats.binomtest(all_secgfd_wins, n=all_secgfd_total, p=0.5,
                                alternative="greater").pvalue
            ),
        }
        print(f"\nGlobal: {all_secgfd_wins}/{all_secgfd_total} wins, "
              f"binomial p (one-sided) = {results['_global']['binomial_p_one_sided']:.4f}")

    out = RESULT_DIR / "real_paired_tests.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
