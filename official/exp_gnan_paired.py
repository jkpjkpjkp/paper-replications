"""
Compute paired statistical tests: FraudGNAM-refined vs GNAN baseline.
Uses per-trial values from both methods (matched seeds).
"""
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats


R = Path("results")


def nadeau_bengio(diffs, train_ratio=0.4, test_ratio=0.4):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 2: return float("nan"), float("nan"), float("nan")
    mean = diffs.mean(); var = diffs.var(ddof=1)
    if var <= 0: return mean, float("nan"), 1.0
    cvar = var * (1.0 / n + test_ratio / train_ratio)
    if cvar <= 0: return mean, float("nan"), 1.0
    t = mean / math.sqrt(cvar)
    p = 2 * (1 - stats.t.cdf(abs(t), n - 1))
    return mean, t, p


def paired(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2: return None
    diffs = a - b
    mean, t, p_nb = nadeau_bengio(diffs)
    try:
        _, p_w = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    except Exception:
        p_w = float("nan")
    wins = int((a > b).sum())
    return {"a_mean": float(np.mean(a)), "b_mean": float(np.mean(b)),
            "delta": float(mean), "t_corr": float(t), "p_NB": float(p_nb),
            "p_Wilcoxon": float(p_w), "wins": wins, "n": int(len(a))}


def main():
    metrics = {"AUROC": "auc_per_trial", "AUPRC": "pre_per_trial", "RecK": "rec_per_trial"}
    print(f"{'Dataset':<10} {'Metric':<6} {'Refined':>8} {'GNAN':>8} {'Δmean':>8} "
          f"{'p_NB':>8} {'p_Wilc':>8} {'wins':>5}")
    out = {}
    total_wins = 0
    total = 0
    for ds in ["yelp", "amazon", "weibo", "tfinance", "reddit", "tolokers", "questions"]:
        try:
            r = json.load(open(R / f"fraudgnam_refined_{ds}.json"))
            g = json.load(open(R / f"gnan/gnan_{ds}.json"))
        except FileNotFoundError:
            continue
        out[ds] = {}
        for m, key in metrics.items():
            res = paired(r[key], g[key])
            if res is None: continue
            out[ds][m] = res
            sig = ("***" if res["p_NB"] < 0.01 else
                   "**"  if res["p_NB"] < 0.05 else
                   "*"   if res["p_NB"] < 0.10 else "")
            print(f"{ds:<10} {m:<6} {res['a_mean']:>8.4f} {res['b_mean']:>8.4f} "
                  f"{res['delta']:>+8.4f} {res['p_NB']:>7.4f}{sig:<3} "
                  f"{res['p_Wilcoxon']:>8.4f} {res['wins']}/{res['n']}")
            total += 1
            if res["delta"] > 0: total_wins += 1
    print(f"\nGlobal: FraudGNAM-refined wins {total_wins}/{total} vs GNAN")
    if total > 0:
        binom = stats.binomtest(total_wins, n=total, p=0.5, alternative="greater").pvalue
        print(f"Exact binomial p (one-sided) = {binom:.6f}")
    out["_global"] = {"wins": total_wins, "total": total,
                       "binomial_p": float(binom) if total > 0 else None}
    with open(R / "paired_vs_gnan.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
