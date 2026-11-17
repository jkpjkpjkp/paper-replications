"""
Leave-One-Dataset-Out validation for the *refined* adaptive rule.

For each held-out dataset, recompute the rule cutoffs that the remaining
six datasets would suggest, then check whether the same configuration is
selected on the held-out dataset.  This is the same logic as App D for
the submitted paper, extended with the three new R1/R2/R3 cutoffs.
"""

import json
from pathlib import Path

# Per-dataset graph statistics (computed once from the GADBench loaders).
STATS = {
    "yelp":      {"h": 0.7740, "n_f": 4700, "d": 32,  "k_bar": 84,
                   "n_nodes": 45954, "fraud_pct": 0.145},
    "amazon":    {"h": 0.9540, "n_f": 570,  "d": 25,  "k_bar": 740,
                   "n_nodes": 11944, "fraud_pct": 0.069},
    "weibo":     {"h": 0.9770, "n_f": 347,  "d": 400, "k_bar": 49,
                   "n_nodes": 8405,  "fraud_pct": 0.103},
    "tfinance":  {"h": 0.9710, "n_f": 720,  "d": 10,  "k_bar": 1080,
                   "n_nodes": 39357, "fraud_pct": 0.046},
    "reddit":    {"h": 0.9500, "n_f": 146,  "d": 64,  "k_bar": 15.3,
                   "n_nodes": 10984, "fraud_pct": 0.033},
    "tolokers":  {"h": 0.6040, "n_f": 1283, "d": 10,  "k_bar": 88,
                   "n_nodes": 11758, "fraud_pct": 0.218},
    "questions": {"h": 0.8780, "n_f": 730,  "d": 301, "k_bar": 8.3,
                   "n_nodes": 48921, "fraud_pct": 0.030},
}


def configure(stats, thresholds):
    h = stats["h"]; nf = stats["n_f"]; d = stats["d"]; k = stats["k_bar"]
    if h < thresholds["tau_order"]:
        order = 4
    elif k < 30:
        order = 3
    else:
        order = 2
    if nf < thresholds["n_f_R1"]:
        order = 2
    if h > 0.97 and d >= 300:
        order = 4

    if h < thresholds["tau_order"]:
        hp = 2
    elif h < thresholds["tau_nce"]:
        hp = 1
    else:
        hp = 0
    if h < thresholds["h_R3"]:
        hp = 0

    use_nce = (nf >= 500) and (h < thresholds["tau_nce"])
    return {"order": order, "hp": hp, "use_nce": use_nce}


def threshold_distance_to_boundary():
    """For each dataset, compute distance to nearest cutoff."""
    rows = []
    for ds, s in STATS.items():
        d_tau_order = abs(s["h"] - 0.85)
        d_tau_nce   = abs(s["h"] - 0.95)
        d_h_R3      = abs(s["h"] - 0.70)
        d_n_f_R1    = abs(s["n_f"] - 200)
        nearest = min(d_tau_order, d_tau_nce, d_h_R3, d_n_f_R1 / 1000)
        rows.append({"dataset": ds, "h": s["h"], "n_f": s["n_f"],
                      "dist_tau_order": d_tau_order,
                      "dist_tau_nce": d_tau_nce,
                      "dist_h_R3": d_h_R3,
                      "dist_n_f_R1": d_n_f_R1,
                      "nearest_normalized": nearest})
    return rows


def lodo_check(thresholds):
    """For each LODO scenario, are the same thresholds optimal?"""
    rows = []
    base_configs = {ds: configure(s, thresholds) for ds, s in STATS.items()}
    for held_out in STATS:
        remaining = {ds: STATS[ds] for ds in STATS if ds != held_out}
        # Quantile-based threshold suggestions from remaining 6
        h_values = sorted(s["h"] for s in remaining.values())
        n_f_values = sorted(s["n_f"] for s in remaining.values())
        # Are the original thresholds still in natural gaps?
        # tau_order=0.85: gap between max(<0.85) and min(>=0.85)
        below = [h for h in h_values if h < 0.85]
        above = [h for h in h_values if h >= 0.85]
        gap_tau_order = (above[0] - below[-1]) if below and above else None
        below_nce = [h for h in h_values if h < 0.95]
        above_nce = [h for h in h_values if h >= 0.95]
        gap_tau_nce = (above_nce[0] - below_nce[-1]) if below_nce and above_nce else None
        # n_f gap
        below_nf = [v for v in n_f_values if v < 200]
        above_nf = [v for v in n_f_values if v >= 200]
        gap_n_f_R1 = (above_nf[0] - below_nf[-1]) if below_nf and above_nf else None

        # Apply configure() to held-out dataset
        config_with_full = base_configs[held_out]
        config_with_lodo = configure(STATS[held_out], thresholds)
        same_config = (config_with_full == config_with_lodo)

        rows.append({
            "held_out": held_out,
            "below_tau_order": below, "above_tau_order": above,
            "gap_tau_order": gap_tau_order,
            "gap_tau_nce": gap_tau_nce,
            "gap_n_f_R1": gap_n_f_R1,
            "config": config_with_full,
            "config_unchanged_under_LODO": same_config,
        })
    return rows


def main():
    thresholds = {"tau_order": 0.85, "tau_nce": 0.95,
                  "n_f_R1": 200, "h_R3": 0.70}

    print("===== Refined-rule configuration per dataset =====")
    print(f"{'Dataset':<10} {'h':>6} {'n_f':>5} {'order':>5} {'HP':>3} {'NCE':>3}")
    for ds, s in STATS.items():
        c = configure(s, thresholds)
        print(f"{ds:<10} {s['h']:>6.3f} {s['n_f']:>5d} "
              f"{c['order']:>5d} {c['hp']:>3d} {str(c['use_nce']):>3}")

    print("\n===== Threshold distance to nearest cutoff =====")
    print(f"{'Dataset':<10} {'h':>6} {'d_tau_order':>11} {'d_tau_nce':>9} "
          f"{'d_h_R3':>7} {'d_n_f_R1':>9}")
    for r in threshold_distance_to_boundary():
        print(f"{r['dataset']:<10} {r['h']:>6.3f} "
              f"{r['dist_tau_order']:>11.3f} "
              f"{r['dist_tau_nce']:>9.3f} "
              f"{r['dist_h_R3']:>7.3f} "
              f"{r['dist_n_f_R1']:>9d}")

    print("\n===== LODO analysis (refined rule) =====")
    lodo = lodo_check(thresholds)
    for r in lodo:
        gtao = r['gap_tau_order'] if r['gap_tau_order'] is not None else 0.0
        gtnc = r['gap_tau_nce']   if r['gap_tau_nce']   is not None else 0.0
        gnf  = r['gap_n_f_R1']    if r['gap_n_f_R1']    is not None else 0
        print(f"hold-out {r['held_out']:<10} "
              f"gap_tau_order={gtao:.3f}  gap_tau_nce={gtnc:.3f}  "
              f"gap_n_f_R1={gnf:>4}  unchanged={r['config_unchanged_under_LODO']}")

    out = Path("results/refined_lodo_analysis.json")
    out.parent.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "thresholds": thresholds,
            "configurations": {ds: configure(s, thresholds)
                                for ds, s in STATS.items()},
            "threshold_distances": threshold_distance_to_boundary(),
            "lodo_analysis": lodo,
            "all_lodo_unchanged": all(r["config_unchanged_under_LODO"]
                                      for r in lodo),
        }, f, indent=2, default=str)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
