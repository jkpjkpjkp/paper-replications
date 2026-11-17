"""Independent, bounded audit of the exact author's released numerical artifacts.

This does not regenerate raw-data states or claim that cached checkpoints were
trained with the current code. All resampling is across the ten original seeds;
none of these observations is presented as a daily trading return.
"""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
UP = ROOT / "upstream"
OUT = ROOT / "results" / "release_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
PIN = "5fca9f59df30ba75b75ca2917a292e041b7cdb39"
REPS = 10000


def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_csv(p, rows):
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def holm(ps):
    p = np.asarray(ps)
    ix = np.argsort(p)
    ans = np.empty(len(p))
    ans[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * (len(p) - np.arange(len(p)))))
    return ans


def main():
    commit = subprocess.check_output(["git", "-C", str(UP), "rev-parse", "HEAD"], text=True).strip()
    assert commit == PIN, commit
    versions = {n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scipy", "torch", "matplotlib", "pyarrow"]}
    (OUT / "environment.json").write_text(json.dumps({"python": sys.version, "executable": sys.executable, "packages": versions}, indent=2))
    test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], cwd=UP, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    (OUT / "upstream_tests.log").write_text(test.stdout)
    # Manifest actual released checkpoints and result files, independent of git mtimes.
    manifest = [{"path": str(p.relative_to(UP)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(UP.rglob("*")) if p.is_file() and ".git" not in p.parts]
    (OUT / "release_manifest.json").write_text(json.dumps(manifest, indent=2))
    inspection = ROOT / "sources" / "CON2.py.inspection"
    source_hash_matches = inspection.read_bytes().rstrip() == (UP / "src" / "CON2.py").read_bytes().rstrip()
    expected = {
        "industry": {"CVaR95_loss": [2.3191, 2.3096, 2.3073], "crowding": [2.2877, 2.2412, 2.2602], "co_crash_freq": [0.1060, 0.1037, 0.1046]},
        "etf": {"CVaR95_loss": [0.9006, 0.8922, 0.8838], "crowding": [0.9967, 0.9847, 0.9868], "co_crash_freq": [0.1743, 0.1733, 0.1730]},
    }
    metrics = ["CVaR95_loss", "crowding", "co_crash_freq", "mean_effort", "mean_payoff", "principal_CE", "policy_objective", "CVaR99_loss", "mean_systematic_exposure"]
    means, contrasts, integrity = [], [], []
    rng = np.random.default_rng(41220261107)
    for dataset in ["industry", "etf"]:
        prefix = "ETF_" if dataset == "etf" else ""
        method_names = {"Feature": prefix + "GraphSignal-ICL-NoPenalty", "Graph": prefix + "GraphSignal-ICL", "Random": prefix + "RandomSignal-ICL", "Constant": prefix + "GraphSignal-ICL-ConstantPenalty", "NoGraph": prefix + "RawNodeMLP_GLS_NoGraph_HardEffort"}
        values = {}
        for seed in SEEDS:
            frozen = read_csv(UP / "results" / "frozen500" / f"seed_{seed}" / "tables" / f"{dataset}_neural_results.csv")
            constant = read_csv(UP / "results" / "constant_score_500" / f"seed_{seed}" / "tables" / f"{dataset}_neural_results.csv")
            rows = {r["method"]: r for r in frozen}
            for r in constant:
                if r["method"] == method_names["Constant"]:
                    rows[r["method"]] = r
            for label, method in method_names.items():
                row = rows[method]
                assert int(float(row["M_paths"])) == 500
                assert int(float(row["B"])) == (228 if dataset == "industry" else 77)
                values[(seed, label)] = row
            detail = read_csv(UP / "results" / "frozen500" / f"seed_{seed}" / "tables" / f"{dataset}_episode_details_neural.csv")
            for label in ["Feature", "Graph", "Random"]:
                dates = sorted(r["date"] for r in detail if r["method"] == method_names[label])
                assert len(dates) == len(set(dates)) == (228 if dataset == "industry" else 77)
            integrity.append({"dataset": dataset, "seed": seed, "all_four_methods": True, "M_paths": 500, "n_states": 228 if dataset == "industry" else 77})
        for metric, paper_values in expected[dataset].items():
            for label, reported in zip(["Feature", "Graph", "Random"], paper_values):
                value = np.mean([float(values[s, label][metric]) for s in SEEDS])
                means.append({"dataset": dataset, "metric": metric, "method": label, "released_mean": value, "paper_rounded": reported, "matches_4dp": round(value, 4) == reported})
        for tr, ba in [("Graph", "Feature"), ("Graph", "Random"), ("Constant", "Feature"), ("Graph", "Constant"), ("Graph", "NoGraph")]:
            for metric in metrics:
                a = np.array([float(values[s, tr][metric]) for s in SEEDS])
                b = np.array([float(values[s, ba][metric]) for s in SEEDS])
                d = a - b
                boot = np.mean(d[rng.integers(0, len(d), size=(REPS, len(d)))], axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                nonzero = d[np.abs(d) > 1e-12]
                signp = stats.binomtest(int(np.sum(nonzero > 0)), len(nonzero), .5).pvalue if len(nonzero) else 1.0
                wp = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                contrasts.append({"dataset": dataset, "contrast": f"{tr} minus {ba}", "metric": metric, "n_seeds": len(d), "mean_treatment": a.mean(), "mean_baseline": b.mean(), "delta": d.mean(), "ci_low": lo, "ci_high": hi, "bootstrap_reps": REPS, "sign_test_p": signp, "wilcoxon_p": wp, "negative_seeds": int(np.sum(d < 0)), "positive_seeds": int(np.sum(d > 0)), "family": "all_90_audited_contrasts"})
    all_adj = holm([r["wilcoxon_p"] for r in contrasts])
    primary = [i for i, r in enumerate(contrasts) if r["contrast"] == "Graph minus Feature" and r["metric"] in ["CVaR95_loss", "crowding", "co_crash_freq"]]
    primary_adj = dict(zip(primary, holm([contrasts[i]["wilcoxon_p"] for i in primary])))
    for i, row in enumerate(contrasts):
        row["holm_all90_p"] = all_adj[i]
        row["holm_primary6_p"] = primary_adj.get(i, "")
    write_csv(OUT / "paper_table1_means.csv", means)
    write_csv(OUT / "all_seed_contrasts.csv", contrasts)
    write_csv(OUT / "released_design_integrity.csv", integrity)
    summary = {
        "commit": commit,
        "scope": "independent audit of author-released cached outputs; not new raw-data experiment",
        "upstream_tests_exit_code": test.returncode,
        "inspection_source_matches_pinned_clone": source_hash_matches,
        "paper_table1_means_matching_4dp": int(sum(r["matches_4dp"] for r in means)),
        "paper_table1_mean_cells": len(means),
        "seed_contrasts": len(contrasts),
        "bootstrap_reps": REPS,
        "primary_six_risk_cells": [contrasts[i] for i in primary],
        "checkpoint_count": sum(x["path"].startswith("checkpoints/source_50path/") and x["path"].endswith("_model.pt") for x in manifest),
        "provenance_caveat": "Release docs explicitly lack code/checkpoint/run-hash linkage for RandomSignal after PR1 correction; regenerated summaries cannot resolve this.",
        "market_neutral_nocost": "not_applicable: modeled principal-agent episode payoffs are not a self-financing calendar-time asset strategy",
        "multiplicity_caveat": "Six original placement cells and90 audited seed-contrast tests, including Graph vs NoGraph, reported separately. No programme-wide correction applied. Seed CIs are risk-diagnostic inference, not HAC trading-return evidence.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if test.returncode:
        raise SystemExit(test.returncode)


if __name__ == "__main__":
    main()
