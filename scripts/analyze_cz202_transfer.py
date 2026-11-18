"""Assess the completed CZ202 transfer experiment without fitting or test tuning.

Month blocks are sampled jointly for all six fitted models. The resulting
intervals condition on these fixed fits; seeds are not independent test markets.
"""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/cz202"
VARIANTS = ("transalpha", "linear")
SEEDS = (0, 1, 2)
METRICS = ("ic", "rank_ic", "gross_sharpe", "net_sharpe")
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261118
BLOCK_LENGTHS = (3, 6, 12)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(x):
    """Last axes are [seed, month, ic/rank_ic/gross/net]."""
    ic = x[..., 0].mean(axis=-1)
    rank_ic = x[..., 1].mean(axis=-1)
    sharpe = np.sqrt(12) * x[..., 2:].mean(axis=-2) / x[..., 2:].std(axis=-2, ddof=1)
    return np.concatenate([ic[..., None], rank_ic[..., None], sharpe], axis=-1)


def main():
    audit = json.loads((OUT / "audit.json").read_text())
    source = json.loads((OUT / "source_verification.json").read_text())
    assert audit["passed"] and source["passed"]
    assert {r["run"] for r in audit["runs"] if r["passed"]} == {
        f"{v}-seed{s}" for v in VARIANTS for s in SEEDS
    }
    expected_months = [year * 100 + month for year in range(2022, 2026) for month in range(1, 13)]
    arrays, provenance, per_run = [], {}, []
    for variant in VARIANTS:
        group = []
        for seed in SEEDS:
            folder = OUT / f"{variant}-seed{seed}"
            metrics = json.loads((folder / "metrics.json").read_text())
            assert metrics["variant"] == variant and metrics["seed"] == seed
            assert sha(folder / "scores.csv") == metrics["scores_sha256"]
            assert sha(folder / "model.pt") == metrics["checkpoint_sha256"]
            assert sha(OUT / "protocol.json") == metrics["protocol_sha256"]
            with (folder / "monthly_returns.csv").open() as f:
                rows = list(csv.DictReader(f))
            assert [int(r["return_month"]) for r in rows] == expected_months
            values = np.array([[float(r[k]) for k in ("ic", "rank_ic", "gross", "net")] for r in rows])
            assert np.isfinite(values).all()
            point = statistics(values)
            np.testing.assert_allclose(point, [metrics[k] for k in METRICS], rtol=0, atol=1e-12)
            group.append(values)
            per_run.append({"variant": variant, "seed": seed, **dict(zip(METRICS, point.tolist())),
                            "net_alpha_ci95": metrics["regression"]["net"]["annual_alpha_ci95"]})
            for filename in ("metrics.json", "monthly_returns.csv", "scores.csv", "model.pt"):
                path = folder / filename
                provenance[str(path.relative_to(ROOT))] = sha(path)
        arrays.append(group)
    arrays = np.asarray(arrays)  # [variant, seed, month, metric]
    per_seed = statistics(arrays)
    means = per_seed.mean(axis=1)
    differences = per_seed[0] - per_seed[1]
    summary = {
        v: {k: {"mean": float(means[i, j]), "seed_sd": float(per_seed[i, :, j].std(ddof=1))}
            for j, k in enumerate(METRICS)}
        for i, v in enumerate(VARIANTS)
    }
    # Independent scalar check of the paired IC estimand.
    np.testing.assert_allclose(differences[:, 0].mean(),
                               (arrays[0, :, :, 0] - arrays[1, :, :, 0]).mean(), atol=1e-15)
    bootstrap = {}
    for block in BLOCK_LENGTHS:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        starts = rng.integers(0, 48, size=(BOOTSTRAP_DRAWS, (48 + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % 48).reshape(BOOTSTRAP_DRAWS, -1)[:, :48]
        assert indices.shape == (BOOTSTRAP_DRAWS, 48)
        # Advanced indexing yields [variant, seed, draw, month, metric].
        sample_statistics = statistics(arrays[:, :, indices, :]).mean(axis=1)
        paired = sample_statistics[0] - sample_statistics[1]
        bootstrap[str(block)] = {
            "model_mean_ci95": {
                v: {k: np.quantile(sample_statistics[i, :, j], [.025, .975]).tolist()
                    for j, k in enumerate(METRICS)} for i, v in enumerate(VARIANTS)
            },
            "difference_ci95": {k: np.quantile(paired[:, j], [.025, .975]).tolist()
                                for j, k in enumerate(METRICS)},
        }
    with (ROOT / "sources/paper_table3_transalpha.csv").open() as f:
        paper = next(r for r in csv.DictReader(f) if r["universe"] == "csi300")
    for path in (OUT / "audit.json", OUT / "source_verification.json", OUT / "protocol.json",
                 ROOT / "sources/paper_table3_transalpha.csv", Path(__file__).resolve()):
        provenance[str(path.relative_to(ROOT))] = sha(path)
    result = {
        "assessment_date": datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat(),
        "scope": "Reanalysis of six existing fresh-training CZ202 fits; no new training or hyperparameter search",
        "question": "Does the reconstructed TransAlpha show predictive/profitability advantage with monthly CZ202 inputs?",
        "conclusion": "No advantage over the linear control was observed under the fixed CZ202 protocol. This does not reproduce the paper's claimed advantage and is not a same-setting numerical replication.",
        "months": expected_months, "seeds": list(SEEDS), "per_run": per_run, "summary": summary,
        "transalpha_minus_linear": {k: {"mean": float(differences[:, j].mean()),
            "per_seed": differences[:, j].tolist(), "positive_seed_count": int((differences[:, j] > 0).sum())}
            for j, k in enumerate(METRICS)},
        "bootstrap_method": {"kind": "Paired circular moving-block percentile bootstrap",
            "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "block_lengths_months": list(BLOCK_LENGTHS),
            "primary_block_length": 6,
            "sampling_unit": "Contiguous months; identical sampled indices across all models and seeds",
            "estimand": "Mean of per-seed IC/rank IC/annualized Sharpe; not an ensemble portfolio",
            "limitations": "Exploratory post-hoc pointwise intervals conditional on fixed fits; only 48 months, no seed-population or training uncertainty. Return resampling retains recorded opening/closing costs. Blocks 3/6/12 are sensitivity analyses, not selected for significance."},
        "bootstrap": bootstrap,
        "paper_reference_only": paper,
        "paper_comparison": {"comparable": False, "absolute_error": None, "relative_error": None,
            "reason": "CZ202 monthly US inputs, different splits and value-weighted deciles replace private Chinese 15-minute factors and long-short 50. The control is not the published DLinear."},
        "checks": {"passed": True, "all_six_saved_run_metrics_recomputed": True,
            "identical_test_months": True, "current_independent_audits_passed": True},
        "input_hashes": provenance,
    }
    (OUT / "transfer_analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["CZ202 transfer assessment — " + result["assessment_date"],
        "No TransAlpha advantage observed in the frozen monthly CZ202 experiment.",
        "Six existing trained fits re-audited; this assessment did not retrain or tune models.",
        "Train 2009-2019; validation 2020-2021; test 2022-2025 (48 shared months).",
        "202 features; 16 calendar-month history; three fixed seeds; validation-IC selection.",
        "Value-weighted D10-D1 (+1/-1); lagged NYSE20 size screen; 10bp/traded dollar.",
        "", "Equal-seed means (sample seed SD in parentheses; SD is not a CI):",
        "Model | IC | Rank IC | Gross Sharpe | Net Sharpe"]
    for v in VARIANTS:
        lines.append(v + " | " + " | ".join(f"{summary[v][k]['mean']:.6f} ({summary[v][k]['seed_sd']:.6f})" for k in METRICS))
    lines += ["", "TransAlpha minus linear; paired month-bootstrap 95% intervals:",
              "Metric | Observed difference | 3-month blocks | 6-month blocks | 12-month blocks"]
    for j, k in enumerate(METRICS):
        cis = [bootstrap[str(b)]["difference_ci95"][k] for b in BLOCK_LENGTHS]
        lines.append(f"{k} | {differences[:, j].mean():.6f} | " + " | ".join(f"[{a:.6f}, {b:.6f}]" for a, b in cis))
    lines += ["", "TransAlpha has lower IC and net Sharpe than the control in all three seed pairs.",
        "All three TransAlpha net FF5+UMD alpha 95% HAC intervals include zero.",
        "Bootstrap: 10,000 circular block draws, RNG 20261118; joint monthly resampling",
        "preserves pairing across six models. Primary block length 6; lengths 3/12 are",
        "sensitivity checks. Intervals are exploratory, pointwise, conditional on these fits;",
        "three seeds share one 48-month market path and are not three independent markets.",
        "", f"Paper CSI300 reference only: IC {paper['ic']}, rank IC {paper['rank_ic']}, Sharpe {paper['sharpe']}.",
        "Different market, frequency, features, dates and portfolio rules prevent treating",
        "these differences as replication errors. The local linear control is not DLinear.",
        "This result does not establish that TransAlpha can never work with CZ202 or that",
        "the original private-data paper fails. It does not test the paper's full baseline",
        "suite or ablations. Model head, optimizer settings and other unpublished choices",
        "are reconstruction assumptions. Missing features/history are neutral-filled, and",
        "the CZ/CRSP source is current-vintage rather than historically as-published.",
        "", "Evidence: transfer_analysis.json (statistics/provenance), audit.json,",
        "source_verification.json, protocol.json, and each seed's saved scores/checkpoint.",
        "Reproduce assessment: PYTHON_CZ scripts/analyze_cz202_transfer.py",
        "PYTHON_CZ=/home/jkp/data/.replication-local/artifacts/ordered-reproductions/.venv-tra/bin/python"]
    (OUT / "TRANSFER_REPORT.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
