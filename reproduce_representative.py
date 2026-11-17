#!/usr/bin/env python3
"""Audit archived LOF outputs, or freshly fit the same paper baseline.

The corporate relationship study is a separate extension. This experiment targets
arXiv:2609.38424v1, Table 1 (LOF AUROC) and Table 15 (LOF AUPRC).
No labels enter LOF fitting; the published baseline fits the labeled-node subset.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "2"

import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE = (ROOT.parent / ".replication-local/artifacts/paper-reproduction-audit/"
           "workspaces/029_graph_anomaly_dynamical_transport")
TARGETS = {"elliptic": (45.1, 10.2, 166), "elliptic_plus_plus": (45.9, 8.5, 182)}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def independent_metrics(y, score):
    """Tie-aware concordance AUROC and grouped-threshold PR integrals."""
    order = np.argsort(score, kind="stable")
    s, labels = score[order], y[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    sizes = np.diff(np.r_[starts, len(s)])
    positives = np.add.reduceat(labels, starts).astype(np.int64)
    negatives = sizes - positives
    neg_below = np.cumsum(negatives) - negatives
    p, n = int(y.sum()), int(len(y) - y.sum())
    # Twice the number of correctly ordered positive/negative pairs; ties get 1/2.
    numerator2 = int(np.sum(positives * (2 * neg_below + negatives)))
    auroc = numerator2 / (2 * p * n)
    tp, fp = np.cumsum(positives[::-1]), np.cumsum(negatives[::-1])
    recall = np.r_[0.0, tp / p]
    precision = np.r_[1.0, tp / (tp + fp)]
    delta = np.diff(recall)
    return {
        "AUROC": auroc,
        "AUPRC_trapezoid": float(np.sum(delta * (precision[:-1] + precision[1:]) / 2)),
        "average_precision": float(np.sum(delta * precision[1:])),
        "concordant_pairs_twice": numerator2,
        "positive_negative_pairs": p * n,
        "distinct_score_thresholds": len(starts),
    }


def check_raw_labels(archive):
    """Rebuild labels in feature-row order independently of the archived loader."""
    raw = archive / "official-code/fraud_data/elliptic_plus_plus"
    manifest = json.loads((archive / "sources/data_manifest.json").read_text())
    expected = {Path(d["file"]).name: d["sha256"] for d in manifest}
    hashes = {}
    for name in ("txs_classes.csv", "txs_features.csv", "txs_edgelist.csv"):
        hashes[name] = sha256(raw / name)
        assert hashes[name] == expected[name], ("raw source hash changed", name)
    with (raw / "txs_classes.csv").open() as f:
        rows = csv.reader(f)
        next(rows)
        classes = {row[0]: {"1": 1, "2": 0, "3": -1}[row[1]] for row in rows}
    with (raw / "txs_features.csv").open() as f:
        next(f)
        ids = [line.partition(",")[0] for line in f]
    assert len(ids) == len(set(ids)) == len(classes) == 203769
    y = np.array([classes[txid] for txid in ids], dtype=np.int64)
    assert (int((y == 1).sum()), int((y == 0).sum()), int((y == -1).sum())) == (4545, 42019, 157205)
    return y, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "fresh"), default="audit")
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw_y, raw_hashes = check_raw_labels(args.archive)
    index = np.flatnonzero(raw_y >= 0)
    assert len(index) == 46564
    from sklearn import __version__ as sklearn_version, config_context
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    results = {}
    for dataset, (paper_auc, paper_pr, dim) in TARGETS.items():
        source = args.archive / "results" / (dataset + "_LOF_scores.npz")
        cached_y = np.load(args.archive / "data-cache" / (dataset + "_y.npy"))
        assert np.array_equal(raw_y, cached_y)
        if args.mode == "fresh":
            from sklearn.neighbors import LocalOutlierFactor
            x = np.load(args.archive / "data-cache" / (dataset + "_x.npy"), mmap_mode="r")
            assert x.shape == (203769, dim) and x.dtype == np.float32
            features = x[index].astype(np.float64)
            assert np.isfinite(features).all()
            with config_context(working_memory=128):
                model = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=1)
                model.fit(features)
            score, y = -model.negative_outlier_factor_, raw_y[index]
            np.savez_compressed(args.output / (dataset + "_LOF_scores.npz"),
                                score=score, y=y, node_index=index)
            del x, features, model
        else:
            with np.load(source, allow_pickle=False) as saved:
                assert np.array_equal(saved["node_index"], index)
                assert np.array_equal(saved["y"], raw_y[index])
                score, y = saved["score"], saved["y"]
        assert score.shape == (46564,) and np.isfinite(score).all()
        metrics = independent_metrics(y, score)
        pr, rc, _ = precision_recall_curve(y, score)
        checks = {"AUROC": float(roc_auc_score(y, score)),
                  "AUPRC_trapezoid": float(auc(rc, pr)),
                  "average_precision": float(average_precision_score(y, score))}
        for key, value in checks.items():
            assert abs(metrics[key] - value) < 1e-12, (dataset, key)
        comparisons = {}
        for metric, target in (("AUROC", paper_auc), ("AUPRC_trapezoid", paper_pr)):
            value = metrics[metric] * 100
            error = abs(value - target)
            comparisons[metric] = {"paper_percent": target, "local_percent": value,
                "absolute_error_percentage_points": error,
                "relative_error_percent": error / target * 100,
                "rounding_tolerance_percentage_points": 0.05,
                "within_tolerance": error <= 0.05}
        results[dataset] = {"metrics": metrics, "comparisons": comparisons,
            "nodes_total": 203769, "nodes_fit_and_evaluate": len(y),
            "positive_nodes": int(y.sum()), "features": dim,
            "archived_score_path": str(source), "archived_score_sha256": sha256(source),
            "independent_metrics_match_sklearn_at_1e_12": True}
        print(dataset, json.dumps(comparisons), flush=True)
    report = {"assessment_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "fresh_fit_performed": args.mode == "fresh",
        "paper": "arXiv:2609.38424v1", "paper_url": "https://arxiv.org/html/2609.38424v1",
        "scope": "LOF baseline only; not EB-GAD or the corporate relationship extension",
        "protocol": {"n_neighbors": 20, "score_direction": "negative negative_outlier_factor_",
            "fit_scope": "all 46564 labeled transactions, as in authors' baseline runner",
            "features": "author-loader float32 values cast to float64, no extra scaling",
            "labels_used_in_model_fit": False, "test_tuning": False,
            "uncertainty": "deterministic baseline; paper gives rounding, not SD/CI"},
        "raw_source_hashes_verified": raw_hashes, "raw_label_alignment_verified": True,
        "results": results, "python": sys.version, "numpy": np.__version__,
        "sklearn": sklearn_version, "script_sha256": sha256(Path(__file__)),
        "seconds": time.time() - start}
    path = args.output / "comparison.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2) + "\n")
    temp.replace(path)


if __name__ == "__main__":
    main()
