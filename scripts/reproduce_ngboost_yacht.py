"""Fresh 20-split reproduction of Duan et al. (2020), Tables 1 and 3.

Uses the production NGB wrapper with an explicit paper configuration. The
primary stopping criterion is validation NLL, as specified in Section 4.
Validation-MSE selection is recorded as a predeclared diagnostic because the
current author script selects it instead. Neither selection sees test labels.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from models import NGB


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_split(task):
    split, data_file, run_dir, outer_train, test, max_estimators = task
    output = Path(run_dir) / f"split-{split:02d}"
    output.mkdir()
    started = time.monotonic()
    data = np.loadtxt(data_file)
    x, y = data[:, :-1], data[:, -1]
    train, validation = train_test_split(outer_train, test_size=.2, random_state=10000+split)
    assert not (set(train) & set(validation) or set(outer_train) & set(test))
    assert set(train) | set(validation) == set(outer_train)
    np.savez(output / "indices.npz", train=train, validation=validation,
             outer_train=outer_train, test=test)

    np.random.seed(20000+split)
    tune = NGB.paper_benchmark(max_estimators, random_state=20000+split).fit(x[train], y[train])
    val_nll, val_mse = [], []
    for dist in tune.mod.staged_pred_dist(x[validation]):
        val_nll.append(float(-dist.logpdf(y[validation]).mean()))
        val_mse.append(float(np.mean((y[validation]-dist.loc)**2)))
    val_nll, val_mse = np.array(val_nll), np.array(val_mse)
    assert np.isfinite(val_nll).all() and np.isfinite(val_mse).all()
    # staged_pred_dist yields the predictions after 1, 2, ... trees.
    best_nll, best_mse = int(np.argmin(val_nll))+1, int(np.argmin(val_mse))+1
    np.savez(output / "validation.npz", nll=val_nll, mse=val_mse)
    del tune

    # Train the whole path so that the two predeclared selection rules use
    # exactly the same model fits, without extra training or test-set tuning.
    np.random.seed(30000+split)
    refit = NGB.paper_benchmark(max_estimators, random_state=30000+split).fit(x[outer_train], y[outer_train])
    result = {"split": split, "n_train": len(train), "n_validation": len(validation),
              "n_refit": len(outer_train), "n_test": len(test)}
    predictions = {"test_index": test, "y": y[test]}
    for name, best in [("paper_nll", best_nll), ("author_mse_diagnostic", best_mse)]:
        mean, variance = refit.predict(x[test], max_iter=best)
        assert np.isfinite(mean).all() and np.isfinite(variance).all() and (variance>0).all()
        nll_by_row = .5*(np.log(2*np.pi*variance)+(y[test]-mean)**2/variance)
        # Independent SciPy scoring verifies the production mean/variance interface.
        library_nll = -refit.mod.pred_dist(x[test], max_iter=best).logpdf(y[test])
        scipy_nll = -norm.logpdf(y[test], loc=mean, scale=np.sqrt(variance))
        score_error = float(max(np.max(np.abs(nll_by_row-library_nll)),
                                np.max(np.abs(nll_by_row-scipy_nll))))
        assert score_error < 1e-10
        result[name] = {"selected_estimators": best,
                        "selection_at_upper_boundary": best == len(val_nll),
                        "nll": float(nll_by_row.mean()),
                        "rmse": float(np.sqrt(np.mean((y[test]-mean)**2))),
                        "score_check_max_abs_error": score_error}
        predictions[name+"_mean"] = mean
        predictions[name+"_variance"] = variance
    np.savez(output / "predictions.npz", **predictions)
    result["elapsed_seconds"] = time.monotonic()-started
    (output / "metrics.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    data = np.loadtxt(args.data_file)
    assert data.shape == (308, 7) and np.isfinite(data).all()
    args.run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "paper": "Duan et al. (2020), NGBoost: Natural Gradient Boosting for Probabilistic Prediction",
        "paper_url": "https://proceedings.mlr.press/v119/duan20a.html",
        "scope": "Fresh training of the project's Gaussian NGBoost component on UCI Yacht; not a CRSP coforecasting replication",
        "primary_selection": "minimum validation NLL; Section 4",
        "diagnostic_selection": "minimum validation MSE; current author script; reported separately regardless of outcome",
        "paper_targets": {"nll": {"mean": .20, "reported_plus_minus": .26, "table": 1},
                          "rmse": {"mean": .50, "reported_plus_minus": .20, "table": 3}},
        "comparison_rule": "Both primary means must be within the corresponding published +/- band for a scoped approximate match. This is a descriptive tolerance, not a significance test or a CI.",
        "uncertainty_caveat": "Table 1 caption calls errors standard errors; current author script prints np.std (ddof=0). Published uncertainty type is unresolved; local SD and SEM are both reported.",
        "data_file": str(args.data_file.resolve()), "data_sha256": sha256(args.data_file),
        "shape": list(data.shape), "target": "last column: residuary resistance per unit displacement",
        "outer_splits": 20, "outer_seed": 1,
        "outer_rule": "RandomState(1).choice(n,n,replace=False); first round(.9*n) train, remaining test; author script's rule",
        "inner_rule": "train_test_split(outer_train,test_size=.2,random_state=10000+split)",
        "tuning_seed": "20000+split", "refit_seed": "30000+split",
        "model": {"distribution": "Normal", "score": "LogScore", "natural_gradient": True,
                  "max_depth": 3, "min_samples_leaf": 1, "learning_rate": .01,
                  "minibatch_frac": 1., "max_estimators": 2000,
                  "target_scale": 1., "train_limit": None, "variance_bounds": None},
        "protocol_limits": ["Modern installed library, not a pinned 2020 runtime",
                            "Inner split and tree seeds explicitly fixed locally, not recovered from the original runs",
                            "2000-stage ceiling from the current author Yacht launcher, not specified in the paper"],
        "python": platform.python_version(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "scipy", "scikit-learn", "ngboost"]},
        "script_sha256": sha256(__file__),
        "models_sha256": sha256(Path(__file__).with_name("models.py")),
    }
    # Freeze targets, protocol and tolerance before fitting any model.
    (args.run_dir / "protocol.json").write_text(json.dumps(config, indent=2)+"\n")
    rng = np.random.RandomState(1)
    tasks = []
    for split in range(20):
        permutation = rng.choice(len(data), len(data), replace=False)
        end = round(.9*len(data))
        tasks.append((split, str(args.data_file), str(args.run_dir),
                      permutation[:end], permutation[end:], 2000))
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_split, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            primary = result["paper_nll"]
            print(f"[{len(results):02d}/20] split={result['split']:02d} "
                  f"M={primary['selected_estimators']} "
                  f"NLL={primary['nll']:.6f} RMSE={primary['rmse']:.6f}", flush=True)
    results.sort(key=lambda r: r["split"])
    summary = {"completed_splits": len(results), "results": {}}
    for selection in ["paper_nll", "author_mse_diagnostic"]:
        metrics = {}
        for metric in ["nll", "rmse"]:
            values = np.array([r[selection][metric] for r in results])
            target = config["paper_targets"][metric]
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            metrics[metric] = {"mean": mean, "split_sd_ddof1": sd,
                               "split_sd_ddof0": float(values.std()), "split_sem": sd/np.sqrt(len(values)),
                               "paper_mean": target["mean"],
                               "absolute_error": abs(mean-target["mean"]),
                               "relative_error_pct": 100*abs(mean-target["mean"])/abs(target["mean"]),
                               "tolerance": target["reported_plus_minus"],
                               "within_published_band": bool(abs(mean-target["mean"]) <= target["reported_plus_minus"])}
        metrics["selected_estimators"] = [r[selection]["selected_estimators"] for r in results]
        metrics["upper_boundary_splits"] = [r["split"] for r in results if r[selection]["selection_at_upper_boundary"]]
        summary["results"][selection] = metrics
    summary["primary_both_metrics_within_band"] = all(
        summary["results"]["paper_nll"][metric]["within_published_band"] for metric in ["nll", "rmse"])
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (args.run_dir / "all_splits.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
