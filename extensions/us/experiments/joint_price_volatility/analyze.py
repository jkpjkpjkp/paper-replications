"""Paired daily forecast comparisons; executed only after every fit completes."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from experiments.joint_price_volatility.common import HERE, TASK_INDEX, VARIANTS, correlation, input_hashes, now, protocol
from usdata import Panel, sha, write_json

OUT = HERE / "analysis"
KEYS = ["date", "realized_date", "symbol"]


def hac(values, lag=20):
    x = np.asarray(values, dtype=float)
    if len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("HAC requires a finite paired daily series")
    lag = min(lag, len(x) - 1)
    fit = sm.OLS(x, np.ones((len(x), 1)), missing="raise").fit(cov_type="HAC",
        cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": True}, use_t=True)
    residual = x - x.mean()
    variance = residual @ residual
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (residual[k:] @ residual[:-k])
    expected_se = np.sqrt(max(0., variance) / (len(x) * (len(x) - 1)))
    np.testing.assert_allclose(fit.bse[0], expected_se, atol=1e-13, rtol=1e-9)
    # Degenerate identical forecasts carry no evidence of superiority.
    if expected_se == 0:
        return dict(gain=float(x.mean()), ci_low=float(x.mean()), ci_high=float(x.mean()),
                    p=1. if x.mean() == 0 else 0.)
    low, high = fit.conf_int()[0]
    return dict(gain=float(x.mean()), ci_low=float(low), ci_high=float(high), p=float(fit.pvalues[0]))


def bootstrap(values, block, samples=5000):
    x = np.asarray(values, dtype=float)
    n = len(x)
    rng = np.random.default_rng(260925617)
    means = []
    for start in range(0, samples, 100):
        size = min(100, samples - start)
        starts = rng.integers(0, n, size=(size, (n + block - 1) // block))
        indices = ((starts[:, :, None] + np.arange(block)) % n).reshape(size, -1)[:, :n]
        means.extend(x[indices].mean(1))
    low, high = np.quantile(means, [.025, .975])
    return float(low), float(high)


def validate_predictions(frame, panel, tasks):
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert not frame[["date", "symbol"]].duplicated().any()
    days = panel.split("test")
    counts = frame.groupby("date", sort=True).size()
    np.testing.assert_array_equal(counts.index.to_numpy(), panel.dates[days])
    np.testing.assert_array_equal(counts.to_numpy(), panel.eligible[:, days].sum(0))
    date_index = {date: t for t, date in enumerate(panel.dates)}
    symbol_index = {symbol: i for i, symbol in enumerate(panel.symbols)}
    tt = frame.date.map(date_index).to_numpy()
    ii = frame.symbol.map(symbol_index).to_numpy()
    assert panel.eligible[ii, tt].all()
    np.testing.assert_array_equal(frame.realized_date, panel.dates[tt + 2])
    for task in tasks:
        assert np.isfinite(frame[f"pred_{task}"]).all()
        np.testing.assert_allclose(frame[f"label_{task}"], panel.labels[ii, tt, TASK_INDEX[task]],
                                   equal_nan=True, rtol=0, atol=1e-7)
    for date, group in frame.groupby("date", sort=True):
        for task in tasks:
            raw = group[f"label_{task}"].to_numpy()
            normalized = (raw - np.nanmean(raw)) / max(np.nanstd(raw, ddof=1), 1e-8)
            np.testing.assert_allclose(group[f"normalized_{task}"], normalized,
                                       equal_nan=True, rtol=1e-5, atol=2e-6)
    return frame


def daily_metrics(frame, tasks):
    rows = []
    for date, group in frame.groupby("date", sort=True):
        for task in tasks:
            prediction = group[f"pred_{task}"].to_numpy(dtype=float)
            raw = group[f"label_{task}"].to_numpy(dtype=float)
            normalized = group[f"normalized_{task}"].to_numpy(dtype=float)
            valid = np.isfinite(raw) & np.isfinite(normalized)
            if not np.isfinite(prediction).all():
                raise ValueError("Cannot score nonfinite predictions")
            rows.append(dict(signal_date=date, date=group.realized_date.iloc[0], target=task,
                stocks=len(group), observed_targets=int(valid.sum()),
                ic=correlation(prediction[valid], raw[valid]),
                rank_ic=correlation(rankdata(prediction[valid]), rankdata(raw[valid])),
                normalized_mse=float(np.mean((prediction[valid] - normalized[valid]) ** 2))))
    return pd.DataFrame(rows)


def paired_metrics(joint, specialists):
    paired = joint.merge(specialists, on=["signal_date", "date", "target"], suffixes=("_joint", "_separate"),
                         how="outer", validate="one_to_one", indicator=True)
    assert (paired._merge == "both").all()
    for name in ["stocks", "observed_targets"]:
        np.testing.assert_array_equal(paired[f"{name}_joint"], paired[f"{name}_separate"])
    paired["gain_ic"] = paired.ic_joint - paired.ic_separate
    paired["gain_rank_ic"] = paired.rank_ic_joint - paired.rank_ic_separate
    paired["gain_normalized_mse"] = paired.normalized_mse_separate - paired.normalized_mse_joint
    return paired.drop(columns="_merge").sort_values(["target", "date"]).reset_index(drop=True)


def summarize_comparison(paired, uncertainty=True):
    rows = []
    for task, group in paired.groupby("target", sort=True):
        for metric in ["ic", "rank_ic", "normalized_mse"]:
            values = group[f"gain_{metric}"].to_numpy()
            row = dict(target=task, metric=metric, days=len(group),
                       joint=float(group[f"{metric}_joint"].mean()),
                       separate=float(group[f"{metric}_separate"].mean()),
                       gain=float(values.mean()))
            if uncertainty:
                row.update(hac(values))
                for lag in [5, 60]:
                    row.update({f"hac{lag}_{key}": value for key, value in hac(values, lag).items()})
                for block in [20, 60]:
                    low, high = bootstrap(values, block)
                    row.update({f"bootstrap{block}_low": low, f"bootstrap{block}_high": high})
            rows.append(row)
    result = pd.DataFrame(rows)
    if uncertainty:
        for is_primary in [True, False]:
            mask = (result.metric == "ic") == is_primary
            result.loc[mask, "p_holm"] = multipletests(result.loc[mask, "p"], method="holm")[1]
        result["family"] = np.where(result.metric == "ic", "primary_two_IC_tests", "secondary_four_tests")
    return result


def load_variant(variant, panel, task_selected=False):
    tasks = VARIANTS[variant]
    total = None
    seed_metrics = []
    resources = []
    frozen = json.loads((HERE / "locked_before_training.json").read_text())["inputs_sha256"]
    filename = "test_task_selected_predictions.parquet" if task_selected else "test_predictions.parquet"
    for seed in protocol()["seeds"]:
        folder = HERE / "results" / variant / f"seed{seed}"
        status = json.loads((folder / "status.json").read_text())
        assert status["phase"] == "complete"
        provenance = json.loads((folder / "provenance.json").read_text())
        assert provenance["inputs_sha256"] == frozen
        for name, expected in provenance["artifacts_sha256"].items():
            assert sha(folder / name) == expected, f"Changed artifact: {folder / name}"
        part = validate_predictions(pd.read_parquet(folder / filename), panel, tasks)
        diagnostics = daily_metrics(part, tasks)
        diagnostics["seed"] = seed
        diagnostics["variant"] = variant
        seed_metrics.append(diagnostics)
        resources.append(dict(variant=variant, seed=seed, parameters=provenance["parameters"],
                              epochs=status["epochs"], best_epoch=status["best_epoch"],
                              training_seconds=status["training_seconds"]))
        if total is None:
            total = part.copy()
            for task in tasks:
                total[f"pred_{task}"] = total[f"pred_{task}"].astype(float)
        else:
            pd.testing.assert_frame_equal(total[KEYS], part[KEYS])
            for task in tasks:
                for prefix in ["label", "normalized"]:
                    np.testing.assert_array_equal(total[f"{prefix}_{task}"], part[f"{prefix}_{task}"])
                total[f"pred_{task}"] += part[f"pred_{task}"].to_numpy(dtype=float)
    for task in tasks:
        total[f"pred_{task}"] /= len(protocol()["seeds"])
    return daily_metrics(total, tasks), pd.concat(seed_metrics, ignore_index=True), resources


def verdict(comparison):
    primary = comparison[comparison.metric == "ic"]
    improved = primary[(primary.gain > 0) & (primary.p_holm < .05)].target.tolist()
    worse = primary[(primary.gain < 0) & (primary.p_holm < .05)].target.tolist()
    if len(improved) == 2:
        return "Joint prediction improves both targets under the declared primary test."
    if improved:
        answer = "Joint prediction improves " + ", ".join(improved) + "; improvement on both targets is not established."
    else:
        answer = "The declared primary test does not establish that joint prediction improves either target."
    if worse:
        answer += " Joint is significantly worse for " + ", ".join(worse) + "."
    return answer


def main():
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert frozen["inputs_sha256"] == input_hashes(), "Experiment source changed"
    OUT.mkdir(exist_ok=True)
    panel = Panel()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        assert sha(ROOT / "data" / name) == expected, f"Changed data array: {name}"
    metrics, diagnostics, resource_rows = {}, [], []
    for variant in VARIANTS:
        metrics[variant], seeds, resources = load_variant(variant, panel)
        diagnostics.append(seeds)
        resource_rows.extend(resources)
        metrics[variant].to_csv(OUT / f"{variant}_ensemble_daily.csv", index=False)
    specialists = pd.concat([metrics["return_only"], metrics["volatility_only"]], ignore_index=True)
    paired = paired_metrics(metrics["joint"], specialists)
    comparison = summarize_comparison(paired)
    comparison.to_csv(OUT / "comparison.csv", index=False)
    paired.to_csv(OUT / "paired_daily.csv", index=False)
    all_seeds = pd.concat(diagnostics, ignore_index=True)
    all_seeds.to_csv(OUT / "per_seed_daily.csv", index=False)
    seed_summaries = []
    for seed in protocol()["seeds"]:
        subset = all_seeds[all_seeds.seed == seed]
        paired_seed = paired_metrics(subset[subset.variant == "joint"].drop(columns=["seed", "variant"]),
            subset[subset.variant != "joint"].drop(columns=["seed", "variant"]))
        seed_summary = summarize_comparison(paired_seed, uncertainty=False)
        seed_summary["seed"] = seed
        seed_summaries.append(seed_summary)
    pd.concat(seed_summaries, ignore_index=True).to_csv(OUT / "per_seed_comparison.csv", index=False)
    year_rows = []
    for year, group in paired.groupby(paired.date.str[:4], sort=True):
        summary = summarize_comparison(group, uncertainty=False)
        summary["realization_year"] = year
        year_rows.append(summary)
    pd.concat(year_rows, ignore_index=True).to_csv(OUT / "calendar_years.csv", index=False)
    selected, _, _ = load_variant("joint", panel, task_selected=True)
    sensitivity_paired = paired_metrics(selected, specialists)
    sensitivity = summarize_comparison(sensitivity_paired)
    sensitivity.to_csv(OUT / "task_selected_sensitivity.csv", index=False)
    sensitivity_paired.to_csv(OUT / "task_selected_paired_daily.csv", index=False)
    resources = pd.DataFrame(resource_rows)
    resources.to_csv(OUT / "resources.csv", index=False)
    outcome = verdict(comparison)
    joint_parameters = int(resources[resources.variant == "joint"].parameters.iloc[0])
    separate_parameters = int(resources[resources.variant != "joint"].groupby("variant").parameters.first().sum())
    joint_hours = float(resources[resources.variant == "joint"].training_seconds.sum() / 3600)
    separate_hours = float(resources[resources.variant != "joint"].training_seconds.sum() / 3600)
    report = ["JOINT PRICE-RETURN / VOLATILITY VS TWO SPECIALISTS", "", outcome, "",
        "Primary: three-seed ensembles, one deployable joint checkpoint per seed, identical held-out dates and stocks.",
        "Positive gain favors joint for every row (specialist-minus-joint for MSE).", "",
        comparison[["target", "metric", "days", "joint", "separate", "gain", "ci_low", "ci_high", "p_holm"]].to_string(index=False),
        "", f"Parameters per seed: joint {joint_parameters:,}; two specialists together {separate_parameters:,}.",
        f"Actual training hours across all three seeds: joint {joint_hours:.3f}; specialists together {separate_hours:.3f}.",
        "Wall times reflect a shared GPU and are descriptive, not a controlled hardware benchmark.", "",
        "METHODS", "Train 2008-2014; validation 2015-2016; test 2017-2025; t+2 label boundaries purged.",
        "Price means CRSP price return, not price level. Volatility means high-low/OHLC4, not realized standard deviation.",
        "Pinned LiMT 256-dimensional encoder, eight experts, dropout .5, Adam 1e-5, full-cross-section batches.",
        "Joint retains volatility-to-return fusion; volume and its supervision are removed from all variants.",
        "Corresponding same-seed common parameters start identically. Joint uses equal task MSE weights; specialists use one MSE.",
        "One joint checkpoint selected by average validation IC; specialists by their own validation IC. Cap100/patience21.",
        "Daily Pearson IC is primary: two two-sided Bartlett HAC20 tests, t intervals, small-sample correction, Holm across targets.",
        "RankIC and normalized MSE are secondary with Holm across four tests; target normalization is per-day cross-sectional.",
        "HAC5/60 and block20/60 bootstrap intervals are provided in comparison.csv; seeds are not independent trading dates.", "",
        "SENSITIVITY: PER-TARGET VALIDATION CHECKPOINTS (exploratory; two joint model snapshots needed)",
        sensitivity[["target", "metric", "joint", "separate", "gain", "ci_low", "ci_high", "p_holm"]].to_string(index=False), "",
        "LIMITATIONS", protocol()["evaluation"]["limits"],
        "This compares the joint LiMT-style design against specialists, including the fusion difference; it does not isolate sharing alone.",
        "No inference about trading profitability follows from predictive accuracy alone.", "",
        "SOFTWARE ACKNOWLEDGMENT", "Design/inference guidance: scientific-critical-thinking and statsmodels skills.",
        "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills:",
        "A Library of Procedural Knowledge for Research Agents. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065",
        "Metadata verified against arXiv on 2026-11-07; current record v2.", ""]
    (HERE / "REPORT.txt").write_text("\n".join(report))
    write_json(OUT / "verification.json", dict(status="passed", created_utc=now(), verdict=outcome,
        statsmodels=statsmodels.__version__, checks=["All nine fits complete; code/protocol locks match.",
        "Every checkpoint, forecast and frozen data array matches its SHA256.",
        "Forecasts cover exactly every eligible test date/stock; labels, normalized targets and offsets reconcile.",
        "All forecasts finite; joint and specialists use identical paired observations.",
        "Selected checkpoints replayed first/middle/last validation and test dates during training finalization.",
        "HAC standard errors independently reconciled to Bartlett arithmetic.",
        "One joint checkpoint is primary; task-selected checkpoint results are separately labeled exploratory."],
        inputs_sha256=input_hashes(), primary_tests=2, secondary_tests=4,
        output_sha256={path.name: sha(path) for path in sorted(OUT.glob("*.csv"))}, report_sha256=sha(HERE / "REPORT.txt")))
    print(outcome, flush=True)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
