"""Repair only label validation; retain all frozen experiment inputs and metrics.

The original validator sorted stocks and reduced one target at a time. The
trainer reduces the original float32 (stocks, 3) array in panel order. Those
operations differ slightly in floating-point arithmetic. Replaying the actual
training transformation permits exact equality instead of relaxing tolerance.
"""
from pathlib import Path
import argparse
import json
import sys
import warnings
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from experiments.joint_price_volatility import analyze
from experiments.joint_price_volatility.common import HERE, TASK_INDEX, VARIANTS, input_hashes, now
from usdata import sha, write_json

RECOVERY = Path(__file__).resolve().parent
AUDIT = dict(forecast_files=0, date_target_checks=0, observed_normalized_values=0,
             max_absolute_normalization_difference=0.)


def validate_predictions(frame, panel, tasks):
    # Retain the frozen validator's universe, raw-label, timing and forecast checks.
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
        t = date_index[date]
        ids = np.flatnonzero(panel.eligible[:, t])
        raw = np.array(panel.labels[ids, t])
        # Exactly the array shape, order, dtype and formula from frozen Panel.day.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            normalized = ((raw - np.nanmean(raw, axis=0)) /
                          np.maximum(np.nanstd(raw, axis=0, ddof=1), 1e-8)).astype(np.float32)
        positions = np.searchsorted(ids, group.symbol.map(symbol_index).to_numpy())
        np.testing.assert_array_equal(ids[positions], group.symbol.map(symbol_index))
        for task in tasks:
            actual = group[f"normalized_{task}"].to_numpy()
            expected = normalized[positions, TASK_INDEX[task]]
            np.testing.assert_array_equal(actual, expected)
            finite = np.isfinite(expected)
            AUDIT["date_target_checks"] += 1
            AUDIT["observed_normalized_values"] += int(finite.sum())
            if finite.any():
                AUDIT["max_absolute_normalization_difference"] = max(
                    AUDIT["max_absolute_normalization_difference"],
                    float(np.max(np.abs(actual[finite] - expected[finite]))))
    AUDIT["forecast_files"] += 1
    return frame


def check_validator():
    """Verify shuffled stocks, missing labels and rejection of label corruption."""
    rng = np.random.default_rng(815)
    symbols = np.array(["P2", "P1", "P3", "P0"])
    dates = np.array(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    labels = rng.normal(size=(4, 4, 3)).astype(np.float32)
    labels[1, 0, 2] = np.nan
    eligible = np.ones((4, 4), dtype=bool)
    panel = SimpleNamespace(dates=dates, symbols=symbols, labels=labels, eligible=eligible,
                            split=lambda _: np.array([0, 1]))
    parts = []
    for t in [0, 1]:
        raw = np.array(labels[:, t])
        z = ((raw - np.nanmean(raw, axis=0)) /
             np.maximum(np.nanstd(raw, axis=0, ddof=1), 1e-8)).astype(np.float32)
        record = dict(date=dates[t], realized_date=dates[t + 2], symbol=symbols)
        for task, k in TASK_INDEX.items():
            record.update({f"pred_{task}": rng.normal(size=4), f"label_{task}": raw[:, k],
                           f"normalized_{task}": z[:, k]})
        parts.append(pd.DataFrame(record))
    frame = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=2)
    before = AUDIT.copy()
    validate_predictions(frame, panel, tuple(TASK_INDEX))
    corrupt = frame.copy()
    corrupt.loc[corrupt.index[0], "normalized_return"] += .001
    try:
        validate_predictions(corrupt, panel, tuple(TASK_INDEX))
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted normalized targets must fail exact validation")
    AUDIT.update(before)
    print("PASS: exact normalization with shuffled stocks and missing labels; corrupted targets rejected.", flush=True)


def main():
    check_validator()
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"], "Original frozen experiment sources changed"
    amendment = dict(created_utc=now(), type="analysis_validation_only", recovery_sha256=sha(__file__),
        original_analysis_sha256=sha(HERE / "analyze.py"),
        explanation="Reproduce Panel.day normalization in original panel stock order on the full three-column float32 target array; require exact equality. Original validation sorted stocks and used one-dimensional reductions, producing harmless rounding differences.",
        training_or_data_or_forecasts_changed=False, metrics_or_tests_or_selection_changed=False,
        original_inputs_sha256=frozen["inputs_sha256"])
    write_json(RECOVERY / "amendment.json", amendment)
    write_json(HERE / "status.json", dict(stage="analyzing", updated_utc=now(),
                                          recovery="exact-normalization validator"))
    try:
        analyze.validate_predictions = validate_predictions
        analyze.main()
        verification_path = HERE / "analysis/verification.json"
        verification = json.loads(verification_path.read_text())
        assert verification["status"] == "passed"
        write_json(RECOVERY / "normalization_audit.json", dict(status="passed", updated_utc=now(), **AUDIT))
        report_path = HERE / "REPORT.txt"
        with report_path.open("a") as report:
            report.write("\nANALYSIS VALIDATION AMENDMENT\n")
            report.write(amendment["explanation"] + "\n")
            report.write("The stored normalized targets match the original training transformation exactly. No training, forecasts, metrics, model selection or statistical tests were changed.\n")
            report.write("Recovery source and audit: analysis_recovery/recover_analysis.py, amendment.json and normalization_audit.json.\n")
        verification.update(analysis_validation_amendment=dict(
            source_sha256=sha(__file__), amendment_sha256=sha(RECOVERY / "amendment.json"),
            normalization_audit_sha256=sha(RECOVERY / "normalization_audit.json"), **AUDIT),
            report_sha256=sha(report_path))
        write_json(verification_path, verification)
        complete = [dict(variant=variant, seed=seed) for seed in [0, 1, 2] for variant in VARIANTS]
        write_json(HERE / "status.json", dict(stage="complete", updated_utc=now(), completed=complete,
            verdict=verification["verdict"], report=str(report_path), analysis_validation_recovered=True))
        print("RECOVERY COMPLETE", verification["verdict"], flush=True)
    except BaseException as error:
        write_json(HERE / "status.json", dict(stage="failed", updated_utc=now(), error=str(error),
                                              recovery="exact-normalization validator"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_validator()
    else:
        main()
