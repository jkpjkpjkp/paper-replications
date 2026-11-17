"""Synthetic end-to-end scoring/inference checks in the analysis environment."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from experiments.joint_price_volatility.analyze import (
    bootstrap, daily_metrics, hac, paired_metrics, summarize_comparison, verdict,
)
from experiments.joint_price_volatility import analyze
from experiments.joint_price_volatility.common import VARIANTS, input_hashes
from usdata import sha


def check_complete_report(joint, separate):
    """Nine synthetic run artifacts exercise validation through the final report."""
    with tempfile.TemporaryDirectory(prefix="limt-inference-check-") as name:
        root = Path(name)
        here = root / "experiment"
        here.mkdir()
        data = root / "data"
        data.mkdir()
        dates = np.array(pd.bdate_range("2020-01-01", periods=92).strftime("%Y-%m-%d"))
        symbols = np.array([f"P{i}" for i in range(20)])
        labels = np.full((20, 92, 3), np.nan)
        for t, (_, group) in enumerate(joint.groupby("date", sort=True)):
            group = group.set_index("symbol").loc[symbols]
            labels[:, t, 0] = group.label_return
            labels[:, t, 2] = group.label_volatility
        eligible = np.ones((20, 92), dtype=bool)
        for key, value in dict(labels=labels, eligible=eligible, dates=dates, symbols=symbols).items():
            np.save(data / f"{key}.npy", value)
        (data / "manifest.json").write_text(json.dumps({"output_sha256": {p.name: sha(p) for p in data.glob("*.npy")}}))
        panel = SimpleNamespace(dates=dates, symbols=symbols, labels=labels, eligible=eligible,
                                split=lambda _: np.arange(90))
        frozen = input_hashes()
        (here / "locked_before_training.json").write_text(json.dumps({"inputs_sha256": frozen}))
        for variant, tasks in VARIANTS.items():
            for seed in [0, 1, 2]:
                folder = here / "results" / variant / f"seed{seed}"
                folder.mkdir(parents=True)
                columns = ["date", "realized_date", "symbol"]
                columns += [f"{prefix}_{task}" for task in tasks for prefix in ["pred", "label", "normalized"]]
                frame = (joint if variant == "joint" else separate)[columns]
                frame.to_parquet(folder / "test_predictions.parquet", index=False)
                if variant == "joint":
                    frame.to_parquet(folder / "test_task_selected_predictions.parquet", index=False)
                (folder / "status.json").write_text(json.dumps(dict(phase="complete", epochs=3,
                    best_epoch=2, training_seconds=1.5)))
                (folder / "provenance.json").write_text(json.dumps(dict(inputs_sha256=frozen,
                    parameters=100 if variant == "joint" else 90,
                    artifacts_sha256={p.name: sha(p) for p in folder.glob("*.parquet")})))
        original = analyze.HERE, analyze.ROOT, analyze.OUT, analyze.Panel
        try:
            analyze.HERE, analyze.ROOT, analyze.OUT, analyze.Panel = here, root, here / "analysis", lambda: panel
            analyze.main()
            verification = json.loads((here / "analysis/verification.json").read_text())
            assert verification["status"] == "passed"
            assert "improves both" in verification["verdict"]
            assert (here / "REPORT.txt").is_file()
            assert len(pd.read_csv(here / "analysis/resources.csv")) == 9
            assert len(pd.read_csv(here / "analysis/comparison.csv")) == 6
        finally:
            analyze.HERE, analyze.ROOT, analyze.OUT, analyze.Panel = original
    print("PASSED: all nine synthetic artifacts through label/coverage/hash checks, ensembles and final report.")


def main():
    rng = np.random.default_rng(19)
    rows = []
    for day, date in enumerate(pd.bdate_range("2020-01-01", periods=90)):
        target = rng.normal(size=20)
        normalized = (target - target.mean()) / target.std(ddof=1)
        for i in range(20):
            rows.append(dict(date=str(date.date()), realized_date=str((date + pd.offsets.BDay(2)).date()),
                symbol=f"P{i}", label_return=target[i], label_volatility=target[i],
                normalized_return=normalized[i], normalized_volatility=normalized[i],
                pred_return=normalized[i] + rng.normal(scale=.1),
                pred_volatility=normalized[i] + rng.normal(scale=.1)))
    joint = pd.DataFrame(rows)
    separate = joint.copy()
    for task in ["return", "volatility"]:
        separate[f"pred_{task}"] += rng.normal(scale=.9, size=len(separate))
    left = daily_metrics(joint, ["return", "volatility"])
    right = daily_metrics(separate, ["return", "volatility"])
    paired = paired_metrics(left, right)
    summary = summarize_comparison(paired)
    assert (summary.gain > 0).all()
    assert (summary.p_holm < .05).all()
    assert "improves both" in verdict(summary)
    reversed_summary = summarize_comparison(paired_metrics(right, left))
    assert (reversed_summary.gain < 0).all()
    assert "significantly worse" in verdict(reversed_summary)
    identical = summarize_comparison(paired_metrics(left, left))
    np.testing.assert_array_equal(identical.gain, 0.)
    np.testing.assert_array_equal(identical.p_holm, 1.)
    assert "does not establish" in verdict(identical)
    try:
        paired_metrics(left, right.iloc[1:])
    except AssertionError:
        pass
    else:
        raise AssertionError("Missing paired dates must fail")
    noisy = rng.normal(size=120)
    for t in range(1, len(noisy)):
        noisy[t] += .7 * noisy[t - 1]
    stats = hac(noisy)
    assert stats["ci_low"] < stats["gain"] < stats["ci_high"]
    for block in [20, 60]:
        np.testing.assert_array_equal(bootstrap(noisy, block), bootstrap(noisy, block))
    print("PASSED: paired IC/RankIC/MSE, gain directions, both superiority verdicts, identical forecasts,")
    print("missing-date rejection, independent HAC arithmetic and reproducible block bootstrap.")
    check_complete_report(joint, separate)


if __name__ == "__main__":
    main()
