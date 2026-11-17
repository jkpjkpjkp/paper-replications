"""Independent raw-formula checks, checkpoint replay and cash/share accounting."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transalpha.data import ROOT, DailyPanel, save_json, sha
from transalpha.model import TransAlpha, LinearControl
from transalpha.evaluation import correlation, summarize_returns


def raw_series(source, symbol, field, calendar_length):
    data = np.fromfile(source / f"features/{symbol.lower()}/{field}.day.bin", dtype="<f4")
    values = np.full(calendar_length, np.nan)
    start = int(data[0])
    end = min(start + len(data) - 1, calendar_length)
    values[start:end] = data[1:1 + end - start]
    return values


def audit_panel(universe, config):
    folder = ROOT / "data" / universe
    panel = DailyPanel(universe, config)
    manifest = panel.manifest
    for name, expected in manifest["array_hashes"].items():
        assert sha(folder / (name + ".npy")) == expected, (universe, name)
    source = Path(config["source"])
    source_hashes = json.loads((folder / "source_hashes.json").read_text())
    assert sha(folder / "source_hashes.json") == manifest["source_hashes_sha256"]
    for relative, expected in source_hashes.items():
        assert sha(source / relative) == expected, relative
    execution = json.loads((folder / "execution_manifest.json").read_text())
    for name in ["portfolio_returns", "execution_tradable"]:
        assert sha(folder / (name + ".npy")) == execution[name + "_sha256"]
    calendar = np.array((source / "calendars/day.txt").read_text().splitlines())
    normalization = np.load(folder / "normalization.npz")
    checks, errors = 0, []
    for split in ["train", "validation", "test"]:
        ids = manifest["indices"][split]
        for t in [ids[0], ids[len(ids) // 2], ids[-1]]:
            eligible = np.flatnonzero(panel.eligible[t])
            for stock in eligible[np.linspace(0, len(eligible) - 1, 3).astype(int)]:
                symbol = panel.symbols[stock]
                raw = {f: raw_series(source, symbol, f, len(calendar)) for f in ["open", "close", "high", "low", "factor"]}
                i = int(np.searchsorted(calendar, panel.dates[t]))
                adjusted = raw["close"] * raw["factor"]
                independently_computed = {
                    "KMID": (raw["close"][i] - raw["open"][i]) / raw["open"][i],
                    "KLEN": (raw["high"][i] - raw["low"][i]) / raw["open"][i],
                    "OPEN0": raw["open"][i] / raw["close"][i],
                    "ROC5": adjusted[i - 5] / adjusted[i],
                    "MA5": np.nanmean(adjusted[i - 4:i + 1]) / adjusted[i],
                }
                for name, value in independently_computed.items():
                    feature = manifest["features"].index(name)
                    expected = np.clip((value - normalization["mean"][feature]) / normalization["std"][feature], -5, 5)
                    actual = float(panel.x[t, stock, feature])
                    error = abs(expected - actual)
                    errors.append(error)
                    assert error < 4e-4, (universe, symbol, str(panel.dates[t]), name, expected, actual)
                    checks += 1
                expected_y = adjusted[i + 2] / adjusted[i + 1] - 1
                np.testing.assert_allclose(panel.y[t, stock], expected_y, rtol=1e-5, atol=1e-7)
    return {"source_files_verified": len(source_hashes), "derived_array_hashes_verified": len(manifest["array_hashes"]) + 2,
            "independent_raw_formula_comparisons": checks, "maximum_normalized_feature_error": max(errors),
            "unknown_eligible_targets": manifest["unknown_eligible_targets"]}


def cash_share_replay(prediction, entry_prices, exit_prices, tradable, rate=.001, topk=50):
    """Use shares plus a cash account; solve fees by bisection, not return formulas."""
    shares = np.zeros(prediction.shape[1], np.float64)
    cash = 1.0
    rows = []
    for step, p in enumerate(prediction):
        ep = np.nan_to_num(entry_prices[step], nan=0.)
        xp = np.nan_to_num(exit_prices[step], nan=0.)
        old_dollars = shares * ep
        before_nav = cash + old_dollars.sum()
        frozen = ~tradable[step] & (shares != 0)
        ids = np.flatnonzero(np.isfinite(p) & tradable[step])
        ids = ids[np.argsort(p[ids], kind="stable")]
        target = np.zeros_like(shares)
        frozen_weights = old_dollars[frozen] / before_nav
        target[frozen] = frozen_weights
        target[ids[-topk:]] = max(0, .5 - frozen_weights[frozen_weights > 0].sum()) / topk
        target[ids[:topk]] = -max(0, .5 + frozen_weights[frozen_weights < 0].sum()) / topk

        def plan(cost):
            dollars = target * (before_nav - cost)
            dollars[frozen] = old_dollars[frozen]
            # Treat long sales and short covers separately in dollar units.
            long_sales = np.maximum(np.maximum(old_dollars, 0) - np.maximum(dollars, 0), 0).sum()
            short_covers = np.maximum(np.maximum(-old_dollars, 0) - np.maximum(-dollars, 0), 0).sum()
            return dollars, rate * (long_sales + short_covers)

        low, high = 0., .01 * before_nav
        for _ in range(60):
            middle = (low + high) / 2
            _, cost = plan(middle)
            if middle > cost:
                high = middle
            else:
                low = middle
        fee = (low + high) / 2
        dollars, _ = plan(fee)
        new_shares = np.divide(dollars, ep, out=np.zeros_like(shares), where=ep > 0)
        assert np.allclose(new_shares[frozen], shares[frozen], atol=1e-12)
        cash -= ((new_shares - shares) * ep).sum() + fee
        shares = new_shares
        assert np.all(xp[shares != 0] > 0)
        after_nav = cash + (shares * xp).sum()
        if step == len(prediction) - 1:
            after_nav -= rate * np.abs(shares * xp).sum()
        rows.append(after_nav / before_nav - 1)
    return np.array(rows)


def audit_run(path, config):
    metrics = json.loads(path.read_text())
    universe = metrics["universe"]
    panel = DailyPanel(universe, config)
    test_ids = panel.manifest["indices"]["test"]
    saved = np.load(path.parent / "test_predictions.npz")
    p, y = saved["prediction"], saved["target"]
    assert sha(path.parent / "model.pt") == metrics["checkpoint_sha256"]
    assert sha(path.parent / "test_predictions.npz") == metrics["predictions_sha256"]
    assert np.array_equal(np.isfinite(p), panel.eligible[test_ids])
    np.testing.assert_allclose(y, panel.y[test_ids], equal_nan=True)
    assert saved["dates"].tolist() == panel.dates[test_ids].tolist()
    daily = pd.read_csv(path.parent / "daily.csv")
    independent_ic = [correlation(a[np.isfinite(a) & np.isfinite(b)], b[np.isfinite(a) & np.isfinite(b)]) for a, b in zip(p, y)]
    np.testing.assert_allclose(daily.ic, independent_ic, atol=1e-12)
    folder = ROOT / "data" / universe
    marks = pd.DataFrame(np.load(folder / "adjusted_close.npy")).ffill().to_numpy()
    rets = cash_share_replay(p, marks[np.array(test_ids) + 1], marks[np.array(test_ids) + 2],
                            np.load(folder / "execution_tradable.npy")[test_ids])
    # The production return panel is float32; this ledger uses float64 prices.
    np.testing.assert_allclose(rets, daily.net_return.to_numpy(), rtol=1e-7, atol=2e-9)
    rescored = summarize_returns(rets)
    for key, value in rescored.items():
        assert np.isclose(value, metrics["net"][key], rtol=1e-7, atol=5e-7), key
    checkpoint = torch.load(path.parent / "model.pt", map_location="cpu", weights_only=False)
    assert checkpoint["fingerprint"] == metrics["fingerprint"]
    expected_fingerprint = hashlib.sha256(json.dumps({"config": config, "variant": metrics["variant"], "seed": metrics["seed"],
        "data": panel.manifest["array_hashes"],
        "execution": json.loads((folder / "execution_manifest.json").read_text()),
        "code": {name: sha(ROOT / name) for name in ["transalpha/model.py", "transalpha/loss.py", "scripts/train.py", "transalpha/evaluation.py"]}}, sort_keys=True).encode()).hexdigest()
    assert expected_fingerprint == metrics["fingerprint"], "Current code/data/config differs from saved run"
    model = (LinearControl(158) if metrics["variant"] == "linear" else
             TransAlpha(158, config["hidden"], config["dropout"], config["threshold"], metrics["variant"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    replay_errors = []
    with torch.no_grad():
        for row in [0, len(test_ids) // 2, len(test_ids) - 1]:
            x, _, mask, stocks = panel.batch([test_ids[row]], "cpu")
            replay = model(x, mask).numpy()[0]
            expected = p[row, stocks[0]]
            np.testing.assert_allclose(replay[:len(expected)], expected, rtol=2e-4, atol=3e-6)
            replay_errors.append(float(np.max(np.abs(replay[:len(expected)] - expected))))
    history = json.loads((path.parent / "history.json").read_text())
    maximum = max(r["validation_ic"] for r in history)
    assert abs(maximum - metrics["validation_ic"]) < 1e-6
    return {"run": str(path.parent.relative_to(ROOT)), "days": len(rets),
            "maximum_cash_share_return_error": float(np.max(np.abs(rets - daily.net_return))),
            "maximum_cpu_checkpoint_replay_error": max(replay_errors),
            "forecast_eligibility_uses_only_past_data": True, "checkpoint_selected_by_validation_ic": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((ROOT / "config.json").read_text())
    result = {"panels": {u: audit_panel(u, config) for u in config["universes"]}, "runs": [], "passed": False}
    if not args.data_only:
        for path in sorted((ROOT / "results").glob("*/*/metrics.json")):
            result["runs"].append(audit_run(path, config))
    result["passed"] = True
    result["precision"] = "Independent cash/share ledger uses float64 prices; saved portfolio return panel is float32; absolute daily-return tolerance 2e-9"
    result["scope"] = "Raw formula samples, full used-source and derived hashes, forecast masks, validation selection, CPU checkpoint replay, cash/share fee and return replay"
    target = "data_audit.json" if args.data_only else "audit.json"
    save_json(ROOT / "results" / target, result)
    print(json.dumps(result, indent=2))
