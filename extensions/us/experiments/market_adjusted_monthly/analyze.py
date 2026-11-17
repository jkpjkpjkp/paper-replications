"""Matched monthly-volatility control, market adjustment and architecture effects."""
from pathlib import Path
import argparse
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from usdata import sha, write_json
from experiments.joint_price_volatility.common import correlation
from experiments.joint_price_volatility.analyze import hac, bootstrap
from experiments.market_adjusted_monthly.common import HERE, INDEX, TASKS, REGIMES, folder, input_hashes, now, protocol
from experiments.market_adjusted_monthly.data import Panel
from experiments.market_adjusted_monthly.sizing import physical_weights
from experiments.separate_three_task_portfolio import portfolio_engine as engine

OUT = HERE / "analysis"
KEYS = ["date", "realized_date", "symbol", "volatility_month", "label_end_date"]
COSTS = {"gross": dict(buy_bp=0., sell_bp=0.), "buy5": dict(buy_bp=5., sell_bp=0.),
         "trade10": dict(buy_bp=10., sell_bp=10.)}


def validate(frame, panel, tasks):
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert not frame[["date", "symbol"]].duplicated().any()
    days = panel.split("test")
    counts = frame.groupby("date", sort=True).size()
    np.testing.assert_array_equal(counts.index, panel.dates[days])
    np.testing.assert_array_equal(counts, panel.eligible[:, days].sum(0))
    di = {d: i for i, d in enumerate(panel.dates)}
    si = {s: i for i, s in enumerate(panel.symbols)}
    tt, ii = frame.date.map(di).to_numpy(), frame.symbol.map(si).to_numpy()
    assert panel.eligible[ii, tt].all()
    np.testing.assert_array_equal(frame.realized_date, panel.dates[tt + 2])
    np.testing.assert_array_equal(frame.label_end_date, panel.dates[panel.label_end[tt]])
    np.testing.assert_array_equal(frame.volatility_month, panel.dates[panel.label_end[tt]].astype("U7"))
    for task in tasks:
        assert np.isfinite(frame[f"pred_{task}"]).all()
        np.testing.assert_array_equal(frame[f"label_{task}"], panel.labels[ii, tt, INDEX[task]])
    return frame


def load(regime, task, seed, panel):
    out = folder(regime, task, seed)
    assert json.loads((out / "status.json").read_text())["phase"] == "complete"
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["inputs_sha256"] == input_hashes()
    for name, expected in provenance["artifacts_sha256"].items():
        assert sha(out / name) == expected, f"Changed artifact: {out / name}"
    tasks = TASKS if task == "joint" else (task,)
    return validate(pd.read_parquet(out / "test_predictions.parquet"), panel, tasks)


def mean_scores(parts, tasks):
    result = parts[0].copy()
    for other in parts[1:]:
        pd.testing.assert_frame_equal(result[KEYS], other[KEYS])
        for task in tasks:
            np.testing.assert_array_equal(result[f"label_{task}"], other[f"label_{task}"])
    columns = [f"pred_{task}" for task in tasks]
    result[columns] = np.mean([p[columns].to_numpy(dtype=float) for p in parts], axis=0)
    return result


def predictions(regime, panel):
    joint = mean_scores([load(regime, "joint", seed, panel) for seed in [0, 1, 2]], TASKS)
    separate = None
    for task in TASKS:
        part = mean_scores([load(regime, task, seed, panel) for seed in [0, 1, 2]], (task,))
        if separate is None:
            separate = part
        else:
            pd.testing.assert_frame_equal(separate[KEYS], part[KEYS])
            separate[f"pred_{task}"] = part[f"pred_{task}"]
            separate[f"label_{task}"] = part[f"label_{task}"]
    pd.testing.assert_frame_equal(joint[KEYS], separate[KEYS])
    return {"joint": joint, "separate": separate}


def forecast_metrics(frame, scale):
    rows = []
    for date, group in frame.groupby("date", sort=True):
        for task in TASKS:
            x, y = group[f"pred_{task}"].to_numpy(), group[f"label_{task}"].to_numpy()
            observed = np.isfinite(y)
            rows.append(dict(date=date, target_month=group.volatility_month.iloc[0], task=task,
                ic=correlation(x[observed], y[observed]),
                rank_ic=correlation(rankdata(x[observed]), rankdata(y[observed])),
                normalized_mse=float(np.mean(((x[observed] - y[observed]) / scale[INDEX[task]]) ** 2)),
                observed_targets=int(observed.sum()), stocks=len(group)))
    daily = pd.DataFrame(rows)
    units = []
    for task in TASKS:
        part = daily[daily.task == task]
        if task == "volatility":
            part = part.groupby("target_month", as_index=False).agg(ic=("ic", "mean"), rank_ic=("rank_ic", "mean"),
                normalized_mse=("normalized_mse", "mean"), observed_targets=("observed_targets", "mean"))
            part["unit"] = "target_month"
            part["date"] = part.target_month
        else:
            part = part.copy()
            part["unit"] = "trading_day"
        part["task"] = task
        units.append(part)
    return daily, pd.concat(units, ignore_index=True)


def backtest(panel, frame, system, cost, strategy="apo"):
    engine.apo_weights = physical_weights
    tag = f"{system}_{strategy}_{cost}"
    try:
        ledger, weights = engine.backtest(panel, frame, strategy=strategy, **COSTS[cost], save_weights=strategy == "apo" and cost == "trade10")
        insolvent = None
    except FloatingPointError as error:
        if str(error) != "NAV exhausted":
            raise
        trace, local = error.__traceback__, None
        while trace is not None:
            if trace.tb_frame.f_code.co_name == "backtest":
                local = trace.tb_frame.f_locals
            trace = trace.tb_next
        if local is None:
            raise
        ledger, weights = pd.DataFrame(local["rows"]), None
        insolvent = dict(date=str(panel.dates[local["t"] + 2]), equity=float(local["nav"]),
                         solvent_days=len(ledger), reason="NAV exhausted under fixed nominal budgets")
        write_json(OUT / f"{tag}_insolvent.json", insolvent)
    ledger.to_csv(OUT / f"{tag}_ledger.csv", index=False)
    if weights is not None:
        np.savez_compressed(OUT / f"{tag}_weights.npz", weights=weights, dates=ledger.date.to_numpy(dtype=str), symbols=panel.symbols)
        lookup = {d: i for i, d in enumerate(panel.dates)}
        ix = np.array([lookup[d] for d in ledger.date])
        raw = np.nan_to_num(np.asarray(panel.total_return[:, ix]).T, nan=0.)
        np.testing.assert_allclose((weights * raw).sum(1), ledger.return_gross, atol=1e-12)
    result = dict(system=system, strategy=strategy, cost=cost, days=len(ledger), insolvent=insolvent is not None)
    if len(ledger):
        result.update(mean_gross_exposure=float(ledger.gross_exposure.mean()),
            annual_cost=float((ledger.fees / ledger.start_nav).mean() * 252), missing_held_count=int(ledger.missing_held_count.sum()))
    if insolvent is not None:
        result.update(insolvency_date=insolvent["date"], terminal_equity=insolvent["equity"])
    else:
        x = ledger.return_net.to_numpy()
        wealth = np.r_[1., np.cumprod(1 + x)]
        result.update(annual_mean=float(x.mean() * 252), cagr=float(wealth[-1] ** (252 / len(x)) - 1),
            sharpe=float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
            max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()))
    print("BACKTESTED", tag, "insolvent", result["insolvent"], flush=True)
    return ledger, result


def compare(left, right, label, cost, left_bad=False, right_bad=False):
    result = dict(contrast=label, cost=cost, available=not (left_bad or right_bad))
    if not result["available"]:
        result.update(p=1., reason="Full-period paired inference unavailable after insolvency")
        return result
    pd.testing.assert_series_equal(left.date, right.date)
    gain = left.return_net.to_numpy() - right.return_net.to_numpy()
    result.update(days=len(gain))
    for lag in [20, 5, 60]:
        prefix = "" if lag == 20 else f"hac{lag}_"
        result.update({prefix + key: value * 252 if key in ["gain", "ci_low", "ci_high"] else value
                       for key, value in hac(gain, lag).items()})
    for block in [20, 60]:
        low, high = bootstrap(gain, block)
        result[f"bootstrap{block}_low"], result[f"bootstrap{block}_high"] = low * 252, high * 252
    return result


def main():
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"]
    assert sha(HERE / "data/manifest.json") == frozen["data_manifest_sha256"]
    for location, manifest in [(ROOT / "data", json.loads((ROOT / "data/manifest.json").read_text())),
                               (HERE / "data", json.loads((HERE / "data/manifest.json").read_text()))]:
        for name, expected in manifest["output_sha256"].items():
            assert sha(location / name) == expected, name
    OUT.mkdir(exist_ok=True)
    forecasts, panels = {}, {}
    forecast_rows, unit_rows = [], []
    for regime in REGIMES:
        panels[regime] = Panel(regime)
        for model, frame in predictions(regime, panels[regime]).items():
            system = f"{regime}_{model}"
            forecasts[system] = frame
            daily, units = forecast_metrics(frame, panels[regime].scale)
            daily["system"], units["system"] = system, system
            forecast_rows.append(daily)
            unit_rows.append(units)
    pd.concat(forecast_rows, ignore_index=True).to_csv(OUT / "daily_forecasts.csv", index=False)
    all_units = pd.concat(unit_rows, ignore_index=True)
    all_units.to_csv(OUT / "forecast_sampling_units.csv", index=False)
    aggregate = all_units.groupby(["system", "task", "unit"]).agg(IC=("ic", "mean"), RankIC=("rank_ic", "mean"),
        normalized_MSE=("normalized_mse", "mean"), units=("ic", "size")).reset_index()
    aggregate.to_csv(OUT / "forecast_summary.csv", index=False)
    cases, summaries, diagnostics = {}, [], []
    for system, frame in forecasts.items():
        panel = panels["market_adjusted" if system.startswith("market_adjusted") else "raw_monthly"]
        for cost in COSTS:
            ledger, summary = backtest(panel, frame, system, cost)
            cases[(system, cost)] = (ledger, summary)
            summaries.append(summary)
            _, diagnostic = backtest(panel, frame, system, cost, strategy="equal")
            diagnostics.append(diagnostic)
    contrasts = [
        ("market_adjusted_joint", "raw_monthly_joint", "adjustment_joint"),
        ("market_adjusted_separate", "raw_monthly_separate", "adjustment_separate"),
        ("market_adjusted_separate", "market_adjusted_joint", "separate_minus_joint_adjusted"),
        ("raw_monthly_separate", "raw_monthly_joint", "separate_minus_joint_raw_monthly")]
    comparisons = []
    for cost in COSTS:
        for first, second, label in contrasts:
            left, a = cases[(first, cost)]
            right, b = cases[(second, cost)]
            comparisons.append(compare(left, right, label, cost, a["insolvent"], b["insolvent"]))
    paired = pd.DataFrame(comparisons)
    for primary in [True, False]:
        mask = (paired.cost == "trade10") == primary
        paired.loc[mask, "p_holm"] = multipletests(paired.loc[mask, "p"], method="holm")[1]
    pd.DataFrame(summaries).to_csv(OUT / "portfolio_summary.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "equal_weight_diagnostics.csv", index=False)
    paired.to_csv(OUT / "paired_profitability.csv", index=False)
    primary = paired[paired.cost == "trade10"]
    verdicts = []
    for label in ["adjustment_joint", "adjustment_separate"]:
        row = primary[primary.contrast == label].iloc[0]
        if not row.available:
            text = "whole-period inference unavailable after portfolio insolvency"
        elif row.p_holm < .05:
            text = "market adjustment " + ("improves" if row.gain > 0 else "reduces") + " net profitability"
        else:
            text = "no significant net-profitability change established"
        verdicts.append(f"{label}: {text}.")
    report = ["MARKET ADJUSTMENT AND NEXT-CALENDAR-MONTH VOLATILITY", "", *verdicts, "",
        "Matched raw-monthly controls and market-adjusted regimes; joint versus three independent models; three seeds each.",
        "Return targets are CRSP total returns at t+2; market is verified Fama-French Mkt-RF+RF.",
        "Volatility is the next calendar month's sample standard deviation of DAILY total returns, unannualized.",
        "Adjusted volatility subtracts the same month's market standard deviation; it is not volatility of excess returns.",
        "Daily cross-sectional normalization would cancel the market terms. Both regimes instead share fixed raw-training mean/std.",
        "All variants purge on the full monthly realization endpoint; December signals at split boundaries are excluded.",
        "No future market value is used to convert forecasts into orders; monthly risk weights are shift-invariant.", "",
        "PORTFOLIOS: annual mean=252*daily mean; CAGR separate; insolvent systems have no whole-period ratios.",
        pd.DataFrame(summaries).to_string(index=False), "", "PAIRED APO COMPARISONS",
        paired.to_string(index=False), "", "FORECASTS: monthly volatility uses target months as sampling units",
        aggregate.to_string(index=False), "", "EQUAL-WEIGHT DIAGNOSTICS", pd.DataFrame(diagnostics).to_string(index=False), "",
        "PRIMARY: four fixed contrasts at 10bp every trade, Holm4, HAC20; gross/buy5 form a separate Holm8 family.",
        "Validation: joint selected by return IC; specialists by own task IC; volume specialist shared because its data and objective are identical.",
        "Monthly volatility is computed from at least15 and80% of month trading days; missing target observations never change inference membership.",
        protocol()["limits"], "", "Workflow attribution: scientific-critical-thinking/statsmodels; software reference in ../joint_price_volatility/README.txt.", ""]
    (HERE / "REPORT.txt").write_text("\n".join(report))
    write_json(OUT / "verification.json", dict(status="passed", created_utc=now(), verdicts=verdicts,
        inputs_sha256=input_hashes(), checks=["All21 fits completed; locked data and artifacts match.",
        "Targets, stock/date universe and monthly realization endpoints reconcile exactly.",
        "Shared scaler fits only purged raw training labels; market subtraction survives scaling.",
        "Physical APO uses only predictions and signal-day observations; no future market denormalization.",
        "Daily P&L, fees, NAV and saved-weight CRSP returns reconcile; insolvent systems explicitly retained.",
        "Volatility forecast sampling units are target months; primary and sensitivity test families fixed."],
        outputs_sha256={p.name: sha(p) for p in OUT.glob("*.csv")}, report_sha256=sha(HERE / "REPORT.txt")))


def self_check():
    rng = np.random.default_rng(23)
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=90):
        row = dict(date=str(date.date()), volatility_month=str(date.to_period("M") + 1))
        for task in TASKS:
            target = rng.normal(size=5)
            row[f"label_{task}"] = target
            row[f"pred_{task}"] = target + rng.normal(scale=.3, size=5)
        rows.append(pd.DataFrame(row))
    frame = pd.concat(rows, ignore_index=True)
    daily, units = forecast_metrics(frame, np.ones(3))
    assert len(daily[daily.task == "volatility"]) == 90
    assert len(units[units.task == "volatility"]) == frame.volatility_month.nunique()
    assert len(units[units.task == "return"]) == 90
    assert (units[units.task == "volatility"].unit == "target_month").all()
    unavailable = compare(pd.DataFrame(), pd.DataFrame(), "fixture", "trade10", left_bad=True)
    assert not unavailable["available"] and unavailable["p"] == 1.
    print("PASS: monthly volatility sampling units, daily return units and explicit unavailable inference after insolvency.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check() if args.self_check else main()
