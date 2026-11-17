"""Same APO and execution assumptions, joint versus three separate forecasters."""
from pathlib import Path
import argparse
import json
import sys
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from experiments.joint_price_volatility.analyze import hac, bootstrap
from experiments.separate_three_task_portfolio.common import HERE, OLD, TASK_INDEX, input_hashes, now, protocol, source_hashes
from experiments.separate_three_task_portfolio.portfolio_engine import backtest
from usdata import Panel, sha, write_json

OUT = HERE / "analysis"
KEYS = ["date", "realized_date", "symbol"]
PRED = [f"pred_{task}" for task in TASK_INDEX]


def validate(frame, panel, tasks):
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    assert not frame[["date", "symbol"]].duplicated().any()
    days = panel.split("test")
    counts = frame.groupby("date", sort=True).size()
    np.testing.assert_array_equal(counts.index, panel.dates[days])
    np.testing.assert_array_equal(counts, panel.eligible[:, days].sum(0))
    di = {date: t for t, date in enumerate(panel.dates)}
    si = {symbol: i for i, symbol in enumerate(panel.symbols)}
    tt = frame.date.map(di).to_numpy()
    ii = frame.symbol.map(si).to_numpy()
    assert panel.eligible[ii, tt].all()
    np.testing.assert_array_equal(frame.realized_date, panel.dates[tt + 2])
    for task in tasks:
        assert np.isfinite(frame[f"pred_{task}"]).all()
        np.testing.assert_array_equal(frame[f"label_{task}"], panel.labels[ii, tt, TASK_INDEX[task]])
    normalized_tasks = [task for task in tasks if f"normalized_{task}" in frame]
    if normalized_tasks:
        for date, group in frame.groupby("date", sort=True):
            t = di[date]
            ids = np.flatnonzero(panel.eligible[:, t])
            raw = np.array(panel.labels[ids, t])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                normalized = ((raw - np.nanmean(raw, axis=0)) /
                    np.maximum(np.nanstd(raw, axis=0, ddof=1), 1e-8)).astype(np.float32)
            position = np.searchsorted(ids, group.symbol.map(si).to_numpy())
            for task in normalized_tasks:
                np.testing.assert_array_equal(group[f"normalized_{task}"], normalized[position, TASK_INDEX[task]])
    return frame


def ensemble(parts, tasks):
    result = parts[0].copy()
    for part in parts[1:]:
        pd.testing.assert_frame_equal(result[KEYS], part[KEYS])
        for task in tasks:
            np.testing.assert_array_equal(result[f"label_{task}"], part[f"label_{task}"])
    # Match the original ensemble's float32 arithmetic exactly.
    columns = [f"pred_{task}" for task in tasks]
    result[columns] = np.mean([part[columns].to_numpy() for part in parts], axis=0)
    return result


def combine(specialists):
    result = specialists["return"].copy()
    for task in ["volume", "volatility"]:
        part = specialists[task]
        pd.testing.assert_frame_equal(result[KEYS], part[KEYS])
        for prefix in ["pred", "label", "normalized"]:
            column = f"{prefix}_{task}"
            if column in part:
                result[column] = part[column]
    assert np.isfinite(result[PRED]).all().all()
    return result


def load_sources(panel):
    joint, independent = [], []
    for seed in protocol()["seeds"]:
        folder = ROOT / "results" / f"seed{seed}"
        assert json.loads((folder / "status.json").read_text())["phase"] == "complete"
        joint.append(validate(pd.read_parquet(folder / "test_predictions.parquet"), panel, tuple(TASK_INDEX)))
        specialists = {}
        for task in TASK_INDEX:
            folder = (HERE / "results/volume_only" if task == "volume" else OLD / "results" / f"{task}_only") / f"seed{seed}"
            assert json.loads((folder / "status.json").read_text())["phase"] == "complete"
            provenance = json.loads((folder / "provenance.json").read_text())
            for name, expected in provenance["artifacts_sha256"].items():
                assert sha(folder / name) == expected, f"Changed artifact: {folder / name}"
            specialists[task] = validate(pd.read_parquet(folder / "test_predictions.parquet"), panel, (task,))
        independent.append(combine(specialists))
        pd.testing.assert_frame_equal(joint[-1][KEYS], independent[-1][KEYS])
    return joint, independent


def summarize(ledger, model, cost, strategy="apo", seed="ensemble"):
    x = ledger.return_net.to_numpy()
    wealth = np.r_[1., np.cumprod(1 + x)]
    stats = hac(x)
    return dict(model=model, cost=cost, strategy=strategy, seed=str(seed), days=len(x),
        annual_mean=float(x.mean() * 252), cagr=float(wealth[-1] ** (252 / len(x)) - 1),
        sharpe=float(x.mean() / x.std(ddof=1) * np.sqrt(252)),
        max_drawdown=float((wealth / np.maximum.accumulate(wealth) - 1).min()),
        mean_gross_exposure=float(ledger.gross_exposure.mean()),
        max_abs_net_exposure=float(ledger.net_exposure.abs().max()),
        annual_two_way_turnover=float(((ledger.buy_value + ledger.sell_value + ledger.liquidation_buy + ledger.liquidation_sell) / ledger.start_nav).mean() * 252),
        annual_cost=float((ledger.fees / ledger.start_nav).mean() * 252),
        missing_held_count=int(ledger.missing_held_count.sum()),
        mean_ci_low=stats["ci_low"] * 252, mean_ci_high=stats["ci_high"] * 252)


def compare(joint, separate, cost):
    pd.testing.assert_frame_equal(joint[["signal_date", "execution_date", "date"]],
                                  separate[["signal_date", "execution_date", "date"]])
    gain = separate.return_net.to_numpy() - joint.return_net.to_numpy()
    result = dict(cost=cost, days=len(gain))
    stats = hac(gain)
    result.update({key: value * 252 if key in ["gain", "ci_low", "ci_high"] else value for key, value in stats.items()})
    for lag in [5, 60]:
        result.update({f"hac{lag}_{key}": value * 252 if key in ["gain", "ci_low", "ci_high"] else value
                       for key, value in hac(gain, lag).items()})
    for block in [20, 60]:
        low, high = bootstrap(gain, block)
        result.update({f"bootstrap{block}_low": low * 252, f"bootstrap{block}_high": high * 252})
    return result


def execute(panel, forecasts, model, cost, strategy="apo", seed="ensemble", adverse=False):
    tag = f"{model}_{seed}_{strategy}_{cost}" + ("_missing_stress" if adverse else "")
    save = seed == "ensemble" and strategy == "apo" and cost in ["gross", "paper_trade10"] and not adverse
    ledger, weights = backtest(panel, forecasts, strategy=strategy,
                              **protocol()["portfolio"]["costs"][cost], adverse_missing=adverse, save_weights=save)
    ledger.to_csv(OUT / f"{tag}_ledger.csv", index=False)
    if weights is not None:
        np.savez_compressed(OUT / f"{tag}_weights.npz", weights=weights, dates=ledger.date.to_numpy(dtype=str), symbols=panel.symbols)
        lookup = {date: t for t, date in enumerate(panel.dates)}
        indices = np.array([lookup[date] for date in ledger.date])
        raw = np.nan_to_num(np.asarray(panel.total_return[:, indices]).T, nan=0.)
        np.testing.assert_allclose((weights * raw).sum(1), ledger.return_gross, atol=1e-12)
    costs = protocol()["portfolio"]["costs"][cost]
    np.testing.assert_allclose(ledger.fees,
        (ledger.buy_value + ledger.liquidation_buy) * costs["buy_bp"] / 1e4 +
        (ledger.sell_value + ledger.liquidation_sell) * costs["sell_bp"] / 1e4, atol=1e-7)
    print("BACKTESTED", tag, flush=True)
    return ledger


def main():
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"]
    assert source_hashes() == frozen["reused_sources_sha256"]
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        assert sha(ROOT / "data" / name) == expected, f"Changed array: {name}"
    OUT.mkdir(exist_ok=True)
    panel = Panel()
    joint_parts, separate_parts = load_sources(panel)
    forecasts = {"joint": ensemble(joint_parts, tuple(TASK_INDEX)),
                 "separate": ensemble(separate_parts, tuple(TASK_INDEX))}
    pd.testing.assert_frame_equal(forecasts["joint"][KEYS], forecasts["separate"][KEYS])
    summaries, comparisons, diagnostics = [], [], []
    for cost in protocol()["portfolio"]["costs"]:
        ledgers = {}
        for model in ["joint", "separate"]:
            ledgers[model] = execute(panel, forecasts[model], model, cost)
            summaries.append(summarize(ledgers[model], model, cost))
            if model == "joint" and cost in ["gross", "legacy_buy5"]:
                old = pd.read_csv(ROOT / "analysis" / f"LiMT_ensemble_apo_buy{0 if cost == 'gross' else 5}_ledger.csv")
                pd.testing.assert_series_equal(ledgers[model].date, old.date)
                for column in ["return_net", "return_gross", "start_nav", "end_nav", "fees", "gross_exposure"]:
                    np.testing.assert_allclose(ledgers[model][column], old[column], rtol=1e-10, atol=1e-10)
        comparisons.append(compare(ledgers["joint"], ledgers["separate"], cost))
        paired = ledgers["joint"][["signal_date", "date"]].copy()
        paired["joint_return"] = ledgers["joint"].return_net
        paired["separate_return"] = ledgers["separate"].return_net
        paired["separate_minus_joint"] = paired.separate_return - paired.joint_return
        paired.to_csv(OUT / f"paired_{cost}.csv", index=False)
        for model in ["joint", "separate"]:
            diagnostic = execute(panel, forecasts[model], model, cost, strategy="equal")
            diagnostics.append(summarize(diagnostic, model, cost, strategy="equal"))
    for task in TASK_INDEX:
        mixed = forecasts["joint"].copy()
        mixed[f"pred_{task}"] = forecasts["separate"][f"pred_{task}"]
        ledger = execute(panel, mixed, f"swap_{task}_only", "paper_trade10")
        diagnostics.append(summarize(ledger, f"swap_{task}_only", "paper_trade10"))
    for model, parts in [("joint", joint_parts), ("separate", separate_parts)]:
        for seed, part in enumerate(parts):
            ledger = execute(panel, part, model, "paper_trade10", seed=str(seed))
            diagnostics.append(summarize(ledger, model, "paper_trade10", seed=str(seed)))
        stress = execute(panel, forecasts[model], model, "paper_trade10", adverse=True)
        diagnostics.append(summarize(stress, model + "_missing_stress", "paper_trade10"))
    summary = pd.DataFrame(summaries)
    paired = pd.DataFrame(comparisons)
    paired["p_holm"] = multipletests(paired.p, method="holm")[1]
    summary.to_csv(OUT / "summary.csv", index=False)
    paired.to_csv(OUT / "paired_profitability.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(OUT / "diagnostics.csv", index=False)
    primary = paired[paired.cost == "paper_trade10"].iloc[0]
    if primary.p_holm < .05:
        verdict = "Three separate models have " + ("higher" if primary.gain > 0 else "lower") + " APO net profitability at 10bp per trade."
    else:
        verdict = "The comparison does not establish an APO net-profitability difference at 10bp per trade."
    report = ["THREE SEPARATE FORECASTERS VERSUS THREE-HEAD LiMT: APO PROFITABILITY", "", verdict, "",
        "Test 2017-2025; three-seed score ensembles, matching dates and stocks. Annual mean=252*daily mean.",
        "Higher gain means separate minus joint. Gross and legacy buy5 are sensitivity cases.", "",
        summary[["model", "cost", "annual_mean", "cagr", "sharpe", "max_drawdown", "mean_gross_exposure", "annual_cost"]].to_string(index=False), "",
        "PAIRED ANNUAL-MEAN DIFFERENCES, HAC20 AND HOLM ACROSS THE THREE COST CASES", paired.to_string(index=False), "",
        "DIAGNOSTICS: EQUAL-WEIGHT RANKING, SINGLE-OUTPUT SWAPS, SEEDS AND MISSING-RETURN STRESS",
        pd.DataFrame(diagnostics).to_string(index=False), "",
        "Return ranks choose top/bottom deciles; predicted volume shock and volatility jointly size APO positions.",
        "APO r=.1; USD1bn nominal per side; 2% target and trade caps; equal-dollar executable legs; residual cash earns zero.",
        "paper_trade10 charges 10bp on every opening/rebalance and terminal liquidation trade, both directions.",
        "Original gross and buy5 joint ledgers independently reproduced; fees, weights, CRSP payoffs and NAV reconciled.",
        "Prices use the declared US OHLC4-for-VWAP substitution. These are idealized retrospective US results.", "",
        "MODEL SELECTION AND LIMITATIONS", protocol()["inference"]["limits"], "",
        "The earlier nine fits were joint return/volatility, return-only and volatility-only, each at three seeds.",
        "They were nine training runs, not nine bug fixes. This experiment adds three volume-only training runs.",
        "The earlier analysis needed one normalization-order validator repair; no trained models or forecasts changed.", "",
        "Workflow attribution: scientific-critical-thinking and statsmodels skills; see the software acknowledgment in ../joint_price_volatility/README.txt.", ""]
    (HERE / "REPORT.txt").write_text("\n".join(report))
    write_json(OUT / "verification.json", dict(status="passed", created_utc=now(), verdict=verdict,
        checks=["Frozen code, reused source artifacts and all data-array hashes match.",
                "All three new volume specialists and all reused joint/return/volatility models complete.",
                "Every prediction matches the exact test universe, target and timing; stored standardized labels replay exactly.",
                "Original joint APO gross and buy5 ledgers reproduce unchanged.",
                "Fees include opening/rebalance and all final liquidation trades.",
                "Dollar neutrality, capital accounting and saved-weight CRSP payoffs reconcile.",
                "Paired unique trading days, independently verified HAC arithmetic and fixed three-test Holm correction."],
        inputs_sha256=input_hashes(), reused_sources_sha256=source_hashes(),
        outputs_sha256={path.name: sha(path) for path in OUT.glob("*.csv")}, report_sha256=sha(HERE / "REPORT.txt")))
    print(verdict, flush=True)


def self_check():
    frame = pd.DataFrame(dict(date=["a", "b"], realized_date=["c", "d"], symbol=["P0", "P0"]))
    parts = {}
    for task in TASK_INDEX:
        part = frame.copy()
        part[f"pred_{task}"] = np.array([.1, .2], dtype=np.float32)
        part[f"label_{task}"] = [.2, .3]
        parts[task] = part
    combined = combine(parts)
    assert len(combined) == 2 and np.isfinite(combined[PRED]).all().all()
    broken = dict(parts)
    broken["volume"] = parts["volume"].iloc[::-1]
    try:
        combine(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Misaligned outputs must not combine")
    equal = ensemble([combined, combined, combined], tuple(TASK_INDEX))
    np.testing.assert_allclose(equal[PRED], combined[PRED], rtol=1e-6)
    print("PASS: forecast alignment, combined outputs and ensemble arithmetic.", flush=True)


def baseline_only():
    global OUT
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert input_hashes() == frozen["inputs_sha256"]
    assert source_hashes() == frozen["reused_sources_sha256"]
    OUT = HERE / "baseline"
    OUT.mkdir(exist_ok=True)
    panel = Panel()
    parts = [validate(pd.read_parquet(ROOT / "results" / f"seed{seed}" / "test_predictions.parquet"),
                      panel, tuple(TASK_INDEX)) for seed in [0, 1, 2]]
    predictions = ensemble(parts, tuple(TASK_INDEX))
    rows = []
    for cost in protocol()["portfolio"]["costs"]:
        ledger = execute(panel, predictions, "joint", cost)
        rows.append(summarize(ledger, "joint", cost))
        if cost in ["gross", "legacy_buy5"]:
            old = pd.read_csv(ROOT / "analysis" / f"LiMT_ensemble_apo_buy{0 if cost == 'gross' else 5}_ledger.csv")
            np.testing.assert_allclose(ledger.return_net, old.return_net, rtol=1e-10, atol=1e-12)
    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.baseline_only:
        baseline_only()
    else:
        main()
