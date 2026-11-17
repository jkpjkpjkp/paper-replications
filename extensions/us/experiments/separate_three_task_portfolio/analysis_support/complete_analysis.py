"""Report insolvent per-seed diagnostics without dropping the main comparison.

This is an analysis-only amendment: a fixed-nominal portfolio can exhaust its
equity under full transaction costs. Such diagnostics have undefined whole-
period CAGR/Sharpe and must be reported as insolvent, rather than crash or be
silently removed. Primary ensemble failures still raise explicitly.
"""
from pathlib import Path
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd
from experiments.separate_three_task_portfolio import analyze
from experiments.separate_three_task_portfolio.common import HERE, now
from usdata import sha, write_json

EXECUTE = analyze.execute
SUMMARIZE = analyze.summarize
INSOLVENT = []


def execute(panel, forecasts, model, cost, strategy="apo", seed="ensemble", adverse=False):
    try:
        return EXECUTE(panel, forecasts, model, cost, strategy=strategy, seed=seed, adverse=adverse)
    except FloatingPointError as error:
        if str(error) != "NAV exhausted" or seed == "ensemble":
            raise
        trace = error.__traceback__
        values = None
        while trace is not None:
            if trace.tb_frame.f_code.co_name == "backtest":
                values = trace.tb_frame.f_locals
            trace = trace.tb_next
        if values is None:
            raise
        record = dict(model=model, cost=cost, strategy=strategy, seed=str(seed), insolvent=True,
            insolvency_date=str(panel.dates[values["t"] + 2]),
            last_equity=float(values["nav"]), solvent_days=len(values["rows"]),
            reason="Fixed nominal portfolio equity exhausted; whole-period return ratios undefined.")
        INSOLVENT.append(record)
        partial = pd.DataFrame(values["rows"])
        partial.to_csv(analyze.OUT / f"{model}_{seed}_{strategy}_{cost}_solvent_prefix.csv", index=False)
        write_json(analyze.OUT / f"{model}_{seed}_{strategy}_{cost}_insolvent.json", record)
        frame = pd.DataFrame()
        frame.attrs["insolvency"] = record
        print("INSOLVENT DIAGNOSTIC", json.dumps(record), flush=True)
        return frame


def summarize(ledger, model, cost, strategy="apo", seed="ensemble"):
    if "insolvency" in ledger.attrs:
        return ledger.attrs["insolvency"].copy()
    return dict(SUMMARIZE(ledger, model, cost, strategy=strategy, seed=seed), insolvent=False)


def check():
    from types import SimpleNamespace
    import numpy as np
    panel = SimpleNamespace(symbols=np.array([f"P{i}" for i in range(20)]),
        dates=np.array(["2020-01-01", "2020-01-02", "2020-01-03"]),
        member=np.ones((20, 3), bool), market=np.ones((20, 3, 7)) * 100,
        total_return=np.zeros((20, 3)), price_return=np.zeros((20, 3)),
        terminal=np.zeros((20, 3), bool), execution_tradable=np.ones((20, 3), bool))
    panel.total_return[:2, 2] = 3.
    forecasts = pd.DataFrame(dict(date=panel.dates[0], realized_date=panel.dates[2], symbol=panel.symbols,
                                 pred_return=np.arange(20), pred_volume=0., pred_volatility=0.))
    original_out = analyze.OUT
    try:
        with tempfile.TemporaryDirectory() as name:
            analyze.OUT = Path(name)
            ledger = execute(panel, forecasts, "fixture", "paper_trade10", strategy="equal", seed="0")
            result = summarize(ledger, "fixture", "paper_trade10", strategy="equal", seed="0")
            assert result["insolvent"] and result["last_equity"] < 0
            assert "sharpe" not in result and "cagr" not in result
    finally:
        analyze.OUT = original_out
        INSOLVENT.clear()
    print("PASS: per-seed insolvency is explicit and undefined whole-period metrics are not invented.", flush=True)


def main():
    check()
    analyze.execute = execute
    analyze.summarize = summarize
    analyze.main()
    verification_path = HERE / "analysis/verification.json"
    verification = json.loads(verification_path.read_text())
    amendment = dict(created_utc=now(), type="analysis-only insolvency reporting for seed diagnostics",
        source_sha256=sha(__file__), gateway_sha256=sha(Path(__file__).with_name("launch.py")),
        training_or_forecasts_or_primary_statistics_changed=False, insolvent_diagnostics=INSOLVENT)
    write_json(Path(__file__).with_name("amendment.json"), amendment)
    report_path = HERE / "REPORT.txt"
    with report_path.open("a") as report:
        report.write("\nPER-SEED INSOLVENCY DIAGNOSTICS\n")
        report.write("Fixed nominal budgets can exhaust NAV after fees. Affected seeds are reported explicitly with the failure date and negative equity; whole-period CAGR and Sharpe are undefined. Primary ensemble tests are unchanged.\n")
        report.write(json.dumps(INSOLVENT, indent=2) + "\n")
    verification.update(insolvency_reporting_amendment=amendment,
                        report_sha256=sha(report_path))
    write_json(verification_path, verification)


if __name__ == "__main__":
    main()
