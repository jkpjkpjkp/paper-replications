"""Post-run annual diagnostics, without changing any frozen experiment input.

Returns are compounded calendar-period returns from the continuous ledger.
Yearly FF5+UMD alpha is the daily intercept *252, fitted separately by year.
Annual hypotheses are descriptive additions, not the original confirmatory family.
"""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from usdata import sha, write_json
from experiments.standard_characteristics.common import HERE, FACTORS, input_hashes, now
from experiments.standard_characteristics.analyze import fit_alpha, performance, NAMES


def main():
    analysis = HERE / "analysis"
    output = analysis / "per_year"
    output.mkdir(exist_ok=True)
    verification = json.loads((analysis / "verification.json").read_text())
    assert verification["status"] == "passed"
    assert verification["inputs_sha256"] == input_hashes(), "Frozen inputs changed"
    factor = pd.read_parquet(FACTORS)
    factor["date"] = factor.date.astype(str)
    assert not factor.date.duplicated().any()
    rows, source_hashes, reconciliations = [], {}, []
    for bp in [0, 10]:
        file = analysis / f"value_cost{bp}_ledger.csv"
        assert sha(file) == verification["output_sha256"][file.name]
        source_hashes[str(file)] = sha(file)
        ledger = pd.read_csv(file)
        assert not ledger.date.duplicated().any()
        assert ledger.date.is_monotonic_increasing
        joined = ledger.merge(factor, on="date", how="left", validate="one_to_one", indicator=True)
        assert joined._merge.eq("both").all()
        assert joined[NAMES["FF5+UMD"]].notna().all().all()
        for year, sample in joined.groupby(joined.date.str[:4].astype(int), sort=True):
            compounded = float(np.prod(1 + sample.return_net.to_numpy()) - 1)
            nav_return = float(sample.end_nav.iloc[-1] / sample.start_nav.iloc[0] - 1)
            np.testing.assert_allclose(compounded, nav_return, atol=1e-12)
            row = {"year": int(year), "weighting": "value", "cost_bp": bp,
                "calendar_return": compounded, **performance(sample), **fit_alpha(sample, "FF5+UMD", 20)}
            rows.append(row)
        year_rows = [r for r in rows if r["cost_bp"] == bp]
        year_product = np.prod([1 + r["calendar_return"] for r in year_rows])
        full_product = np.prod(1 + ledger.return_net.to_numpy())
        np.testing.assert_allclose(year_product, full_product, atol=1e-12)
        assert sum(r["days"] for r in year_rows) == len(ledger) == 2260
        # Reproduce the reported full-sample regression as a scale/date check.
        full = fit_alpha(joined)
        original = pd.read_csv(analysis / "alpha.csv")
        original = original[(original.weighting == "value") & (original.cost_bp == bp) &
            (original.window == "full-2017-2025") & (original.factor_model == "FF5+UMD")].iloc[0]
        for name in ["annual_alpha", "annual_alpha_se", "t", "p", "ci_low", "ci_high"]:
            np.testing.assert_allclose(full[name], original[name], rtol=1e-10, atol=1e-12)
        reconciliations.append(dict(cost_bp=bp, total_days=len(ledger), yearly_days=sum(r["days"] for r in year_rows),
            full_wealth_multiple=float(full_product), product_of_yearly_wealth=float(year_product),
            full_alpha_reproduced=full["annual_alpha"]))
    result = pd.DataFrame(rows).sort_values(["cost_bp", "year"])
    assert len(result) == 18
    result["alpha_p_holm18_diagnostic"] = multipletests(result.p.to_numpy(), method="holm")[1]
    prior_p = pd.read_csv(analysis / "alpha.csv").p.to_numpy()
    assert len(prior_p) == 36
    result["alpha_p_holm54_with_original_family"] = multipletests(np.r_[prior_p, result.p.to_numpy()], method="holm")[1][36:]
    result.to_csv(output / "value_weighted_yearly_stats.csv", index=False)
    gross, net = (result[result.cost_bp == bp].set_index("year") for bp in [0, 10])
    display = pd.DataFrame(index=gross.index)
    display["days"] = gross.days
    display["return_before_cost_pct"] = gross.calendar_return * 100
    display["return_after_10bp_pct"] = net.calendar_return * 100
    display["vol_before_cost_pct"] = gross.annual_vol * 100
    display["sharpe_before_cost"] = gross.sharpe
    display["max_drawdown_before_cost_pct"] = gross.max_drawdown * 100
    display["ff5_umd_alpha_before_cost_pct"] = gross.annual_alpha * 100
    display["alpha_t_before_cost"] = gross.t
    display["alpha_p_before_cost"] = gross.p
    display["ff5_umd_alpha_after_10bp_pct"] = net.annual_alpha * 100
    display["alpha_t_after_10bp"] = net.t
    display["alpha_p_after_10bp"] = net.p
    display.to_csv(output / "yearly_display.csv")
    notes = ["VALUE-WEIGHTED D10-D1: CALENDAR-YEAR DIAGNOSTICS", "",
        "Same Chen-Zimmermann three-seed ensemble, explicit nonmicrocap screen, +100% long/-100% short (200% gross).",
        "Before costs and 10bp on all traded dollars, including initial and final trades; no artificial year-end liquidation.",
        "Calendar return = product(1+daily ledger net return)-1. Volatility and Sharpe annualized using 252 days.",
        "Within-year drawdown resets the initial high-water mark to one at the beginning of each year.",
        "FF5+UMD daily regressions are fitted separately within each year. Alpha = daily intercept*252, not compounded.",
        "Bartlett HAC20, small-sample correction, t-reference confidence intervals and p-values. No second RF subtraction.",
        "18 post-run yearly alpha tests are descriptive. Separate Holm18 and combined original36+new18 Holm54 adjustments are saved.",
        "2017 begins January 5 because of the pre-specified signal/execution/realization delay.", "",
        display.to_string(float_format=lambda x: f"{x:.4f}"), "",
        "Workflow: existing statsmodels skill; reference retained from the frozen parent report:",
        "Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026). Scientific Agent Skills:",
        "A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065", ""]
    (output / "README.txt").write_text("\n".join(notes))
    outputs = [output / name for name in ["value_weighted_yearly_stats.csv", "yearly_display.csv", "README.txt"]]
    write_json(output / "verification.json", dict(status="passed", created_utc=now(),
        calculation_sha256=sha(__file__), source_sha256={**source_hashes, str(FACTORS): sha(FACTORS)},
        parent_verification_sha256=sha(analysis / "verification.json"),
        reconciliations=reconciliations, yearly_alpha_test_count=18,
        output_sha256={f.name: sha(f) for f in outputs}, frozen_input_hashes_unchanged=input_hashes() == verification["inputs_sha256"]))
    print(display.to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nBefore-cost yearly alpha nominal p<.05:", gross.loc[gross.p < .05, ["annual_alpha", "t", "p"]].to_string())
    print("After-cost yearly alpha nominal p<.05:", net.loc[net.p < .05, ["annual_alpha", "t", "p"]].to_string())
    print("Holm18 p<.05:", result.loc[result.alpha_p_holm18_diagnostic < .05,
        ["year", "cost_bp", "annual_alpha", "p", "alpha_p_holm18_diagnostic"]].to_string(index=False))
    print("PASS: yearly NAV and day-count reconciliations; original full-period alpha reproduced; frozen inputs unchanged.")


if __name__ == "__main__":
    main()
