"""Independent final arithmetic checks and concise findings; no model changes."""
from pathlib import Path
import sys
import json
from datetime import datetime, timezone
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import t
from statsmodels.stats.multitest import multipletests
from usdata import sha, write_json


def main():
    out = HERE / "results"
    verification = json.loads((out / "verification.json").read_text())
    assert verification["status"] == "passed"
    for path, expected in verification["inputs_sha256"].items():
        assert sha(path) == expected, path
    for name, expected in verification["outputs_sha256"].items():
        assert sha(out / name) == expected, name
    audit = json.loads((out / "source_audit.json").read_text())
    assert audit["status"] == "passed"
    assert json.loads((HERE / "status.json").read_text())["stage"] == "complete"
    u = pd.read_csv(out / "utility_and_return_tests.csv")
    pairs = pd.read_csv(out / "paired_tests.csv")
    factors = pd.read_csv(out / "factor_diagnostics.csv")
    plan = json.loads((HERE / "protocol.json").read_text())
    years = []
    for case in plan["cases"]:
        ledger = pd.read_csv(out / f"{case}_ensemble_ledger.csv")
        for _, row in u[u.case == case].iterrows():
            a, b = plan["windows"][row.window]
            frame = ledger[ledger.date.between(a, b)]
            r = frame.return_net.to_numpy()
            flow = r - row.gamma / 2 * (r - r.mean())**2
            np.testing.assert_allclose(row.estimate, flow.mean() * 252, atol=1e-12)
            resid = flow - flow.mean()
            n = len(r)
            variance = resid @ resid
            for lag in range(1, 21):
                variance += 2 * (1 - lag/21) * (resid[lag:] @ resid[:-lag])
            se = np.sqrt(variance/(n*(n-1))) * 252
            np.testing.assert_allclose(row.se, se, atol=1e-12)
            np.testing.assert_allclose(row.p, 2 * t.sf(abs(row.estimate / se), n-1), atol=1e-12)
        for year, g in ledger.groupby(ledger.date.str[:4], sort=True):
            r = g.return_net.to_numpy()
            years.append(dict(case=case, year=year, days=len(r),
                              calendar_return=np.prod(1+r)-1, annual_mean=r.mean()*252,
                              utility10=(r.mean()-5*r.var(ddof=0))*252))
    family = np.r_[u.p, pairs.p, factors.p]
    np.testing.assert_allclose(np.r_[u.p_holm132, pairs.p_holm132, factors.p_holm132],
                               multipletests(family, method="holm")[1], atol=1e-12)
    pd.DataFrame(years).to_csv(out / "calendar_years.csv", index=False)
    full = u[u.window == "full-2017-2025"]
    mean = full[(full.case == "LiMT_equal") & (full.gamma == 0)].iloc[0]
    utility = full[(full.case == "LiMT_equal") & (full.gamma == 10)].iloc[0]
    recent = u[(u.case == "LiMT_equal") & (u.gamma == 10) & (u.window == "recent-2021-2025")].iloc[0]
    ff6 = factors[(factors.case == "LiMT_equal") & (factors.model == "FF5+Mom") &
                  (factors.window == "full-2017-2025")].iloc[0]
    perf = pd.read_csv(out / "performance.csv")
    full_perf = perf[perf.window == "full-2017-2025"].set_index("case")
    p_equal = pairs[(pairs.strategy == "equal") & (pairs.gamma == 0) &
                    (pairs.window == "full-2017-2025")].iloc[0]
    lines = ["LIMT RFS-GUIDED BEFORE-COST TEST: FINDINGS", "",
             "Yes: the existing US decile ensemble retains significantly positive full-period before-cost gains.",
             f"2017-01-05 to 2025-12-31, 2260 days: annual arithmetic mean {mean.estimate:.2%}, "
             f"95% HAC interval [{mean.ci_low:.2%}, {mean.ci_high:.2%}].",
             f"Raw p={mean.p:.6g}; Holm p={mean.p_holm132:.6g} over 132 audit hypotheses; "
             f"Holm p={mean.p_holm153_with_prior:.6g} after appending all 21 prior aggregate tests.",
             f"Annual volatility {full_perf.loc['LiMT_equal','volatility']:.2%}, Sharpe "
             f"{full_perf.loc['LiMT_equal','sharpe']:.2f}, CAGR {full_perf.loc['LiMT_equal','cagr']:.2%}, "
             f"maximum drawdown {full_perf.loc['LiMT_equal','max_drawdown']:.2%}.", "",
             "The RFS utility test supports a narrower conclusion:",
             f"At gamma=10, annual empirical utility is {utility.estimate:.2%}, HAC95% "
             f"[{utility.ci_low:.2%}, {utility.ci_high:.2%}], raw p={utility.p:.4g}.",
             f"Its Holm p={utility.p_holm132:.3g}; positive nominal evidence does not survive the full test-family correction.",
             f"Recent 2021-2025 utility is {recent.estimate:.2%}, HAC95% [{recent.ci_low:.2%}, "
             f"{recent.ci_high:.2%}], p={recent.p:.4g}; not significantly positive.",
             "Utility is mean return less a risk penalty. It is not a cash return or a transaction cost.", "",
             "All five ensemble constructions (annual before-cost mean / volatility / Sharpe / U10):"]
    for case in plan["cases"]:
        p = full_perf.loc[case]
        v = full[(full.case == case) & (full.gamma == 10)].iloc[0]
        lines.append(f"  {case}: {p.annual_mean:.2%} / {p.volatility:.2%} / {p.sharpe:.2f} / {v.estimate:.2%}")
    lines += ["", "The new rank-weighted version is positive, but its before-cost mean fails the family correction.",
              "The capacity-constrained APO version has much lower deployed exposure and no significant U10.",
              f"LiMT minus LightGBM decile mean: {p_equal.estimate:.2%} per year, HAC95% "
              f"[{p_equal.ci_low:.2%}, {p_equal.ci_high:.2%}], p={p_equal.p:.4g}. Superiority is inconclusive.", "",
              "MARKET NEUTRALITY", "Primary means dollar neutral: target +50% long/-50% short, gross 100%.",
              "This does not impose zero beta. Full-period CAPM beta is -0.023 (95% -0.076 to 0.030).",
              f"FF5+momentum alpha is {ff6.alpha:.2%} annually, HAC95% [{ff6.alpha_low:.2%}, "
              f"{ff6.alpha_high:.2%}], Holm153 p={ff6.p_holm153_with_prior:.5g}.",
              "These are ex-post regression diagnostics, not a beta-neutral trading rule.", "",
              "RELATION TO THE REQUESTED PAPER", plan["paper"],
              "Applied: empirical mean-variance utility (Eq.11), return/risk/turnover/gross exposure (Table 2),",
              "rank-weighted comparison, paired utility comparisons, and descriptive equal-risk curves (Figure 1).",
              "Preserved daily frozen LiMT predictions/splits and execution timing; no retraining or winner selection.",
              "Zero costs are the user's primary case. Thus this audit does not establish the paper's central",
              "nonzero-cost implementability claim, reproduce its dynamic Portfolio-ML algorithm, or prove a globally efficient frontier.",
              "The monthly 1981-2020 protocol, nested annual refitting, characteristic risk model and AUM/impact grid are not reproduced.",
              "The saved 'paper-period-2017-2020' window is inherited from the LiMT protocol and ends 2020-08-01; it is not the RFS sample.",
              "Validation-only 10% risk scaling overshoots to 16.73% test volatility for decile LiMT. Ex-post equal-risk plots are descriptive.", "",
              "VERIFICATION AND LIMITS", "21 tests pass; all 20 seed/ensemble portfolio runs complete; original ensemble ledgers reproduce.",
              f"All {audit['distinct_held_stock_dates']:,} distinct ensemble held stock-date payoffs match original CRSP source partitions.",
              f"{audit['missing_source_payoffs']} distinct source payoffs are missing; they are preserved and adverse stress results are saved.",
              "Primary LiMT adverse stress retains an 11.66% annual before-cost mean; no future-outcome deletion.",
              "Ideal closes, assumed shorts, current-vintage CRSP and OHLC4 VWAP substitution remain limitations.",
              "The historical test was already inspected; this is retrospective evidence and no fresh prospective holdout.",
              "The independent joint price/volatility experiment is unfinished and excluded from these conclusions.", "",
              "FILES", "REPORT.txt: complete tables and methods; sources/protocol-mapping.txt: source anchors and deviations.",
              "results/frontier_and_utility.png: descriptive zero-cost risk curves and utility path.",
              "results/utility_and_return_tests.csv, paired_tests.csv, factor_diagnostics.csv: full inference.",
              "results/verification.json, source_audit.json and final_checks.json: audit evidence.", "",
              "Scientific-critical-thinking and statsmodels skills informed this audit; workflow attribution:",
              "Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.",
              "https://doi.org/10.48550/arXiv.2609.00065 (current arXiv v2 metadata verified)."]
    (HERE / "FINDINGS.txt").write_text("\n".join(lines) + "\n")
    artifacts = list(out.glob("*")) + [HERE / "REPORT.txt", HERE / "FINDINGS.txt", HERE / "protocol.json",
                 HERE / "locked_before_evaluation.json", HERE / "sources/protocol-mapping.txt",
                 ROOT / "tests/test_rfs_frontier.py"] + list((HERE / "checks").glob("*.py"))
    write_json(out / "final_checks.json", dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
        checks=["21 model/financial tests passed", "90 utility/mean estimates and HAC SE/p-values independently recomputed",
                "132-test Holm adjustment independently recomputed", "Original locked inputs and output hashes unchanged",
                "All newly held stock-date source payoffs reconciled", "Plot visually inspected; no clipped labels"],
        artifact_sha256={str(p.relative_to(HERE)) if p.is_relative_to(HERE) else str(p):sha(p)
                         for p in artifacts if p.is_file() and p.name != "final_checks.json"}))
    print((HERE / "FINDINGS.txt").read_text())


if __name__ == "__main__":
    main()
