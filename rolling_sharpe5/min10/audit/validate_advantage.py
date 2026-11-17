"""Independent numerical/chronology audit of min10 advantage analysis.

Runs synthetic experiments only; does not execute the main backtest or mutate
its inputs/results. Output is restricted to this audit directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
MODULE_PATH = HERE.parent / "analyze.py"
sys.path.insert(0, str(HERE.parent))
spec = importlib.util.spec_from_file_location("min10_advantage_under_audit", MODULE_PATH)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def manual_raw_moment_sharpe_delta(a, b, lag):
    """Independent 4-moment gradient and Bartlett HAC covariance.

    Coordinates are E[a], E[a^2], E[b], E[b^2], unlike the implementation's
    centered scalar influence-function regression. The n/(n-1) correction
    matches its explicitly requested one-intercept HAC finite-sample option.
    """
    n = len(a)
    obs = np.column_stack((a, a*a, b, b*b))
    theta = obs.mean(axis=0)
    gradient = np.empty(4)
    scale = np.sqrt(12) * np.sqrt((n-1)/n)
    for start, sign in ((0, 1.), (2, -1.)):
        mu, raw_second = theta[start:start+2]
        sigma = np.sqrt(raw_second-mu*mu)
        gradient[start] = sign * scale * raw_second/sigma**3
        gradient[start+1] = -sign * scale * mu/(2*sigma**3)
    centered = obs-theta
    longrun = centered.T@centered/n
    for h in range(1, lag+1):
        cov = centered[h:].T@centered[:-h]/n
        longrun += (1-h/(lag+1))*(cov+cov.T)
    variance = float(gradient@longrun@gradient/n * n/(n-1))
    se = np.sqrt(max(0., variance))
    delta = float(np.sqrt(12)*(a.mean()/a.std(ddof=1)-b.mean()/b.std(ddof=1)))
    critical = student_t.ppf(.975, n-1)
    return delta, se, (delta-critical*se, delta+critical*se)


def main():
    checks = []
    rng = np.random.default_rng(828115)
    n = 480
    shocks = rng.standard_t(8, size=(n, 2))*.025
    ar = np.zeros_like(shocks)
    for t in range(n):
        ar[t] = shocks[t]+(.27*ar[t-1] if t else 0.)
    b = .006+ar[:, 0]
    a = .003+.65*b+.35*ar[:, 1]

    same = analysis.paired_sharpe(b, b, lag=6, bootstrap_n=800)
    zero_fields = ["sharpe_difference", "sharpe_delta_se", "sharpe_delta_ci_low",
                   "sharpe_delta_ci_high", "sharpe_boot_ci_low", "sharpe_boot_ci_high"]
    assert all(same[key] == 0. for key in zero_fields)
    assert same["sharpe_delta_p"] == 1.
    checks.append({"check": "identical_paths_zero_difference_and_ci", "pass": True})

    scaled = analysis.paired_sharpe(2.37*b, b, lag=6, bootstrap_n=800)
    scale_error = max(abs(scaled[key]) for key in zero_fields)
    assert scale_error < 2e-13
    checks.append({"check": "positive_scaling_preserves_sharpe", "pass": True,
                   "maximum_absolute_numerical_error": scale_error,
                   "note": "Pure multiplicative scale; an additive cash shift need not preserve Sharpe."})

    # An asymptotic Wald ratio of two floating-point zeros is meaningless.
    # This exact construction previously produced p=.0091 for a=.1*b.
    degenerate_rng = np.random.default_rng(17)
    degenerate_b = degenerate_rng.normal(.005, .03, 240)
    degenerate_pvalues = []
    for scale in (.1, 2.37, 10.):
        value = analysis.paired_sharpe(scale*degenerate_b, degenerate_b, lag=6)
        assert abs(value["sharpe_difference"]) < 2e-13
        assert value["sharpe_delta_p"] == 1., "Numerical zero-over-zero Wald ratio spuriously rejects scale invariance"
        degenerate_pvalues.append(value["sharpe_delta_p"])
    checks.append({"check": "numerical_zero_sharpe_delta_does_not_create_false_significance",
                   "pass": True, "pvalues": degenerate_pvalues})

    manual_errors = []
    for lag in (0, 3, 6, 12):
        actual = analysis.paired_sharpe(a, b, lag=lag)
        delta, se, ci = manual_raw_moment_sharpe_delta(a, b, lag)
        errors = [abs(actual["sharpe_difference"]-delta),
                  abs(actual["sharpe_delta_se"]-se),
                  abs(actual["sharpe_delta_ci_low"]-ci[0]),
                  abs(actual["sharpe_delta_ci_high"]-ci[1])]
        assert max(errors) < 2e-12
        manual_errors.append({"hac_lag": lag, "maximum_error": max(errors), "se": se})
    checks.append({"check": "independent_four_raw_moment_HAC_delta_covariance",
                   "pass": True, "details": manual_errors})

    forward = analysis.paired_sharpe(a, b, lag=6, bootstrap_n=800)
    reverse = analysis.paired_sharpe(b, a, lag=6, bootstrap_n=800)
    swap_errors = [abs(forward["sharpe_difference"]+reverse["sharpe_difference"]),
                   abs(forward["sharpe_delta_se"]-reverse["sharpe_delta_se"]),
                   abs(forward["sharpe_delta_p"]-reverse["sharpe_delta_p"])]
    for low, high in (("sharpe_delta_ci_low", "sharpe_delta_ci_high"),
                      ("sharpe_boot_ci_low", "sharpe_boot_ci_high")):
        swap_errors.extend([abs(forward[low]+reverse[high]), abs(forward[high]+reverse[low])])
    assert max(swap_errors) < 2e-12
    checks.append({"check": "paired_path_swap_negates_estimate_and_intervals",
                   "pass": True, "maximum_error": max(swap_errors)})

    idx = pd.period_range("2010-01", "2015-12", freq="M")
    names = [f"factor_{i:02}" for i in range(10)]
    f = pd.DataFrame(.007+rng.normal(0., .035, (72, 10)), index=idx, columns=names)
    e = .002+f.to_numpy()@np.linspace(-.2, .3, 10)+rng.normal(0., .025, 72)
    paper = pd.DataFrame({"gross": e, "5bps": e-.001}, index=idx)
    common = {"target_year": 2015, "method": "min10_annual_iid",
              "availability": "published_before_year", "universe": "all_factors"}
    counts = pd.DataFrame([common | {"factor_count": 10}])
    members = pd.DataFrame([common | {"factor": name, "selection_rank": i+1}
                            for i, name in enumerate(names)])
    paths, weights, betas, diagnostics = analysis.make_paths(f, paper, counts, members)
    f_changed, p_changed = f.copy(), paper.copy()
    f_changed.iloc[-12:] += rng.normal(.08, .12, (12, 10))
    p_changed.iloc[-12:] += rng.normal(-.07, .13, (12, 2))
    changed_paths, changed_weights, changed_betas, changed_diag = analysis.make_paths(
        f_changed, p_changed, counts, members)
    pd.testing.assert_frame_equal(weights, changed_weights, check_exact=True)
    pd.testing.assert_frame_equal(betas, changed_betas, check_exact=True)
    pd.testing.assert_frame_equal(diagnostics, changed_diag, check_exact=True)
    assert not np.allclose(paths.augmented_ew, changed_paths.augmented_ew)
    assert set(betas.train_end) == {"2014-12"}
    assert diagnostics.n_months.eq(60).all()
    checks.append({"check": "test_year_perturbation_leaves_all_weights_betas_and_fit_diagnostics_invariant",
                   "pass": True, "test_year": 2015, "training_months": 60, "factor_count": 10})

    errors = {"equal_weight_gain": [], "simplex_baseline": [],
              "simplex_augmented": [], "oos_residual_retains_train_intercept": []}
    for scenario in paper:
        z = paths.loc[paths.scenario.eq(scenario)].reset_index(drop=True)
        test_f = f.iloc[-12:].to_numpy()
        test_p = paper[scenario].iloc[-12:].to_numpy()
        expected_base = test_f.mean(axis=1)
        expected_aug = (10*expected_base+test_p)/11
        np.testing.assert_allclose(z.base_ew, expected_base, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(z.augmented_ew, expected_aug, atol=1e-14, rtol=1e-13)
        gain_error = np.max(np.abs(z.delta_ew.to_numpy()-(test_p-expected_base)/11))
        errors["equal_weight_gain"].append(float(gain_error))
        base_w = weights.loc[weights.model.eq("baseline")].set_index("factor").loc[names].weight.to_numpy()
        aug_w = weights.loc[weights.model.eq("augmented") & weights.scenario.eq(scenario)].set_index("factor").loc[names+["EARC"]].weight.to_numpy()
        np.testing.assert_allclose(z.base_optimized, test_f@base_w, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(z.augmented_optimized, np.column_stack((test_f, test_p))@aug_w,
                                   atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(z.delta_optimized, z.augmented_optimized-z.base_optimized,
                                   atol=1e-14, rtol=1e-13)
        assert min(base_w) >= -1e-14 and min(aug_w) >= -1e-14
        errors["simplex_baseline"].append(float(abs(base_w.sum()-1)))
        errors["simplex_augmented"].append(float(abs(aug_w.sum()-1)))
        train_x = np.column_stack((np.ones(60), f.iloc[:60].to_numpy()))
        coeff = np.linalg.solve(train_x.T@train_x, train_x.T@paper[scenario].iloc[:60].to_numpy())
        expected_residual = test_p-test_f@coeff[1:]
        err = np.max(np.abs(expected_residual-z.oos_residual.to_numpy()))
        assert abs(coeff[0]) > 1e-5
        assert err < 2e-12
        errors["oos_residual_retains_train_intercept"].append(float(err))
    checks.append({"check": "portfolio_gain_weight_and_residual_identities", "pass": True,
                   "maximum_errors": {key: max(vals) for key, vals in errors.items()}})

    result = {"pass": True, "checks": checks,
              "tested_module": str(MODULE_PATH),
              "tested_module_sha256": hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
              "scope": "Synthetic numerical and information-chronology tests only. No main analysis run or real result files changed.",
              "limits": "Fixed-path Sharpe inference does not reselect factors or refit historical portfolios in bootstrap; sparse publication/vintage history and missing OOS factors require separate source audits."}
    destination = HERE/"validate_advantage.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
