"""Zero-cost RFS equation-11 evaluation of the completed, frozen US transfer.

Run: $LIMT_ANALYSIS_PYTHON experiments/rfs_frontier/run.py
Uses no GPU and never modifies the base reproduction's files.
"""
from pathlib import Path
import sys
import json
import itertools
import hashlib
import platform
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from usdata import Panel, sha, write_json
from analyze import load_predictions, ensemble
from portfolio import backtest
from experiments.rfs_frontier.engine import rank_target, rank_backtest, utility_flow

OUT = HERE / "results"
PLAN = json.loads((HERE / "protocol.json").read_text())
CHECKS = []






def simulate(panel, pred, strategy, **kwargs):
    if strategy == "rank":
        return rank_backtest(panel, pred, **kwargs)
    return backtest(panel, pred, strategy=strategy, **kwargs)


def hac(x, lag=20):
    x = np.asarray(x, float)
    fit = sm.OLS(x, np.ones((len(x), 1)), missing="raise").fit(
        cov_type="HAC", cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": True},
        use_t=True)
    u = x - x.mean()
    variance = u @ u
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k])
    np.testing.assert_allclose(fit.bse[0], np.sqrt(variance / (len(x) * (len(x) - 1))),
                               atol=1e-13, rtol=1e-9)
    low, high = fit.conf_int()[0]
    return dict(estimate=float(x.mean() * 252), se=float(fit.bse[0] * 252),
                ci_low=float(low * 252), ci_high=float(high * 252),
                t=float(fit.tvalues[0]), p=float(fit.pvalues[0]))




def bootstrap_utilities(x, block):
    """Joint date resamples; re-estimate the mean/variance in each draw."""
    x = np.asarray(x, float)
    n, m = x.shape
    rng = np.random.default_rng(PLAN["inference"]["bootstrap_seed"] + block)
    result = np.empty((5000, m, len(PLAN["gammas"])))
    gammas = np.array(PLAN["gammas"])
    for first in range(0, 5000, 100):
        starts = rng.integers(0, n, size=(100, (n + block - 1) // block))
        ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(100, -1)[:, :n]
        sampled = x[ix]
        means = sampled.mean(1)
        variance = sampled.var(1, ddof=0)
        result[first:first + 100] = 252 * (means[:, :, None] - variance[:, :, None] * gammas / 2)
    return result


def masks(dates):
    dates = np.asarray(dates, dtype=str)
    for name, (a, b) in PLAN["windows"].items():
        yield name, (dates >= a) & (dates <= b)


def get_inputs():
    paths = [ROOT / "protocol.json", ROOT / "data/manifest.json", ROOT / "portfolio.py",
             ROOT / "limt/neutral.py", ROOT / "usdata.py", ROOT / "analyze.py",
             ROOT / "analysis/summary.csv", ROOT / "analysis/forecast_comparison.csv",
             ROOT / "analysis/verification.json", ROOT / "analysis/final-checks.json",
             ROOT.parent / "factors-daily/ff5_momentum_daily.parquet",
             ROOT.parent / "factors-daily/verification.json"]
    paths += list(HERE.glob("*.py")) + [HERE / "protocol.json"]
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, expected in manifest["output_sha256"].items():
        path = ROOT / "data" / name
        assert sha(path) == expected, ("Data changed", name)
        paths.append(path)
    for model in ["results", "baseline"]:
        for seed in PLAN["seeds"]:
            folder = ROOT / model / f"seed{seed}"
            paths += [folder / "test_predictions.parquet", folder / "valid_predictions.parquet"]
    return {str(p): sha(p) for p in sorted(set(paths))}


def base_turnover(ledger, weights, panel):
    """Include both final long sales and short covers, even at zero fees."""
    dates = {str(d): i for i, d in enumerate(panel.dates)}
    t = dates[str(ledger.date.iloc[-1])]
    ending = weights[-1] * ledger.start_nav.iloc[-1] * (
        1 + np.nan_to_num(np.asarray(panel.price_return[:, t]), nan=0.))
    ending[panel.terminal[:, t]] = 0.
    sale = float(np.maximum(ending, 0).sum())
    ledger["liquidation_sell"] = 0.
    ledger.loc[ledger.index[-1], "liquidation_sell"] = sale
    return ledger


def verify_weights(ledger, weights, panel):
    lookup = {str(d): i for i, d in enumerate(panel.dates)}
    ix = np.array([lookup[str(d)] for d in ledger.date])
    payoff = np.nan_to_num(np.asarray(panel.total_return[:, ix]).T, nan=0.)
    np.testing.assert_allclose((weights * payoff).sum(1), ledger.return_gross, atol=1e-12)
    np.testing.assert_allclose(weights.sum(1), ledger.net_exposure, atol=1e-12)
    np.testing.assert_allclose(abs(weights).sum(1), ledger.gross_exposure, atol=1e-12)
    np.testing.assert_array_equal(ledger.return_net, ledger.return_gross)
    assert ledger.fees.eq(0).all()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = get_inputs()
    lock = HERE / "locked_before_evaluation.json"
    if lock.exists():
        assert json.loads(lock.read_text())["inputs_sha256"] == inputs, "Locked inputs changed"
    else:
        write_json(lock, dict(created_utc=datetime.now(timezone.utc).isoformat(),
                              prior_test_results_already_inspected=True, inputs_sha256=inputs))
    write_json(HERE / "status.json", dict(stage="running", pid=__import__("os").getpid()))
    panel = Panel()
    ledgers, stresses, validations = {}, {}, {}
    seed_records = []
    for model in ["LiMT", "LightGBM"]:
        test_parts = [load_predictions(model, seed, panel) for seed in PLAN["seeds"]]
        valid_parts = []
        for seed in PLAN["seeds"]:
            folder = ROOT / ("results" if model == "LiMT" else "baseline") / f"seed{seed}"
            pred = pd.read_parquet(folder / "valid_predictions.parquet").sort_values(
                ["date", "symbol"]).reset_index(drop=True)
            assert not pred[["date", "symbol"]].duplicated().any()
            np.testing.assert_array_equal(pred.groupby("date").size().index,
                                          panel.dates[panel.split("valid")])
            valid_parts.append(pred)
        parts = [("ensemble", ensemble(test_parts))] + [(str(s), test_parts[s]) for s in PLAN["seeds"]]
        for strategy in (["equal", "rank", "apo"] if model == "LiMT" else ["equal", "rank"]):
            case = f"{model}_{strategy}"
            for seed, pred in parts:
                ledger, weights = simulate(panel, pred, strategy, save_weights=seed == "ensemble")
                if seed == "ensemble":
                    verify_weights(ledger, weights, panel)
                    if strategy != "rank":
                        old = pd.read_csv(ROOT / f"analysis/{model}_ensemble_{strategy}_buy0_ledger.csv")
                        pd.testing.assert_frame_equal(ledger, old, check_dtype=False,
                                                      rtol=1e-10, atol=1e-9)
                        ledger = base_turnover(ledger, weights, panel)
                    ledgers[case] = ledger
                    np.savez_compressed(OUT / f"{case}_weights.npz", weights=weights,
                                        dates=ledger.date.to_numpy(dtype=str), symbols=panel.symbols)
                    stress, _ = simulate(panel, pred, strategy, adverse_missing=True)
                    stresses[case] = stress
                    stress.to_csv(OUT / f"{case}_missing_stress.csv", index=False)
                ledger.to_csv(OUT / f"{case}_{seed}_ledger.csv", index=False)
                x = ledger.return_net.to_numpy()
                seed_records.append(dict(case=case, seed=seed, annual_mean=x.mean() * 252,
                                         volatility=x.std(ddof=0) * np.sqrt(252),
                                         utility10=utility_flow(x, 10).mean() * 252))
                print("REPLAYED", case, seed, flush=True)
            valid, _ = simulate(panel, ensemble(valid_parts), strategy)
            valid.to_csv(OUT / f"{case}_validation_ledger.csv", index=False)
            validations[case] = valid
    CHECKS.append("All three seeds of every declared strategy rerun; three legacy ensemble ledgers reproduce unchanged.")
    CHECKS.append("Every ensemble weight/P&L/NAV/exposure reconciles independently to panel total returns; all fees zero.")
    pd.DataFrame(seed_records).to_csv(OUT / "seed_diagnostics.csv", index=False)
    cases = PLAN["cases"]
    dates = ledgers[cases[0]].date.to_numpy()
    for case in cases:
        np.testing.assert_array_equal(ledgers[case].date, dates)
    x = np.column_stack([ledgers[c].return_net.to_numpy() for c in cases])
    endpoint_rows, pair_rows, factor_rows, performance, stress_rows = [], [], [], [], []
    factors = pd.read_parquet(ROOT.parent / "factors-daily/ff5_momentum_daily.parquet")
    factors["date"] = factors.date.astype(str)
    for window, mask in masks(dates):
        sample = x[mask]
        boots = {b: bootstrap_utilities(sample, b) for b in [20, 60]}
        for j, case in enumerate(cases):
            r = sample[:, j]
            g = ledgers[case].loc[mask]
            n = len(r)
            wealth = np.r_[1., np.cumprod(1 + r)]
            perf = dict(case=case, window=window, days=n, start=str(dates[mask][0]), end=str(dates[mask][-1]),
                        annual_mean=r.mean() * 252, volatility=r.std(ddof=0) * np.sqrt(252),
                        sharpe=r.mean() / r.std(ddof=0) * np.sqrt(252),
                        cagr=wealth[-1] ** (252 / n) - 1,
                        max_drawdown=(wealth / np.maximum.accumulate(wealth) - 1).min(),
                        mean_gross=g.gross_exposure.mean(), max_abs_net=g.net_exposure.abs().max(),
                        max_abs_drift=g.end_net_drift.abs().max(),
                        daily_turnover=((g.buy_value + g.sell_value + g.liquidation_buy +
                                         g.liquidation_sell) / g.start_nav).mean(),
                        annual_turnover=((g.buy_value + g.sell_value + g.liquidation_buy +
                                          g.liquidation_sell) / g.start_nav).mean() * 252,
                        missing_count=int(g.missing_held_count.sum()),
                        max_missing_weight=g.missing_return_gross_weight.max())
            performance.append(perf)
            for k, gamma in enumerate(PLAN["gammas"]):
                flow = utility_flow(r, gamma)
                rec = dict(case=case, window=window, gamma=gamma, days=n, **hac(flow))
                for lag in [5, 60]:
                    rec.update({f"hac{lag}_{key}": value for key, value in hac(flow, lag).items()})
                for block in [20, 60]:
                    rec[f"block{block}_low"], rec[f"block{block}_high"] = np.quantile(
                        boots[block][:, j, k], [.025, .975])
                endpoint_rows.append(rec)
            sr = stresses[case].loc[mask, "return_net"].to_numpy()
            stress_rows.append(dict(case=case, window=window, annual_mean=sr.mean() * 252,
                                    utility10=utility_flow(sr, 10).mean() * 252,
                                    **{f"mean_{k}": v for k, v in hac(sr).items()}))
            joined = pd.DataFrame(dict(date=dates[mask], r=r)).merge(factors, on="date", how="left", validate="one_to_one")
            assert len(joined) == n and joined.notna().all().all()
            for factor_model, names in [("CAPM", ["mkt_rf"]), ("FF5+Mom", ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])]:
                design = sm.add_constant(joined[names], has_constant="add")
                fit = sm.OLS(joined.r, design, missing="raise").fit(
                    cov_type="HAC", cov_kwds={"maxlags": 20, "use_correction": True}, use_t=True)
                ci = fit.conf_int()
                factor_rows.append(dict(case=case, window=window, model=factor_model, days=n,
                                        alpha=fit.params["const"] * 252,
                                        alpha_low=ci.loc["const", 0] * 252, alpha_high=ci.loc["const", 1] * 252,
                                        p=fit.pvalues["const"], beta_market=fit.params["mkt_rf"],
                                        beta_market_low=ci.loc["mkt_rf", 0], beta_market_high=ci.loc["mkt_rf", 1],
                                        beta_market_p=fit.pvalues["mkt_rf"], r_squared=fit.rsquared,
                                        condition_number=fit.condition_number,
                                        **{f"loading_{name}": fit.params[name] for name in names}))
        for strategy, gamma in itertools.product(["equal", "rank"], [0, 10]):
            a, b = cases.index(f"LiMT_{strategy}"), cases.index(f"LightGBM_{strategy}")
            flow = utility_flow(sample[:, a], gamma) - utility_flow(sample[:, b], gamma)
            rec = dict(strategy=strategy, window=window, gamma=gamma, days=len(sample), **hac(flow))
            k = PLAN["gammas"].index(gamma)
            for block in [20, 60]:
                rec[f"block{block}_low"], rec[f"block{block}_high"] = np.quantile(
                    boots[block][:, a, k] - boots[block][:, b, k], [.025, .975])
            pair_rows.append(rec)
        print("INFERENCE", window, flush=True)
    endpoints, pairs, alpha = map(pd.DataFrame, [endpoint_rows, pair_rows, factor_rows])
    family = np.r_[endpoints.p, pairs.p, alpha.p]
    assert len(family) == 132
    prior = pd.read_csv(ROOT / "analysis/summary.csv")
    prior = prior[prior.seed.astype(str) == "ensemble"].p.to_numpy()
    prior = np.r_[prior, pd.read_csv(ROOT / "analysis/forecast_comparison.csv").paired_p]
    assert len(prior) == 21
    adjusted = multipletests(family, method="holm")[1]
    combined = multipletests(np.r_[family, prior], method="holm")[1][:len(family)]
    offset = 0
    for frame in [endpoints, pairs, alpha]:
        frame["p_holm132"] = adjusted[offset:offset + len(frame)]
        frame["p_holm153_with_prior"] = combined[offset:offset + len(frame)]
        offset += len(frame)
    endpoints.to_csv(OUT / "utility_and_return_tests.csv", index=False)
    pairs.to_csv(OUT / "paired_tests.csv", index=False)
    alpha.to_csv(OUT / "factor_diagnostics.csv", index=False)
    pd.DataFrame(performance).to_csv(OUT / "performance.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(OUT / "missing_stress_summary.csv", index=False)
    CHECKS.append("132 hypotheses and all 21 prior base hypotheses accounted for; seeds never pooled as independent dates.")
    CHECKS.append("Bootstrap recomputes variance inside 5000 shared date resamples; HAC SEs independently match Bartlett arithmetic.")
    CHECKS.append("CAPM/FF6 diagnostics use all trading dates and correct zero-investment return units; no double risk-free subtraction.")
    scales, rays = [], []
    for j, case in enumerate(cases):
        r = x[:, j]
        valid_r = validations[case].return_net.to_numpy()
        scale = .10 / (valid_r.std(ddof=0) * np.sqrt(252))
        scaled = scale * r
        scales.append(dict(case=case, validation_days=len(valid_r), validation_start=validations[case].date.iloc[0],
                           validation_end=validations[case].date.iloc[-1], fixed_scale=scale,
                           validation_volatility=valid_r.std(ddof=0) * np.sqrt(252),
                           test_volatility=scaled.std(ddof=0) * np.sqrt(252), annual_mean=scaled.mean() * 252,
                           utility10=utility_flow(scaled, 10).mean() * 252,
                           implied_mean_gross=scale * ledgers[case].gross_exposure.mean(),
                           note="Constant return-stream overlay, ideal frictionless linear scaling; not new execution-cap simulation."))
        for target in np.linspace(0, .10, 11):
            scale_expost = target / (r.std(ddof=0) * np.sqrt(252))
            rays.append(dict(case=case, target_expost_volatility=target, scale_expost=scale_expost,
                             annual_mean=r.mean() * 252 * scale_expost,
                             implied_mean_gross=scale_expost * ledgers[case].gross_exposure.mean(),
                             utility10=utility_flow(r * scale_expost, 10).mean() * 252))
    pd.DataFrame(scales).to_csv(OUT / "validation_risk_scale_diagnostic.csv", index=False)
    pd.DataFrame(rays).to_csv(OUT / "zero_cost_strategy_rays.csv", index=False)
    draw_plots(ledgers, pd.DataFrame(rays), endpoints)
    for path, expected in inputs.items():
        assert sha(path) == expected, ("Input changed during run", path)
    report(pd.DataFrame(performance), endpoints, pairs, alpha, pd.DataFrame(scales))
    write_json(OUT / "verification.json", dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
               checks=CHECKS, inputs_sha256=inputs, python=platform.python_version(), numpy=np.__version__,
               pandas=pd.__version__, statsmodels=statsmodels.__version__, hypotheses=132,
               with_prior_hypotheses=153, daily_test_observations=len(dates),
               seed_strategy_runs=len(seed_records),
               outputs_sha256={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "verification.json"}))
    write_json(HERE / "status.json", dict(stage="complete", created_utc=datetime.now(timezone.utc).isoformat()))
    print((HERE / "REPORT.txt").read_text(), flush=True)


def draw_plots(ledgers, rays, endpoints):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    for case, g in rays.groupby("case", sort=False):
        axes[0].plot(g.target_expost_volatility * 100, g.annual_mean * 100, label=case.replace("_", " "))
    axes[0].set(xlabel="Ex-post annual volatility (%)", ylabel="Annual mean before costs (%)",
                title="Zero-cost strategy rays (descriptive scaling)")
    axes[0].legend(fontsize=8)
    for case, g in ledgers.items():
        r = g.return_net.to_numpy()
        flow = utility_flow(r, 10)
        axes[1].plot(pd.to_datetime(g.date), np.cumsum(flow), label=case.replace("_", " "))
    axes[1].set(xlabel="Payoff date", ylabel="Cumulative sum of empirical utility",
                title="Risk-adjusted utility, gamma = 10, zero costs")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(OUT / "frontier_and_utility.png", dpi=180)
    plt.close(fig)


def report(performance, endpoints, pairs, factors, scales):
    full = endpoints[endpoints.window == "full-2017-2025"]
    returns = full[full.gamma == 0]
    util = full[full.gamma == 10]
    perf = performance[performance.window == "full-2017-2025"]
    columns = ["case", "estimate", "ci_low", "ci_high", "p", "p_holm132", "p_holm153_with_prior", "block20_low", "block20_high"]
    lines = ["LiMT US: RFS ZERO-COST PORTFOLIO EVALUATION", "",
             "SOURCE AND SCOPE", PLAN["paper"],
             "Apply Eq.11 utility, Table 2 return/risk/turnover/leverage and Figure 1 risk comparisons to frozen forecasts.",
             "This is the requested frictionless limit and a daily dollar-neutral adaptation, not the paper's full cost-aware protocol.",
             "No new training, model selection, signal reversal, winning-seed selection or test-chosen risk allocation.",
             "The full 2017-2025 test has 2260 calendar observations, and was already inspected in the original run.", "",
             "BEFORE-COST MEAN RETURN (annual arithmetic; inferential CIs in same units)", returns[columns].to_string(index=False), "",
             "EMPIRICAL UTILITY AT GAMMA=10", util[columns].to_string(index=False), "",
             "Utility = 252*(mean daily return - gamma/2 * daily population variance).",
             "The empirical risk penalty is not a cash charge or a realized investable return.", "",
             "RETURN, RISK, EXPOSURE AND TURNOVER", perf.to_string(index=False), "",
             "Turnover = all buy and sell dollar changes / start NAV, including initial trades and both terminal legs.",
             "Table 2 uses monthly turnover; these are daily figures and must not be compared without frequency conversion.", "",
             "PAIRED LiMT MINUS LIGHTGBM (all predeclared windows; gamma=0 mean, gamma=10 utility)", pairs.to_string(index=False), "",
             "FACTOR DIAGNOSTICS (full period)", factors[factors.window == "full-2017-2025"].to_string(index=False), "",
             "Equal long/short dollars do not guarantee market beta zero. Ex-post factor alpha is not an ex-ante beta hedge.",
             "Do not subtract RF again from zero-investment long/short P&L; the common RF term already cancels.", "",
             "FIXED VALIDATION RISK SCALE (diagnostic return-stream overlays)", scales.to_string(index=False), "",
             "The multiplier uses only 2015-2016 validation risk; test risk is allowed to differ.",
             "The separate 0%-10% strategy-ray chart uses ex-post risk scaling, only for descriptive equal-risk comparison.",
             "It does not establish the globally efficient frontier or a deployable risk target.",
             "APO scaled overlays do not re-solve capacity at a different AUM and cannot establish capacity.", "",
             "INFERENCE AND MULTIPLICITY",
             "Two-sided HAC20 Bartlett with small-sample correction and t reference; lag5/60 sensitivities saved.",
             "5000 paired circular block20/block60 bootstrap resamples, re-estimating means and variances within draws.",
             "132-hypothesis Holm family includes all cases/windows/gammas, paired endpoints and factor alphas.",
             "153-hypothesis sensitivity appends all 21 original aggregate tests, conservatively retaining overlapping hypotheses.",
             "This does not account for all method searches across papers or earlier unpublished trials.",
             "Individual seeds, alternative lags, missing-payoff stresses and scaled curves are diagnostic only.", "",
             "PROTOCOL COVERAGE AND LIMITATIONS"]
    lines += ["- " + s for s in PLAN["limitations"]]
    lines += ["", "Missing-payoff adverse stress is saved for all five portfolios without future-outcome filtering.",
              "All costs and cash interest are zero; dividends and short dividend obligations remain in CRSP total P&L.",
              "The base equal/APO P&L reproduces the original ledgers; all original artifacts remain untouched.", "",
              "REPRODUCE", "From repo root:",
              "  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $LIMT_ANALYSIS_PYTHON experiments/rfs_frontier/run.py", "",
              "ARTIFACTS: results/performance.csv, utility_and_return_tests.csv, paired_tests.csv, factor_diagnostics.csv,",
              "zero_cost_strategy_rays.csv, validation_risk_scale_diagnostic.csv, *_ledger.csv, *_weights.npz,",
              "missing_stress_summary.csv, seed_diagnostics.csv, frontier_and_utility.png and verification.json.", "",
              "WORKFLOW AND SOFTWARE", "Scientific-critical-thinking and statsmodels skills informed the audit and robust inference.",
              "Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). Scientific Agent Skills:",
              "A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065",
              "Metadata checked against current arXiv v2; attribution is not evidence of investment performance."]
    (HERE / "REPORT.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
