"""Weighted decile portfolios and realized-date FF5+UMD HAC inference."""
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from usdata import sha, write_json
from experiments.standard_characteristics.common import HERE, DATA, FACTORS, config, input_hashes, now
from experiments.standard_characteristics.data import Panel
from experiments.standard_characteristics.portfolio import backtest

NAMES = {"mean": [], "FF5": ["mkt_rf", "smb", "hml", "rmw", "cma"],
         "FF5+UMD": ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]}


def fit_alpha(joined, factor_model="FF5+UMD", lag=20):
    names = NAMES[factor_model]
    y = joined.return_net.to_numpy(float)
    x = np.column_stack([np.ones(len(joined)), joined[names].to_numpy(float)])
    assert np.isfinite(y).all() and np.isfinite(x).all()
    assert np.linalg.matrix_rank(x) == x.shape[1]
    result = sm.OLS(y, x, missing="raise").fit(cov_type="HAC",
        cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": True}, use_t=True)
    interval = result.conf_int()[0] * 252
    row = dict(factor_model=factor_model, days=len(y), hac_lag=lag, annual_alpha=float(result.params[0] * 252),
        annual_alpha_se=float(result.bse[0] * 252), t=float(result.tvalues[0]), p=float(result.pvalues[0]),
        ci_low=float(interval[0]), ci_high=float(interval[1]), annual_alpha_gross100=float(result.params[0] * 126),
        r_squared=float(result.rsquared), condition_number=float(np.linalg.cond(x)))
    row.update({f"beta_{name}": float(value) for name, value in zip(names, result.params[1:])})
    return row


def performance(ledger):
    r = ledger.return_net.to_numpy()
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1., curve])[1:]
    return dict(days=len(r), first_date=ledger.date.iloc[0], last_date=ledger.date.iloc[-1],
        annual_mean=float(r.mean() * 252), annual_mean_gross100=float(r.mean() * 126),
        cagr=float(curve[-1] ** (252 / len(r)) - 1), annual_vol=float(r.std(ddof=1) * np.sqrt(252)),
        sharpe=float(r.mean() / r.std(ddof=1) * np.sqrt(252)), max_drawdown=float((curve / peak - 1).min()),
        mean_gross=float(ledger.gross_exposure.mean()), max_abs_net=float(ledger.net_exposure.abs().max()),
        mean_turnover=float(ledger.turnover.mean()), missing_held_count=int(ledger.missing_held_count.sum()),
        missing_weight_days=float(ledger.missing_gross_weight.sum()),
        max_target_deviation=float(ledger.target_deviation_l1.max()))


def ensemble():
    frames = []
    for seed in config()["training"]["seeds"]:
        out = HERE / "results" / f"seed{seed}"
        assert json.loads((out / "status.json").read_text())["phase"] == "complete"
        provenance = json.loads((out / "provenance.json").read_text())
        assert sha(out / "test_predictions.parquet") == provenance["prediction_sha256"]
        frame = pd.read_parquet(out / "test_predictions.parquet").sort_values(["date", "symbol"]).reset_index(drop=True)
        assert not frame.duplicated(["date", "symbol"]).any()
        if frames:
            pd.testing.assert_frame_equal(frame[["date", "symbol", "realized_date"]], frames[0][["date", "symbol", "realized_date"]])
        frames.append(frame)
    result = frames[0].copy()
    cols = ["pred_return", "pred_volume", "pred_volatility"]
    result[cols] = np.mean([f[cols].to_numpy() for f in frames], axis=0)
    return result, frames


def run():
    out = HERE / "analysis"
    out.mkdir(exist_ok=True)
    frozen = json.loads((HERE / "locked_before_training.json").read_text())
    assert frozen["inputs_sha256"] == input_hashes()
    assert frozen["data_manifest_sha256"] == sha(DATA / "manifest.json")
    panel = Panel()
    pred, seeds = ensemble()
    factors = pd.read_parquet(FACTORS)
    factors["date"] = factors.date.astype(str)
    assert not factors.date.duplicated().any()
    alpha_rows, lag_rows, perf_rows, stress_rows, seed_rows = [], [], [], [], []
    ledgers = {}
    p = config()
    for weighting in p["portfolio"]["weights"]:
        for bp in p["portfolio"]["cost_bp_all_trades"]:
            case = f"{weighting}_cost{bp}"
            ledger, weights, deciles = backtest(panel, pred, weighting, bp, save_weights=True)
            ledger.to_csv(out / f"{case}_ledger.csv", index=False)
            np.savez_compressed(out / f"{case}_weights.npz", weights=weights,
                dates=ledger.date.to_numpy(dtype="U10"), symbols=panel.symbols)
            if bp == 0:
                deciles.to_csv(out / f"{weighting}_target_deciles.csv", index=False)
                summary = deciles.groupby("decile").agg(days=("target_return", "size"),
                    daily_mean=("target_return", "mean"), missing_weight=("missing_weight", "sum"))
                summary["annual_mean"] = summary.daily_mean * 252
                summary.to_csv(out / f"{weighting}_decile_summary.csv")
            ledgers[case] = ledger
            joined = ledger.merge(factors, on="date", how="left", validate="one_to_one", indicator=True)
            assert joined._merge.eq("both").all(), "Factor coverage gap"
            assert joined[list(NAMES["FF5+UMD"]) + ["rf"]].notna().all().all()
            for window, (a, b) in p["inference"]["windows"].items():
                sample = joined.loc[joined.date.between(a, b)]
                perf_rows.append(dict(weighting=weighting, cost_bp=bp, window=window, **performance(sample)))
                for model in NAMES:
                    alpha_rows.append(dict(weighting=weighting, cost_bp=bp, window=window,
                        **fit_alpha(sample, model, p["inference"]["primary_hac_lag"])))
                    for lag in p["inference"]["sensitivity_hac_lags"]:
                        lag_rows.append(dict(weighting=weighting, cost_bp=bp, window=window, **fit_alpha(sample, model, lag)))
        # Stress before costs is a robustness diagnostic with unchanged score direction.
        stress, _, _ = backtest(panel, pred, weighting, 0, adverse_missing=True)
        stress.to_csv(out / f"{weighting}_missing_stress_ledger.csv", index=False)
        sj = stress.merge(factors, on="date", how="left", validate="one_to_one")
        stress_rows.append(dict(weighting=weighting, **performance(stress), **{
            f"stress_{k}": v for k, v in fit_alpha(sj).items()}))
        for seed, frame in enumerate(seeds):
            ledger, _, _ = backtest(panel, frame, weighting, 0)
            ledger.to_csv(out / f"{weighting}_seed{seed}_ledger.csv", index=False)
            seed_rows.append(dict(weighting=weighting, seed=seed, **performance(ledger)))
    alpha = pd.DataFrame(alpha_rows)
    assert len(alpha) == 36
    alpha["p_holm36"] = multipletests(alpha.p.to_numpy(), method="holm")[1]
    alpha["nominal_positive"] = (alpha.annual_alpha > 0) & (alpha.p < .05)
    alpha["family_positive"] = (alpha.annual_alpha > 0) & (alpha.p_holm36 < .05)
    alpha.to_csv(out / "alpha.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(out / "hac_sensitivity.csv", index=False)
    perf = pd.DataFrame(perf_rows)
    perf.to_csv(out / "performance.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(out / "missing_stress.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(out / "seed_diagnostics.csv", index=False)
    # Independently replay saved holdings against original total-return arrays.
    source_checks = []
    date_map = {str(d): i for i, d in enumerate(panel.dates)}
    for case, ledger in ledgers.items():
        archive = np.load(out / f"{case}_weights.npz")
        payoff = np.asarray(panel.total_return[:, [date_map[d] for d in ledger.date]], float).T
        replay = np.sum(archive["weights"] * np.where(np.isfinite(payoff), payoff, 0), axis=1)
        np.testing.assert_allclose(replay, ledger.return_gross, rtol=1e-10, atol=1e-12)
        source_checks.append(dict(case=case, max_replay_error=float(abs(replay - ledger.return_gross).max())))
    primary = alpha[(alpha.window == "full-2017-2025") & (alpha.factor_model == "FF5+UMD")]
    verdicts = primary[["weighting", "cost_bp", "annual_alpha", "ci_low", "ci_high", "t", "p", "p_holm36",
        "nominal_positive", "family_positive"]].to_dict("records")
    report(perf, alpha, verdicts)
    plot(ledgers)
    write_json(out / "verification.json", dict(status="passed", updated_utc=now(),
        inputs_sha256=input_hashes(), data_manifest_sha256=sha(DATA / "manifest.json"),
        source_replay=source_checks, alpha_family_size=36, verdicts=verdicts,
        output_sha256={f.name: sha(f) for f in sorted(out.glob("*.csv"))},
        report_sha256=sha(HERE / "REPORT.txt")))
    print(json.dumps(verdicts, indent=2), flush=True)


def report(perf, alpha, verdicts):
    universe = pd.read_csv(DATA / "universe_audit.csv")
    manifest = json.loads((DATA / "manifest.json").read_text())
    lines = ["LiMT / CHEN-ZIMMERMANN / NONMICROCAP / WEIGHTED DECILES", "",
        "Primary weighting requested by the user: VALUE (market capitalization). Volume weighting is supplementary.",
        "Completed three-seed ensemble. Retrospective specification; no winning-seed selection.",
        "D10-D1 uses +100%/-100%, gross 200%. The gross100 columns divide arithmetic means and alpha by two.", "",
        "FULL-PERIOD FF5+UMD RESULTS", pd.DataFrame(verdicts).to_string(index=False), "",
        "Significantly positive requires alpha>0 and a two-sided HAC p<0.05; family_positive additionally requires Holm36 p<0.05.",
        "A positive estimate alone does not establish alpha. All cost levels and weights remain reported.", "",
        "UNIVERSE SCREEN", universe.to_string(index=False), "",
        f"209 official CZ source characteristics; {len(manifest['feature_names'])} retained after training-only constant-column removal.",
        "Public source omits Price, Size, STreversal. No Alpha158 input or factor-portfolio-return substitution.",
        "Preserves dynamic US large-500 universe. Results do not generalize automatically to the entire nonmicrocap CRSP universe.",
        "Monthly features use previous calendar month only; the authors already apply the source accounting publication lags.",
        "Within-month cross-sectional ranks over observed common-stock nonmicrocap population; missing values neutral zero, no stale carry.",
        "Value weights use previous-month-end capitalization; volume weights use signal-day raw shares.",
        "Raw share-volume weighting depends on share units/splits and is distinct from dollar-volume and capitalization weighting.", "",
        "PERFORMANCE", perf.to_string(index=False), "", "HAC20 ALPHAS", alpha.to_string(index=False), "",
        "Signal t; assumed execution t+1 close; total-return payoff t+2. Frozen positions use original dollar-neutral projection.",
        "D10-D1 ledger reflects executable holdings; target-decile tables are unconstrained before-cost diagnostics.",
        "Zero-net spreads are regressed directly on realized-date FF5 and UMD; RF is not subtracted again.",
        "Annual alpha is arithmetic daily intercept *252. HAC Bartlett20 with small-sample correction, t-reference 95% CI; lags5/60 saved.",
        "One Holm36 family covers all 2 weights x2 costs x3 windows x3 intercept specifications. Prior research choices are outside this family.",
        "All-trade10 sensitivity charges 10bp for buys, sells and final liquidation. Financing, borrow, impact and collateral yield omitted.",
        "Missing held payoffs remain zero in primary accounting, with counts/weights and adverse long-100%/short+100% stress saved.",
        "Current-vintage CRSP/CZ data cannot certify historical as-published availability; option signal coverage ends early in some columns.", "",
        "SOURCES", "Chen and Zimmermann (2022), Open Source Cross-Sectional Asset Pricing, Critical Finance Review 11(2), 207-264.",
        "https://www.openassetpricing.com/data/", "https://huggingface.co/datasets/OpenSourceAP/CrossSection/tree/Data-Release-2026.10",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_5_factors_2x3.html",
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor_daily.html",
        "Software workflow: statsmodels skill; Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026).",
        "Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. https://doi.org/10.48550/arXiv.2609.00065", "",
        "REPRODUCE", "$LIMT_MODEL_PYTHON experiments/standard_characteristics/run.py", ""]
    (HERE / "REPORT.txt").write_text("\n".join(lines))


def plot(ledgers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, weighting in zip(axes, config()["portfolio"]["weights"]):
        for bp in [0, 10]:
            ledger = ledgers[f"{weighting}_cost{bp}"]
            ax.plot(pd.to_datetime(ledger.date), np.cumprod(1 + ledger.return_net), label=f"{bp} bp all trades")
        ax.set_title(f"{weighting.capitalize()} weighted D10-D1 (+100%/-100%)")
        ax.axhline(1, color="grey", lw=.5)
        ax.legend()
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Wealth per initial dollar")
    fig.tight_layout()
    fig.savefig(HERE / "analysis/performance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    run()
