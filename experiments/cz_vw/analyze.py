"""Full-period value-weighted D10-D1 and FF5+UMD HAC alpha report."""
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from common import HERE, DATA, FACTORS, settings, check_lock, sha, save_json, now
from panel import Panel
from portfolio import backtest


def fit_alpha(ledger, factors, lag=20):
    joined = ledger.merge(factors, on='date', how='left', validate='one_to_one', indicator=True)
    assert joined._merge.eq('both').all(), 'Unmatched realized factor dates'
    names = settings()['inference']['factors']
    x = np.column_stack([np.ones(len(joined)), joined[names].to_numpy(float)])
    y = joined.return_net.to_numpy(float)
    assert np.isfinite(x).all() and np.isfinite(y).all() and np.linalg.matrix_rank(x) == 7
    fit = sm.OLS(y, x, missing='raise').fit(cov_type='HAC',
        cov_kwds=dict(maxlags=lag, kernel='bartlett', use_correction=True), use_t=True)
    ci = fit.conf_int()[0] * 252
    result = dict(days=len(y), hac_lag=lag, annual_alpha=float(fit.params[0] * 252),
        annual_alpha_se=float(fit.bse[0] * 252), hac_t=float(fit.tvalues[0]), p=float(fit.pvalues[0]),
        ci_low=float(ci[0]), ci_high=float(ci[1]), annual_alpha_gross100=float(fit.params[0] * 126),
        r_squared=float(fit.rsquared), condition_number=float(np.linalg.cond(x)),
        max_leverage=float(fit.get_influence().hat_matrix_diag.max()),
        residual_lag1=float(pd.Series(fit.resid).autocorr(1)))
    result.update({f'beta_{name}': float(beta) for name, beta in zip(names, fit.params[1:])})
    return result


def performance(ledger):
    r = ledger.return_net.to_numpy()
    wealth = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1., wealth])[1:]
    return dict(days=len(r), first_date=ledger.date.iloc[0], last_date=ledger.date.iloc[-1],
        annual_mean=float(r.mean() * 252), annual_mean_gross100=float(r.mean() * 126),
        cagr=float(wealth[-1] ** (252 / len(r)) - 1), annual_vol=float(r.std(ddof=1) * np.sqrt(252)),
        sharpe=float(np.sqrt(252) * r.mean() / r.std(ddof=1)), max_drawdown=float(np.max(1 - wealth / peak)),
        cumulative_return=float(wealth[-1] - 1), mean_turnover=float(ledger.turnover.mean()),
        mean_gross_exposure=float(ledger.gross_exposure.mean()), max_abs_net=float(ledger.net_exposure.abs().max()),
        missing_held_count=int(ledger.missing_held_count.sum()), missing_gross_weight_days=float(ledger.missing_gross_weight.sum()),
        missing_price_drift_count=int(ledger.missing_price_drift_count.sum()),
        max_target_deviation=float(ledger.target_deviation_l1.max()),
        final_unliquidated_dollars=float(ledger.final_unliquidated_dollars.iloc[-1]))


def run():
    check_lock()
    out = HERE / 'analysis'
    out.mkdir(exist_ok=True)
    folder = HERE / 'results/sp500/stage2/seed0'
    metrics = json.loads((folder / 'metrics.json').read_text())
    assert not metrics['engineering_smoke']
    assert metrics['predictions_sha256'] == sha(folder / 'predictions.parquet')
    pred = pd.read_parquet(folder / 'predictions.parquet')
    panel = Panel()
    factors = pd.read_parquet(FACTORS)
    factors['date'] = factors.date.astype(str)
    assert not factors.date.duplicated().any()
    rows, sensitivity, performances, ledgers, stresses = [], [], [], {}, []
    for bp in settings()['portfolio']['cost_bp_all_trades']:
        ledger, weights, deciles = backtest(panel, pred, bp)
        ledger.to_csv(out / f'cost{bp}_ledger.csv', index=False)
        np.savez_compressed(out / f'cost{bp}_weights.npz', weights=weights,
            dates=ledger.date.to_numpy(dtype='U10'), symbols=panel.symbols)
        ledgers[bp] = ledger
        performances.append(dict(cost_bp=bp, **performance(ledger)))
        rows.append(dict(cost_bp=bp, **fit_alpha(ledger, factors, settings()['inference']['primary_hac_lag'])))
        for lag in settings()['inference']['sensitivity_hac_lags']:
            sensitivity.append(dict(cost_bp=bp, **fit_alpha(ledger, factors, lag)))
        if bp == 0:
            deciles.to_csv(out / 'target_deciles.csv', index=False)
            d = deciles.groupby('decile').agg(days=('target_return', 'size'), daily_mean=('target_return', 'mean'),
                missing_weight=('missing_weight', 'sum'), stocks_mean=('stocks', 'mean'))
            d['annual_mean'] = d.daily_mean * 252
            d.to_csv(out / 'decile_summary.csv')
        stress, _, _ = backtest(panel, pred, bp, adverse_missing=True)
        stresses.append(dict(cost_bp=bp, **fit_alpha(stress, factors)))
    alpha = pd.DataFrame(rows)
    alpha['p_holm2'] = multipletests(alpha.p.to_numpy(), method='holm')[1]
    alpha.to_csv(out / 'alpha.csv', index=False)
    perf = pd.DataFrame(performances)
    perf.to_csv(out / 'performance.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(out / 'hac_sensitivity.csv', index=False)
    pd.DataFrame(stresses).to_csv(out / 'missing_return_stress.csv', index=False)
    report(alpha, perf, metrics)
    plot(ledgers, alpha)
    save_json(out / 'analysis_receipt.json', dict(created_utc=now(), predictions_sha256=sha(folder / 'predictions.parquet'),
        protocol_sha256=sha(HERE / 'protocol.json'), outputs_sha256={p.name: sha(p) for p in sorted(out.iterdir())
            if p.is_file() and p.name not in ['analysis_receipt.json', 'verification.json']},
        report_sha256=sha(HERE / 'REPORT.txt'), primary_results=alpha.to_dict('records')))
    print(alpha.to_string(index=False), flush=True)


def report(alpha, perf, metrics):
    manifest = json.loads((DATA / 'manifest.json').read_text())
    history1 = json.loads((HERE / 'results/sp500/stage1/status.json').read_text())
    compact = alpha[['cost_bp', 'annual_alpha', 'hac_t', 'ci_low', 'ci_high', 'p', 'p_holm2']]
    lines = ['PRISM-VQ / CZ202 / VALUE-WEIGHTED D10-D1', '',
        'One completed full-period US experiment, Stage1 seed42 / Stage2 seed0.',
        'New retrospective adaptation of the released PRISM-VQ architecture; not exact published reproduction.',
        'User acceptance protocol: ../protocol.txt relative to the PRISM-VQ workspace. Completion is subject to analysis/verification.json passing.', '',
        'TEST 2022-2024: FF5+UMD, HAC20', compact.to_string(index=False), '',
        'Alpha and intervals are decimal annual arithmetic returns. p is two-sided; p_holm2 corrects the two gross/net tests.',
        'D10-D1 is +100% long and -100% short, 200% gross. Divide arithmetic returns and alpha by two for 100% gross.',
        'Annual alpha = daily OLS intercept *252. Bartlett HAC20, small-sample correction, t-reference 95% CI.',
        'Zero-net spread returns are regressed directly; no second subtraction of the risk-free return.', '',
        'PERFORMANCE', perf.to_string(index=False), '',
        f"Ranking diagnostic: RankIC={metrics['RankIC']:.6f}, RankICIR={metrics['RankICIR']:.6f}; {metrics['days']} forecast dates.",
        f"Stage1 selected epoch {history1['best_epoch']} of {history1['epochs_completed']} trained (zero-based).",
        f"Stage2 selected epoch {metrics['best_epoch']} of {metrics['trained_epochs']} trained (zero-based), solely by validation loss.", '',
        'PROTOCOL AND SCOPE',
        'Historical S&P500 membership, further screened to US corporate ordinary common shares and observed lagged capitalization/CZ.',
        'Explicitly exclude size below previous-month-end NYSE common-stock 20th percentile; unknown size is excluded separately.',
        'This large-cap sample is not an all-US-nonmicrocap CRSP experiment. See data/universe_audit.csv for every exclusion.',
        '209 public CZ characteristics; remove seven training-constant columns and retain exactly 202.',
        'Dropped: ' + ', '.join(manifest['dropped_training_constant']) + '.',
        'Public source omits Price, Size, STreversal. Inputs are stock characteristics, not portfolio factor returns.',
        'Each historical daily input uses only its immediately previous calendar month CZ record; no stale-month forward fill.',
        'Monthly ranks use contemporaneous eligible nonmicrocap common shares, average ties, missing neutral zero; no return data.',
        '20 daily input sessions; monthly CZ values repeat within months. This changes the feature time scale from Alpha158.',
        '13 lagged JKP US theme priors, model architecture, losses, original close-to-close targets and split purges retained.',
        'Train2009-2019, validation2020-2021, test2022-2024. Frozen Stage1 remains in eval mode; custom RMSNorm is respected.',
        'Scores at t execute at t+1 close and earn daily CRSP total return at t+2; five-day labels never stand in for daily P&L.',
        'Deciles use signal-known previous-month-end capitalization; ties are deterministic by PERMNO; direction fixed high-minus-low.',
        'Untradable positions remain held; minimum-change projection maintains dollar neutrality when feasible.',
        'Dividends enter P&L; price-only returns drift share positions. Dated terminal events settle to cash.',
        'Cost0 is gross. Cost10 charges 10bp on each traded dollar on both sides, including entry and executable final liquidation.',
        'Target exposures use pre-fee NAV; resulting post-fee exposure differs slightly. Borrow, financing, impact, collateral yield omitted.',
        'Final unexecutable positions remain marked and are reported, without invented liquidation trades.',
        'Missing held total returns are zero in the main ledger; counts, weights, missing-price drift and adverse stress are saved.',
        'Lags5/60 are diagnostics, not alternative result selection. One seed does not measure seed uncertainty.',
        'Current-vintage CZ/CRSP data retain vendor revisions; historical as-published availability is not established.', '',
        'SOURCES AND SOFTWARE',
        'Chen and Zimmermann (2022), Open Source Cross-Sectional Asset Pricing, Critical Finance Review 11(2), 207-264.',
        'Official source: https://huggingface.co/datasets/OpenSourceAP/CrossSection/tree/Data-Release-2026.10',
        'CRSP CIZ daily/monthly, local WRDS snapshots, retained transfer receipts and hashes.',
        'Kenneth French official daily FF5+Mom, independently checked against local WRDS factors; decimal units.',
        'PRISM-VQ author implementation commit 7d02635d0cdeec2f4e7278a9e73acd0eb67a8fa6, retained unmodified.',
        'PyTorch2.4.1; statsmodels0.14.6 OLS/HAC. Full versions and input/code hashes: locked_before_training.json.', '',
        'ARTIFACTS: analysis/alpha.csv, performance.csv, cost*_ledger.csv, cost*_weights.npz,',
        'target_deciles.csv, decile_summary.csv, hac_sensitivity.csv, missing_return_stress.csv, results.png, verification.json.',
        'Updated UTC: ' + now()]
    (HERE / 'REPORT.txt').write_text('\n'.join(lines) + '\n')


def plot(ledgers, alpha):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout='constrained')
    colors = ['#26766a', '#c46039']
    for (bp, ledger), color in zip(ledgers.items(), colors):
        axes[0].plot(pd.to_datetime(ledger.date), ledger.end_nav, label=f'{bp} bp / traded dollar', color=color)
    axes[0].set(title='Value-weighted D10-D1, 200% gross', ylabel='NAV (initial = 1)')
    axes[0].axhline(1, color='.7', lw=.8)
    axes[0].legend(frameon=False)
    for i, row in enumerate(alpha.itertuples()):
        axes[1].errorbar(row.annual_alpha * 100, i, xerr=[[100 * (row.annual_alpha - row.ci_low)],
            [100 * (row.ci_high - row.annual_alpha)]], fmt='o', color=colors[i], capsize=5)
    axes[1].axvline(0, color='.5', lw=1)
    axes[1].set(yticks=[0, 1], yticklabels=['Gross', 'After 10 bp'], ylim=(-.7, 1.7),
        xlabel='Annual alpha (%) and 95% CI', title='FF5+UMD / HAC20')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(alpha=.15)
    fig.savefig(HERE / 'analysis/results.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    run()
