#!/usr/bin/env python3
"""Named EARC outliers, chronological stability, and selection-aware inference."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/earnings-company-time-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import S_hac_simple
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from company_time.analyze import OUT as PARENT

OUT = PARENT/'outliers'
LAGS = 4
MIN_CYCLES = 8
OUTCOMES = ['spread_raw', 'spread_market_adjusted']


def time_covariance(y, dates, split=False, maxlags=LAGS):
    """OLS mean(s) with Bartlett covariance of calendar-quarter score sums."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    q = dates.dt.year.to_numpy()*4+dates.dt.quarter.to_numpy()
    q = q-q.min()
    lag = min(maxlags, int(q.max()))
    if split:
        first = np.arange(n) < n//2
        x = np.column_stack([first, ~first]).astype(float)
        contrast = np.array([-1., 1.])
    else:
        x = np.ones((n, 1))
        contrast = np.array([1.])
    k = x.shape[1]
    bread = np.diag(1/x.sum(axis=0))
    params = bread @ x.T @ y
    residual = y-x @ params
    scores = np.column_stack([np.bincount(q, weights=x[:, j]*residual) for j in range(k)])
    meat = scores.T @ scores
    for h in range(1, lag+1):
        cross = scores[h:].T @ scores[:-h]
        meat += (1-h/(lag+1))*(cross+cross.T)
    covariance = bread @ meat @ bread*n/(n-k)
    estimate = float(contrast @ params)
    se = float(np.sqrt(max(contrast @ covariance @ contrast, 0.)))
    statistic = estimate/se
    df = n-k
    p = float(2*t.sf(abs(statistic), df))
    interval = t.ppf(.975, df)*se
    return dict(cycles=n, estimate=estimate, se=se, t_stat=statistic, df=df,
        p_unadjusted=p, ci_low=estimate-interval, ci_high=estimate+interval,
        hac_quarter_lags=lag, first_event=str(dates.min().date()),
        last_event=str(dates.max().date())), (x, contrast, covariance)


def independent_check(y, dates, result, design):
    x, contrast, covariance = design
    fit = sm.OLS(np.asarray(y), x, missing='raise').fit()
    # Use explicit calendar expansion: group_sums can factorize sparse integer
    # time labels and thereby compress gaps when max(label)>2*nobs.
    index = pd.PeriodIndex(dates, freq='Q')
    scores = pd.DataFrame(fit.model.exog*fit.resid[:, None], index=index).groupby(level=0).sum()
    scores = scores.reindex(pd.period_range(index.min(), index.max(), freq='Q'), fill_value=0.)
    kernel = S_hac_simple(scores.to_numpy(), nlags=result['hac_quarter_lags'])
    bread = fit.normalized_cov_params
    expected = bread @ kernel @ bread*len(y)/(len(y)-x.shape[1])
    np.testing.assert_allclose(covariance, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(result['estimate'], contrast @ fit.params, rtol=1e-10, atol=1e-15)


def estimate_all(companies, cycles):
    rows = []
    for permco, group in cycles.loc[cycles.permco.isin(companies.permco)].groupby('permco'):
        g = group.sort_values('event_date').reset_index(drop=True)
        assert len(g) >= MIN_CYCLES
        halves = {'all': g, 'first_half': g.iloc[:len(g)//2], 'second_half': g.iloc[len(g)//2:]}
        for outcome in OUTCOMES:
            for period, sample in halves.items():
                result, _ = time_covariance(sample[outcome], sample.event_date)
                rows.append(dict(permco=permco, outcome=outcome, period=period, **result))
            result, _ = time_covariance(g[outcome], g.event_date, split=True)
            rows.append(dict(permco=permco, outcome=outcome, period='late_minus_early', **result))
    inference = pd.DataFrame(rows)
    inference['p_holm_all_tests'] = multipletests(inference.p_unadjusted, method='holm')[1]
    return inference.merge(companies[['permco', 'latest_ticker', 'conm']],
        on='permco', how='left', validate='many_to_one')


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    companies = pd.read_csv(PARENT/'company_distribution.csv')
    candidates = companies.loc[companies.period.eq('all') & companies.sufficient_history
        & companies.complete_cycles.ge(MIN_CYCLES)].copy()
    cycles = pd.read_csv(PARENT/'company_cycles.csv', parse_dates=['event_date'])
    long = candidates.loc[candidates.complete_cycles.ge(40)]
    selections = [('extreme_positive', candidates.nlargest(2, 'spread_raw')),
                  ('extreme_negative', candidates.nsmallest(2, 'spread_raw')),
                  ('long_history_positive', long.nlargest(2, 'spread_raw')),
                  ('long_history_negative', long.nsmallest(2, 'spread_raw'))]
    selected = pd.concat([frame.assign(selection=label) for label, frame in selections], ignore_index=True)
    assert not selected.permco.duplicated().any()
    inference = estimate_all(candidates, cycles)
    inference.to_csv(OUT/'all_company_inference.csv', index=False)
    selected_inference = inference.loc[inference.permco.isin(selected.permco)]
    selected_inference.to_csv(OUT/'selected_company_inference.csv', index=False)
    summary, sensitivity, annual = [], [], []
    independent_checks = 0
    for company in selected.itertuples():
        g = cycles.loc[cycles.permco.eq(company.permco)].sort_values('event_date').reset_index(drop=True)
        row = dict(permco=company.permco, ticker=company.latest_ticker, name=company.conm,
            selection=company.selection, original_all_stock_day_spread_bp=company.spread_raw*1e4,
            cycles=len(g), positive_cycles=int(g.spread_raw.gt(0).sum()),
            first_event=str(g.event_date.min().date()), last_event=str(g.event_date.max().date()))
        for outcome in OUTCOMES:
            for period in ['all', 'first_half', 'second_half', 'late_minus_early']:
                z = selected_inference.loc[selected_inference.permco.eq(company.permco)
                    & selected_inference.outcome.eq(outcome) & selected_inference.period.eq(period)].iloc[0]
                label = period if outcome == 'spread_raw' else f'adjusted_{period}'
                for key in ['estimate', 'ci_low', 'ci_high']:
                    row[f'{label}_{key}_bp'] = z[key]*1e4
                for key in ['cycles', 'p_unadjusted', 'p_holm_all_tests', 'first_event', 'last_event']:
                    row[f'{label}_{key}'] = z[key]
                sample = g if period in ['all', 'late_minus_early'] else (
                    g.iloc[:len(g)//2] if period == 'first_half' else g.iloc[len(g)//2:])
                result, design = time_covariance(sample[outcome], sample.event_date,
                    split=period == 'late_minus_early')
                independent_check(sample[outcome], sample.event_date, result, design)
                independent_checks += 1
            for lag in [0, 1, 4]:
                result, _ = time_covariance(g[outcome], g.event_date, maxlags=lag)
                sensitivity.append(dict(permco=company.permco, name=company.conm,
                    outcome=outcome, requested_lags=lag, **result))
        extreme = g.spread_raw.abs().idxmax()
        row['largest_absolute_cycle_date'] = str(g.loc[extreme, 'event_date'].date())
        row['largest_absolute_cycle_bp'] = g.loc[extreme, 'spread_raw']*1e4
        row['mean_excluding_largest_absolute_cycle_bp'] = g.drop(extreme).spread_raw.mean()*1e4
        row['median_cycle_bp'] = g.spread_raw.median()*1e4
        summary.append(row)
        for year, y in g.groupby(g.event_date.dt.year):
            annual.append(dict(permco=company.permco, name=company.conm, year=year,
                cycles=len(y), mean_raw_bp=y.spread_raw.mean()*1e4,
                mean_market_adjusted_bp=y.spread_market_adjusted.mean()*1e4,
                positive_cycles=int(y.spread_raw.gt(0).sum())))
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT/'selected_companies.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT/'hac_sensitivity.csv', index=False)
    pd.DataFrame(annual).to_csv(OUT/'annual_effects.csv', index=False)
    cycles.loc[cycles.permco.isin(selected.permco)].to_csv(OUT/'selected_cycles.csv', index=False)
    design = dict(sample='1985-2025 primary EARC; gross complete-cycle phase contrasts.',
        company_candidates=len(candidates), statistical_tests=len(inference),
        selection='Top/bottom 2 raw all-stock-day company spreads; repeat within >=40 complete-cycle firms.',
        minimum_complete_cycles=MIN_CYCLES,
        outcome='Mean of 15 daily long-phase returns minus mean of 15 daily short-phase returns for same announcement. Complete share classes averaged at company/event.',
        split='First floor(n/2) qualifying cycles versus remaining cycles, chronological; not calendar midpoint.',
        inference='OLS mean / late-minus-early difference; Bartlett HAC scores aggregated by calendar quarter, <=4 quarter lags; n/(n-k) correction; Student t df=n-k. Approximate small-sample inference.',
        multiplicity='One Holm family: all eligible companies x raw/market-adjusted x full/first half/second half/difference. Covers this stated exploratory family, conditional on approximate marginal p-values.',
        independent_statsmodels_covariance_checks=independent_checks,
        covariance_maxlags=LAGS,
        selected_nominal_full_raw_significant=int((summary.all_p_unadjusted < .05).sum()),
        selected_holm_full_raw_significant=int((summary.all_p_holm_all_tests < .05).sum()),
        all_nominal_full_raw_significant=int(((inference.period == 'all') &
            (inference.outcome == 'spread_raw') & (inference.p_unadjusted < .05)).sum()),
        all_holm_full_raw_significant=int(((inference.period == 'all') &
            (inference.outcome == 'spread_raw') & (inference.p_holm_all_tests < .05)).sum()),
        limitations='Selected on entire history; earlier/later slices are descriptive, not independent out-of-sample confirmation. Short samples, shared shocks, historical linkage/versions and fat tails limit inference. Qualified cycles are intermittent. These daily contrasts are not annualized investment returns. Company effects are gross.',
        source_files=['../company_distribution.csv', '../company_cycles.csv', '../../README.txt'])
    (OUT/'inference_design.json').write_text(json.dumps(design, indent=2))
    plot(summary, cycles)
    report(summary, design)
    print(summary[['ticker', 'name', 'cycles', 'all_estimate_bp', 'all_p_unadjusted',
        'all_p_holm_all_tests', 'first_half_estimate_bp', 'first_half_p_unadjusted',
        'second_half_estimate_bp', 'second_half_p_unadjusted',
        'late_minus_early_p_unadjusted', 'mean_excluding_largest_absolute_cycle_bp']].to_string(index=False))
    print(json.dumps(design, indent=2))


def plot(summary, cycles):
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), constrained_layout=True)
    for axis, row in zip(axes.flat, summary.itertuples()):
        g = cycles.loc[cycles.permco.eq(row.permco)].sort_values('event_date')
        axis.scatter(g.event_date, g.spread_raw*1e4, s=22, c='#777777', alpha=.7,
                     label='One complete cycle')
        axis.axhline(0, c='#9d2d2d', lw=.8)
        for label, color in [('first_half', '#2867a0'), ('second_half', '#bc5e22')]:
            x = [pd.Timestamp(getattr(row, label+'_first_event')),
                 pd.Timestamp(getattr(row, label+'_last_event'))]
            mean = getattr(row, label+'_estimate_bp')
            low, high = getattr(row, label+'_ci_low_bp'), getattr(row, label+'_ci_high_bp')
            axis.plot(x, [mean]*2, c=color, lw=2, label=label.replace('_', ' '))
            axis.fill_between(x, [low]*2, [high]*2, color=color, alpha=.12)
        axis.set(title=f'{row.name.title()} ({row.ticker}), {row.cycles} cycles',
                 ylabel='Cycle spread (bp/day)', xlabel='Announcement date')
        axis.text(.02, .97,
            f'Early {row.first_half_estimate_bp:+.1f}, p={row.first_half_p_unadjusted:.3g}\n'
            f'Later {row.second_half_estimate_bp:+.1f}, p={row.second_half_p_unadjusted:.3g}',
            transform=axis.transAxes, va='top', fontsize=8,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
        axis.tick_params(axis='x', rotation=20)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Named EARC outliers: earlier versus later complete cycles\n'
        'Shading: approximate 95% quarter-HAC intervals; displayed p-values are unadjusted', fontsize=14)
    fig.savefig(OUT/'effects_over_time.png', dpi=150)
    plt.close(fig)


def report(summary, design):
    compact = []
    for row in summary.itertuples():
        compact.append(dict(company=f'{row.name.title()} ({row.ticker})',
            cycles=row.cycles, sample=f'{row.first_event} to {row.last_event}',
            full_bp=row.all_estimate_bp, full_p=row.all_p_unadjusted,
            holm_p=row.all_p_holm_all_tests,
            early_bp=row.first_half_estimate_bp, early_p=row.first_half_p_unadjusted,
            later_bp=row.second_half_estimate_bp, later_p=row.second_half_p_unadjusted,
            change_p=row.late_minus_early_p_unadjusted,
            mean_without_largest_cycle_bp=row.mean_excluding_largest_absolute_cycle_bp))
    compact = pd.DataFrame(compact)
    compact.to_csv(OUT/'compact_summary.csv', index=False)
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Named EARC outliers</title>
    <style>body{{font:16px/1.5 system-ui;max-width:1350px;margin:40px auto;padding:0 20px;color:#202830}}
    img{{max-width:100%}}table{{border-collapse:collapse;font-size:12px}}td,th{{padding:5px 8px;border-bottom:1px solid #ddd;text-align:right}}
    h1,h2{{color:#174d76}}.scroll{{overflow-x:auto}}</style></head><body>
    <h1>Named EARC company outliers and significance over time</h1>
    <p>Eight historical examples: two highest and two lowest company raw return spreads,
    plus the corresponding four among companies with at least 40 complete cycles.
    Candidate universe: {design['company_candidates']:,} firms with at least 120 stock-days in each phase
    and eight complete cycles. Names/tickers come from local CRSP/Compustat, including delisted companies.
    Dates describe the qualifying sample.</p>
    <p><b>Effect:</b> for each announcement, mean return on trading days 6–20 minus mean return on days 36–50.
    Require all 15 sessions on each side. Equal-weight complete cycles; basis points per selected day,
    before fees. The windows occur at different dates. This historical phase contrast is not a simultaneous
    self-financing investment return. Complete-cycle means can differ from earlier all-stock-day means
    because partial windows are excluded.</p>
    <p><b>Earlier/later:</b> first floor(n/2) qualifying cycles versus the rest. These are nonoverlapping
    halves of eligible observations, not equal calendar-length windows. Qualified cycles can be sparse.
    Missing years are not filled with zero.</p>
    <h2>Effects and significance</h2>
    <div class="scroll">{compact.to_html(index=False, float_format=lambda x: f'{x:.4g}')}</div>
    <p>Full, early and later p-values test zero mean spread; change_p tests later minus earlier = 0.
    Approximate Student t tests use Bartlett HAC score covariance at actual calendar-quarter distances,
    at most four quarter lags, correction n/(n−k), and n−k reference degrees of freedom.
    Short 8–12-cycle histories limit HAC inference. See
    <a href="hac_sensitivity.csv">lag 0/1/4 sensitivity</a> and
    <a href="selected_company_inference.csv">95% intervals and market-adjusted results</a>.</p>
    <p><b>Selection:</b> firms were picked for extreme full-history estimates. Holm adjustment covers
    {design['statistical_tests']:,} tests: every candidate firm, raw and market-adjusted effects,
    full/early/later mean and change. Among these eight,
    {design['selected_nominal_full_raw_significant']} full-sample raw effects have unadjusted p&lt;0.05;
    {design['selected_holm_full_raw_significant']} survive this Holm adjustment. Displayed intervals and
    unadjusted p-values are not simultaneous or selection-adjusted confidence statements. Correction covers
    this exploratory family, not every prior project analysis, and marginal HAC tests remain approximate.</p>
    <img src="effects_over_time.png" alt="Eight company cycle series, early/later means and intervals">
    <h2>Influence of single cycles</h2>
    <p>The last column recomputes the mean after dropping the largest absolute cycle. This is an influence
    diagnostic, not a preferred deletion rule. Seres' extreme cycle starts at the 2020-07-28 announcement;
    Scholar Rock's at 2024-08-08. The phase windows extend after those dates. A stock move inside a window
    need not be caused by that earnings announcement.</p>
    <h2>Files and validation</h2>
    <p><a href="selected_companies.csv">Selected estimates, intervals, dates and influence</a> ·
    <a href="annual_effects.csv">Annual descriptive effects</a> ·
    <a href="selected_cycles.csv">Every selected complete cycle</a> ·
    <a href="all_company_inference.csv">All-company test family</a> ·
    <a href="inference_design.json">Design and checks</a> ·
    <a href="../report.html">Main report</a></p>
    <p>Mean/difference covariance and OLS estimates independently match statsmodels OLS plus its HAC kernel
    on explicitly expanded calendar-quarter score sums in
    {design['independent_statsmodels_covariance_checks']} fits. Saved complete-cycle data inherit the primary
    validation. Historical versions/linkage and intermittent coverage remain limitations. Earlier/later
    comparisons are retrospective: firms were selected using both halves. Gross company effects omit fees.</p>
    <p>Software workflow: statsmodels 0.15.0; methods citation in
    <a href="../../README.txt">company_time/README.txt</a>.</p></body></html>'''
    (OUT/'report.html').write_text(page)
    main = PARENT/'report.html'
    text = main.read_text()
    text = text.replace('No company-specific hypothesis search or significance ranking is performed.',
        'Named-company follow-up tests and selection correction are reported in the linked outlier study.')
    link = '<p><a href="outliers/report.html">Named outlier companies: effects and significance over time</a></p>'
    if 'outliers/report.html' not in text:
        text = text.replace('</body>', link+'</body>')
    main.write_text(text)


if __name__ == '__main__':
    run()
