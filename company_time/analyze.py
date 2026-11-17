#!/usr/bin/env python3
"""Company and calendar-time diagnostics of the fixed primary EARC strategy.

Run from the project root with .venv/bin/python company_time/analyze.py.
Raw inputs and earlier outputs are read-only. No large stock-day cache is saved.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/earnings-company-time-mpl')
import duckdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from rolling_sharpe5 import build_paper_factor as primary

OUT = Path(__file__).resolve().parent / 'results'
MIN_DAYS_PER_PHASE = 120  # At least eight 15-session windows on each side.
MIN_SPLIT_CYCLES = 8
PERIODS = {'all': ('1985-01-01', '2025-12-31'),
           'pre2005': ('1985-01-01', '2004-12-31'),
           'post2005': ('2005-01-01', '2025-12-31'),
           'recent2015': ('2015-01-01', '2025-12-31')}


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def distribution(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    q = np.quantile(x, [.05, .10, .25, .5, .75, .90, .95])
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)),
                positive_fraction=float(np.mean(x > 0)),
                **dict(zip(['p05', 'p10', 'p25', 'median', 'p75', 'p90', 'p95'], q.tolist())))


def grouped_stats(frame, keys):
    """Unweighted Pearson phase/return correlation within each requested group."""
    z = frame.assign(s2=frame.sign**2, r2=frame.ret**2,
                     sr=frame.sign*frame.ret, a2=frame.abnormal_vw**2,
                     sa=frame.sign*frame.abnormal_vw)
    g = z.groupby(keys, observed=True)
    out = g.agg(n=('ret', 'size'), sum_s=('sign', 'sum'), sum_r=('ret', 'sum'),
                sum_r2=('r2', 'sum'), sum_sr=('sr', 'sum'), sum_a=('abnormal_vw', 'sum'),
                sum_a2=('a2', 'sum'), sum_sa=('sa', 'sum'),
                contribution=('contribution', 'sum')).reset_index()
    out['n_long'] = (out.n + out.sum_s) / 2
    out['n_short'] = (out.n - out.sum_s) / 2
    both = out.n_long.gt(0) & out.n_short.gt(0)
    for value, total, cross, squared in [('raw', 'sum_r', 'sum_sr', 'sum_r2'),
                                          ('market_adjusted', 'sum_a', 'sum_sa', 'sum_a2')]:
        denom = np.sqrt(np.maximum(out.n-out.sum_s**2/out.n, 0)
                        * np.maximum(out[squared]-out[total]**2/out.n, 0))
        out[f'ic_{value}'] = np.where(both & (denom > 0),
             (out[cross]-out.sum_s*out[total]/out.n)/denom, np.nan)
        out[f'long_{value}'] = np.where(out.n_long > 0,
                                            (out[total]+out[cross])/(2*out.n_long), np.nan)
        out[f'short_{value}'] = np.where(out.n_short > 0,
                                            (out[total]-out[cross])/(2*out.n_short), np.nan)
        out[f'spread_{value}'] = out[f'long_{value}']-out[f'short_{value}']
    return out.drop(columns=['sum_s', 'sum_r', 'sum_r2', 'sum_sr', 'sum_a', 'sum_a2', 'sum_sa'])


def hac_mean(x, annualization=1, lags=21):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    fit = sm.OLS(x, np.ones((len(x), 1)), missing='raise').fit(
        cov_type='HAC', cov_kwds={'maxlags': lags, 'use_correction': True})
    ci = fit.conf_int()[0] * annualization
    return dict(n=len(x), mean=float(fit.params[0]*annualization),
                se=float(fit.bse[0]*annualization), t=float(fit.tvalues[0]),
                p=float(fit.pvalues[0]), ci_low=float(ci[0]), ci_high=float(ci[1]),
                hac_lags=lags, annualization=annualization)


def split_persistence(cycles, column, rng):
    z = cycles.assign(split=np.where(cycles.event_date < pd.Timestamp('2005-01-01'),
                                     'pre2005', 'post2005'))
    means = z.groupby(['permco', 'split'])[column].agg(['mean', 'size']).unstack('split')
    keep = means['size'].min(axis=1).ge(MIN_SPLIT_CYCLES) & means['size'].notna().all(axis=1)
    m = means.loc[keep, 'mean'].dropna()
    x, y = m.pre2005.to_numpy(), m.post2005.to_numpy()
    pearson = float(stats.pearsonr(x, y).statistic)
    spearman = float(stats.spearmanr(x, y).statistic)
    boot = np.empty(2000)
    for b in range(len(boot)):
        ix = rng.integers(0, len(x), len(x))
        boot[b] = np.corrcoef(x[ix], y[ix])[0, 1]
    return dict(n_companies=len(m), pearson=pearson, spearman=spearman,
                bootstrap_ci_low=float(np.quantile(boot, .025)),
                bootstrap_ci_high=float(np.quantile(boot, .975)),
                minimum_complete_cycles_each_period=MIN_SPLIT_CYCLES), m


def variance_components(cycles, column):
    """Descriptive fixed-effect R²; categorical IDs are never Pearson regressors."""
    firm, firms = pd.factorize(cycles.permco)
    quarter, quarters = pd.factorize(cycles.event_date.dt.to_period('Q'))
    y = cycles[column].to_numpy(dtype=float)
    n = len(y)
    centered = y-y.mean()
    sst = float(centered @ centered)
    residuals = {}
    for label, ids in [('company', firm), ('announcement_quarter', quarter)]:
        count = np.bincount(ids)
        means = np.bincount(ids, weights=y)/count
        e = y-means[ids]
        residuals[label] = e
    e = centered.copy()
    counts = [np.bincount(firm), np.bincount(quarter)]
    for iteration in range(1000):
        old = e.copy()
        for ids, count in zip([firm, quarter], counts):
            e -= (np.bincount(ids, weights=e)/count)[ids]
        if np.max(np.abs(old-e)) < 1e-12:
            break
    else:
        raise AssertionError('Two-way demeaning did not converge')
    residuals['company_and_quarter'] = e
    # Rank of the two-way dummy design accounts for disconnected components.
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = firm
    b = quarter+len(firms)
    graph = coo_matrix((np.ones(2*n), (np.r_[a, b], np.r_[b, a])),
                       shape=(len(firms)+len(quarters),)*2).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    ranks = {'company': len(firms), 'announcement_quarter': len(quarters),
             'company_and_quarter': len(firms)+len(quarters)-components}
    output = []
    for label, residual in residuals.items():
        sse = float(residual @ residual)
        r2 = 1-sse/sst
        output.append(dict(outcome=column, model=label, observations=n,
            companies=len(firms), announcement_quarters=len(quarters),
            r_squared=r2, correlation_ratio=float(np.sqrt(max(r2, 0))),
            adjusted_r_squared=1-(sse/(n-ranks[label]))/(sst/(n-1)),
            rank=ranks[label], demeaning_iterations=iteration+1))
    return output


def run():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute('SET threads=4')
    con.execute("SET memory_limit='6GB'")
    calendar = pd.read_csv(PROJECT/'results/trading_calendar.csv', parse_dates=['date'])
    con.register('calendar_frame', calendar)
    con.execute('CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame')
    event_audit = primary.build_events(con, 'compustat')
    log(f"Qualified announcements: {event_audit['qualified_events']:,}")
    parts = []
    for year in range(1985, 2026):
        df = primary.year_data(con, year)
        selected = (df.eligible & df.regular_paper & df.coverage90_at_announcement.ge(5)
                    & (df.age.between(6, 20) | df.age.between(36, 50)))
        slim = df.loc[selected, ['permno', 'date', 'idx', 'event_idx', 'age', 'ret']].copy()
        del df
        con.register('slim_frame', slim)
        p = con.execute(f"""SELECT s.*, r.permco, r.ticker
            FROM slim_frame s JOIN read_parquet('{primary.RAW}/{year}.parquet') r
            ON s.permno=r.permno AND s.date=r.dlycaldt""").df()
        assert len(p) == len(slim)
        assert not p.duplicated(['permno', 'date']).any()
        assert p.permco.notna().all()
        parts.append(p)
        if year % 5 == 0 or year == 2025:
            log(f'Extracted through {year}: {sum(len(z) for z in parts):,} selected stock-days')
    panel = pd.concat(parts, ignore_index=True)
    del parts
    panel['missing_return'] = panel.ret.isna()
    panel['ret'] = panel.ret.fillna(0.)  # Match primary execution convention.
    panel['sign'] = np.where(panel.age.le(20), 1, -1)
    panel['event_idx'] = panel.event_idx.astype(int)
    panel['event_date'] = pd.to_datetime(calendar.set_index('idx').date.reindex(panel.event_idx).to_numpy())
    panel = panel.sort_values(['date', 'permno']).reset_index(drop=True)
    assert np.isfinite(panel.ret).all()
    assert (panel.ret >= -1).all()
    count = panel.groupby(['date', 'sign']).ret.transform('size')
    panel['contribution'] = panel.sign*panel.ret/count
    market = con.execute(f"SELECT date, vw_market FROM read_parquet('{PROJECT}/results/panel/market_*.parquet')").df()
    panel = panel.merge(market, on='date', how='left', validate='many_to_one')
    assert panel.vw_market.notna().all()
    panel['abnormal_vw'] = panel.ret-panel.vw_market

    daily = grouped_stats(panel, ['date'])
    prior = pd.read_csv(PROJECT/'rolling_sharpe5/paper_factor_daily.csv', parse_dates=['date'])
    primary_gross = prior.loc[prior.scenario.eq('gross')].set_index('date')
    daily = daily.set_index('date').sort_index()
    assert daily.index.equals(primary_gross.index)
    max_diff = float(np.max(np.abs(daily.spread_raw-primary_gross.returns)))
    assert max_diff < 1e-12, f'Stock-day reconstruction differs: {max_diff}'
    assert np.array_equal(daily.n_long.to_numpy(), primary_gross.n_long.to_numpy())
    assert np.array_equal(daily.n_short.to_numpy(), primary_gross.n_short.to_numpy())
    assert np.max(np.abs(daily.spread_market_adjusted-daily.spread_raw)) < 1e-12
    assert np.max(np.abs(daily.contribution-daily.spread_raw)) < 1e-12
    daily['net_5bps'] = prior.loc[prior.scenario.eq('5bps')].set_index('date').returns
    daily.reset_index().to_csv(OUT/'daily_cross_section.csv', index=False)
    log(f'Reconciled {len(daily):,} dates to primary; max difference {max_diff:.3g}')

    # A paired complete cycle is a historical phase contrast within one security.
    phases = panel.groupby(['permco', 'permno', 'event_idx', 'event_date', 'sign']).agg(
        n=('ret', 'size'), mean_return=('ret', 'mean'),
        mean_adjusted=('abnormal_vw', 'mean'),
        compound_return=('ret', lambda x: np.prod(1+x.to_numpy())-1),
        missing_returns=('missing_return', 'sum')).reset_index()
    paired = phases.pivot(index=['permco', 'permno', 'event_idx', 'event_date'], columns='sign')
    complete = paired['n'].eq(15).all(axis=1) & paired['n'].notna().all(axis=1)
    paired = paired.loc[complete]
    cycles = pd.DataFrame({
        'spread_raw': paired.mean_return[1]-paired.mean_return[-1],
        'spread_market_adjusted': paired.mean_adjusted[1]-paired.mean_adjusted[-1],
        'spread_15session_compound': paired.compound_return[1]-paired.compound_return[-1],
        'missing_returns': paired.missing_returns.sum(axis=1),
    }).reset_index()
    security_cycles = len(cycles)
    cycles = cycles.groupby(['permco', 'event_idx', 'event_date']).agg(
        spread_raw=('spread_raw', 'mean'), spread_market_adjusted=('spread_market_adjusted', 'mean'),
        spread_15session_compound=('spread_15session_compound', 'mean'),
        share_classes=('permno', 'size'), missing_returns=('missing_returns', 'sum')).reset_index()
    cycles['event_year'] = cycles.event_date.dt.year
    cycles.to_csv(OUT/'company_cycles.csv', index=False)
    identities = panel.groupby('permco').agg(latest_ticker=('ticker', 'last'),
        securities=('permno', 'nunique'), first_selected_date=('date', 'min'),
        last_selected_date=('date', 'max')).reset_index()
    names = con.execute(f"""SELECT permno,conm FROM read_parquet(
        '{PROJECT.parent}/compustat/ccm/quarterly_announcements_crsp_daily.parquet')
        WHERE permno IS NOT NULL AND conm IS NOT NULL
        QUALIFY row_number() OVER(PARTITION BY permno ORDER BY rdq DESC NULLS LAST,datadate DESC)=1""").df()
    firm_names = panel[['permco', 'permno']].drop_duplicates().merge(names, on='permno', how='left')
    firm_names = firm_names.dropna(subset=['conm']).drop_duplicates('permco')[['permco', 'conm']]
    identities = identities.merge(firm_names, on='permco', how='left', validate='one_to_one')
    company_rows = []
    aggregate_rows = []
    inference_rows = []
    for period, (start, end) in PERIODS.items():
        p = panel.loc[panel.date.between(start, end)]
        d = daily.loc[start:end]
        firms = grouped_stats(p, ['permco']).merge(identities, on='permco', validate='one_to_one')
        firms['period'] = period
        firms['sufficient_history'] = firms.n_long.ge(MIN_DAYS_PER_PHASE) & firms.n_short.ge(MIN_DAYS_PER_PHASE)
        firms['annual_gross_contribution'] = firms.contribution/len(d)*252
        cyc = cycles.loc[cycles.event_date.between(start, end)].groupby('permco').agg(
            complete_cycles=('spread_raw', 'size'), paired_spread_raw=('spread_raw', 'mean'),
            paired_spread_market_adjusted=('spread_market_adjusted', 'mean'),
            paired_spread_15session_compound=('spread_15session_compound', 'mean')).reset_index()
        firms = firms.merge(cyc, on='permco', how='left', validate='one_to_one')
        company_rows.append(firms)
        assert abs(firms.annual_gross_contribution.sum()-d.spread_raw.mean()*252) < 1e-12
        for dimension, data in [('company', firms.loc[firms.sufficient_history]), ('time', d)]:
            for metric in ['ic_raw', 'ic_market_adjusted', 'spread_raw', 'spread_market_adjusted']:
                aggregate_rows.append(dict(period=period, dimension=dimension, metric=metric,
                                            **distribution(data[metric])))
        for metric, a in [('spread_raw', 252), ('net_5bps', 252), ('ic_raw', 1)]:
            inference_rows.append(dict(period=period, metric=metric, **hac_mean(d[metric], a)))
    companies = pd.concat(company_rows, ignore_index=True)
    companies.to_csv(OUT/'company_distribution.csv', index=False)
    pd.DataFrame(aggregate_rows).to_csv(OUT/'distribution_summary.csv', index=False)
    pd.DataFrame(inference_rows).to_csv(OUT/'mean_inference.csv', index=False)

    rng = np.random.default_rng(20261109)
    persistence = {}
    for column in ['spread_raw', 'spread_market_adjusted']:
        result, points = split_persistence(cycles, column, rng)
        persistence[column] = result
        points.reset_index().to_csv(OUT/f'company_split_{column}.csv', index=False)
    vc = []
    for column in ['spread_raw', 'spread_market_adjusted']:
        vc.extend(variance_components(cycles, column))
    pd.DataFrame(vc).to_csv(OUT/'categorical_variance.csv', index=False)

    yearly = []
    for year, d in daily.groupby(daily.index.year):
        p = panel.loc[panel.date.dt.year.eq(year)]
        f = grouped_stats(p, ['permco'])
        f = f.loc[f.n_long.ge(15) & f.n_short.ge(15)]
        yearly.append(dict(year=year, days=len(d),
            gross_annual_arithmetic=d.spread_raw.mean()*252,
            net_5bps_annual_arithmetic=d.net_5bps.mean()*252,
            gross_compounded=float((1+d.spread_raw).prod()-1),
            net_5bps_compounded=float((1+d.net_5bps).prod()-1),
            mean_cross_section_ic=d.ic_raw.mean(), median_cross_section_ic=d.ic_raw.median(),
            positive_day_fraction=d.spread_raw.gt(0).mean(),
            company_count=len(f), positive_company_spread_fraction=f.spread_raw.gt(0).mean()))
    yearly = pd.DataFrame(yearly)
    yearly.to_csv(OUT/'annual_time_distribution.csv', index=False)
    monthly = daily[['spread_raw', 'net_5bps']].resample('ME').agg(lambda x: (1+x).prod()-1)
    monthly.to_csv(OUT/'monthly_time_distribution.csv')
    time_correlations = dict(daily_return_lag1=float(daily.spread_raw.autocorr()),
        monthly_gross_return_lag1=float(monthly.spread_raw.autocorr()),
        annual_gross_return_lag1=float(yearly.gross_compounded.autocorr()),
        cross_section_ic_lag1=float(daily.ic_raw.autocorr()),
        calendar_year_vs_gross_return_pearson=float(stats.pearsonr(
            yearly.year, yearly.gross_annual_arithmetic).statistic),
        positive_gross_years=int(yearly.gross_compounded.gt(0).sum()),
        positive_net_years=int(yearly.net_5bps_compounded.gt(0).sum()), years=len(yearly))
    audit = dict(created_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        event_audit=event_audit, stock_days=len(panel), companies=int(panel.permco.nunique()),
        securities=int(panel.permno.nunique()), dates=len(daily),
        missing_selected_returns=int(panel.missing_return.sum()),
        complete_security_cycles=security_cycles, complete_company_cycles=len(cycles),
        companies_with_complete_cycles=int(cycles.permco.nunique()),
        max_daily_return_difference=max_diff, min_days_each_phase=MIN_DAYS_PER_PHASE,
        first_date=str(daily.index.min().date()), last_date=str(daily.index.max().date()),
        persistence=persistence, time_correlations=time_correlations,
        source_fingerprints=[primary.fingerprint(Path(event_audit['path'])),
            primary.fingerprint(PROJECT/'coverage5/results/coverage_states.parquet'),
            primary.fingerprint(PROJECT/'rolling_sharpe5/paper_factor_daily.csv')],
        elapsed_seconds=time.monotonic()-started)
    save_json('audit.json', audit)
    plot_and_report(companies, daily, yearly, pd.DataFrame(aggregate_rows), persistence, vc, audit)
    log(f'COMPLETE in {time.monotonic()-started:.1f}s')
    print(json.dumps(dict(distributions=aggregate_rows, persistence=persistence,
                         time_correlations=time_correlations, variance=vc), indent=2), flush=True)


def plot_and_report(companies, daily, yearly, summary, persistence, vc, audit):
    f = companies.loc[companies.period.eq('all') & companies.sufficient_history]
    points = pd.read_csv(OUT/'company_split_spread_market_adjusted.csv')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for a, x, title in [(ax[0, 0], f.spread_raw*1e4,
                         f'Company return spreads ({len(f):,} companies)'),
                        (ax[0, 1], daily.ic_raw,
                         f'Daily cross-sectional phase / return correlation ({len(daily):,} dates)')]:
        lo, hi = x.quantile([.01, .99])
        a.hist(x, bins=np.linspace(lo, hi, 60), color='#2867a0', alpha=.85)
        a.axvline(0, color='#9d2d2d', lw=1)
        a.axvline(x.median(), color='#cb8a19', lw=1.5, label=f'Median {x.median():.4g}')
        a.set(title=title, ylabel='Count', xlabel='bp per selected day' if a is ax[0, 0] else 'Pearson r')
        a.legend(frameon=False)
        a.text(.02, .95, 'Display: 1st–99th percentiles; statistics use all observations',
               transform=a.transAxes, fontsize=8, va='top')
    ax[1, 0].plot(yearly.year, yearly.gross_annual_arithmetic*100, color='#2867a0', label='Gross')
    ax[1, 0].plot(yearly.year, yearly.net_5bps_annual_arithmetic*100, color='#bc5e22', label='Exact 5bp fees')
    ax[1, 0].axhline(0, color='#999999', lw=1)
    ax[1, 0].set(title='Calendar-time portfolio returns', xlabel='Year', ylabel='Annual arithmetic return (%)')
    ax[1, 0].legend(frameon=False)
    x, y = points.pre2005*1e4, points.post2005*1e4
    density = ax[1, 1].hexbin(x, y, gridsize=38, mincnt=1, cmap='Blues', bins='log')
    fig.colorbar(density, ax=ax[1, 1], label='Companies per hexagon (log scale)', shrink=.8)
    ax[1, 1].axhline(0, color='#999999', lw=.7)
    ax[1, 1].axvline(0, color='#999999', lw=.7)
    r = persistence['spread_market_adjusted']['pearson']
    ax[1, 1].set(title=f'Company persistence after market adjustment: r = {r:.3f}',
        xlabel='Company mean cycle spread, 1985–2004 (bp/day)',
        ylabel='Same company, 2005–2025 (bp/day)')
    fig.suptitle('EARC: company distribution and time cross-section, 1985–2025', fontsize=15)
    fig.savefig(OUT/'company_time.png', dpi=165)
    plt.close(fig)
    lines = []
    for period in PERIODS:
        for dimension in ['company', 'time']:
            s = summary.loc[(summary.period == period) & (summary.dimension == dimension)]
            r = s.loc[s.metric.eq('ic_raw')].iloc[0]
            b = s.loc[s.metric.eq('spread_raw')].iloc[0]
            lines.append(dict(period=period, dimension=dimension, observations=int(r.n),
                mean_r=r['mean'], median_r=r['median'], mean_spread_bp=b['mean']*1e4,
                median_spread_bp=b['median']*1e4, positive_fraction=b.positive_fraction,
                p10_spread_bp=b.p10*1e4, p90_spread_bp=b.p90*1e4))
    compact = pd.DataFrame(lines)
    compact.to_csv(OUT/'compact_summary.csv', index=False)
    high = f.sort_values('annual_gross_contribution', ascending=False)
    low = f.sort_values('annual_gross_contribution')
    columns = ['permco', 'latest_ticker', 'conm', 'n_long', 'n_short', 'ic_raw',
               'spread_raw', 'annual_gross_contribution']
    report = f'''<!doctype html><html><head><meta charset="utf-8"><title>EARC company and time</title>
    <style>body{{font:16px/1.5 system-ui;max-width:1150px;margin:40px auto;padding:0 20px;color:#202830}}
    img{{width:100%}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:6px 9px;border-bottom:1px solid #ddd;text-align:right}}
    h1,h2{{color:#174d76}}code{{background:#eee;padding:2px 4px}}</style></head><body>
    <h1>EARC: company distribution and time cross-section</h1>
    <p>Sample {audit['first_date']} through {audit['last_date']}; {audit['stock_days']:,} selected stock-days,
    {audit['companies']:,} CRSP companies (PERMCO), {audit['securities']:,} securities (PERMNO).
    Latest primary Compustat EARC: long age 6–20 and short age 36–50 trading sessions; regular announcements,
    at least five analysts frozen at announcement. All windows and screens reuse the primary implementation.</p>
    <p><b>Correlation definition:</b> phase S=+1 in the long window, −1 in the short window.
    Company r is corr(S, stock daily total return) across that company's eligible stock-days.
    Time r is the same correlation across stocks on each date. These are conditional on the two trading windows;
    neutral phases are excluded. Company IDs and dates are categorical groupings, never numerical regressors.
    A positive correlation means higher returns during the long phase. Spread is mean(long underlying return)
    minus mean(short underlying return), in bp per selected day. Company averages give each company equal weight;
    time averages give each date equal weight. Multiple share classes enter as separate securities.</p>
    <p>Company distribution requires ≥{MIN_DAYS_PER_PHASE} observations in each phase (eight full windows' worth;
    not necessarily eight complete paired cycles). Time distribution includes all {len(daily):,} dates.
    Market-adjusted metrics subtract the retained local CRSP value-weighted market return each day;
    these are market-adjusted returns, not risk-model alphas.</p>
    <img src="company_time.png" alt="Company distributions, time returns and persistence">
    <h2>Company and time distributions</h2>{compact.to_html(index=False, float_format=lambda v: f'{v:.5f}')}
    <p>Mean company spread does not equal the daily equal-weight portfolio spread: companies have different
    eligible histories and daily portfolio weights. Median daily IC and mean daily IC can differ from the
    fraction of positive days due to skewness. Gross stock-level correlations exclude costs.
    Portfolio net returns use the saved exact 5bp-per-dollar-traded path, with no borrow charge.</p>
    <h2>Company persistence</h2>
    <p>For each security/announcement, require all 15 long-phase and all 15 short-phase sessions. Its cycle
    contrast is the difference of these two average daily returns. Multiple complete share classes at the same
    company/event are averaged. Compare company mean contrasts before and after 2005, requiring at least eight
    complete company cycles on each side. These paired contrasts compare two different calendar windows;
    they are descriptive historical phase contrasts and are not simultaneous portfolio returns.</p>
    {pd.DataFrame(persistence).T.to_html(float_format=lambda v: f'{v:.5f}')}
    <p>Intervals use 2,000 company-resampling bootstrap draws, seed 20261109. They describe uncertainty across
    companies conditional on the observed periods; they do not capture all common calendar-time dependence.
    Split-history comparison includes companies observed on both sides and is not a tradable selection backtest.</p>
    <h2>Descriptive categorical association</h2>
    <p>One-way and two-way fixed-effect R² on complete company-cycle contrasts. Company and announcement-quarter
    effects measure in-sample variance of realized cycle returns explained by category means. They are not a
    partition of expected strategy profits. Adjusted R² accounts for fitted category count; an in-sample
    company effect can reflect noise and is assessed separately by split-period persistence.</p>
    {pd.DataFrame(vc).to_html(index=False, float_format=lambda v: f'{v:.5f}')}
    <h2>Time dependence</h2><pre>{html.escape(json.dumps(audit['time_correlations'], indent=2))}</pre>
    <p>Lag-one correlations pair consecutive daily, monthly or annual observations at the stated frequency.
    Year-versus-return correlation measures a linear calendar trend across 41 annual arithmetic returns.
    It is descriptive and does not measure cross-sectional stock correlation. Annual positive counts use
    compounded returns. Newey–West mean intervals (21 daily lags) are in <a href="mean_inference.csv">mean_inference.csv</a>.</p>
    <h2>Largest and smallest gross contributions</h2>
    <p>Annual gross contribution is the sum of each company's daily signed portfolio-weighted returns divided by
    all sample trading days, times 252. Contributions across <em>all</em> companies exactly sum to the strategy's
    annual arithmetic gross return. Rankings below use the history-qualified company subset; annual contribution
    and spread columns are decimal returns. Names/tickers are descriptive latest available labels.</p>
    {high.head(12)[columns].to_html(index=False, float_format=lambda v: f'{v:.6f}')}
    {low.head(12)[columns].to_html(index=False, float_format=lambda v: f'{v:.6f}')}
    <h2>Validation and files</h2>
    <p>Rebuilt daily gross returns and long/short counts match the saved primary on every date;
    maximum absolute return difference {audit['max_daily_return_difference']:.3g}.
    Company contributions reconcile exactly. {audit['missing_selected_returns']} missing selected daily returns
    are treated as zero, matching the primary. Complete cycles: {audit['complete_company_cycles']:,} company events.
    Raw inputs and prior reports are unchanged. Large regenerated stock-day data are not retained.</p>
    <p><a href="company_distribution.csv">Company distribution (all companies, all four periods)</a> ·
    <a href="daily_cross_section.csv">Daily cross-sections</a> ·
    <a href="annual_time_distribution.csv">Annual distribution</a> ·
    <a href="company_cycles.csv">Complete company-cycle contrasts</a> ·
    <a href="audit.json">Audit</a></p>
    <p>Limitations inherited from the primary: retrospective CCM/Compustat dates and incomplete historical
    versions; analyst update counts approximate active coverage; original study and this analysis are
    retrospective. Company distributions are noisy estimates, not a list of predictably profitable firms.
    No company-specific hypothesis search or significance ranking is performed.</p>
    <p>Method inputs: <a href="../../rolling_sharpe5/returns_spec.txt">primary return specification</a> and
    <a href="../../rolling_sharpe5/build_paper_factor.py">primary construction code</a>.
    Python statistical workflow: statsmodels 0.15.0; Scientific Agent Skills statsmodels skill.
    Skill reference is recorded in <a href="../README.txt">README.txt</a>.</p>
    </body></html>'''
    (OUT/'report.html').write_text(report)


if __name__ == '__main__':
    run()
