#!/usr/bin/env python3
"""Describe departures from normality in the saved company/time distributions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from company_time.analyze import OUT, PERIODS


def shape(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    # Bias-corrected standardized central moments, verified independently.
    centered = x-x.mean()
    m2, m3, m4 = (float(np.mean(centered**k)) for k in (2, 3, 4))
    g1 = m3/m2**1.5
    g2 = m4/m2**2-3
    corrected_skew = np.sqrt(n*(n-1))/(n-2)*g1
    corrected_excess = (n-1)/((n-2)*(n-3))*((n+1)*g2+6)
    skew = float(stats.skew(x, bias=False))
    excess = float(stats.kurtosis(x, fisher=True, bias=False))
    assert np.isclose(skew, corrected_skew, rtol=1e-10, atol=1e-12)
    assert np.isclose(excess, corrected_excess, rtol=1e-10, atol=1e-12)
    jb = stats.jarque_bera(x)
    return dict(n=n, skewness=skew, excess_kurtosis=excess,
        pearson_kurtosis=excess+3,
        fraction_beyond_3sd=float(np.mean(np.abs(centered/x.std(ddof=1)) > 3)),
        jarque_bera_statistic=float(jb.statistic),
        jarque_bera_iid_reference_p=float(jb.pvalue))


def update_report():
    results = pd.read_csv(OUT/'normality.csv')
    table = results.loc[results.period.eq('all') & results.metric.eq('spread_raw'),
        ['dimension', 'n', 'skewness', 'excess_kurtosis', 'pearson_kurtosis']]
    table = pd.concat([pd.DataFrame([dict(dimension='Normal reference', n=np.nan,
        skewness=0., excess_kurtosis=0., pearson_kurtosis=3.)]), table], ignore_index=True)
    section = '''<!-- NORMALITY_START -->
    <h2>Departure from a normal distribution</h2>
    <p>These statistics describe the distributions of gross return spreads: across the history-qualified
    companies and across all dates. Company IDs remain unordered grouping labels. Skewness measures asymmetry
    (normal reference 0); excess kurtosis measures the fourth standardized moment minus 3 (normal reference 0).
    Pearson kurtosis is excess kurtosis plus 3. Positive excess kurtosis describes more extreme standardized
    observations than the normal reference; it is not a percentage deviation or a measure of predictive power.</p>
    TABLE
    <p>Estimates use SciPy bias-corrected skewness and kurtosis, checked against independent central-moment
    formulas. Complete untrimmed samples are used. Shared market shocks, different company history lengths,
    and time-varying volatility can influence these pooled distributions. Jarque–Bera statistics and IID
    reference p-values are available in the CSV; company and time dependence means those p-values are not
    calibrated dependence-adjusted inference. Numerical p=0 denotes floating-point underflow.</p>
    <p><a href="normality.csv">Shape statistics for all periods, spreads and phase correlations</a></p>
    <!-- NORMALITY_END -->'''.replace('TABLE', table.to_html(index=False,
        na_rep='—', float_format=lambda x: f'{x:.4f}'))
    path = OUT/'report.html'
    report = path.read_text()
    if '<!-- NORMALITY_START -->' in report:
        start = report.index('<!-- NORMALITY_START -->')
        end = report.index('<!-- NORMALITY_END -->')+len('<!-- NORMALITY_END -->')
        report = report[:start]+section+report[end:]
    else:
        report = report.replace('<h2>Company persistence</h2>', section+'<h2>Company persistence</h2>')
    path.write_text(report)


def run():
    companies = pd.read_csv(OUT/'company_distribution.csv')
    daily = pd.read_csv(OUT/'daily_cross_section.csv', parse_dates=['date']).set_index('date')
    rows = []
    for period, (start, end) in PERIODS.items():
        company = companies.loc[companies.period.eq(period) & companies.sufficient_history]
        for dimension, frame in [('company', company), ('time', daily.loc[start:end])]:
            for metric in ['spread_raw', 'spread_market_adjusted', 'ic_raw', 'ic_market_adjusted']:
                rows.append(dict(period=period, dimension=dimension, metric=metric, **shape(frame[metric])))
    results = pd.DataFrame(rows)
    results.to_csv(OUT/'normality.csv', index=False)
    update_report()
    (OUT/'normality_validation.json').write_text(json.dumps(dict(
        distributions_checked=len(results),
        bias_corrected_moments_match_independent_formulas=True,
        normal_reference=dict(skewness=0, excess_kurtosis=0, pearson_kurtosis=3),
        methods='SciPy skew(bias=False), kurtosis(fisher=True,bias=False); full finite samples',
        note='JB p-values use IID reference; zero indicates numerical underflow.'), indent=2))
    print(results.loc[results.period.eq('all') & results.metric.eq('spread_raw')].to_string(index=False))


if __name__ == '__main__':
    run()
