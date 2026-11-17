#!/usr/bin/env python3
"""Independent SciPy correlations and direct cycle-contrast spot checks."""
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from company_time.analyze import primary, PROJECT, OUT


def run():
    con = duckdb.connect()
    con.execute('SET threads=4')
    calendar = pd.read_csv(PROJECT/'results/trading_calendar.csv', parse_dates=['date'])
    con.register('calendar_frame', calendar)
    con.execute('CREATE TEMP TABLE calendar AS SELECT * FROM calendar_frame')
    primary.build_events(con, 'compustat')
    r = primary.year_data(con, 2025)
    r = r.loc[r.eligible & r.regular_paper & r.coverage90_at_announcement.ge(5)
        & (r.age.between(6, 20) | r.age.between(36, 50))].copy()
    r['ret'] = r.ret.fillna(0.)
    r['sign'] = np.where(r.age.le(20), 1, -1)
    daily = pd.read_csv(OUT/'daily_cross_section.csv', parse_dates=['date']).set_index('date')
    ic_checks = []
    for date in sorted(r.date.unique())[::25]:
        g = r.loc[r.date.eq(date)]
        actual = float(pearsonr(g.sign.to_numpy(), g.ret.to_numpy()).statistic)
        error = abs(actual-daily.loc[date, 'ic_raw'])
        assert error < 1e-12
        ic_checks.append(error)
    cycles = pd.read_csv(OUT/'company_cycles.csv')
    mapping = con.execute(f"""SELECT DISTINCT permno,permco
        FROM read_parquet('{primary.RAW}/2025.parquet')""").df()
    cycle_checks = []
    for (permno, event_idx), g in r.groupby(['permno', 'event_idx']):
        if len(g) != 30 or g.loc[g.sign.eq(1)].shape[0] != 15:
            continue
        ids = mapping.loc[mapping.permno.eq(permno), 'permco']
        if len(ids) != 1:
            continue
        x = cycles.loc[cycles.event_idx.eq(event_idx) & cycles.permco.eq(ids.iloc[0])
                       & cycles.share_classes.eq(1)]
        if len(x) != 1:
            continue
        actual = g.loc[g.sign.eq(1), 'ret'].mean()-g.loc[g.sign.eq(-1), 'ret'].mean()
        error = abs(actual-float(x.spread_raw.iloc[0]))
        assert error < 1e-12
        cycle_checks.append(error)
        if len(cycle_checks) == 10:
            break
    assert len(cycle_checks) == 10
    firms = pd.read_csv(OUT/'company_distribution.csv')
    inference = pd.read_csv(OUT/'mean_inference.csv')
    for period, g in firms.groupby('period'):
        expected = inference.loc[inference.period.eq(period)
            & inference.metric.eq('spread_raw'), 'mean'].iloc[0]
        assert abs(g.annual_gross_contribution.sum()-expected) < 1e-12

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ['src', 'href']:
                    self.links.append(value)

    parser = Links()
    parser.feed((OUT/'report.html').read_text())
    missing = [v for v in parser.links if '://' not in v and not (OUT/v).exists()]
    assert not missing, missing
    results = dict(independent_2025_scipy_ic_dates=len(ic_checks),
        max_ic_error=max(ic_checks), independent_complete_cycle_checks=len(cycle_checks),
        max_cycle_error=max(cycle_checks), all_company_contributions_reconcile=True,
        report_local_links_resolve=len(parser.links))
    (OUT/'validation.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    run()
