"""Re-score existing reproducibility experiments at zero implementation costs."""
from pathlib import Path
import hashlib
import json
import sys
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
INPUTS = {}
ROWS = []
CHECKS = []
SERIES = []
WARNINGS = []


def source(path):
    path = Path(path).resolve()
    if str(path) not in INPUTS:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(2**20), b''):
                h.update(block)
        INPUTS[str(path)] = h.hexdigest()
    return path


def read_json(path):
    return json.loads(source(path).read_text())


def csv(path, **kwargs):
    return pd.read_csv(source(path), **kwargs)


def hac(x, lag, periods):
    fit = sm.OLS(x, np.ones((len(x), 1)), missing='raise').fit(
        cov_type='HAC', cov_kwds={'maxlags': lag, 'use_correction': True,
                                 'kernel': 'bartlett'}, use_t=True)
    # Independent intercept-only Bartlett calculation checks library arithmetic.
    n = len(x)
    u = x - x.mean()
    v = u @ u
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (u[k:] @ u[:-k])
    se = np.sqrt(v / (n * (n-1)))
    np.testing.assert_allclose(fit.bse[0], se, atol=1e-13, rtol=1e-9)
    ci = fit.conf_int()[0] * periods
    return dict(p=float(fit.pvalues[0]), t=float(fit.tvalues[0]),
                ci_low=float(ci[0]), ci_high=float(ci[1]))


def boot(x, periods, block, seed, reps=5000):
    rng = np.random.default_rng(seed)
    n = len(x)
    estimates = []
    for j in range(0, reps, 250):
        starts = rng.integers(0, n, size=(min(250, reps-j), (n+block-1)//block))
        ix = ((starts[:, :, None] + np.arange(block)) % n).reshape(len(starts), -1)[:, :n]
        estimates.extend(x[ix].mean(1) * periods)
    return np.quantile(estimates, [.025, .975]).tolist()


def add(project, case, dates, returns, frequency='daily', diagnostic=False, **extra):
    x = np.asarray(returns, dtype=float)
    dates = np.asarray(dates).astype(str)
    assert len(x) == len(dates) and len(x) > 60 and np.isfinite(x).all()
    assert len(set(dates)) == len(dates) and np.all(dates[1:] > dates[:-1])
    assert x.min() > -1
    periods, lag, block = (252, 20, 20) if frequency == 'daily' else (12, 6, 12)
    seed = int(hashlib.sha256((project+case).encode()).hexdigest()[:8], 16)
    wealth = np.r_[1., np.cumprod(1+x)]
    low, high = boot(x, periods, block, seed)
    row = dict(project=project, case=case, diagnostic=diagnostic, frequency=frequency,
               n=len(x), start=dates[0], end=dates[-1],
               annual_mean=float(x.mean()*periods),
               cagr=float(wealth[-1]**(periods/len(x))-1),
               sharpe=float(x.mean()/x.std(ddof=1)*np.sqrt(periods)),
               max_drawdown=float(np.min(wealth/np.maximum.accumulate(wealth)-1)),
               periodic_mean=float(x.mean()), lag=lag, **hac(x, lag, periods),
               bootstrap_ci_low=low, bootstrap_ci_high=high, **extra)
    for l in ([5, 60] if frequency == 'daily' else [3, 12]):
        row.update({f'hac{l}_{k}': v for k, v in hac(x, l, periods).items()})
    if frequency == 'daily':
        a, b = boot(x, periods, 60, seed)
        row.update(bootstrap60_ci_low=a, bootstrap60_ci_high=b)
    ROWS.append(row)
    SERIES.append(pd.DataFrame(dict(project=project, case=case, date=dates, return_gross=x)))
    print(project, case, len(x), f'mean={row["annual_mean"]:.3%}',
          f'SR={row["sharpe"]:.3f}', f'p={row["p"]:.5g}', flush=True)


def alphaprobe():
    root = ROOT/'alphaprobe-reproduction/strategy/results/longshort-20day'
    read_json(root/'summary.json')
    for universe in ['csi300', 'csi500', 'csi1000']:
        f = csv(root/universe/'gross-equity.csv')
        np.testing.assert_allclose(np.cumprod(1+f.daily_return)*1e6, f.equity, rtol=1e-10)
        assert (f.transaction_cost == 0).all() and (f.borrow_cost == 0).all()
        assert f.open_net_exposure.abs().max() < 1e-10
        assert f.open_gross_exposure.min() > 0 and f.open_gross_exposure.max() < 1.1
        add('AlphaPROBE', universe, f.date, f.daily_return,
            max_net_exposure=float(f.open_net_exposure.abs().max()),
            max_end_of_day_net_drift=float(f.close_net_exposure.abs().max()), gross_target=1.,
            mean_gross_exposure=float(f.open_gross_exposure.mean()),
            max_gross_exposure=float(f.open_gross_exposure.max()))
    CHECKS.append('AlphaPROBE: gross NAV recurrence, zero costs and open net=0 for all three universes; execution restrictions/stale holdings can move gross exposure slightly above/below target, separately reported.')


def hermes():
    root = ROOT/'hermes-reproduction/results'
    old = {}
    for p in sorted(root.glob('*/test_predictions.npz')):
        if not p.parent.name.startswith(('sp500-', 'companion-')):
            continue
        z = np.load(source(p), allow_pickle=False)
        w, r = z['portfolio_weights'], z['portfolio_return']
        np.testing.assert_allclose(w.sum(1), 0, atol=1e-12)
        np.testing.assert_allclose(abs(w).sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(np.sum(w*z['pnl_target'], axis=1), r, atol=1e-9)
        old[p.parent.name] = (z['dates'].copy(), r.copy(), w.copy())
        add('HERMES', p.parent.name, z['dates'], r,
            gross_target=1., max_net_exposure=float(abs(w.sum(1)).max()))
    for reduction in ['', '-meanrank']:
        parts = [old[f'sp500-full-seed{s}{reduction}'] for s in range(3)]
        assert all(np.array_equal(parts[0][0], a[0]) for a in parts)
        netted = np.mean([a[2] for a in parts], axis=0)
        add('HERMES', f'sp500-full-3seed{reduction}', parts[0][0], np.mean([a[1] for a in parts], axis=0),
            seeds=3, mean_gross_exposure=float(abs(netted).sum(1).mean()))
    rolling = {}
    for p in sorted((root/'recent').glob('*/test_predictions.npz')):
        z = np.load(source(p), allow_pickle=False)
        w, r = z['portfolio_weights'], z['portfolio_return']
        np.testing.assert_allclose(w.sum(1), 0, atol=1e-12)
        np.testing.assert_allclose(abs(w).sum(1), 1, atol=1e-12)
        np.testing.assert_allclose(np.sum(w*z['pnl_target'], axis=1), r, atol=1e-9)
        if p.parent.name.startswith('frozen'):
            add('HERMES', 'frozen-2022-2025', z['dates'], r, gross_target=1.)
        else:
            rolling.setdefault(p.parent.name[-1], []).append((z['dates'].copy(), r.copy(), w.copy()))
    combined = []
    for seed, parts in sorted(rolling.items()):
        parts.sort(key=lambda a: a[0][0])
        dates = np.concatenate([a[0] for a in parts])
        r = np.concatenate([a[1] for a in parts])
        add('HERMES', f'rolling-2023-2025-seed{seed}', dates, r, diagnostic=True, gross_target=1.)
        combined.append((dates, r, np.concatenate([a[2] for a in parts])))
    assert all(np.array_equal(combined[0][0], a[0]) for a in combined)
    netted = np.mean([a[2] for a in combined], axis=0)
    add('HERMES', 'rolling-2023-2025-3seed', combined[0][0], np.mean([a[1] for a in combined], axis=0),
        seeds=3, mean_gross_exposure=float(abs(netted).sum(1).mean()), gross_target=1.)
    CHECKS.append('HERMES: all saved neutral weight sums/gross exposures verified; original weighted labels reconcile gross returns; recent seeds aligned by calendar date.')


def limt():
    root = ROOT/'limt-reproduction'
    sys.path.insert(0, str(root))
    from limt.data import Panel
    import limt.neutral as engine
    neutral_backtest = engine.neutral_backtest
    source(root/'limt/neutral.py')
    source(root/'limt/data.py')
    for u in ['csi300', 'csi500']:
        for name in ['market.npy', 'dates.npy', 'symbols.npy', 'manifest.json']:
            source(root/'data'/u/name)
        panel = Panel(root/'data'/u)
        series = []
        for seed in range(3):
            path = root/'results/main'/u/f'seed{seed}/test_predictions.parquet'
            if not path.exists():
                WARNINGS.append(f'LiMT {u} seed{seed}: no completed test predictions; not retrained.')
                continue
            pred = pd.read_parquet(source(path))
            positions = []
            projection = engine.neutral_projection
            def record_projection(*args, **kwargs):
                value = projection(*args, **kwargs)
                positions.append(value.copy())
                return value
            engine.neutral_projection = record_projection
            try:
                daily, metrics = neutral_backtest(panel, pred, buy_fee=0., sell_fee=0.)
            finally:
                engine.neutral_projection = projection
            daily.to_csv(OUT/f'limt_{u}_seed{seed}_zero_ledger.csv', index=False)
            # Existing engine holds fixed dollar risk budgets, independent of fees.
            # Rebuild the gross NAV from the previous charged ledger independently.
            charged = root/f'results/main/market_neutral/{u}_seed{seed}_buy5_sell5.csv'
            if charged.exists():
                old = csv(charged)
                np.testing.assert_array_equal(old.date, daily.date)
                nav = 2e9*np.r_[1., np.cumprod(1+old.return_net.to_numpy())[:-1]]
                gross_pnl = old.return_net.to_numpy()*nav + old.fees.to_numpy()
                gross_nav = 2e9 + np.r_[0., np.cumsum(gross_pnl)[:-1]]
                np.testing.assert_allclose(gross_pnl/gross_nav, daily.return_net, atol=1e-12)
            np.testing.assert_allclose(daily.long_notional, daily.short_notional, atol=.3)
            assert (daily.fees == 0).all()
            exposure = (daily.long_notional + daily.short_notional).to_numpy() / (
                2e9*np.r_[1., np.cumprod(1+daily.return_net.to_numpy())[:-1]])
            # Verify portfolio payoffs directly against the raw adjusted closes,
            # with the engine's documented forward mark carry for missing closes.
            lookup = {d:i for i,d in enumerate(panel.dates)}
            ix = np.array([lookup[d] for d in daily.date])
            marks = pd.DataFrame(np.asarray(panel.market[:, :, 0]).T).ffill().to_numpy()
            realized = np.nan_to_num(marks[ix]/marks[ix-1]-1, nan=0.)
            nav = 2e9*np.r_[1., np.cumprod(1+daily.return_net.to_numpy())[:-1]]
            weights = np.asarray(positions)/nav[:, None]
            np.testing.assert_allclose(np.sum(weights*realized, axis=1), daily.return_net, atol=1e-12)
            np.testing.assert_array_equal(panel.dates[ix-1], daily.execution_date)
            assert (daily.signal_date < daily.execution_date).all()
            assert (daily.execution_date < daily.date).all()
            np.savez_compressed(OUT/f'limt_{u}_seed{seed}_zero_weights.npz',
                weights=weights, dates=daily.date.to_numpy(dtype=str), symbols=panel.symbols)
            add('LiMT', f'{u}-seed{seed}', daily.date, daily.return_net, diagnostic=True,
                mean_gross_exposure=float(exposure.mean()), max_net_exposure=float(daily.net_exposure.abs().max()))
            series.append((daily.date.to_numpy(), daily.return_net.to_numpy(), exposure, weights))
        assert all(np.array_equal(series[0][0], s[0]) for s in series)
        add('LiMT', f'{u}-{len(series)}seed', series[0][0], np.mean([s[1] for s in series], axis=0),
            seeds=len(series), mean_gross_exposure=float(abs(np.mean([s[3] for s in series], axis=0)).sum(1).mean()),
            mean_sleeve_gross_exposure=float(np.mean([s[2].mean() for s in series])))
    CHECKS.append('LiMT: zero-fee engine rerun with existing predictions; prior fee ledgers independently converted to gross P&L/NAV where available; recorded holdings independently dotted against raw adjusted close returns; information/execution/return dates and equal executed legs checked.')


def arbnn():
    f = csv(ROOT/'arboreal-reproduction/finance/results/returns.csv')
    f = f[(f.scenario == 'gross') & f.method.str.startswith('arbnn')]
    for (method, fraction), g in f.groupby(['method', 'fraction'], sort=True):
        np.testing.assert_allclose(g.net_exposure, 0, atol=1e-10)
        np.testing.assert_allclose(g.gross_exposure, 2, atol=1e-10)
        assert (g.transaction_cost == 0).all() and (g.borrow_cost == 0).all()
        add('ArbNN', f'{method}-q{fraction:g}', g.date, .5*g.gross_return,
            frequency='monthly', diagnostic=method != 'arbnn', gross_target=1.)
    CHECKS.append('ArbNN: original zero-cost 200%-gross stream normalized to 100% gross; zero-net exposures and costs verified.')


def prophets():
    f = csv(ROOT/'prophets-profit-reproduction/equity/results/monthly_returns.csv')
    f = f[(f.cost_bps == 0) & (f.method != 'momentum_12_2')]
    for (method, neutral), g in f.groupby(['method', 'neutrality'], sort=True):
        np.testing.assert_allclose(g.net_exposure, 0, atol=1e-10)
        np.testing.assert_allclose(g.gross_exposure, 1, atol=1e-10)
        # cost_bps=0 still contains an assumed borrow cost: use gross_return.
        np.testing.assert_allclose(g.net_return+g.borrow_cost, g.gross_return, atol=1e-12)
        add('Prophets Profit', f'{method}-{neutral}', g.date, g.gross_return,
            frequency='monthly', gross_target=1.)
    CHECKS.append('Prophets Profit: zero commission rows still charge borrowing; actual gross_return used and borrowing identity reconciled for every sizing/neutrality variant.')


def sparse_inputs(market, ids):
    root = ROOT/'sparse-tangent-reproduction/reproduction'
    if market == 'us':
        p = read_json(root/'protocol.json')
        raw = pd.read_parquet(source(p['source']))
        raw['dlycaldt'] = pd.to_datetime(raw.dlycaldt)
        dates = pd.DatetimeIndex(sorted(raw.dlycaldt.unique()))
        r = raw.pivot(index='dlycaldt', columns='permno', values='dlyret').reindex(index=dates, columns=ids).astype(float)
        members = raw.pivot(index='dlycaldt', columns='permno', values='membership_active').reindex(index=dates, columns=ids).fillna(False).to_numpy(bool)
        trade = np.isfinite(r.to_numpy())  # only used for verification, not US formation eligibility
    else:
        folder = root/'global_data'/market
        p = read_json(root/'global_results'/market/'protocol.json')
        raw = csv(folder/'daily.csv.gz', dtype={'gvkey':str, 'iid':str, 'prcstd':str}, parse_dates=['datadate'])
        raw['id'] = raw.gvkey+'_'+raw.iid
        n = len(ids)
        if market == 'kospi200':
            counts = raw.assign(open_quote=raw.prcstd=='10').groupby('datadate').open_quote.sum()
            dates = pd.DatetimeIndex(counts.index[counts >= n/2]).sort_values()
        else:
            dates = pd.DatetimeIndex(sorted(raw.loc[raw.prcstd=='10', 'datadate'].unique()))
        level = raw.assign(level=raw.prccd/raw.ajexdi*raw.trfd).pivot(index='datadate', columns='id', values='level').reindex(index=dates, columns=ids)
        r = level.pct_change(fill_method=None)
        trade = (raw.pivot(index='datadate', columns='id', values='prcstd').reindex(index=dates, columns=ids) == '10').to_numpy()
        h = csv(folder/'membership.csv.gz', dtype={'gvkey':str, 'iid':str}, parse_dates=['from','thru'])
        h['id'] = h.gvkey+'_'+h.iid
        members = np.zeros(r.shape, bool)
        for _, row in h.iterrows():
            if row['id'] in ids:
                members[(dates>=row['from']) & ((dates<=row['thru']) if pd.notna(row['thru']) else True), ids.index(row['id'])] = True
        if p.get('membership_mode') == 'fixed_initial_cohort':
            members[:] = True
            for event in p['corporate_actions']:
                j = ids.index(event['id'])
                members[dates>=pd.Timestamp(event['exit_close']), j] = False
                trade[dates>=pd.Timestamp(event['exclude_from']), j] = False
    values = r.to_numpy(dtype=float)
    b = np.zeros(r.shape)
    for i in range(101, len(dates)):
        eligible = members[i-1] & (np.isfinite(values[i-101:i-1]).sum(0) >= 95)
        if market != 'us':
            eligible &= trade[i-1]
        if eligible.any():
            b[i, eligible] = 1/eligible.sum()
    return dates.strftime('%Y-%m-%d').to_numpy(), values, b, trade


def sparse():
    root = ROOT/'sparse-tangent-reproduction/reproduction'
    for market in ['us', 'euro50', 'nikkei225', 'kospi200']:
        folder = root/('clean_results' if market == 'us' else 'global_results/'+market)
        metrics = csv(folder/'test_metrics.csv')
        first = next((folder/'test').glob('dfl*/weights.npz'))
        with np.load(source(first)) as z:
            ids = z['permnos'].tolist()
        all_dates, returns, benchmark, tradable = sparse_inputs(market, ids)
        lookup = {d:i for i,d in enumerate(all_dates)}
        groups = {}
        for rec in metrics.itertuples():
            if rec.method not in ['dfl', 'pfl']:
                continue
            name = f'{rec.method}_k{rec.k}_seed{rec.seed}'
            dest = folder/'test'/name
            with np.load(source(dest/'weights.npz')) as z:
                w, dates = z['weights'], z['dates']
                assert z['permnos'].tolist() == ids
            ix = np.array([lookup[d] for d in dates])
            r, b = returns[ix], benchmark[ix]
            original = csv(dest/'daily.csv')
            np.testing.assert_array_equal(original.date, dates)
            # Saved weights are actual held weights (including skipped rebalances).
            reconstructed = np.sum(w*np.nan_to_num(r), axis=1) + (1-w.sum(1))*original.rf.to_numpy()
            np.testing.assert_allclose(reconstructed, original.gross_return, atol=1e-9)
            assert not np.any((abs(w)>1e-9) & ~np.isfinite(r))
            u = w - w.sum(1)[:, None]*b
            scale = abs(u).sum(1)
            assert scale.min() > 1e-8
            u /= scale[:, None]
            np.testing.assert_allclose(u.sum(1), 0, atol=1e-12)
            np.testing.assert_allclose(np.maximum(u, 0).sum(1), .5, atol=1e-12)
            np.testing.assert_allclose(np.maximum(-u, 0).sum(1), .5, atol=1e-12)
            assert not np.any((abs(u)>1e-9) & ~np.isfinite(r))
            # Retain original status-based execution restrictions as a disclosed
            # diagnostic; the adaptation assumes ideal daily close rebalancing.
            y = np.sum(u*np.nan_to_num(r), axis=1)
            groups.setdefault((rec.method, rec.k), []).append(y)
            np.savez_compressed(OUT/f'sparse_{market}_{name}_zero_weights.npz', dates=dates, ids=ids, weights=u)
            add('Sparse Tangent', f'{market}-{name}', dates, y, diagnostic=True, gross_target=1.)
        for (method, k), parts in groups.items():
            add('Sparse Tangent', f'{market}-{method}-k{k}-5seed', dates, np.mean(parts, axis=0), seeds=5, gross_target=1.)
    CHECKS.append('Sparse Tangent: all 120 saved DFL/PFL gross ledgers reconciled to source returns before adaptation; netted hedge weights have +50%/-50% legs and all held payoffs are finite.')


def scope():
    history = Path('/home/jkp/.codex/history.jsonl')
    ids = ['01a1ab67-e49c-7c72-ba1c-e418e87ba102', '01a1ab81-81fb-7cb2-a6ad-d908429bec37',
           '01a1aeda-1aef-71f2-9129-dfab47a7b2ee', '01a1af08-f74b-7143-9a27-6a9ed53be550',
           '01a1af09-bdeb-7982-b99e-99c026c6cb8f', '01a1abc0-c834-7032-9b5c-2ded7247d517',
           '01a1aad6-a045-7dc0-878e-ab4698e4e73c', '01a1b0f0-18d8-7bc2-bdfc-8ebf66a6f4c6']
    rows = [json.loads(l) for l in history.read_text().splitlines()]
    selected = [r for r in rows if r['session_id'] in ids and
                any(s in r['text'].lower() for s in ['neutral', '5bp', '5 bp', 'reproduce'])]
    (OUT/'session-scope.json').write_text(json.dumps(selected, indent=2, ensure_ascii=False)+'\n')
    reviews = ROOT/'paper-reproduction-audit/reviews'
    counts = {}
    for p in reviews.glob('*.json'):
        d = json.loads(p.read_text()).get('market_neutral_5bp', {})
        status = d.get('status', 'missing')
        counts[status] = counts.get(status, 0)+1
    (OUT/'programme-coverage.json').write_text(json.dumps(dict(
        coverage=counts, note='Registry status only; the programme is not 437 completed negative trading tests. Canonical reviews preserved.'), indent=2)+'\n')


def main():
    scope()
    for name, fn in [('alphaprobe', alphaprobe), ('hermes', hermes), ('limt', limt),
                     ('arbnn', arbnn), ('prophets', prophets), ('sparse', sparse)]:
        print('START', name, flush=True)
        fn()
    result = pd.DataFrame(ROWS)
    aggregates = result.index[~result.diagnostic]
    result.loc[aggregates, 'p_holm_audit'] = multipletests(result.loc[aggregates, 'p'], method='holm')[1]
    for _, group in result.loc[aggregates].groupby('project'):
        result.loc[group.index, 'p_holm_project'] = multipletests(group.p, method='holm')[1]
    result['positive_nominal'] = (result.annual_mean > 0) & (result.ci_low > 0)
    result['positive_audit_adjusted'] = (result.annual_mean > 0) & (result.p_holm_audit < .05)
    result.to_csv(OUT/'summary.csv', index=False)
    pd.concat(SERIES, ignore_index=True).to_csv(OUT/'return-series.csv', index=False)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    scipy=scipy.__version__, statsmodels=statsmodels.__version__,
                    source_sha256=INPUTS, checks=CHECKS, warnings=WARNINGS,
                    rows=len(result), aggregate_tests=len(aggregates),
                    all_samples_retrospective=True, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(result.loc[aggregates, ['project','case','annual_mean','sharpe','p','p_holm_audit']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
