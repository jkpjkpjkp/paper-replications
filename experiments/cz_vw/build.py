"""Build compact monthly CZ features with explicit lagged NYSE20 screening."""
import json
import warnings
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from common import HERE, ROOT, DATA, BASE, CZ, SIGNALS, settings, sha, save_json, now


def prior_month(dates):
    month = pd.DatetimeIndex(dates).to_period('M') - 1
    return np.array(month.year * 100 + month.month, dtype=np.int32)


def common_shares(f):
    return (f.securitytype.eq('EQTY') & f.securitysubtype.eq('COM') & f.sharetype.eq('NS') &
            f.issuertype.eq('CORP') & f.usincflg.eq('Y') & f.primaryexch.isin(['N', 'A', 'Q']) &
            f.conditionaltype.eq('RW') & f.tradingstatusflg.eq('A') & f.mthdelflg.eq('N') & f.mthcap.gt(0))


def rank_signals(f):
    count = f.notna().sum()
    value = 2 * (f.rank(method='average') - 1) / (count - 1).replace(0, np.nan) - 1
    return value.fillna(0).to_numpy(np.float32)


def build():
    DATA.mkdir(exist_ok=True)
    if (DATA / 'manifest.json').exists():
        m = json.loads((DATA / 'manifest.json').read_text())
        assert m['protocol_sha256'] == sha(HERE / 'protocol.json')
        for name, digest in m['output_sha256'].items():
            assert sha(DATA / name) == digest, name
        print('Prepared CZ data verified', m['shape'], flush=True)
        return
    receipt = json.loads((SIGNALS.parent / 'download_receipt.json').read_text())
    assert sha(SIGNALS) == receipt['sha256']
    base_manifest = json.loads((BASE / 'manifest.json').read_text())
    shared = ['dates', 'symbols', 'close', 'priors', 'valid_priors', 'tradable', 'totalret', 'terminal_events']
    for name in shared + ['member']:
        assert sha(BASE / f'{name}.npy') == base_manifest['output_hashes'][f'{name}.npy']
    dates, symbols = [np.load(BASE / f'{name}.npy') for name in ['dates', 'symbols']]
    permnos = np.array([int(s[1:]) for s in symbols])
    months = np.unique(prior_month(dates))
    columns = ['permno', 'yyyymm', 'mthcap', 'primaryexch', 'securitytype', 'securitysubtype',
        'sharetype', 'issuertype', 'usincflg', 'conditionaltype', 'tradingstatusflg', 'mthdelflg']
    source = ds.dataset(ROOT.parent / 'crsp-monthly/parquet', format='parquet')
    monthly = source.to_table(columns=columns, filter=(ds.field('yyyymm') >= int(months[0])) &
        (ds.field('yyyymm') <= int(months[-1]))).to_pandas()
    monthly['mthcap'] = pd.to_numeric(monthly.mthcap).astype(float)
    assert not monthly.duplicated(['permno', 'yyyymm']).any()
    population = monthly.loc[common_shares(monthly)].copy()
    thresholds = population.loc[population.primaryexch.eq('N')].groupby('yyyymm').mthcap.quantile(.2)
    assert thresholds.reindex(months).notna().all()
    population['nyse20'] = population.yyyymm.map(thresholds)
    population['nonmicro'] = population.mthcap.ge(population.nyse20)
    thresholds.rename('nyse20').to_csv(DATA / 'nyse20_breakpoints.csv')
    key = pd.MultiIndex.from_product([permnos, months], names=['permno', 'yyyymm'])
    cap = monthly.set_index(['permno', 'yyyymm']).reindex(key).mthcap.to_numpy(float).reshape(len(symbols), -1)
    screen = population.set_index(['permno', 'yyyymm']).reindex(key).nonmicro.fillna(False).to_numpy(bool).reshape(cap.shape)
    source_names = [n for n in pq.read_schema(SIGNALS).names if n not in ['permno', 'yyyymm']]
    docs = pd.read_csv(CZ / 'SignalDoc.csv')
    assert len(source_names) == 209 and set(source_names) <= set(docs.loc[docs['Cat.Signal'].eq('Predictor'), 'Acronym'])
    features = np.zeros((*cap.shape, len(source_names)), np.float32)
    observed = np.zeros(cap.shape, bool)
    coverage = []
    for year in sorted(set(months // 100)):
        f = pd.read_parquet(SIGNALS, filters=[('yyyymm', '>=', int(year * 100 + 1)),
                                             ('yyyymm', '<=', int(year * 100 + 12))])
        assert not f.duplicated(['permno', 'yyyymm']).any()
        eligible = population.loc[population.nonmicro & (population.yyyymm // 100 == year), ['permno', 'yyyymm']]
        f = f.merge(eligible, on=['permno', 'yyyymm'], how='inner', validate='one_to_one')
        f[source_names] = f[source_names].replace([np.inf, -np.inf], np.nan)
        for month, group in f.groupby('yyyymm', sort=True):
            j = int(np.searchsorted(months, month))
            if j >= len(months) or months[j] != month:
                continue
            rank = pd.DataFrame(rank_signals(group[source_names]), index=group.permno, columns=source_names)
            features[:, j] = rank.reindex(permnos).fillna(0).to_numpy(np.float32)
            observed[:, j] = np.isin(permnos, group.permno)
            coverage.append(dict(yyyymm=int(month), rank_population=len(group),
                target_records=int(observed[:, j].sum()), observed_fraction=float(group[source_names].notna().to_numpy().mean())))
        print('CZ month alignment', year, flush=True)
    ix = np.searchsorted(months, prior_month(dates))
    original = np.load(BASE / 'member.npy')
    member = original & screen[:, ix] & observed[:, ix]
    p = settings()
    train_days = (dates >= p['train'][0]) & (dates <= p['train'][1])
    train_mask = np.zeros(cap.shape, bool)
    # Union over all training sessions, so entrants within a month are included.
    for t in np.flatnonzero(train_days):
        train_mask[:, ix[t]] |= member[:, t]
    values = features[train_mask]
    keep = np.ptp(values, axis=0) > 1e-12
    assert keep.sum() == 202, f'Training retains {keep.sum()}, expected 202; do not arbitrarily drop columns'
    for name in shared:
        (DATA / f'{name}.npy').symlink_to(BASE / f'{name}.npy')
    # New capitalization/screen inputs and compact monthly features are owned here.
    for name, array in dict(monthly_features=features[:, :, keep], month_index=ix, source_months=months,
            member=member, lagged_cap=cap[:, ix], monthly_cap=cap, monthly_screen=screen,
            monthly_observed=observed, training_feature_mask=train_mask).items():
        np.save(DATA / f'{name}.npy', array)
    # Exact CRSP price-only return is needed to drift positions through dividends/splits.
    price_return = np.full(original.shape, np.nan, np.float32)
    date_map = pd.Index(dates)
    id_map = pd.Index(permnos)
    daily_sources = {}
    for year in range(int(dates[0][:4]), int(dates[-1][:4]) + 1):
        path = ROOT.parent / f'crsp-daily/data/crsp_a_stock/dsf_v2/{year}.parquet'
        receipt = json.loads(path.with_suffix('.done.json').read_text())
        assert sha(path) == receipt['sha256']
        daily_sources[str(path)] = receipt['sha256']
        f = pd.read_parquet(path, columns=['permno', 'dlycaldt', 'dlyretx'], filters=[('permno', 'in', permnos.tolist())])
        a, b = id_map.get_indexer(f.permno), date_map.get_indexer(f.dlycaldt.astype(str))
        ok = (a >= 0) & (b >= 0)
        price_return[a[ok], b[ok]] = pd.to_numeric(f.dlyretx).to_numpy(np.float32)[ok]
    np.save(DATA / 'price_return.npy', price_return)
    pd.DataFrame(coverage).to_csv(DATA / 'coverage.csv', index=False)
    audit = []
    lag_cap = cap[:, ix]
    for split in ['train', 'valid', 'test']:
        days = (dates >= p[split][0]) & (dates <= p[split][1])
        before = original[:, days]
        known = np.isfinite(lag_cap[:, days]) & (lag_cap[:, days] > 0)
        below = known & (lag_cap[:, days] < thresholds.reindex(months).to_numpy()[ix[days]][None, :])
        audit.append(dict(split=split, original_stock_days=int(before.sum()), microcap_removed=int((before & below).sum()),
            unknown_lag_cap_removed=int((before & ~known).sum()),
            other_common_share_screen_removed=int((before & known & ~below & ~screen[:, ix[days]]).sum()),
            missing_cz_removed=int((before & screen[:, ix[days]] & ~observed[:, ix[days]]).sum()),
            retained_stock_days=int(member[:, days].sum()), min_stocks=int(member[:, days].sum(0).min()),
            max_stocks=int(member[:, days].sum(0).max())))
    pd.DataFrame(audit).to_csv(DATA / 'universe_audit.csv', index=False)
    source_paths = [SIGNALS, CZ / 'SignalDoc.csv', BASE / 'manifest.json', ROOT.parent / 'crsp-monthly/manifest.json',
        ROOT.parent / 'crsp-monthly/validation.json']
    monthly_manifest = json.loads((ROOT.parent / 'crsp-monthly/manifest.json').read_text())
    # Preserve the immutable transfer manifest as the monthly-source receipt.
    outputs = list(DATA.glob('*.npy')) + list(DATA.glob('*.csv'))
    manifest = dict(created_utc=now(), protocol_sha256=sha(HERE / 'protocol.json'), builder_sha256=sha(__file__),
        feature_names=np.array(source_names)[keep].tolist(), dropped_training_constant=np.array(source_names)[~keep].tolist(),
        omitted_from_public_source=['Price', 'Size', 'STreversal'], shape=list(features[:, :, keep].shape),
        source_sha256={str(f): sha(f) for f in source_paths}, daily_source_sha256=daily_sources,
        output_sha256={f.name: sha(f) for f in outputs}, universe_audit=audit,
        feature_selection='Training-period eligible S&P500 observations only; constant after missing-neutral monthly ranking')
    save_json(DATA / 'manifest.json', manifest)
    print('Data complete', manifest['shape'], audit, flush=True)


if __name__ == '__main__':
    build()
