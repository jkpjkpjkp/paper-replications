"""Independent forecast replay, raw data checks, ledger and HAC recomputation."""
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from scipy import stats
from common import HERE, ROOT, DATA, SIGNALS, FACTORS, settings, check_lock, sha, save_json, now
from panel import Panel
from runtime import temporal_model, rankic, ranking_metrics, encode, from_codes
from train import seed_all


def independent_hac(y, factors, lag):
    x = np.column_stack([np.ones(len(y)), factors])
    n, k = x.shape
    inverse = np.linalg.inv(x.T @ x)
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    scores = x * (y - x @ beta)[:, None]
    meat = scores.T @ scores
    for j in range(1, lag + 1):
        cross = scores[j:].T @ scores[:-j]
        meat += (1 - j / (lag + 1)) * (cross + cross.T)
    cov = inverse @ meat @ inverse * (n / (n - k))
    se = np.sqrt(cov[0, 0])
    t = beta[0] / se
    width = stats.t.ppf(.975, n - k) * se
    return np.array([beta[0] * 252, t, 2 * stats.t.sf(abs(t), n - k),
        (beta[0] - width) * 252, (beta[0] + width) * 252])


def raw_cz_check(panel):
    columns = ['permno', 'yyyymm', 'mthcap', 'primaryexch', 'securitytype', 'securitysubtype',
        'sharetype', 'issuertype', 'usincflg', 'conditionaltype', 'tradingstatusflg', 'mthdelflg']
    results = []
    for month in [202112, 202306, 202411]:
        f = ds.dataset(ROOT.parent / 'crsp-monthly/parquet', format='parquet').to_table(
            columns=columns, filter=ds.field('yyyymm') == month).to_pandas()
        cap = pd.to_numeric(f.mthcap).astype(float)
        common = ((f.securitytype == 'EQTY') & (f.securitysubtype == 'COM') & (f.sharetype == 'NS') &
            (f.issuertype == 'CORP') & (f.usincflg == 'Y') & f.primaryexch.isin(['N', 'A', 'Q']) &
            (f.conditionaltype == 'RW') & (f.tradingstatusflg == 'A') & (f.mthdelflg == 'N') & (cap > 0))
        threshold = np.quantile(cap[common & (f.primaryexch == 'N')], .2)
        saved = pd.read_csv(DATA / 'nyse20_breakpoints.csv').set_index('yyyymm').loc[month, 'nyse20']
        np.testing.assert_allclose(threshold, saved, rtol=1e-12)
        raw = pd.read_parquet(SIGNALS, filters=[('yyyymm', '==', month)])
        raw = raw.loc[raw.permno.isin(f.loc[common & (cap >= threshold), 'permno'])].set_index('permno')
        names = panel.manifest['feature_names']
        values = raw[names].replace([np.inf, -np.inf], np.nan)
        count = values.notna().sum()
        expected = (2 * (values.rank(method='average') - 1) / (count - 1).replace(0, np.nan) - 1).fillna(0)
        expected = expected.reindex([int(s[1:]) for s in panel.symbols]).fillna(0).to_numpy(np.float32)
        j = int(np.flatnonzero(panel.source_months == month)[0])
        np.testing.assert_array_equal(expected, panel.monthly_features[:, j])
        results.append(dict(source_month=month, rank_population=len(raw), nyse20=float(threshold)))
    return results


def run():
    check_lock()
    panel = Panel()
    out = HERE / 'analysis'
    receipt = json.loads((out / 'analysis_receipt.json').read_text())
    for name, digest in receipt['outputs_sha256'].items():
        assert sha(out / name) == digest
    assert sha(HERE / 'REPORT.txt') == receipt['report_sha256']
    for name, digest in panel.manifest['output_sha256'].items():
        assert sha(DATA / name) == digest, name
    for name, digest in panel.manifest['source_sha256'].items():
        assert sha(name) == digest, name
    folder = HERE / 'results/sp500/stage2/seed0'
    metrics = json.loads((folder / 'metrics.json').read_text())
    for name, key in [('best.pt', 'checkpoint_sha256'), ('predictions.parquet', 'predictions_sha256')]:
        assert sha(folder / name) == metrics[key]
    assert sha(HERE / 'results/sp500/stage1/best.pt') == metrics['stage1_sha256']
    pred = pd.read_parquet(folder / 'predictions.parquet')
    assert not pred.duplicated(['date', 'symbol']).any() and np.isfinite(pred.prediction).all()
    by_date = dict(tuple(pred.groupby('date', sort=True)))
    assert list(by_date) == list(panel.dates[panel.plans['test']])
    daily = []
    for t in panel.plans['test']:
        f = by_date[str(panel.dates[t])]
        ids = panel.ids(t, 'test')
        np.testing.assert_array_equal(f.symbol, panel.symbols[ids])
        target = panel.close[ids, t + 5] / panel.close[ids, t + 1] - 1
        target[~np.isfinite(target)] = np.nan
        np.testing.assert_allclose(f.label, target, rtol=0, atol=0, equal_nan=True)
        daily.append(rankic(f.prediction.to_numpy(), target))
    recomputed = ranking_metrics(daily)
    for name in ['RankIC', 'RankICIR']:
        np.testing.assert_allclose(metrics[name], recomputed[name], atol=1e-14)
    np.testing.assert_allclose(pd.read_csv(folder / 'daily_rankic.csv').RankIC, daily, atol=1e-14, equal_nan=True)
    torch.set_num_threads(2)
    seed_all(0)
    cfg = json.loads((folder / 'config.json').read_text())
    model = temporal_model(cfg, len(panel.plans['train']) * settings()['training']['epochs']).to('cuda')
    state = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    replay = []
    for t in panel.plans['test'][[0, len(panel.plans['test']) // 2, -1]]:
        x, p, y, ids = panel.batch(t, 'test', 'cuda')
        with torch.no_grad():
            native = model(x, p)[0]
            z, _ = encode(model, x)
            cached, _ = from_codes(model, x, p, z)
        torch.testing.assert_close(native, cached, rtol=0, atol=0)
        np.testing.assert_allclose(native.cpu().numpy(), by_date[str(panel.dates[t])].prediction, rtol=1e-5, atol=2e-6)
        replay.append(str(panel.dates[t]))
    for stage in [1, 2]:
        path = HERE / f'results/sp500/stage{stage}'
        if stage == 2:
            path /= 'seed0'
        history = json.loads((path / 'history.json').read_text())
        best = torch.load(path / 'best.pt', map_location='cpu', weights_only=False)
        assert best['epoch'] == int(np.argmin([h['valid_loss'] for h in history]))
        assert len(history) <= 50
    # Build raw test-period total/price-return arrays independently of Panel.
    raw_returns = np.full(panel.totalret.shape, np.nan, np.float32)
    raw_price = np.full(panel.totalret.shape, np.nan, np.float32)
    ids = pd.Index([int(s[1:]) for s in panel.symbols])
    dates = pd.Index(panel.dates)
    for year in [2022, 2023, 2024]:
        path = ROOT.parent / f'crsp-daily/data/crsp_a_stock/dsf_v2/{year}.parquet'
        assert sha(path) == panel.manifest['daily_source_sha256'][str(path)]
        f = pd.read_parquet(path, columns=['permno', 'dlycaldt', 'dlyret', 'dlyretx'], filters=[('permno', 'in', ids.tolist())])
        a, b = ids.get_indexer(f.permno), dates.get_indexer(f.dlycaldt.astype(str))
        assert (a >= 0).all() and (b >= 0).all()
        raw_returns[a, b] = pd.to_numeric(f.dlyret).to_numpy(np.float32)
        raw_price[a, b] = pd.to_numeric(f.dlyretx).to_numpy(np.float32)
    mask = (panel.dates >= '2022-01-01') & (panel.dates <= '2024-12-31')
    np.testing.assert_array_equal(raw_returns[:, mask], panel.totalret[:, mask])
    np.testing.assert_array_equal(raw_price[:, mask], panel.price_return[:, mask])
    factors = pd.read_parquet(FACTORS)
    factors['date'] = factors.date.astype(str)
    alpha = pd.read_csv(out / 'alpha.csv')
    ledger_checks = []
    for bp in [0, 10]:
        ledger = pd.read_csv(out / f'cost{bp}_ledger.csv')
        archive = np.load(out / f'cost{bp}_weights.npz')
        weights = archive['weights']
        at = dates.get_indexer(ledger.date)
        payoffs = np.asarray(raw_returns[:, at], float).T
        replay_pnl = (weights * np.where(np.isfinite(payoffs), payoffs, 0)).sum(1)
        np.testing.assert_allclose(replay_pnl, ledger.return_gross, atol=1e-12)
        np.testing.assert_allclose(ledger.return_net, ledger.return_gross - ledger.fees / ledger.start_nav, atol=1e-12)
        np.testing.assert_allclose(ledger.fees, (ledger.traded_dollars + ledger.final_liquidation_dollars) * bp / 1e4, atol=1e-12)
        # Independent trade replay from saved weights and raw price returns.
        dollars = weights * ledger.start_nav.to_numpy()[:, None]
        previous = np.zeros(len(ids))
        for j, t in enumerate(at):
            traded = abs(dollars[j] - previous).sum()
            np.testing.assert_allclose(traded, ledger.traded_dollars.iloc[j], rtol=1e-9, atol=1e-10)
            price = np.asarray(raw_price[:, t], float)
            previous = dollars[j] * (1 + np.where(np.isfinite(price), price, 0))
            previous[panel.terminal_events[:, t]] = 0
        joined = ledger.merge(factors, on='date', validate='one_to_one')
        assert len(joined) == len(ledger)
        estimate = independent_hac(joined.return_net.to_numpy(), joined[settings()['inference']['factors']].to_numpy(), 20)
        saved = alpha.loc[alpha.cost_bp == bp, ['annual_alpha', 'hac_t', 'p', 'ci_low', 'ci_high']].iloc[0].to_numpy(float)
        np.testing.assert_allclose(estimate, saved, rtol=1e-8, atol=1e-11)
        ledger_checks.append(dict(cost_bp=bp, days=len(ledger), max_raw_return_replay_error=float(abs(replay_pnl - ledger.return_gross).max()),
            annual_alpha=float(estimate[0]), final_unliquidated_dollars=float(ledger.final_unliquidated_dollars.iloc[-1])))
    result = dict(passed=True, updated_utc=now(), forecasts=len(pred), forecast_dates=len(daily),
        features=202, native_checkpoint_replay_dates=replay, validation_only_epoch_selection=True,
        raw_cz_checks=raw_cz_check(panel), ledger_checks=ledger_checks,
        independent_hac='NumPy OLS and Bartlett sandwich plus SciPy t-reference match statsmodels',
        exact_published_results_reproduced=False)
    save_json(out / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    run()
