"""Independent, same-universe review of every declared completed experiment."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gap-resolution-20261113'


def load(path):
    return json.loads(path.read_text())


def main():
    chain = load(OUT / 'completion-chain.json')
    assert chain['stage'] == 'experiments_complete_pending_review'
    baseline_path = ROOT / 'results/doubleadapt-script-matched'
    baseline = pd.read_pickle(baseline_path / 'predictions-raw-labels.pkl')
    ledger0 = pd.read_pickle(next((baseline_path / 'mlruns').rglob('report_normal_1day.pkl')))
    paper = load(baseline_path / 'protocol.json')['paper_target']
    paths = [ROOT / 'results/doubleadapt-original', baseline_path,
             OUT / 'validation-rates-script-matched', OUT / 'available-inputs',
             OUT / 'validation-rates-available-inputs', OUT / 'validation-rates-available-inputs-coherent']
    if not chain['step5_unavailable']:
        paths += [OUT / 'available-inputs-step5', OUT / 'validation-rates-available-inputs-step5']
    da = {}
    for path in paths:
        summary = load(path / 'summary.json')
        pred = pd.read_pickle(path / 'predictions-raw-labels.pkl')
        assert pred.index.equals(baseline.index), path
        assert np.array_equal(pred.raw_label.values, baseline.raw_label.values), path
        assert pred.prediction.notna().all() and np.isfinite(pred.prediction).all()
        daily = pred.groupby(level='datetime').apply(lambda g: pd.Series(dict(IC=g.prediction.corr(g.raw_label), RankIC=g.prediction.corr(g.raw_label, method='spearman'))))
        metrics = summary['metrics']
        computed = {'IC':daily.IC.mean(), 'ICIR':daily.IC.mean()/daily.IC.std(),
                    'Rank IC':daily.RankIC.mean(), 'Rank ICIR':daily.RankIC.mean()/daily.RankIC.std()}
        for key, value in computed.items():
            assert np.isclose(value, metrics[key], atol=1e-9, rtol=1e-7), (path, key)
        ledgers = list((path / 'mlruns').rglob('report_normal_1day.pkl'))
        assert len(ledgers) == 1, path
        ledger = pd.read_pickle(ledgers[0])
        assert ledger.index.equals(ledger0.index)
        assert np.array_equal(ledger.bench.values, ledger0.bench.values)
        gross = ledger['return'] - ledger.bench
        net = gross - ledger.cost
        annual = net.mean()*238
        ir = net.mean()/net.std()*np.sqrt(238)
        assert np.isclose(annual, metrics['1day.excess_return_with_cost.annualized_return'], atol=1e-10)
        assert np.isclose(ir, metrics['1day.excess_return_with_cost.information_ratio'], atol=1e-9)
        name = path.name
        ledger.to_parquet(OUT / (name + '-reviewed-ledger.parquet'))
        da[name] = dict(metrics={k:metrics[k] for k in paper}, validation_selection=summary.get('selection'),
            rows=len(pred), days=len(daily), identical_test_keys_labels_benchmark=True, annualizer=238,
            gross_annual_excess=float(gross.mean()*238), annual_cost_drag=float(ledger.cost.mean()*238),
            predictions_sha256=hashlib.sha256((path / 'predictions-raw-labels.pkl').read_bytes()).hexdigest(),
            differences_from_paper={k:metrics[k]-v for k,v in paper.items()})

    nb = load(ROOT / 'sources/TRA-qlib/examples/benchmarks/TRA/Reports.ipynb')
    source = ''.join(nb['cells'][1]['source']).replace('%matplotlib inline','')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'func')
    namespace = dict(pd=pd, np=np)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'inspected_author_portfolio_formula', 'exec'), namespace)
    returns = pd.read_pickle(ROOT / 'data/TRA/kdd21/ret.pkl').clip(-.1,.1)
    original = ROOT / 'results/tra-original'
    original_protocol = load(original / 'protocol.json')
    tra = {}
    for path in [original, ROOT / 'results/tra-boundary-corrected', OUT / 'tra-calendar-corrected']:
        assert load(path / 'protocol.json')['data_sha256'] == original_protocol['data_sha256']
        for stage in ['initialization', 'tra']:
            pred = pd.read_pickle(path / stage / 'pred.pkl')
            old = pd.read_pickle(original / stage / 'pred.pkl')
            assert pred.index.equals(old.index) and np.array_equal(pred.label.values, old.label.values)
            pred['ret'] = returns.reindex(pred.index)
            assert pred.ret.notna().all()
            daily = pred.groupby(level='datetime').apply(namespace['func'])
            r = daily.R.reindex(pd.date_range(daily.index.min(), daily.index.max())).fillna(0)
            metrics = {key:float(daily[key].mean()) for key in ['MSE','MAE','IC']}
            metrics.update(ICIR=float(daily.IC.mean()/daily.IC.std()), AR=float(r.mean()*365),
                AV=float(r.std()*np.sqrt(365)), SR=float(r.mean()/r.std()*np.sqrt(365)),
                MDD=float((r.cumsum().cummax()-r.cumsum()).max()))
            native = load(path / stage / 'info.json')['metric']
            for key, value in native.items():
                assert np.isclose(metrics[key], value, atol=1e-8, rtol=1e-6)
            history = pd.read_csv(path / stage / 'logs.csv', header=[0,1])
            valid = history[('valid','IC')]
            tra[path.name + '-' + stage] = dict(metrics=metrics, epochs=len(history),
                best_validation_ic=float(valid.max()), best_validation_epoch=int(valid.idxmax()),
                identical_test_keys_labels=True, rows=len(pred), days=len(daily))
            daily.to_parquet(OUT / (path.name + '-' + stage + '-reviewed-daily.parquet'))

    # The checkpoint diagnostic must compare exactly the same selected weights.
    import torch
    native = torch.load(OUT / 'available-inputs/offline-model.pt', map_location='cpu')
    coherent = torch.load(OUT / 'available-inputs/best-coherent-state.pt', map_location='cpu')
    assert all(torch.equal(value, coherent['framework'][key]) for key,value in native['framework'].items())
    opt_diff = {}
    for key in ['framework_opt','opt']:
        first = native[key]['state']; second = coherent[key]['state']
        assert first.keys() == second.keys()
        steps = sorted(set(float(first[k]['step'])-float(second[k]['step']) for k in first))
        opt_diff[key] = dict(step_differences=steps, identical_state=False if any(steps) else None)
    picked = load(OUT / 'validation-selected-clean-model.json')
    assert picked['winner']['validation_ic'] == max(c['validation_ic'] for c in picked['all_candidates'])
    evidence = dict(passed=True, DoubleAdapt=da, TRA=tra, selected_clean_candidate=picked['winner'],
        checkpoint_diagnostic=dict(identical_model_weights=True, optimizer_differences=opt_diff),
        paper_targets=dict(DoubleAdapt=paper, TRA=dict(MSE=.157,MAE=.318,IC=.059,ICIR=.460,AR=.124,SR=.885)),
        limitations=['All new results are single seed, not the authors multi-run means.',
                     'The five-session candidate, if present, uses the available modern CUDA runtime.',
                     'Original gates remain unchanged pending review; no full-CZ promotion.'])
    (OUT / 'final-verification.json').write_text(json.dumps(evidence, indent=2, default=str, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, selected_clean_candidate=picked['winner'], checkpoint=opt_diff), indent=2))


if __name__ == '__main__':
    main()
