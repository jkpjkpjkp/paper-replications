"""Audit completed experiment artifacts and recompute all reported test metrics."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.metrics import evaluate, market_neutral_portfolio, PORTFOLIO_PROTOCOL
from hermes.costs import buy_only_costs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text())
    assert sha(ROOT / 'data/sp500.npz') == manifest['prepared_sha256']
    assert sha(ROOT / 'upstream/estimate/src/data/US/sp500/baseline_data_sp500.npy') == manifest['raw_sha256']
    audit = {'data_hashes': 'pass', 'runs': [], 'code_sha256': {}, 'evaluation_protocol':PORTFOLIO_PROTOCOL}
    main_paths = sorted((ROOT / 'results').glob('sp500-*/metrics.json'))
    companion_paths = sorted((ROOT / 'results').glob('companion-*/metrics.json'))
    assert len(main_paths) == 12, f'expected 12 S&P500 runs, found {len(main_paths)}'
    assert len(companion_paths) in (0, 2), 'companion suite is incomplete'
    for path in main_paths + companion_paths:
        run = path.parent
        metrics = json.loads(path.read_text())
        config = json.loads((run / 'config.json').read_text())
        assert metrics['evaluation_protocol']==PORTFOLIO_PROTOCOL
        predictions = np.load(run / 'test_predictions.npz')
        data = np.load(config['data'])
        expected_shape = (len(data['returns']) - config['val_end'],len(data['tickers']))
        assert predictions['prediction'].shape == expected_shape
        assert predictions['target'].shape == expected_shape
        np.testing.assert_array_equal(predictions['dates'], data['dates'][config['val_end']:])
        np.testing.assert_array_equal(predictions['tickers'], data['tickers'])
        np.testing.assert_array_equal(predictions['target'], data['returns'][config['val_end']:])
        np.testing.assert_array_equal(predictions['valid'], data['valid'][config['val_end']:])
        assert np.isfinite(predictions['prediction']).all()
        recomputed, ic, portfolio = evaluate(predictions['prediction'], predictions['target'], predictions['valid'], config['top_k'])
        for key, value in recomputed.items():
            if value is None:
                assert metrics['test'][key] is None
            else:
                np.testing.assert_allclose(value, metrics['test'][key], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ic, predictions['daily_ic'])
        np.testing.assert_allclose(portfolio, predictions['portfolio_return'])
        book=market_neutral_portfolio(predictions['prediction'],predictions['target'],predictions['valid'],config['top_k'])
        for key,value in book.items():np.testing.assert_allclose(value,predictions[key])
        weights=predictions['portfolio_weights']
        np.testing.assert_allclose(np.maximum(weights,0).sum(1),.5,atol=1e-12)
        np.testing.assert_allclose(np.minimum(weights,0).sum(1),-.5,atol=1e-12)
        np.testing.assert_allclose(weights.sum(1),0,atol=1e-12)
        np.testing.assert_allclose(np.abs(weights).sum(1),1,atol=1e-12)
        cost=buy_only_costs(weights,predictions['target'])
        for key,value in cost.items():np.testing.assert_allclose(value,predictions[key],atol=1e-14)
        np.testing.assert_allclose(predictions['transaction_fees'],.0005*predictions['buy_turnover'],atol=1e-14)
        validation=np.load(run/'validation_predictions.npz')
        np.testing.assert_array_equal(validation['dates'],data['dates'][config['train_end']:config['val_end']])
        val_metrics,_,_=evaluate(validation['prediction'],validation['target'],validation['valid'],config['top_k'])
        for key,value in val_metrics.items():
            if value is None:assert metrics['validation'][key] is None
            else:np.testing.assert_allclose(value,metrics['validation'][key],rtol=1e-10,atol=1e-12)
        control=evaluate(data['returns'][config['val_end']-1:-1],predictions['target'],predictions['valid'],config['top_k'])[0]
        for key,value in control.items():
            if value is None:assert metrics['previous_day_return_control'][key] is None
            else:np.testing.assert_allclose(value,metrics['previous_day_return_control'][key],rtol=1e-10,atol=1e-12)
        archive=ROOT/'results/archive/long_only_2026-11-05'/run.name
        if (archive/'test_predictions.npz').exists():
            old=np.load(archive/'test_predictions.npz')
            for key in ['prediction','target','valid','dates','tickers']:
                np.testing.assert_array_equal(old[key],predictions[key])
            assert sha(run/'best.pt')==sha(archive/'best.pt')
        history = [json.loads(line) for line in (run / 'history.jsonl').read_text().splitlines()]
        best = min(history, key=lambda row: row['val_loss'])
        assert best['epoch'] == metrics['best_epoch']
        checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        assert checkpoint['epoch'] == metrics['best_epoch']
        assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
        audit['runs'].append({'run': run.name, 'metric_recomputation': 'pass',
                              'validation_selection': 'pass', 'safe_checkpoint_load': 'pass',
                              'daily_dollar_neutrality':'pass','buy_only_fees':'pass','historical_predictions_and_checkpoints_preserved':'pass',
                              'checkpoint_sha256': sha(run / 'best.pt'),
                              'predictions_sha256': sha(run / 'test_predictions.npz')})
    for market in ['nasdaq','nyse']:
        path = ROOT / 'data' / f'{market}-manifest.json'
        if path.exists():
            m = json.loads(path.read_text())
            assert sha(ROOT / 'data' / f'{market}.npz') == m['prepared_sha256']
    for directory in ['hermes', 'scripts', 'tests', 'configs','analysis']:
        for path in sorted((ROOT / directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                audit['code_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (ROOT / 'results/verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'PASS: {len(audit["runs"])} runs; gross/net metrics, daily long=short exposure, buy-only fees, validation selection, and unchanged original predictions/checkpoints.')


if __name__ == '__main__':
    main()
