"""Audit a completed full-universe NASDAQ case without reusing metrics.evaluate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def verify(folder, panel):
    config = json.loads((folder / 'config.json').read_text())
    reported = json.loads((folder / 'metrics.json').read_text())
    history = json.loads((folder / 'history.json').read_text())
    assert config['market'] == 'NASDAQ' and config['stocks'] == 1026
    assert config['train_steps'] == 0 and config['epochs'] == 300
    assert reported['status'] == 'reconstructed_protocol'
    assert hashlib.sha256(panel.read_bytes()).hexdigest() == config['data_sha256']
    assert [r['epoch'] for r in history] == list(range(len(history)))
    assert len(history) == 300 or len(history) >= config['patience'] + 1
    best = min(history, key=lambda r: r['validation_mse'])
    assert best['epoch'] == reported['selected_epoch']
    assert best['validation_mse'] == reported['validation_mse']
    if len(history) < 300:
        assert len(history) - 1 - best['epoch'] == config['patience']
    z = np.load(folder / 'predictions.npz', allow_pickle=False)
    source = np.load(panel, allow_pickle=False)
    p, y, mask = z['prediction'], z['truth'], z['mask']
    assert p.shape == y.shape == mask.shape == (1026, 216)
    assert np.isfinite(p).all() and np.isfinite(y).all()
    np.testing.assert_array_equal(z['dates'], source['dates'][1008:])
    np.testing.assert_array_equal(z['tickers'], source['tickers'])
    np.testing.assert_array_equal(y, source['truth'][:, 1008:].astype(np.float32))
    expected_mask = np.stack([source['mask'][:, d-4:d+1].min(1)
                              for d in range(1008, 1224)], axis=1)
    np.testing.assert_array_equal(mask, expected_mask.astype(np.float32))
    returns, reciprocal = [], []
    for day in range(216):
        ids = np.flatnonzero(mask[:, day] > .5)
        ordered = ids[np.argsort(-p[ids, day], kind='stable')]
        assert len(ordered) >= 5
        returns.append(float(y[ordered[:5], day].mean(dtype=np.float64)))
        selected_truth = y[ordered[0], day]
        rank = 1 + np.count_nonzero(y[ids, day] > selected_truth)
        rank += .5 * (np.count_nonzero(y[ids, day] == selected_truth) - 1)
        reciprocal.append(1 / rank)
    returns = np.asarray(returns)
    independent = {
        'sharpe5': float(np.sqrt(252) * returns.mean() / returns.std(ddof=0)),
        'mrr': float(np.mean(reciprocal)),
        'simple_return': float(returns.sum()),
        'compound_return': float(np.prod(1 + returns) - 1),
        'downside_deviation_annual': float(np.sqrt(252 * np.mean(np.minimum(returns, 0)**2))),
    }
    for name, value in independent.items():
        np.testing.assert_allclose(value, reported[name], rtol=2e-6, atol=2e-8)
    csv_returns = np.loadtxt(folder / 'daily_returns.csv', delimiter=',', skiprows=1,
                             usecols=1)
    np.testing.assert_allclose(csv_returns, returns, rtol=2e-6, atol=2e-8)
    refs = {'sharpe5': (2.2112, .1283), 'mrr': (.0407, .0005)}
    comparison = {name: {
        'local': independent[name], 'paper_mean': mean, 'paper_seed_sd': sd,
        'absolute_error': abs(independent[name] - mean),
        'relative_error': abs(independent[name] - mean) / abs(mean),
        'within_two_published_seed_sd': bool(abs(independent[name] - mean) <= 2 * sd),
    } for name, (mean, sd) in refs.items()}
    result = {
        'scope': 'One full-universe NASDAQ seed, reconstructed protocol',
        'software_and_artifact_checks_passed': True,
        'epochs_completed': len(history), 'optimizer_updates': len(history) * 752,
        'selected_epoch': best['epoch'], 'seed': config['seed'],
        'independent_metrics': independent, 'paper_comparison': comparison,
        'comparison_rule': 'Predeclared descriptive screen: BOTH SR and MRR within two published seed SDs; not a confidence interval or equivalence test.',
        'descriptive_screen_passed': all(v['within_two_published_seed_sd'] for v in comparison.values()),
        'exact_paper_replication': False,
        'limitations': [
            'Author-prepared panel and exact preprocessing are unavailable.',
            'Repaired architecture, validation-MSE selection and patience 20 are reconstruction choices.',
            'Realized returns are close-to-close; paper describes close-to-next-open.',
            'Corrected NDCG/DD definitions are not established as the paper definitions.',
            'One seed provides no measured seed uncertainty.',
        ],
        'artifact_sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                            for name in ['config.json', 'history.json', 'metrics.json',
                                         'predictions.npz', 'best.pt']},
    }
    (folder / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--panel', type=Path, default=Path(__file__).parent / 'data/NASDAQ.npz')
    args = parser.parse_args()
    print(json.dumps(verify(args.folder, args.panel), indent=2))
