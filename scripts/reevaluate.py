"""Apply the user-requested dollar-neutral protocol to saved, fixed checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.evaluation import predict_checkpoint, evaluated_artifacts
from hermes.metrics import evaluate, PORTFOLIO_PROTOCOL


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_arrays(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    paths = sorted((ROOT / 'results').glob('sp500-*/metrics.json'))
    paths += sorted((ROOT / 'results').glob('companion-*/metrics.json'))
    audit = {'protocol': PORTFOLIO_PROTOCOL, 'change_date': '2026-11-06',
             'source': 'same saved predictions and validation-loss-selected checkpoints; no retraining', 'runs': []}
    for path in paths:
        run = path.parent
        result = json.loads(path.read_text())
        config = json.loads((run / 'config.json').read_text())
        checkpoint_hash = sha(run / 'best.pt')
        data = np.load(config['data'])
        original = read_arrays(run / 'test_predictions.npz')
        test_metrics, test_arrays = evaluated_artifacts(
            original['prediction'], original['target'], original['valid'],
            original['dates'], original['tickers'], config['top_k'])
        for key in ['prediction','target','valid','dates','tickers']:
            np.testing.assert_array_equal(test_arrays[key], original[key])
        for key in ['IC','ICIR','mse']:
            np.testing.assert_allclose(test_metrics[key], result['test'][key], rtol=1e-10, atol=1e-12)
        val_path = run / 'validation_predictions.npz'
        if val_path.exists():
            validation = read_arrays(val_path)
            val_prediction = validation['prediction']
        else:
            val_prediction, saved = predict_checkpoint(run / 'best.pt', data, config['train_end'], config['val_end'], args.device)
            assert saved['epoch'] == result['best_epoch']
        val_days = slice(config['train_end'], config['val_end'])
        returns, valid, dates, tickers = data['returns'], data['valid'], data['dates'], data['tickers']
        val_metrics, val_arrays = evaluated_artifacts(val_prediction, returns[val_days], valid[val_days], dates[val_days], tickers, config['top_k'])
        np.testing.assert_allclose(val_metrics['IC'], result['validation']['IC'], atol=1e-6, rtol=1e-5)
        test_days = np.arange(config['val_end'], len(returns))
        control, _, _ = evaluate(returns[test_days-1], original['target'], original['valid'], config['top_k'])
        np.savez_compressed(run / 'test_predictions.npz', **test_arrays)
        np.savez_compressed(val_path, **val_arrays)
        result.update(test=test_metrics, validation=val_metrics, previous_day_return_control=control,
                      evaluation_protocol=PORTFOLIO_PROTOCOL,
                      reevaluation={'date':'2026-11-06','retrained':False,
                                    'checkpoint_sha256':checkpoint_hash,'test_predictions_changed':False})
        config['evaluation_protocol'] = PORTFOLIO_PROTOCOL
        path.write_text(json.dumps(result,indent=2)+'\n')
        (run / 'config.json').write_text(json.dumps(config,indent=2)+'\n')
        assert sha(run / 'best.pt') == checkpoint_hash
        audit['runs'].append({'run':run.name,'checkpoint_sha256':checkpoint_hash,
                              'test_predictions_unchanged':True,'IC_ICIR_mse_unchanged':True,
                              'weights_sha256':hashlib.sha256(test_arrays['portfolio_weights'].tobytes()).hexdigest()})
        print(run.name, 'neutral SR=',test_metrics['SR_annualized'],flush=True)
    (ROOT/'results/market_neutral_reevaluation.json').write_text(json.dumps(audit,indent=2)+'\n')


if __name__ == '__main__':
    main()
