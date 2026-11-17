"""Replay a saved checkpoint without training or changing checkpoint selection."""
from pathlib import Path
import numpy as np
import torch
from .model import Hermes, LinearBaseline
from .metrics import evaluate, market_neutral_portfolio
from .costs import buy_only_costs


def predict_checkpoint(checkpoint, data, start, end, device='cpu'):
    saved = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    config = saved['config']
    features = data['features']
    if start < config['lookback'] or end > len(features) or end <= start:
        raise ValueError('invalid evaluation interval')
    if config['ablation'] == 'linear':
        model = LinearBaseline(config['lookback'], features.shape[-1])
    else:
        model = Hermes(data['incidence'], config['lookback'], features.shape[-1], config['hidden'],
                       config['scales'], config['windows'], config['ablation'])
    model.load_state_dict(saved['model'])
    model.to(device).eval()
    predictions = []
    offsets = np.arange(-config['lookback'], 0)
    with torch.no_grad():
        for begin in range(start, end, config['batch_size']):
            days = np.arange(begin, min(begin + config['batch_size'], end))
            x = features[days[:, None] + offsets[None, :]].transpose(0, 2, 1, 3)
            predictions.append(model(torch.as_tensor(x, device=device)).cpu().numpy())
    return np.concatenate(predictions), saved


def evaluated_artifacts(prediction, target, valid, dates, tickers, top_k=5, pnl_target=None):
    pnl_target = target if pnl_target is None else pnl_target
    metrics, daily_ic, _ = evaluate(prediction, target, valid, top_k, pnl_target)
    book = market_neutral_portfolio(prediction,pnl_target,valid,top_k)
    arrays = {'prediction': prediction, 'target': target, 'valid': valid,
              'dates': dates, 'tickers': tickers, 'daily_ic': daily_ic,
              'pnl_target':pnl_target, **book,
              **buy_only_costs(book['portfolio_weights'],pnl_target,price_returns=target)}
    return metrics, arrays
