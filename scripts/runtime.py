"""Data and model adapters around the pinned, unmodified author implementation."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream'))


class Panel:
    def __init__(self, universe):
        self.folder = ROOT / 'data' / universe
        self.manifest = json.loads((self.folder / 'manifest.json').read_text())
        for name in ['dates', 'symbols', 'features', 'close', 'priors', 'member',
                     'tradable', 'valid_priors', 'totalret']:
            setattr(self, name, np.load(self.folder / f'{name}.npy', mmap_mode='r'))
        terminal = self.folder / 'terminal_events.npy'
        self.terminal_events = np.load(terminal, mmap_mode='r') if terminal.exists() else None
        self.plans = {}
        for split, start, stop in [('train', '2009-01-01', '2019-12-31'),
                                    ('valid', '2020-01-01', '2021-12-31'),
                                    ('test', '2022-01-01', '2024-12-31')]:
            days = []
            for t in range(19, len(self.dates)-10):
                if not start <= self.dates[t] <= stop or not self.valid_priors[t]:
                    continue
                if split != 'test' and self.dates[t+10] > stop:
                    continue
                ids = self.ids(t, split)
                if len(ids) >= 30:
                    days.append(t)
            self.plans[split] = np.array(days, dtype=np.int64)

    def ids(self, t, split):
        mask = self.member[:, t] & self.tradable[:, t]
        if split != 'test':
            tail = self.close[:, t+1:t+11]
            mask = mask & np.isfinite(tail).all(1) & (tail > 0).all(1)
        return np.flatnonzero(mask)

    def batch(self, t, split, device='cpu'):
        ids = self.ids(t, split)
        x = np.array(self.features[ids[:, None], np.arange(t-19, t+1)[None, :]], copy=True)
        prior = np.repeat(self.priors[t:t+1], len(ids), axis=0).copy()
        with np.errstate(invalid='ignore', divide='ignore'):
            labels = np.asarray(self.close[ids, t+1:t+11] / self.close[ids, t+1:t+2] - 1)
        labels[~np.isfinite(labels)] = np.nan
        if split != 'test':
            labels = ((pd.DataFrame(labels).rank(pct=True) - 0.5) * 3.46).to_numpy(dtype=np.float32)
        return (torch.from_numpy(x).to(device), torch.from_numpy(prior).to(device),
                torch.from_numpy(labels.copy()).to(device), ids)

    def summary(self):
        return {s: {'days': len(ds), 'first': str(self.dates[ds[0]]),
                    'last': str(self.dates[ds[-1]]),
                    'stock_days': sum(len(self.ids(t, s)) for t in ds)} for s, ds in self.plans.items()}


def config(universe, seed=0, epochs=50, stage1=None):
    cfg = yaml.safe_load((ROOT / 'upstream/configs/config.yaml').read_text())
    cfg['data']['universe'] = universe
    cfg['train'].update(seed=seed, num_epochs=epochs)
    cfg['predictor'].update(n_expert=2 if universe == 'csi300' else 8,
                            k=1 if universe == 'csi300' else 4,
                            aux_weight=0.01 if universe == 'csi300' else 0.001)
    cfg['predictor']['transformer']['num_heads'] = 2 if universe == 'csi300' else 4
    if stage1:
        cfg['predictor']['saved_model'] = str(Path(stage1).resolve())
    return cfg


def spatial_model(cfg):
    from module.autoencoder import VQVAE
    return VQVAE(copy.deepcopy(cfg))


def temporal_model(cfg, total_steps):
    from trainer.train_ypred import GenerateReturn

    class FrozenGenerateReturn(GenerateReturn):
        def train(self, mode=True):
            super().train(mode)
            # requires_grad=False does not stop the released VQ anchor update.
            # Keep all pretrained modules in eval mode after recursive train().
            self.encoder.eval()
            self.quantizer.eval()
            self.revin.eval()
            return self

    return FrozenGenerateReturn(copy.deepcopy(cfg), T_max=total_steps)


def encode(model, x):
    with torch.no_grad():
        h = model.encoder(model.revin(x, mode='norm'))
        z, _, (_, _, idx) = model.quantizer(h)
    return z, idx


def from_codes(model, x, prior, z):
    from trainer.train_ypred import softcap_log1p
    a, bp, bl, loss = model.loadings(x, z)
    pred = model.return_predictor(a, bp, bl, model.z_prior_norm(prior), model.latent_value_head(z))
    return pred, softcap_log1p(loss, model.aux_imp)


def rankic(pred, target):
    good = np.isfinite(pred) & np.isfinite(target)
    if good.sum() < 3:
        return np.nan
    a = pd.Series(pred[good]).rank().to_numpy()
    b = pd.Series(target[good]).rank().to_numpy()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def ranking_metrics(daily):
    values = np.array(daily, dtype=float)
    good = values[np.isfinite(values)]
    return dict(RankIC=float(good.mean()), RankICIR=float(good.mean()/good.std(ddof=0)),
                days=len(good), undefined_days=int(len(values)-len(good)))
