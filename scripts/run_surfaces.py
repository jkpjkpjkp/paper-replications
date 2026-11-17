#!/usr/bin/env python3
"""Illustrative Figures 1/3: learn across a declared nondegenerate initial box."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from krrdp.market import Market, payoff
from krrdp.pricing import Training, train
from scripts.run_suite import atomic_json

with threadpool_limits(2):
    market = Market(dimension=2)
    training = Training(n=800, m=256, validation_n=128, batch_size=4096,
                        initial_box=(60.,140.))
    axis = np.linspace(60., 140., 51)
    xx, yy = np.meshgrid(axis, axis)
    inputs = np.c_[xx.ravel(), yy.ravel()]
    for index, kind in enumerate(['max_call','geometric_put']):
        meta_path = Path('results/surfaces')/(kind+'.json')
        if meta_path.exists():
            continue
        result = train(market, training, kind, seed=9419+index)
        arrays = dict(axis=axis)
        for stage in [0,2,4,6,8,9]:
            values = (payoff(inputs, kind) if stage == market.steps
                      else result.models[stage].predict(inputs))
            arrays[f'value_{stage}'] = values.reshape(xx.shape)
        meta = result.asdict()
        meta['description'] = ('Illustrative value surfaces with independent Uniform[60,140] initial '
                               'asset prices propagated by GBM. Not the deterministic-spot table experiment; '
                               'the paper does not state its surface training distribution.')
        meta_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(meta_path.with_suffix('.npz'), **arrays)
        # Portable model weights for all nonterminal stages.
        weights = {}
        for stage, model in enumerate(result.models[:-1]):
            weights[f'centers_{stage}'] = model.centers
            weights[f'coefficients_{stage}'] = model.coefficients
            weights[f'lengthscale_{stage}'] = np.asarray(model.lengthscale)
            weights[f'clip_{stage}'] = np.asarray(model.clip)
        np.savez_compressed(meta_path.with_name(kind+'_weights.npz'), **weights)
        atomic_json(meta_path, meta)
        print(f'{kind}: surfaces generated in {result.seconds:.2f} sec', flush=True)
