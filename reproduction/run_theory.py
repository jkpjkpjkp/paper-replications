#!/usr/bin/env python3
"""Reproduce the paper's signal/noise mechanism with independently fitted OLS.

Synthetic evidence only. Equation 29 uses a ratio of ensemble moments; it is
also plotted, but is not assumed equal to mean conditional IC squared.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import save_json

HERE = Path(__file__).resolve().parent


def run(args):
    rng = np.random.default_rng(args.seed)
    n, d, count = 512, 32, 60
    sigma, delta0 = 0.5, 1.0
    horizon = np.arange(1, count + 1)
    profiles = {'fast_realization': np.ones(count),
                'slow_realization': horizon / count,
                'saturating_realization': 1 - np.exp(-horizon / 12)}
    truth = rng.normal(size=d)
    truth /= np.linalg.norm(truth)
    # Orthogonalized Gaussian factor design gives S.T@S = N*I exactly.
    q, _ = np.linalg.qr(rng.normal(size=(n, d)))
    design = q * np.sqrt(n)
    simulations = {name: [] for name in profiles}
    for _ in range(args.repetitions):
        initial = rng.normal(scale=sigma * np.sqrt(delta0), size=(n, 1))
        noise = initial + rng.normal(scale=sigma, size=(n, count)).cumsum(axis=1)
        noise_weights = design.T @ noise / n
        for name, alpha in profiles.items():
            # This is OLS (S.T S)^(-1) S.T y for every proxy horizon.
            fitted = truth[:, None] * alpha[None] + noise_weights
            target_variance = alpha[-1] ** 2 + sigma ** 2 * (count + delta0)
            population_ic = (truth @ fitted) * alpha[-1] / (
                np.linalg.norm(fitted, axis=0) * np.sqrt(target_variance))
            simulations[name].append(population_ic)
    args.output.mkdir(parents=True, exist_ok=True)
    summaries, records = {}, []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for ax, (name, alpha) in zip(axes, profiles.items()):
        ic = np.asarray(simulations[name])
        mean = ic.mean(0)
        se = ic.std(0, ddof=1) / np.sqrt(len(ic))
        target_var = alpha[-1] ** 2 + sigma ** 2 * (count + delta0)
        k = d * sigma ** 2 / n
        moment_ic = np.sqrt(alpha ** 2 * alpha[-1] ** 2 /
                            ((alpha ** 2 + k * (horizon + delta0)) * target_var))
        summaries[name] = {'empirical_peak_horizon': int(mean.argmax() + 1),
                           'equation29_peak_horizon': int(moment_ic.argmax() + 1),
                           'peak_mean_ic': float(mean.max()), 'final_horizon_mean_ic': float(mean[-1]),
                           'peak_minus_final_ic': float(mean.max() - mean[-1])}
        for h in range(count):
            records.append({'scenario': name, 'horizon': h + 1, 'mean_ic': mean[h],
                            'ic_se': se[h], 'mean_squared_ic': np.mean(ic[:, h] ** 2),
                            'equation29_squared_ic': moment_ic[h] ** 2})
        ax.plot(horizon, mean, color='#146b91', label='OLS Monte Carlo mean IC')
        ax.fill_between(horizon, mean - 1.96 * se, mean + 1.96 * se, color='#146b91', alpha=0.2)
        ax.plot(horizon, moment_ic, '--', color='#c86c30', label='Eq. 29 moment ratio, square root')
        ax.axvline(mean.argmax() + 1, color='gray', ls=':', alpha=0.7)
        ax.set(title=name.replace('_', ' ').capitalize(), xlabel='Training horizon (fixed target = 60)', ylabel='Population IC on target')
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=7, loc='lower left')
    fig.suptitle('Synthetic reproduction of the label-horizon mechanism', fontsize=13)
    fig.savefig(args.output / 'signal_noise_mechanism.png', dpi=180)
    plt.close(fig)
    pd.DataFrame(records).to_csv(args.output / 'horizon_curves.csv', index=False)
    result = {'evidence_type': 'synthetic mechanism verification; no financial data or financial test metrics',
              'seed': args.seed, 'repetitions': args.repetitions, 'training_stocks': n,
              'factors': d, 'horizons': count, 'noise_sigma': sigma, 'noise_offset': delta0,
              'design': 'whitened Gaussian factors; exact OLS; shared Brownian noise across horizons',
              'evaluation': 'conditional population IC of each fitted model on independent target data',
              'caveat': 'Equation 29 is a ratio of ensemble moments, distinct from averaging per-fit squared correlations',
              'scenarios': summaries}
    save_json(args.output / 'summary.json', result)
    print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--repetitions', type=int, default=500)
    p.add_argument('--output', type=Path, default=HERE / 'results/theory')
    run(p.parse_args())
