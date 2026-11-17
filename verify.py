#!/usr/bin/env python3
"""Deterministic numerical reproduction/counterexamples; no empirical results.

Run: python verify.py
All fixtures are synthetic and all checks assert mathematical properties.
A PASS for a counterexample means the claimed defect was reproduced.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from stable_core import (
    bl_posterior, checked_sharpe_weights, ddim_step, guided_noise,
    kalman_embedding, literal_budget_weights, ridge_covariance, sharpe,
    view_moments,
)

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/jkp/data/paper-reproduction-audit/workspaces/105-stable')
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
CHECKS = []


def record(name, evidence, kind='implementation_check'):
    CHECKS.append(dict(name=name, status='PASS', kind=kind, evidence=evidence))


def must_reject(call):
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise AssertionError('Expected rejection did not occur')


def positive_matrix(rng, size, scale=1):
    x = rng.normal(size=(size, size))
    return scale * (x @ x.T / size + np.eye(size))


def run():
    rng = np.random.default_rng(20261117)

    # Independent fixture for Eqs. 3-4, including log-return compounding.
    simple = np.array([[[.10, -.05], [-.03, .04]],
                       [[.02, .08], [.07, -.06]],
                       [[-.01, .02], [.03, .01]]])
    mu, cov = view_moments(np.log1p(simple))
    truth = np.prod(1 + simple, axis=1) - 1
    np.testing.assert_allclose(mu, truth.mean(axis=0), atol=1e-15)
    np.testing.assert_allclose(cov, np.cov(truth, rowvar=False), atol=1e-15)
    record('path_compounding_and_unbiased_moments', {'max_mean_error': float(abs(mu - truth.mean(0)).max())})

    # Paper dimensions, including all selected markets. Raw inv() may return
    # rubbish rather than raising; matrix rank and residual are the evidence.
    rank_rows = []
    rank_fixtures = {}
    for market, size in [('US', 55), ('China', 55), ('Europe', 37), ('Korea', 44)]:
        paths = rng.normal(.0003, .01, size=(50, 20, size))
        mean, covariance = view_moments(paths)
        rank = int(np.linalg.matrix_rank(covariance))
        assert rank == min(size, 49)
        row = dict(market=market, assets=size, paths=50, rank=rank,
                   nullity=size-rank, condition_number=float(np.linalg.cond(covariance)))
        try:
            inverse = np.linalg.inv(covariance)
            row['raw_inverse_identity_residual_inf'] = float(np.linalg.norm(
                covariance @ inverse - np.eye(size), ord=np.inf))
        except np.linalg.LinAlgError:
            row['raw_inverse_error'] = 'LinAlgError'
        if size == 55:
            row['guard'] = must_reject(lambda: bl_posterior(
                np.zeros(size), np.eye(size), mean, covariance))
        rank_rows.append(row)
        rank_fixtures[market] = (mean, covariance)
    record('paper_k50_covariance_rank_defect', rank_rows, 'paper_counterexample')

    # Ridge is explicitly a repair, never relabelled as the paper method.
    ridge_rows = []
    mean, covariance = rank_fixtures['US']
    prior_covariance = positive_matrix(rng, 55, .002)
    prior_mean = prior_covariance @ np.full(55, 1 / 55)
    for ridge in [1e-6, 1e-4, 1e-2, 1e-1]:
        repaired = ridge_covariance(covariance, ridge)
        residual = np.linalg.norm(repaired @ np.linalg.solve(repaired, np.eye(55)) - np.eye(55), np.inf)
        assert residual < 1e-7
        posterior_mean, posterior_cov = bl_posterior(prior_mean, prior_covariance, mean, repaired)
        assert np.linalg.eigvalsh(posterior_cov).min() > 0
        direction = np.linalg.solve(posterior_cov, posterior_mean)
        weights = literal_budget_weights(posterior_mean, posterior_cov)
        row = dict(relative_ridge=ridge, inverse_residual_inf=float(residual),
                   min_eigenvalue=float(np.linalg.eigvalsh(repaired).min()),
                   normalization_denominator=float(direction.sum()),
                   synthetic_weight_gross=float(abs(weights).sum()),
                   positive_sharpe_normalization=bool(direction.sum() > 0))
        ridge_rows.append(row)
    record('explicit_ridge_repair_is_numerically_solvable', ridge_rows, 'repair_check')

    # Independent Gaussian MAP oracle: whiten the two distributions and solve
    # least squares with SVD instead of adding precisions.
    size = 7
    cp, cv = positive_matrix(rng, size), positive_matrix(rng, size)
    mp, mv = rng.normal(size=size), rng.normal(size=size)
    post_m, post_c = bl_posterior(mp, cp, mv, cv)
    whiten_p = np.linalg.solve(np.linalg.cholesky(cp), np.eye(size))
    whiten_v = np.linalg.solve(np.linalg.cholesky(cv), np.eye(size))
    design = np.vstack([whiten_p, whiten_v])
    observations = np.concatenate([whiten_p @ mp, whiten_v @ mv])
    oracle, *_ = np.linalg.lstsq(design, observations, rcond=None)
    _, singular_values, vt = np.linalg.svd(design, full_matrices=False)
    oracle_cov = (vt.T / singular_values**2) @ vt
    np.testing.assert_allclose(post_m, oracle, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(post_c, oracle_cov, rtol=1e-12, atol=1e-12)
    record('BL_matches_independent_whitened_least_squares', {
        'max_mean_error': float(abs(post_m-oracle).max()),
        'max_covariance_error': float(abs(post_c-oracle_cov).max())})

    # Appendix A.3's exact equilibrium prior makes prior covariance cancel
    # from the final allocation. Both posteriors still depend on the prior.
    eq = np.full(size, 1 / size)
    mv = cv @ (np.linspace(.2, 1.2, size) - eq)
    reference = None
    errors = []
    for scale in [.01, 1, 100]:
        cp = positive_matrix(rng, size, scale)
        pm, pc = bl_posterior(cp @ eq, cp, mv, cv)
        weights = checked_sharpe_weights(pm, pc)
        direction = eq + np.linalg.solve(cv, mv)
        expected = direction / direction.sum()
        np.testing.assert_allclose(weights, expected, atol=1e-12, rtol=1e-12)
        if reference is None:
            reference = weights
        errors.append(float(abs(weights - reference).max()))
    record('A3_prior_covariance_cancels_from_final_weights', {
        'scales': [.01, 1, 100], 'max_weight_change': max(errors),
        'identity': 'w = normalize(w_equal + solve(Sigma_view, mu_view))'}, 'paper_structural_result')

    # Independent exhaustive search over a 2-asset budget plane, positive case.
    covariance = np.array([[.04, .008], [.008, .09]])
    mean = np.array([.05, .04])
    optimum = checked_sharpe_weights(mean, covariance)
    grid = np.linspace(-10, 10, 200001)
    candidates = np.stack([grid, 1-grid], axis=1)
    scores = candidates @ mean / np.sqrt(np.einsum('ni,ij,nj->n', candidates, covariance, candidates))
    theoretical = sharpe(mean, covariance, optimum)
    assert theoretical >= scores.max() - 1e-12
    assert theoretical - scores.max() < 1e-8
    record('positive_denominator_tangency_matches_grid_search', {
        'formula_sharpe': theoretical, 'best_grid_sharpe': float(scores.max()), 'grid_points': len(grid)})

    # Theorem counterexample from valid BL posterior inputs, no singularities.
    pm, pc = bl_posterior(np.array([.5, .5]), np.eye(2),
                          np.array([-1.5, -2.5]), np.eye(2))
    literal = literal_budget_weights(pm, pc)
    alternative = np.array([3., -2.])
    literal_score, alternative_score = sharpe(pm, pc, literal), sharpe(pm, pc, alternative)
    assert literal_score < 0 < alternative_score
    np.testing.assert_allclose(literal_score, -np.sqrt(pm @ np.linalg.solve(pc, pm)))
    rejection = must_reject(lambda: checked_sharpe_weights(pm, pc))
    record('negative_denominator_returns_minimum_sharpe', {
        'posterior_mean': pm.tolist(), 'posterior_covariance': pc.tolist(),
        'paper_weights': literal.tolist(), 'paper_sharpe': literal_score,
        'feasible_alternative_weights': alternative.tolist(), 'alternative_sharpe': alternative_score,
        'corrected_guard': rejection}, 'paper_counterexample')
    message = must_reject(lambda: checked_sharpe_weights(np.array([1., -1.]), np.eye(2)))
    record('zero_budget_tangency_direction_rejected', {'message': message}, 'repair_check')

    # Marginal samples are identical, but their pairing defines different joint laws.
    values = np.array([-2., -1., 1., 2.]) * .01
    joint_pos = np.stack([values, values], axis=1)
    joint_neg = np.stack([values, values[::-1]], axis=1)
    np.testing.assert_allclose(np.sort(joint_pos, axis=0), np.sort(joint_neg, axis=0))
    corr_pos, corr_neg = np.corrcoef(joint_pos.T)[0, 1], np.corrcoef(joint_neg.T)[0, 1]
    assert corr_pos > .99999 and corr_neg < -.99999
    record('per_stock_marginals_do_not_identify_joint_covariance', {
        'same_marginals': True, 'coupling_1_correlation': float(corr_pos),
        'coupling_2_correlation': float(corr_neg)}, 'specification_counterexample')

    macro, full = np.array([1., 2.]), np.array([3., 6.])
    np.testing.assert_array_equal(guided_noise(macro, full, 0), macro)
    np.testing.assert_array_equal(guided_noise(macro, full, 1), full)
    midpoint = guided_noise(macro, full, .5)
    assert not np.array_equal(midpoint, full)
    record('fixed_half_gate_is_not_full_condition_denoiser', {
        'macro': macro.tolist(), 'full': full.tolist(), 'z_half': midpoint.tolist(),
        'full_condition_gate': 1}, 'paper_counterexample')

    # Verify the deterministic reverse equation with an exact oracle, not a
    # learned model. It reconstructs a whole paper-sized 50x20x55 path tensor.
    clean = rng.normal(.0003, .01, size=(50, 20, 55))
    epsilon = rng.normal(size=clean.shape)
    schedule = np.linspace(.01, 1, 31)
    state = np.sqrt(schedule[0]) * clean + np.sqrt(1-schedule[0]) * epsilon
    for now, previous in zip(schedule[:-1], schedule[1:]):
        state, _ = ddim_step(state, epsilon, now, previous)
    np.testing.assert_allclose(state, clean, atol=1e-13, rtol=1e-12)
    view_m, view_c = view_moments(state)
    repaired = ridge_covariance(view_c, .01)
    # Shift the synthetic horizon mean to a prescribed positive tangency
    # direction; no forecast-quality claim is made by this integration fixture.
    target = np.linspace(.01, .04, 55)
    view_m = repaired @ target
    cp = positive_matrix(rng, 55, .002)
    pm, pc = bl_posterior(cp @ np.full(55, 1/55), cp, view_m, repaired)
    weights = checked_sharpe_weights(pm, pc)
    np.testing.assert_allclose(weights, (target + 1/55) / (target + 1/55).sum(), atol=1e-12)
    record('DDIM_oracle_to_explicitly_repaired_allocation', {
        'shape': list(clean.shape), 'DDIM_updates': 30,
        'max_reconstruction_error': float(abs(state-clean).max()),
        'weight_sum': float(weights.sum()), 'learned_denoiser': False,
        'view_mean_override': 'repaired covariance @ prescribed positive direction; synthetic fixture only'})

    # Kalman Q=0 must agree with exact Bayesian linear regression; future
    # changes must not change past filtered betas.
    x, y = rng.normal(size=(60, 5)), rng.normal(size=60)
    initial_mean, initial_covariance, variance = np.zeros(5), np.eye(5), .3
    betas, beta_cov = kalman_embedding(x, y, initial_mean, initial_covariance, np.zeros((5, 5)), variance)
    batch_cov = np.linalg.solve(np.eye(5) + x.T @ x / variance, np.eye(5))
    batch_mean = batch_cov @ (x.T @ y / variance)
    np.testing.assert_allclose(betas[-1], batch_mean, atol=1e-12)
    np.testing.assert_allclose(beta_cov[-1], batch_cov, atol=1e-12)
    y_changed = y.copy()
    y_changed[30:] += 100
    changed, _ = kalman_embedding(x, y_changed, initial_mean, initial_covariance, np.zeros((5, 5)), variance)
    np.testing.assert_array_equal(betas[:30], changed[:30])
    record('causal_Kalman_matches_batch_Gaussian_reference', {
        'final_beta_error': float(abs(betas[-1]-batch_mean).max()), 'unchanged_prefix_observations': 30,
        'parameters': 'synthetic Q=0, R=.3, initial beta=0, P=I; not paper settings'})

    # Population optimal denoiser in a Gaussian DDPM fixture. E[eps|x_t]
    # need not itself have unit variance: law of total variance.
    count, alpha = 200000, .8
    x0, noise = rng.normal(size=count), rng.normal(size=count)
    xt = np.sqrt(alpha) * x0 + np.sqrt(1-alpha) * noise
    optimal_prediction = np.sqrt(1-alpha) * xt
    empirical_variance = float(optimal_prediction.var())
    assert abs(empirical_variance - .2) < .005
    record('MSE_optimal_noise_prediction_need_not_be_standard_normal', {
        'alpha_bar': alpha, 'analytic_prediction_variance': 1-alpha,
        'empirical_prediction_variance': empirical_variance, 'samples': count,
        'scope': 'counterexample to Section 3.3 distribution assertion; not an objection to DDIM itself'},
        'paper_counterexample')

    # Audit arithmetic of already extracted targets, never call it reproduction.
    with (SOURCE / 'audit/paper_targets.csv').open() as stream:
        targets = list(csv.DictReader(stream))
    headline = []
    for table, metric in [('3', 'ASR'), ('4', 'MSE_x1e-4')]:
        for market in ['S&P500', 'CSI300', 'EUROSTOXX', 'KOSPI200']:
            rows = [r for r in targets if r['table'] == table and r['metric'] == metric and r['universe'] == market]
            proposed = float(next(r['paper_value'] for r in rows if r['method'].startswith('STABLE')))
            competitors = [r for r in rows if not r['method'].startswith('STABLE')]
            best = (max if table == '3' else min)(competitors, key=lambda r: float(r['paper_value']))
            baseline = float(best['paper_value'])
            improvement = ((proposed-baseline) / abs(baseline) if table == '3'
                           else (baseline-proposed) / baseline) * 100
            headline.append(dict(table=table, metric=metric, market=market, paper_STABLE=proposed,
                                 best_baseline=best['method'], paper_baseline=baseline,
                                 improvement_percent=improvement,
                                 convention='gain / abs(baseline)' if table == '3' else 'relative MSE reduction'))
    # Source-value anchors prevent accidentally inspecting the wrong table.
    assert next(r for r in headline if r['table']=='3' and r['market']=='EUROSTOXX')['paper_baseline'] == 1.42
    assert next(r for r in headline if r['table']=='4' and r['market']=='CSI300')['paper_baseline'] == 4.57
    record('published_table_headline_arithmetic', headline, 'source_arithmetic_only')

    result = dict(status='partial_equation_reproduction_with_counterexamples',
                  full_empirical_reproduction=False, published_metric_cells_reproduced=0,
                  training_runs=0, market_backtests=0, author_code_available=False,
                  fixture_seed=20261117, passed=len(CHECKS), checks=CHECKS,
                  environment=dict(python=platform.python_version(), numpy=np.__version__),
                  source_pdf_sha256=hashlib.sha256((SOURCE/'sources/paper.pdf').read_bytes()).hexdigest(),
                  files_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [ROOT/'stable_core.py', ROOT/'verify.py']})
    (OUT/'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for name, rows in [('covariance_rank.csv', rank_rows), ('ridge_sensitivity.csv', ridge_rows), ('headline_arithmetic.csv', headline)]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (OUT/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ['status','passed','published_metric_cells_reproduced','training_runs','market_backtests']}))


if __name__ == '__main__':
    run()
