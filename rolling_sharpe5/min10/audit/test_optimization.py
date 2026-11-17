"""Independent deterministic checks for the fixed training-only allocator.

Run: .venv/bin/python rolling_sharpe5/min10/audit/test_optimization.py
Writes optimization_validation.json next to this file.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / 'optimization.py'
spec = importlib.util.spec_from_file_location('min10_optimization', MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
optimize_weights = module.optimize_weights


def independent_objective(returns, weights):
    annual_mean = returns.mean(axis=0) * 12.0
    sample = np.atleast_2d(np.cov(returns, rowvar=False, ddof=1))
    covariance = (sample + np.diag(np.diag(sample))) * 6.0
    return float(annual_mean @ weights - 2.5 * weights @ covariance @ weights)


def run():
    checks = {}
    rng = np.random.default_rng(20261103)
    common = rng.normal(0.0, 0.025, (60, 1))
    returns = common + rng.normal(0.0, 0.07, (60, 12))
    # Keep all training means identical to encourage a genuine interior mix,
    # making permutation invariance more informative than a one-hot solution.
    returns = returns - returns.mean(axis=0) + 0.009
    original = returns.copy()
    weights, diagnostics = optimize_weights(returns)
    assert np.array_equal(returns, original), 'caller input was mutated'
    assert np.isfinite(weights).all() and np.isfinite(list(filter(
        lambda value: isinstance(value, float), diagnostics.values()))).all()
    assert np.min(weights) >= 0.0 and abs(weights.sum() - 1) < 1e-12
    assert (weights > 1e-6).sum() >= 6, 'test did not create a diversified solution'
    checks['simplex_finite_no_input_mutation'] = True
    actual = independent_objective(returns, weights)
    ew = independent_objective(returns, np.full(12, 1/12))
    assert abs(actual - diagnostics['objective']) < 1e-12
    assert actual >= ew - 1e-10
    assert diagnostics['kkt_residual'] < 1e-6
    checks['independent_objective_and_equal_weight_dominance'] = {
        'optimized_objective': actual, 'equal_weight_objective': ew,
        'kkt_residual': diagnostics['kkt_residual'],
    }

    permutation = rng.permutation(12)
    permuted, _ = optimize_weights(returns[:, permutation])
    restored = np.empty_like(permuted)
    restored[permutation] = permuted
    perm_error = float(np.max(np.abs(restored - weights)))
    assert perm_error < 2e-6
    checks['permutation_invariance_max_error'] = perm_error

    single = rng.normal(.008, .04, 60)
    symmetric = np.repeat(single[:, None], 6, axis=1)
    symmetric_weights, _ = optimize_weights(symmetric)
    assert np.allclose(symmetric_weights, np.full(6, 1/6), atol=1e-10, rtol=0)
    checks['equal_asset_symmetry'] = symmetric_weights.tolist()

    zero = np.zeros((60, 7))
    zero_weights, zero_diagnostics = optimize_weights(zero)
    zero_repeat, _ = optimize_weights(zero)
    assert np.array_equal(zero_weights, zero_repeat)
    assert np.allclose(zero_weights, np.full(7, 1/7), atol=1e-16, rtol=0)
    assert zero_diagnostics['objective'] == 0.0
    assert zero_diagnostics['annual_volatility'] == 0.0
    checks['zero_input_deterministic_equal_weights'] = True

    base = np.linspace(-.01, .01, 60) + .015
    dominated = np.column_stack([base, base - .05])
    dominant_weights, _ = optimize_weights(dominated)
    assert dominant_weights[0] > 1 - 1e-9 and dominant_weights[1] < 1e-9
    checks['clearly_dominated_asset_zero_weight'] = dominant_weights.tolist()

    one_weights, one_diagnostics = optimize_weights(single[:, None])
    assert np.array_equal(one_weights, np.ones(1))
    assert abs(one_diagnostics['objective'] - independent_objective(single[:, None], one_weights)) < 1e-12
    checks['single_asset'] = True

    # With two assets, a scalar quadratic has a closed-form constrained optimum.
    two = rng.normal(0, .08, (60, 2))
    two = two - two.mean(axis=0) + np.array([.009, .011])
    covariance = np.cov(two, rowvar=False, ddof=1)
    covariance = 6.0 * (covariance + np.diag(np.diag(covariance)))
    mu = two.mean(axis=0) * 12
    denominator = 5.0 * (covariance[0,0] - 2*covariance[0,1] + covariance[1,1])
    numerator = mu[0] - mu[1] + 5.0*(covariance[1,1] - covariance[0,1])
    w0 = float(np.clip(numerator / denominator, 0, 1))
    analytic = np.array([w0, 1-w0])
    two_weights, _ = optimize_weights(two)
    analytic_error = float(np.max(np.abs(two_weights - analytic)))
    assert analytic_error < 2e-7
    checks['two_asset_closed_form_max_error'] = analytic_error

    rejected = []
    invalid = {
        'one_dimensional': np.zeros(60),
        'zero_months': np.zeros((0, 2)),
        'one_month': np.zeros((1, 2)),
        'zero_assets': np.zeros((60, 0)),
        'nan': np.array([[0.0, np.nan], [0.01, 0.0]]),
        'positive_infinity': np.array([[0.0, np.inf], [0.01, 0.0]]),
        'negative_infinity': np.array([[0.0, -np.inf], [0.01, 0.0]]),
    }
    for name, data in invalid.items():
        try:
            optimize_weights(data)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(f'Invalid input accepted: {name}')
    checks['invalid_inputs_rejected'] = rejected

    # A fake unsuccessful solver must propagate failure, never return EW silently.
    original_solver = module.minimize
    class FailedResult:
        success = False
        status = 9
        message = 'synthetic iteration limit'
    module.minimize = lambda **kwargs: FailedResult()
    try:
        try:
            optimize_weights(returns)
        except RuntimeError as exc:
            assert 'synthetic iteration limit' in str(exc)
        else:
            raise AssertionError('Optimizer failure was not rejected')
    finally:
        module.minimize = original_solver
    checks['optimizer_failure_rejected'] = True

    result = {'passed': True, 'seed': 20261103, 'checks': checks,
              'fixed_specification': {'gamma': 5, 'covariance_shrinkage': .5,
                                     'annualization': 12, 'simplex': 'long-only sum1'}}
    output = Path(__file__).with_name('optimization_validation.json')
    output.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    run()
