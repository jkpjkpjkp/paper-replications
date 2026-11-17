"""Training-only, fixed-specification long-only factor allocation.

Each input column is a factor's MONTHLY return in decimal units. Each factor
sleeve has 200% gross exposure; simplex weights preserve that sleeve-level gross
exposure before any constituent netting. The caller must pass only the prior
60 complete months for each annual decision. No test-year observations, cost
optimization, hyperparameter selection or target-volatility rescaling occurs
inside this module.

The allocation maximizes annualized mean minus gamma/2 times annualized
variance, with gamma=5 and 50% sample-covariance shrinkage to its own diagonal.
This is a deliberately fixed mean-variance sensitivity, not estimated maximum
Sharpe allocation. It does not infer unavailable factor transaction costs.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

GAMMA = 5.0
COVARIANCE_SHRINKAGE = 0.5
MONTHS_PER_YEAR = 12.0


def optimize_weights(train_returns: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return nonnegative sum-one weights and serializable fit diagnostics.

    Parameters
    ----------
    train_returns : array, shape (nmonths, nassets)
        Finite monthly arithmetic returns in decimal units. At least two
        months and one asset are required. The rolling caller is responsible
        for using exactly the prescribed 60 prior months and for consistent
        sample alignment. Missing observations are rejected, never imputed.

    Raises
    ------
    ValueError
        For malformed, empty or nonfinite input.
    RuntimeError
        If SLSQP fails, yields invalid weights, or loses to its equal-weight
        feasible initialization beyond numerical tolerance.

    Notes
    -----
    Fixed equal-weight initialization makes all-zero returns deterministic:
    the returned weights are equal weights. Constant/duplicate asset columns
    are allowed. The fixed diagonal shrinkage does not require matrix inversion.
    """
    try:
        returns = np.asarray(train_returns, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('train_returns must be a finite numeric 2D array') from exc
    if returns.ndim != 2:
        raise ValueError('train_returns must have shape (nmonths, nassets)')
    nmonths, nassets = returns.shape
    if nmonths < 2 or nassets < 1:
        raise ValueError('at least two months and one asset are required')
    if not np.isfinite(returns).all():
        raise ValueError('train_returns contains NaN or infinite values')

    monthly_mean = returns.mean(axis=0)
    centered = returns - monthly_mean
    sample_covariance = centered.T @ centered / (nmonths - 1)
    diagonal = np.diag(np.diag(sample_covariance))
    annual_mean = MONTHS_PER_YEAR * monthly_mean
    annual_covariance = MONTHS_PER_YEAR * (
        (1.0 - COVARIANCE_SHRINKAGE) * sample_covariance
        + COVARIANCE_SHRINKAGE * diagonal
    )
    # Explicitly symmetric after floating-point accumulation.
    annual_covariance = 0.5 * (annual_covariance + annual_covariance.T)
    if not np.isfinite(annual_mean).all() or not np.isfinite(annual_covariance).all():
        raise ValueError('training moments overflowed; check return units')

    def objective(weights: np.ndarray) -> float:
        return float(annual_mean @ weights - 0.5 * GAMMA * weights @ annual_covariance @ weights)

    def objective_gradient(weights: np.ndarray) -> np.ndarray:
        return annual_mean - GAMMA * (annual_covariance @ weights)

    equal_weights = np.full(nassets, 1.0 / nassets, dtype=np.float64)
    result = minimize(
        fun=lambda weights: -objective(weights),
        x0=equal_weights,
        jac=lambda weights: -objective_gradient(weights),
        method='SLSQP',
        bounds=[(0.0, 1.0)] * nassets,
        constraints=[{
            'type': 'eq',
            'fun': lambda weights: float(weights.sum() - 1.0),
            'jac': lambda weights: np.ones(nassets, dtype=np.float64),
        }],
        options={'ftol': 1e-12, 'maxiter': 2000, 'disp': False},
    )
    if not result.success:
        raise RuntimeError(f'SLSQP allocation failed (status={result.status}): {result.message}')
    raw_weights = np.asarray(result.x, dtype=np.float64)
    raw_simplex_error = float(abs(raw_weights.sum() - 1.0))
    if (
        not np.isfinite(raw_weights).all()
        or raw_weights.min() < -1e-8
        or raw_weights.max() > 1.0 + 1e-8
        or raw_simplex_error > 1e-8
    ):
        raise RuntimeError('SLSQP returned invalid simplex weights')
    # Remove only floating-point boundary noise; never repair failed solutions.
    weights = np.clip(raw_weights, 0.0, 1.0)
    weights /= weights.sum()
    value = objective(weights)
    equal_weight_value = objective(equal_weights)
    if value < equal_weight_value - 1e-10:
        raise RuntimeError('SLSQP objective is worse than equal-weight initialization')

    gradient = objective_gradient(weights)
    multiplier = float(weights @ gradient)
    positive = weights > 1e-8
    active_error = float(np.max(np.abs(gradient[positive] - multiplier)))
    inactive_error = (
        float(max(0.0, np.max(gradient[~positive] - multiplier)))
        if (~positive).any() else 0.0
    )
    annual_variance = float(weights @ annual_covariance @ weights)
    if annual_variance < -1e-12:
        raise RuntimeError('Shrunk sample covariance produced negative variance')
    diagnostics = {
        'success': bool(result.success),
        'status': int(result.status),
        'message': str(result.message),
        'iterations': int(result.nit),
        'n_months': int(nmonths),
        'n_assets': int(nassets),
        'gamma': GAMMA,
        'covariance_shrinkage_to_diagonal': COVARIANCE_SHRINKAGE,
        'annualization': MONTHS_PER_YEAR,
        'covariance_ddof': 1,
        'annual_expected_return': float(annual_mean @ weights),
        'annual_volatility': float(np.sqrt(max(0.0, annual_variance))),
        'objective': value,
        'equal_weight_objective': equal_weight_value,
        'objective_gain_over_equal_weight': value - equal_weight_value,
        'raw_simplex_error': raw_simplex_error,
        'simplex_error': float(abs(weights.sum() - 1.0)),
        'min_weight': float(weights.min()),
        'max_weight': float(weights.max()),
        'positive_weight_count': int(positive.sum()),
        'kkt_residual': max(active_error, inactive_error),
        'lagrange_multiplier': multiplier,
        'all_zero_training_returns': bool(np.count_nonzero(returns) == 0),
    }
    return weights, diagnostics
