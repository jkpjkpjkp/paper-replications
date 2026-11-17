"""Executable reconstruction of specified STABLE equations, not author code.

This module contains equation kernels. The independently reconstructed learned
denoiser and market backtest are in diffusion_model.py and run_empirical.py.
Corrections are explicit opt-ins. All returns and covariances passed to BL must
refer to the same investment horizon.
"""
from __future__ import annotations

import numpy as np


def guided_noise(macro_noise, full_noise, gate):
    """Section 3.4; gate must be broadcastable to the noise tensors."""
    return np.asarray(macro_noise) + np.asarray(gate) * (
        np.asarray(full_noise) - np.asarray(macro_noise)
    )


def ddim_step(state, predicted_noise, alpha_bar, previous_alpha_bar):
    """Section 3.3 deterministic (eta=0) DDIM step, including skipped steps."""
    if not 0 < alpha_bar <= previous_alpha_bar <= 1:
        raise ValueError("Require 0 < alpha_bar <= previous_alpha_bar <= 1")
    clean = (state - np.sqrt(1 - alpha_bar) * predicted_noise) / np.sqrt(alpha_bar)
    previous = (np.sqrt(previous_alpha_bar) * clean
                + np.sqrt(1 - previous_alpha_bar) * predicted_noise)
    return previous, clean


def kalman_embedding(macro, returns, initial_mean, initial_covariance,
                     process_covariance, observation_variance):
    """Causal random-walk-beta Kalman filter; all unspecified parameters explicit.

    Models y_t = m_t @ beta_t + noise, beta_t = beta_(t-1) + innovation.
    This is one standard realization of the paper's under-specified filter.
    The beta output at t includes y_t, so trading must occur after y_t is known.
    """
    mean = np.array(initial_mean, dtype=float, copy=True)
    covariance = np.array(initial_covariance, dtype=float, copy=True)
    macro, returns = np.asarray(macro), np.asarray(returns)
    if observation_variance <= 0 or macro.shape[0] != len(returns):
        raise ValueError("Invalid observation variance or input length")
    means, covariances = [], []
    identity = np.eye(len(mean))
    for x, y in zip(macro, returns):
        covariance = covariance + process_covariance
        innovation_variance = x @ covariance @ x + observation_variance
        gain = covariance @ x / innovation_variance
        mean = mean + gain * (y - x @ mean)
        # Joseph form preserves PSD better than subtracting rank-one matrices.
        residual = identity - np.outer(gain, x)
        covariance = (residual @ covariance @ residual.T
                      + observation_variance * np.outer(gain, gain))
        means.append(mean.copy())
        covariances.append(covariance.copy())
    return np.asarray(means), np.asarray(covariances)


def view_moments(log_paths):
    """Eqs. (3)-(4); shape [joint sample, day, stock].

    Convert daily log returns to horizon simple returns using exp(sum)-1.
    The joint-sample index MUST represent an explicitly defined cross-asset
    coupling. Per-stock marginals alone do not identify this covariance.
    """
    paths = np.asarray(log_paths, dtype=float)
    if paths.ndim != 3 or paths.shape[0] < 2 or not np.isfinite(paths).all():
        raise ValueError("Need finite [k >= 2, horizon, assets] log paths")
    cumulative = np.expm1(paths.sum(axis=1))
    mean = cumulative.mean(axis=0)
    centered = cumulative - mean
    covariance = centered.T @ centered / (len(cumulative) - 1)
    return mean, covariance


def ridge_covariance(covariance, relative_ridge):
    """Explicit repair C + relative_ridge * trace(C)/S * I; not in the paper."""
    covariance = np.asarray(covariance, dtype=float)
    scale = np.trace(covariance) / len(covariance)
    if relative_ridge <= 0 or scale <= 0:
        raise ValueError("Positive ridge and positive average variance required")
    return covariance + relative_ridge * scale * np.eye(len(covariance))


def require_positive_definite(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14)):
        raise ValueError("A finite symmetric covariance matrix is required")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 1e-12 * max(abs(eigenvalues[-1]), np.finfo(float).tiny):
        raise ValueError("Singular or ill-conditioned covariance: explicit repair required")


def bl_posterior(prior_mean, prior_covariance, view_mean, view_covariance):
    """Eqs. (6)-(7), checked SPD inputs and linear solves."""
    require_positive_definite(prior_covariance)
    require_positive_definite(view_covariance)
    identity = np.eye(len(prior_mean))
    prior_precision = np.linalg.solve(prior_covariance, identity)
    view_precision = np.linalg.solve(view_covariance, identity)
    precision = prior_precision + view_precision
    information = prior_precision @ prior_mean + view_precision @ view_mean
    mean = np.linalg.solve(precision, information)
    covariance = np.linalg.solve(precision, identity)
    return mean, (covariance + covariance.T) / 2


def literal_budget_weights(mean, covariance):
    """Paper's normalization; NOT a maximum-Sharpe guarantee for negative sum."""
    direction = np.linalg.solve(covariance, mean)
    denominator = direction.sum()
    if abs(denominator) <= 1e-12 * max(np.linalg.norm(direction, 1), 1):
        raise ValueError("Tangency direction has zero/near-zero budget")
    return direction / denominator


def checked_sharpe_weights(mean, covariance):
    """Only return the claimed maximizer when its positivity condition holds.

    With a nonpositive denominator, reject rather than silently inventing a
    cash asset, long-only rule, gross cap, or alternative optimization problem.
    """
    require_positive_definite(covariance)
    direction = np.linalg.solve(covariance, mean)
    if direction.sum() <= 1e-12 * max(np.linalg.norm(direction, 1), 1):
        raise ValueError("Maximum-Sharpe normalization requires a positive denominator")
    return direction / direction.sum()


def sharpe(mean, covariance, weights):
    return float(weights @ mean / np.sqrt(weights @ covariance @ weights))
