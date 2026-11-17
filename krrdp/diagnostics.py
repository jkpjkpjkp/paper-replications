"""Exact finite-state diagnostics for Eq. (23), separate from option tables."""
from dataclasses import dataclass
import numpy as np
from .kernel import KRR


@dataclass
class Chain:
    states: np.ndarray
    transition: np.ndarray
    mu: list
    terminal: np.ndarray
    steps: int
    discount: float
    moves: np.ndarray


def example_chain(steps=9, state_count=61, rate=.05):
    states = np.linspace(60., 140., state_count)
    # Reflected/clipped boundary random walk, with exact forward marginals.
    moves = np.clip(np.arange(state_count)[:, None]+np.array([-1, 0, 1]), 0, state_count-1)
    p = np.zeros((state_count, state_count))
    for k, weight in enumerate([.25, .5, .25]):
        p[np.arange(state_count), moves[:, k]] += weight
    mu0 = np.exp(-.5*((states-100)/12)**2)
    mu0 /= mu0.sum()
    mu = [mu0]
    for _ in range(steps):
        mu.append(mu[-1]@p)
    return Chain(states, p, mu, np.maximum(100-states, 0)/100, steps,
                 np.exp(-rate/steps), moves)


def error_decomposition(n=128, m=32, seed=0, chain=None):
    c = example_chain() if chain is None else chain
    rng = np.random.default_rng(seed)
    v = [None]*(c.steps+1)
    w = [None]*(c.steps+1)
    v[-1] = w[-1] = c.terminal.copy()
    stages = []
    for t in reversed(range(c.steps)):
        # Exact Bellman, empirical Bellman and fitted regressor are separate.
        v[t] = np.maximum(c.terminal, c.discount*c.transition@v[t+1])
        counts = np.bincount(rng.choice(3, m, p=[.25, .5, .25]), minlength=3)/m
        empirical_q = c.discount*np.sum(w[t+1][c.moves]*counts, axis=1)
        target = np.maximum(c.terminal, empirical_q)
        sampled = rng.choice(len(c.states), n, p=c.mu[t])
        fit = KRR.fit(c.states[sampled, None], target[sampled], lengthscale=15.,
                      regularization=1e-6, clip=1.)
        w[t] = fit.predict(c.states[:, None])
        exact_bellman_w = np.maximum(c.terminal, c.discount*c.transition@w[t+1])
        reg = w[t]-target
        mc = target-exact_bellman_w
        prop = exact_bellman_w-v[t]
        error = w[t]-v[t]
        sq = lambda e, measure: float(measure@(e*e))
        e = sq(error, c.mu[t])
        terms = [sq(z, c.mu[t]) for z in [reg, mc, prop]]
        next_e = sq(w[t+1]-v[t+1], c.mu[t+1])
        stages.append(dict(stage=t, total_mse=e, regression_mse=terms[0],
                           mc_mse=terms[1], propagation_mse=terms[2],
                           triangle_bound=3*sum(terms),
                           contraction_bound=c.discount**2*next_e,
                           identity_residual=float(np.max(np.abs(error-reg-mc-prop))),
                           signed_cross_terms=e-sum(terms)))
    return dict(n=n, m=m, seed=seed, stages=stages,
                initial_mse=stages[-1]['total_mse'])


def mc_scaling(repetitions=2000, sample_sizes=(8,16,32,64,128,256), seed=37):
    """Unbiased conditional expectation MSE has exact variance/M reference."""
    c = example_chain()
    f = np.sin(c.states/10)+c.terminal
    next_values = f[c.moves]
    exact = c.transition@f
    variance = np.sum(next_values**2*np.array([.25,.5,.25]), axis=1)-exact**2
    coefficient = float(c.mu[0]@variance)
    rng = np.random.default_rng(seed)
    rows = []
    for m in sample_sizes:
        counts = rng.multinomial(m, [.25,.5,.25], size=repetitions)/m
        errors = counts@next_values.T-exact[None, :]
        squared = errors**2@c.mu[0]
        rows.append(dict(m=m, mse=float(squared.mean()),
                         se=float(squared.std(ddof=1)/np.sqrt(repetitions)),
                         exact_mse=coefficient/m, repetitions=repetitions))
    return rows
