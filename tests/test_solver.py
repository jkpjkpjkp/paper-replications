from dataclasses import replace
import numpy as np
import cvxpy as cp
from stco.solver import (PSGAConfig, project_simplex, prox_exact, prox_admm,
                         solve_psga, objective, _topk)


def test_prox_against_independent_convex_solver():
    rng = np.random.default_rng(20261106)
    for m in [3, 10, 36]:
        for tau in [0., 1e-5, .05, 1.]:
            h = rng.dirichlet(np.ones(m))
            z = rng.normal(0, .4, m)
            actual = prox_exact(z, h, tau)
            w = cp.Variable(m)
            problem = cp.Problem(cp.Minimize(.5*cp.sum_squares(w-z)
                                             + tau*cp.norm1(w-h)),
                                 [w >= 0, cp.sum(w) == 1])
            problem.solve(solver="CLARABEL", tol_gap_abs=1e-11,
                          tol_feas=1e-11, tol_gap_rel=1e-11)
            assert problem.status == "optimal"
            np.testing.assert_allclose(actual, w.value, atol=3e-6)
            assert abs(actual.sum()-1) < 1e-11
            assert actual.min() >= 0


def test_admm_solves_fixed_anchor_subproblem():
    rng = np.random.default_rng(171)
    for m in [5, 24]:
        for lam in [0., .025, .05]:
            h = rng.dirichlet(np.ones(m))
            anchor = rng.dirichlet(np.ones(m))
            q = rng.uniform(.85, 1.15, m)
            v = _topk(anchor, min(3, m))
            actual, info = prox_admm(anchor, q, h, v, lam=lam)
            expected = prox_exact(anchor+.0025*(q+8e-6*v), h, .0025*lam)
            assert info["converged"]
            np.testing.assert_allclose(actual, expected, atol=2e-9)


def test_event_skipping_matches_direct_psga():
    rng = np.random.default_rng(288)
    for i in range(32):
        m = int(rng.integers(3, 15))
        cfg = PSGAConfig(k=int(rng.integers(1, m+1)), alpha=.03,
                         penalty=float(rng.choice([8e-6, .01, .15])),
                         tol=1e-9, max_iter=200_000)
        h = rng.dirichlet(np.ones(m))
        q = rng.uniform(.8, 1.2, m)
        lam = float(rng.choice([0, .025, .1]))
        fast, info = solve_psga(q, h, lam, cfg)
        slow, direct_info = solve_psga(q, h, lam, replace(cfg, accelerate=False))
        assert info["converged"] and direct_info["converged"]
        np.testing.assert_allclose(fast, slow, atol=2e-8)
        assert abs(info["iterations"]-direct_info["iterations"]) <= 3


def test_paper_step_size_and_zero_prediction():
    h = np.array([.05, .15, .2, .25, .35])
    q = np.array([1.01, .985, 1.05, .995, .96])
    cfg = PSGAConfig(k=2)
    fast, info = solve_psga(q, h, .025, cfg)
    slow, si = solve_psga(q, h, .025, replace(cfg, accelerate=False))
    np.testing.assert_allclose(fast, slow, atol=2e-8)
    assert info["prox_evaluations"] < si["prox_evaluations"]/10
    w, info = solve_psga(np.ones(5), h, .025, cfg)
    assert info["converged"]
    np.testing.assert_allclose(w, h, atol=1e-7)


def test_objective_descent_for_exact_psga():
    rng = np.random.default_rng(117)
    m, k, alpha, lam, c = 8, 3, .1, .025, .001
    q, h = rng.uniform(.8, 1.2, m), rng.dirichlet(np.ones(m))
    w = np.ones(m)/m
    for _ in range(500):
        old = objective(w, q, h, lam, c, k)
        new = prox_exact(w+alpha*(q+c*_topk(w, k)), h, alpha*lam)
        assert objective(new, q, h, lam, c, k) <= old + 2e-12
        w = new


def test_simplex_projection():
    np.testing.assert_allclose(project_simplex(np.array([.2, -.5, 2.])), [0, 0, 1])
    np.testing.assert_allclose(project_simplex(np.ones(4)), np.ones(4)/4)
