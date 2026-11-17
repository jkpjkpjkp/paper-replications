import numpy as np
from scripts.validate import lp_objective, small_problem_audit


def test_fixed_subgradient_gap_formula_against_lp():
    rng = np.random.default_rng(781)
    for lam in [0., .025, .2]:
        for m in [4, 9]:
            q = rng.uniform(.8, 1.2, m)
            h = rng.dirichlet(np.ones(m))
            actual, _ = lp_objective(q, h, lam, range(m))
            formula = -q@h-np.sum(h*np.maximum(q.max()-q-2*lam, 0))
            assert abs(actual-formula) < 1e-10


def test_exact_penalty_global_equivalence_on_small_oracle():
    result = small_problem_audit()
    assert result["penalty"] > result["sufficient_bound"]
    assert result["global_equivalence_error"] < 1e-9
    assert result["psga_support"] <= result["k"]
    assert result["solver"]["converged"]
