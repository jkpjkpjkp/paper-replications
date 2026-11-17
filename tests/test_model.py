import json

import numpy as np
import pytest
import torch
import xgboost as xgb
from torch.autograd import gradcheck

from arboreal.model import ArborCell, ArborealNetwork, SymbolicTree


def stump(threshold=0.0, left=-2.0, right=3.0):
    return SymbolicTree({"nodeid": 0, "split": "f0", "split_condition": threshold,
                         "yes": 1, "no": 2, "missing": 1,
                         "children": [{"nodeid": 1, "leaf": left},
                                      {"nodeid": 2, "leaf": right}]})


def unbalanced():
    return SymbolicTree({"nodeid": 0, "split": "f0", "split_condition": 0.0,
                         "yes": 1, "no": 2, "missing": 1,
                         "children": [{"nodeid": 1, "leaf": -1.0},
                                      {"nodeid": 2, "split": "f1", "split_condition": 0.0,
                                       "yes": 3, "no": 4, "missing": 3,
                                       "children": [{"nodeid": 3, "leaf": 2.0},
                                                    {"nodeid": 4, "leaf": 5.0}]}]})


def test_algorithm1_path_weights_and_signs():
    _, _, P, _, _, _ = unbalanced().arrays(2)
    np.testing.assert_array_equal(P, [[-1, 1, 1], [0, -0.5, 0.5]])


def test_literal_equation2_reverses_stump_routing():
    X = torch.tensor([[-1.0], [1.0]])
    correct = ArborCell(stump(), 1)
    literal = ArborCell(stump(), 1, routing="equation2")
    assert torch.equal(correct.leaf_probabilities(X).argmax(-1).flatten(), torch.tensor([0, 1]))
    assert torch.equal(literal.leaf_probabilities(X).argmax(-1).flatten(), torch.tensor([1, 0]))


def test_soft_forward_matches_direct_matrix_equations():
    cell = ArborCell(unbalanced(), 2, tau1=2, tau2=3)
    X = torch.tensor([[-1.0, 0.5], [0.2, -0.1], [1.0, 1.0]])
    W = torch.eye(2)
    d = torch.tanh(2 * (X @ W.T - cell.f[0]))
    alpha = torch.softmax(3 * d @ cell.P[0], dim=-1)
    torch.testing.assert_close(cell(X), alpha @ cell.v[0])


def test_gradient_against_finite_difference_for_thresholds_and_values():
    cell = ArborCell(unbalanced(), 2, tau1=2, tau2=3).double()
    X = torch.tensor([[0.01, -0.02], [-0.03, 0.04]], dtype=torch.float64,
                     requires_grad=True)
    params = {"f": cell.f, "v": cell.v}

    def functional(X, f, v):
        return torch.func.functional_call(cell, {"f": f, "v": v}, (X,))

    assert gradcheck(functional, (X, params["f"], params["v"]), eps=1e-6, atol=1e-5)


def test_constant_tree_and_padding_have_correct_probability_mass():
    model = ArborealNetwork([unbalanced(), stump(), SymbolicTree({"nodeid": 0, "leaf": 7.0})], 2)
    X = torch.tensor([[-2., 2.], [2., -2.]])
    alpha = model.leaf_probabilities(X)
    torch.testing.assert_close(alpha.sum(-1), torch.ones(2, 3))
    assert (alpha[:, 1, 2] == 0).all()
    assert (alpha[:, 2, 0] == 1).all()
    torch.testing.assert_close(model(X),
                               ArborCell(unbalanced(), 2)(X) + ArborCell(stump(), 2)(X) + 7)


def test_decompilation_updates_rules_and_round_trip(tmp_path):
    model = ArborealNetwork([unbalanced(), stump()], 2)
    with torch.no_grad():
        model.f[0, 0] = 0.25
        model.v[0, 0] = -4.0
    trees = model.decompile()
    assert trees[0].root["split_condition"] == 0.25
    assert trees[0].root["children"][0]["leaf"] == -4.0
    rebuilt = ArborealNetwork(trees, 2)
    X = torch.tensor([[-0.1, 0.1], [0.3, -0.4]])
    torch.testing.assert_close(model(X), rebuilt(X), rtol=0, atol=0)
    path = tmp_path / "rules.json"
    model.export_rules(path)
    rules = json.loads(path.read_text())
    assert rules["trees"][0]["rules"][0]["if"][0]["threshold"] == 0.25
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    torch.testing.assert_close(model(X), ArborealNetwork.load(checkpoint)(X), rtol=0, atol=0)


@pytest.mark.parametrize("task", ["binary:logistic", "reg:squarederror"])
def test_xgboost_compile_hard_matches_with_nondefault_intercept(task):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(600, 4)).astype(np.float32)
    score = X[:, 0] + X[:, 1] * X[:, 2]
    y = (score > 0.8).astype(np.float32) if task == "binary:logistic" else score + 2.5
    booster = xgb.train({"objective": task, "max_depth": 3, "eta": 0.2,
                         "base_score": 0.3 if task == "binary:logistic" else 2.5,
                         "nthread": 1}, xgb.DMatrix(X, label=y), num_boost_round=7)
    model = ArborealNetwork.from_xgboost(booster, X[:100])
    np.testing.assert_allclose(model.predict_numpy(X, hard=True),
                               booster.predict(xgb.DMatrix(X), output_margin=True), atol=2e-6)


def test_hard_threshold_equality_and_missing_branch():
    tree = stump()
    np.testing.assert_array_equal(tree.predict([[-1], [0], [1], [np.nan]]), [-2, 3, 3, -2])
    model = ArborCell(tree, 1)
    with pytest.raises(ValueError, match="finite"):
        model(torch.tensor([[float("nan")]]))


def test_finite_temperature_is_not_lossless_prediction_equivalence():
    model = ArborCell(stump(), 1)
    X = np.array([[-1e-7], [1e-7]], dtype=np.float32)
    soft = model.predict_numpy(X)
    hard = model.predict_numpy(X, hard=True)
    assert np.max(np.abs(soft - hard)) > 2.0


def test_saturated_decisions_select_correct_leaf_on_unbalanced_tree():
    X = torch.tensor([[-10., -10.], [-10., 10.], [10., -10.], [10., 10.]])
    model = ArborCell(unbalanced(), 2)
    chosen = model.leaf_probabilities(X).argmax(-1).flatten()
    torch.testing.assert_close(chosen, torch.tensor([0, 0, 1, 2]))


def test_leaf_values_include_xgboost_shrinkage():
    # A stump's dumped leaf values must not be multiplied by eta again.
    X = np.array([[-2], [-1], [1], [2]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    booster = xgb.train({"objective": "reg:squarederror", "max_depth": 1,
                         "eta": 0.1, "base_score": 0.5, "nthread": 1},
                        xgb.DMatrix(X, label=y), num_boost_round=1)
    model = ArborealNetwork.from_xgboost(booster, X)
    assert 0 < model.v.abs().max() < 0.1
    np.testing.assert_allclose(model.predict_numpy(X, hard=True),
                               booster.predict(xgb.DMatrix(X), output_margin=True), atol=1e-7)
