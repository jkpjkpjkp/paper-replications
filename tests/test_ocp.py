import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import betaln

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "upstream"))
from ocp import UniversalPortfolio, run, metrics, controller


def test_algorithm1_wealth_against_beta_integral():
    # An independent closed-form integral checks the portfolio recurrence.
    model = UniversalPortfolio(0.05)
    scores = np.r_[np.zeros(10), np.arange(50), np.full(20, 1000.0)]
    for n, score in enumerate(scores, 1):
        radius = model.predict()
        assert radius >= 0
        model.update(score)
        misses = model.misses
        expected = (betaln(misses + 0.5, n - misses + 0.5) - betaln(0.5, 0.5)
                    - misses * np.log(model.alpha) - (n - misses) * np.log(1 - model.alpha))
        assert model.log_wealth == pytest.approx(expected, abs=2e-12)


@pytest.mark.parametrize("method", ["UP", "UP-paper", "KT", "SFOGD", "DtACI"])
def test_predict_before_observe(method):
    a = np.abs(np.random.default_rng(3).normal(10, 4, 100))
    b = a.copy()
    b[50:] *= 100
    np.testing.assert_array_equal(run(a, method)[:51], run(b, method)[:51])


def test_algorithm1_class_and_kernel():
    s = np.maximum(0, 7 + 10 * np.sin(np.arange(500) / 20))
    p = UniversalPortfolio()
    expected = []
    for score in s:
        expected.append(p.predict())
        p.update(score)
    np.testing.assert_allclose(run(s, "UP-paper"), expected, rtol=1e-11)


def test_zero_score_long_stream_remains_finite():
    r = run(np.zeros(30000), "UP-paper")
    assert np.isfinite(r).all()
    assert np.all(r >= 0)


@pytest.mark.parametrize("method,clsname,kwargs", [
    ("UP", "UniversalPortfolio", {"alpha": 0.065}),  # cancel upstream's -0.015
    ("KT", "KrichevskyTrofimov", {"alpha": 0.05}),
    ("SFOGD", "ScaleFreeOnlineGradientDescent", {"alpha": 0.05, "lr": 25}),
    ("DtACI", "DynamicallyTunedAdaptiveConformalInference", {"alpha": 0.05}),
])
def test_accelerated_kernel_matches_authors(method, clsname, kwargs):
    from core import methods
    p = getattr(methods, clsname)(**kwargs)
    scores = np.r_[np.maximum(0, 7 + 10 * np.sin(np.arange(80) / 10)),
                   np.zeros(10), np.full(10, 7), np.arange(40)]
    expected = []
    for s in scores:
        expected.append(p.predict())
        p.update(s)
    np.testing.assert_allclose(run(scores, method, lr=25), expected, atol=2e-8, rtol=2e-9)


def test_infinite_and_empty_intervals():
    m = metrics([0, 1, 3, 4], [-1, 0, np.inf, np.inf])
    assert m["coverage"] == 0.5
    assert m["longest_error_streak"] == 2
    assert m["mean_width"] == np.inf
    assert m["q95_width"] == np.inf
    assert m["infinite_fraction"] == 0.5


@pytest.mark.parametrize("ki", [0.0, 10.0])
def test_controllers_match_original_pid_reference(ki):
    # Load the downloaded, unedited reference. A local NumPy proxy restores
    # its old np.infty spelling without modifying NumPy or the source.
    import importlib.util
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location("reference_pid", ROOT / "reference/pid/methods.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = SimpleNamespace(**{**vars(np), "infty": np.inf})
    module.tqdm = lambda x: x
    scores = np.maximum(0, 7 + 10 * np.sin(np.arange(500) / 10))
    expected = module.quantile_integrator_log(scores, 0.05, 0.1, 1.0, ki, 1, 100)["q"]
    actual = controller(scores, lr=0.1, window=100, ki=ki)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
    changed = scores.copy()
    changed[250:] *= 10
    np.testing.assert_array_equal(actual[:251], controller(changed, lr=0.1, window=100, ki=ki)[:251])
