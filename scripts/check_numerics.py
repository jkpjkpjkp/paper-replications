"""Fast independent checks for the numerical primitives used by the rerun.

The historical README referred to an absent tests/ directory. These checks
validate observable mathematical contracts and do not replace the full saved
experiment audit in verify.py.
"""
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from krrdp.kernel import KRR, rbf
from krrdp.market import Market
from krrdp.reference import geometric_european, geometric_tree, max_call_european


def main():
    x = np.array([[-1.0], [0.0], [1.0], [2.0]])
    y = np.array([0.2, -0.5, 0.8, 1.1])
    regularization = 0.03
    model = KRR.fit(x, y, lengthscale=1.3, regularization=regularization)
    kernel = rbf(x, x, 1.3)
    expected = kernel @ np.linalg.solve(kernel + len(x) * regularization * np.eye(len(x)), y)
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-12, atol=1e-12)
    repeated = KRR.fit(np.ones((4, 1)), y, regularization=regularization)
    np.testing.assert_allclose(repeated.predict(np.ones((1, 1))), y.mean() / (1 + regularization))

    market = Market(dimension=1)
    sigma_t = market.volatility * np.sqrt(market.maturity)
    d1 = (np.log(market.spot / market.strike) +
          (market.rate + market.volatility**2 / 2) * market.maturity) / sigma_t
    d2 = d1 - sigma_t
    call = market.spot * ndtr(d1) - market.strike * np.exp(-market.rate * market.maturity) * ndtr(d2)
    actual, _ = max_call_european(market)
    np.testing.assert_allclose(actual, call, rtol=1e-9, atol=1e-9)
    put = market.strike * np.exp(-market.rate * market.maturity) * ndtr(-d2) - market.spot * ndtr(-d1)
    np.testing.assert_allclose(geometric_european(market), put, rtol=1e-12, atol=1e-12)
    assert geometric_tree(market, subdivisions=40) >= put - 0.02
    print("PASS: regularized KRR, coincident inputs, Black-Scholes call/put, exercise lower bound")


if __name__ == "__main__":
    main()
