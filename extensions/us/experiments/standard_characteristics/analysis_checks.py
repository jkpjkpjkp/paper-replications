"""Known-signal regression checks, including the zero-investment RF convention."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from experiments.standard_characteristics.analyze import fit_alpha, NAMES


def main():
    rng = np.random.default_rng(713)
    x = rng.normal(0, .01, (2000, 6))
    beta = np.array([.1, -.2, .3, .15, -.1, .4])
    y = .0003 + x @ beta + rng.normal(0, .001, len(x))
    frame = pd.DataFrame(x, columns=NAMES["FF5+UMD"])
    frame["return_net"], frame["rf"] = y, .02
    ours = fit_alpha(frame)
    reference = sm.OLS(y, np.column_stack([np.ones(len(x)), x])).fit(cov_type="HAC",
        cov_kwds={"maxlags": 20, "kernel": "bartlett", "use_correction": True}, use_t=True)
    np.testing.assert_allclose(ours["annual_alpha"], reference.params[0] * 252)
    np.testing.assert_allclose(ours["annual_alpha_se"], reference.bse[0] * 252)
    np.testing.assert_allclose(ours["p"], reference.pvalues[0])
    assert abs(ours["annual_alpha"] - .0756) < .02
    frame["rf"] = 999  # RF must not be subtracted from a zero-investment spread.
    assert fit_alpha(frame) == ours
    half = frame.copy()
    half["return_net"] *= .5
    scaled = fit_alpha(half)
    np.testing.assert_allclose(scaled["annual_alpha"], ours["annual_alpha"] * .5)
    np.testing.assert_allclose(scaled["t"], ours["t"])
    for name in NAMES:
        assert np.isfinite(fit_alpha(frame, name)["p"])
    print("Analysis checks passed: known alpha, HAC covariance, gross scaling, no duplicate RF subtraction")


if __name__ == "__main__":
    main()
