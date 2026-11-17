"""Calendar and information-boundary checks for the monthly adapter."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('cz_runner', Path(__file__).resolve().parents[1]/'scripts/run_cz202.py')
cz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cz)


def test_monthly_history_keeps_gaps_and_excludes_future():
    panel = cz.MonthlyPanel.__new__(cz.MonthlyPanel)
    panel.lookup = np.arange(1, 41, dtype=np.int32).reshape(20,2)
    panel.lookup[8,0] = 0  # A missing month must not compress calendar time.
    panel.stocks = [np.array([0,1]) for _ in range(20)]
    panel.rows = [np.array([2*t,2*t+1]) for t in range(20)]
    panel.x = np.arange(41*202,dtype=np.float32).reshape(41,202)
    panel.x[0] = 0
    panel.y = np.arange(40,dtype=np.float32)
    x,y = panel.batch(16)
    assert x.shape == (2,16,202)
    np.testing.assert_array_equal(x[0,7], np.zeros(202))
    np.testing.assert_array_equal(x[1,-1],panel.x[34])
    panel.lookup[17:] = 0
    panel.y[:] = -999  # Neither future rows nor labels enter the input.
    after,_ = panel.batch(16)
    np.testing.assert_array_equal(x,after)
    assert y.shape == (2,)
