"""APO sizing from physical shock/monthly-volatility predictions."""
import numpy as np
from limt.portfolio import linear_norm


def physical_weights(pred, ids, market, member, t, r=.1, participation=.02, capital=1e9, volume_multiplier=1.):
    volume = np.asarray(market[ids, t - 4:t + 1, 1], dtype=float)
    assert np.isfinite(volume).all() and (volume > 0).all()
    logmean = np.log(volume).mean(1)
    with np.errstate(over="raise", invalid="raise"):
        amount = np.exp(pred[:, 1] + logmean) * market[ids, t, 6] * volume_multiplier
    raw = r * linear_norm(-pred[:, 2]) + (1 - r) * linear_norm(amount)
    cap = participation * market[ids, t, 1] * market[ids, t, 6] * volume_multiplier / capital
    return np.minimum(raw, cap)
