from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Market:
    dimension: int = 2
    spot: float = 100.0
    strike: float = 100.0
    rate: float = 0.05
    volatility: float = 0.2
    correlation: float = 0.2
    maturity: float = 1.0
    steps: int = 9

    def __post_init__(self):
        if self.dimension < 1 or self.steps < 1 or self.maturity <= 0:
            raise ValueError("Positive dimension, steps and maturity required")
        if not 0 <= self.correlation < 1:
            raise ValueError("Common-factor sampler supports 0 <= correlation < 1")
        if self.spot <= 0 or self.strike <= 0 or self.volatility <= 0:
            raise ValueError("Positive spot, strike and volatility required")

    @property
    def dt(self):
        return self.maturity / self.steps

    @property
    def discount(self):
        return np.exp(-self.rate * self.dt)

    @property
    def geometric_variance(self):
        return self.volatility**2 * (self.correlation + (1-self.correlation)/self.dimension)

    def normals(self, rng, n):
        """Exact equicorrelation via a shared independent Gaussian factor."""
        return (np.sqrt(self.correlation)*rng.standard_normal((n, 1))
                + np.sqrt(1-self.correlation)*rng.standard_normal((n, self.dimension)))

    def marginal(self, rng, n, stage):
        time = stage * self.dt
        if stage == 0:
            return np.full((n, self.dimension), self.spot)
        return self.spot * np.exp((self.rate-0.5*self.volatility**2)*time
                                 + self.volatility*np.sqrt(time)*self.normals(rng, n))

    def factors(self, rng, n):
        return np.exp((self.rate-0.5*self.volatility**2)*self.dt
                      + self.volatility*np.sqrt(self.dt)*self.normals(rng, n))

    def asdict(self):
        return asdict(self)


def payoff(x, kind, strike=100.0):
    if kind == "geometric_put":
        return np.maximum(strike - np.exp(np.log(x).mean(axis=-1)), 0.0)
    if kind == "max_call":
        return np.maximum(np.max(x, axis=-1) - strike, 0.0)
    raise ValueError(f"Unknown payoff {kind}")
