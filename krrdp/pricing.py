from dataclasses import asdict, dataclass, field
from time import perf_counter
import numpy as np
from .kernel import KRR
from .market import Market, payoff


@dataclass(frozen=True)
class Training:
    n: int = 200
    m: int = 50
    lengthscales: tuple = (40.0, 80.0)
    regularization: float = 1e-6
    max_centers: int | None = None
    validation_n: int = 128
    batch_size: int = 2048
    max_call_clip: float = 10000.0
    mc_mode: str = "shared"
    initial_box: tuple | None = None

    def asdict(self):
        return asdict(self)


@dataclass
class Result:
    kind: str
    market: Market
    training: Training
    seed: int
    price: float
    seconds: float
    stages: list
    models: list = field(repr=False)

    def asdict(self):
        return dict(kind=self.kind, market=self.market.asdict(), training=self.training.asdict(),
                    seed=self.seed, price=self.price, seconds=self.seconds, stages=self.stages)


def continuation(x, factors, next_model, market, kind, batch_size=2048):
    """Algorithm 1: the SAME M innovations are shared across all input states."""
    result = np.empty(len(x))
    # Keep (states * M * d) and kernel workspaces bounded.
    m = factors.shape[-2]
    state_batch = max(1, batch_size // m)
    for start in range(0, len(x), state_batch):
        stop = min(start+state_batch, len(x))
        innovations = factors[None, :, :] if factors.ndim == 2 else factors[start:stop]
        children = (x[start:stop, None, :] * innovations).reshape(-1, market.dimension)
        values = (payoff(children, kind, market.strike) if next_model is None
                  else next_model.predict(children, batch_size=batch_size))
        result[start:stop] = market.discount*values.reshape(stop-start, -1).mean(axis=1)
    return result


def train(market=Market(), training=Training(), kind="geometric_put", seed=0):
    """Regress the Bellman VALUE max(payoff, continuation), not continuation alone.

    mu_t is the uncontrolled GBM marginal starting at the deterministic spot.
    Holdout inputs share the training stage's empirical Bellman operator, as
    required to select a regressor for that fixed target; no reference prices
    are used for hyperparameter selection.
    """
    if training.n < 1 or training.m < 1 or training.regularization <= 0:
        raise ValueError("Positive n, M and regularization required")
    if training.validation_n < 1 or not training.lengthscales:
        raise ValueError("Positive validation_n and nonempty lengthscale grid required")
    if training.mc_mode not in ("shared", "independent"):
        raise ValueError("mc_mode must be shared or independent")
    rng = np.random.default_rng(seed)
    started = perf_counter()
    clip = market.strike if kind == "geometric_put" else training.max_call_clip
    models = [None]*(market.steps+1)
    stages = []
    def sample_inputs(n, stage):
        x = market.marginal(rng, n, stage)
        if training.initial_box is not None:
            lo, hi = training.initial_box
            x *= rng.uniform(lo, hi, (n, market.dimension))/market.spot
        return x

    for stage in reversed(range(market.steps)):
        n = training.n if (stage or training.initial_box is not None
                           or training.mc_mode == "independent") else 1
        x = sample_inputs(n, stage)
        factors = market.factors(rng, training.m if training.mc_mode == "shared" else n*training.m)
        if training.mc_mode == "independent":
            factors = factors.reshape(n, training.m, market.dimension)
        next_model = models[stage+1]
        y = np.maximum(payoff(x, kind, market.strike), continuation(
            x, factors, next_model, market, kind, training.batch_size))
        if (stage or training.initial_box is not None) and len(training.lengthscales) > 1:
            xv = sample_inputs(training.validation_n, stage)
            fv = factors
            if training.mc_mode == "independent":
                fv = market.factors(rng, training.validation_n*training.m).reshape(
                    training.validation_n, training.m, market.dimension)
            yv = np.maximum(payoff(xv, kind, market.strike), continuation(
                xv, fv, next_model, market, kind, training.batch_size))
        else:
            xv, yv = x, y
        candidates = [KRR.fit(x, y, lengthscale=ell, regularization=training.regularization,
                             clip=clip, max_centers=training.max_centers, rng=rng)
                      for ell in training.lengthscales]
        errors = [float(np.mean((model.predict(xv)-yv)**2)) for model in candidates]
        best = int(np.argmin(errors))
        models[stage] = candidates[best]
        stages.append(dict(stage=stage, lengthscale=models[stage].lengthscale,
                           validation_mse=errors[best], candidate_mse=errors,
                           solver=models[stage].solver,
                           target_mean=float(y.mean()), target_min=float(y.min()),
                           target_max=float(y.max()),
                           clipped_validation=int(np.sum(np.abs(models[stage].predict(xv)) >= clip))))
    price = float(models[0].predict(np.full((1, market.dimension), market.spot))[0])
    return Result(kind, market, training, seed, price, perf_counter()-started, stages, models)
