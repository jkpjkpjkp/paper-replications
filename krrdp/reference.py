"""Independent reference algorithms; none are used to fit KRR-DP."""
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from .market import Market


def geometric_european(market, spot=None, remaining=None):
    g = market.spot if spot is None else np.asarray(spot)
    time = market.maturity if remaining is None else remaining
    if time <= 0:
        return np.maximum(market.strike-g, 0.)
    variance = market.geometric_variance
    log_mean = (market.rate-0.5*market.volatility**2)*time
    s = np.sqrt(variance*time)
    d2 = (np.log(g/market.strike)+log_mean)/s
    return np.exp(-market.rate*time)*(market.strike*ndtr(-d2)
           - g*np.exp(log_mean+0.5*s*s)*ndtr(-d2-s))


def geometric_tree(market, subdivisions=400):
    """CRR tree for the EXACT scalar geometric-basket reduction.

    Discrete exercise at k*maturity/steps; no claimed American closed form.
    """
    total = market.steps*subdivisions
    dt = market.maturity/total
    sigma_g = np.sqrt(market.geometric_variance)
    log_up = sigma_g*np.sqrt(dt)
    up, down = np.exp(log_up), np.exp(-log_up)
    drift_g = market.rate - 0.5*(market.volatility**2-market.geometric_variance)
    probability = (np.exp(drift_g*dt)-down)/(up-down)
    if not 0 < probability < 1:
        raise ValueError("Invalid tree probability; increase subdivisions")
    nodes = market.spot*np.exp((2*np.arange(total+1)-total)*log_up)
    values = np.maximum(market.strike-nodes, 0.)
    discount = np.exp(-market.rate*dt)
    for step in range(total-1, -1, -1):
        values = discount*((1-probability)*values[:-1]+probability*values[1:])
        if step % subdivisions == 0:
            nodes = market.spot*np.exp((2*np.arange(step+1)-step)*log_up)
            values = np.maximum(values, market.strike-nodes)
    return float(values[0])


def max_call_european(market):
    """One-dimensional quadrature over maximum of iid normal idiosyncratic factors.

    Conditional on their maximum u, the common factor gives a lognormal asset.
    For zero dividends and nonnegative r the convex payoff has no early-exercise
    advantage, so this also prices the target Bermudan max-call.
    """
    a = market.volatility*np.sqrt(market.correlation*market.maturity)
    b = market.volatility*np.sqrt((1-market.correlation)*market.maturity)
    mu = (market.rate-0.5*market.volatility**2)*market.maturity
    def integrand(u):
        density = market.dimension*np.exp(-u*u/2)/np.sqrt(2*np.pi)*ndtr(u)**(market.dimension-1)
        conditional_log_moneyness = np.log(market.spot/market.strike)+mu+b*u
        if a > 0:
            call = (market.spot*np.exp(mu+b*u+a*a/2)*ndtr((conditional_log_moneyness+a*a)/a)
                    - market.strike*ndtr(conditional_log_moneyness/a))
        else:
            call = max(market.spot*np.exp(mu+b*u)-market.strike, 0.)
        return call*density
    value, quadrature_error = quad(integrand, -12., 12., epsabs=1e-9, epsrel=1e-10)
    return float(np.exp(-market.rate*market.maturity)*value), float(quadrature_error)
