"""Cost-aware within-leg allocation and exact, self-financing daily ledgers.

The covariance penalty is a diagonal *tracking-risk surrogate*, not an estimate
of full portfolio covariance. Weekly positions target equal dollars on each
side; between rebalances share holdings drift and need not remain neutral.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def allocate_leg(mu, sigma, previous, cost, gamma=10., horizon=5., cap=.2,
                 lower_multiplier=.5, upper_multiplier=1.5):
    """Maximize H mu'a-H gamma/2 sum(sigma²(a-b)²/b)-c|a-previous|.

    mu and previous are (models,names); sigma can be (names,) or the same shape.
    Each positive leg sums to one. Bounds are [.5/n,min(1.5/n,cap)].
    ``previous`` is the same-side, pretrade dollar weight on prior NAV, and
    may be negative when a name has changed from the other side. Costs of
    exiting names outside this leg are constant in the optimization.
    """
    mu = np.asarray(mu, dtype=float)
    one = mu.ndim == 1
    if one:
        mu = mu[None, :]
    if mu.ndim != 2 or not np.isfinite(mu).all():
        raise ValueError('mu must be a finite vector or matrix')
    n = mu.shape[1]
    if not n:
        return mu[0] if one else mu
    if cost < 0 or gamma <= 0 or horizon <= 0:
        raise ValueError('Invalid optimization settings')
    b = 1. / n
    lower, upper = lower_multiplier * b, min(upper_multiplier * b, cap)
    if lower * n > 1 + 1e-12 or upper * n < 1 - 1e-12:
        raise ValueError('Position constraints cannot fund this leg')
    if abs(upper * n - 1) < 1e-12 or abs(lower * n - 1) < 1e-12:
        result = np.full(mu.shape, b)
        return result[0] if one else result
    prev = np.broadcast_to(np.asarray(previous, dtype=float), mu.shape)
    if not np.isfinite(prev).all():
        raise ValueError('Nonfinite previous position')
    sigma = np.clip(np.nan_to_num(np.asarray(sigma, dtype=float), nan=.03,
                                 posinf=.20, neginf=.005), .005, .20)
    q = np.broadcast_to(horizon * gamma * sigma ** 2 / b, mu.shape)
    drift = horizon * mu
    lo = np.min(drift - q * (upper - b) - cost, axis=1) - 1e-10
    hi = np.max(drift - q * (lower - b) + cost, axis=1) + 1e-10

    def at(lagrange):
        z = b + (drift - lagrange[:, None]) / q - prev
        return np.clip(prev + np.sign(z) * np.maximum(abs(z) - cost / q, 0),
                       lower, upper)

    for _ in range(48):
        mid = (lo + hi) / 2
        a = at(mid)
        too_large = a.sum(axis=1) > 1
        lo = np.where(too_large, mid, lo)
        hi = np.where(too_large, hi, mid)
    result = at((lo + hi) / 2)
    if np.max(abs(result.sum(axis=1) - 1)) > 2e-10:
        raise ArithmeticError('Allocator equality constraint did not converge')
    return result[0] if one else result


class Ledger:
    """Vectorized independent model accounts, all normalized to their own NAV.

    ``step`` earns the day's return after any requested rebalance. Its inputs
    must already respect the user's information-set timing. Missing held rows
    close at the last observed mark and pay the scenario's transaction cost;
    an observed row with missing return earns zero and is separately counted.
    Cash distributions from ret minus retx remain cash until next rebalance.
    """
    def __init__(self, n_models, cost, borrow_rate=None):
        self.n_models = int(n_models)
        self.cost = float(cost)
        self.borrow_rate = (.01 if cost > 0 else 0.) if borrow_rate is None else float(borrow_rate)
        self.ids = np.empty(0, dtype=np.int64)
        self.hold = np.empty((n_models, 0), dtype=float)
        self.nav = np.ones(n_models, dtype=float)
        self.dead = np.zeros(n_models, dtype=bool)
        self.max_accounting_error = 0.

    def lookup(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        result = np.zeros((self.n_models, len(ids)))
        if len(self.ids) and len(ids):
            loc = np.searchsorted(self.ids, ids)
            ok = (loc < len(self.ids)) & (self.ids[np.minimum(loc, len(self.ids)-1)] == ids)
            result[:, ok] = self.hold[:, loc[ok]]
        return result

    def step(self, ids_all, ret, retx, target_ids=None, weights=None, terminal=False):
        ids_all = np.asarray(ids_all, dtype=np.int64)
        ret = np.asarray(ret, dtype=float)
        retx = np.asarray(retx, dtype=float)
        if len(ids_all) != len(ret) or len(ret) != len(retx):
            raise ValueError('Return array lengths differ')
        if len(ids_all) and not np.all(np.diff(ids_all) > 0):
            raise ValueError('Market IDs must be sorted and unique')
        rebalance = weights is not None
        if (target_ids is None) != (weights is None):
            raise ValueError('Provide both target_ids and weights for a rebalance')
        if rebalance:
            target_ids = np.asarray(target_ids, dtype=np.int64)
            weights = np.asarray(weights, dtype=float)
            if weights.shape != (self.n_models, len(target_ids)):
                raise ValueError('Target shape mismatch')
            if not np.isfinite(weights).all():
                raise ValueError('Targets must be finite')
            if len(target_ids) and not np.all(np.diff(target_ids) > 0):
                raise ValueError('Target IDs must be sorted and unique')
            if not np.isin(target_ids, ids_all).all():
                raise ValueError('Target lacks a current market row')
            union = np.union1d(self.ids, target_ids)
        else:
            union = self.ids
        h = np.zeros((self.n_models, len(union)))
        if len(self.ids):
            h[:, np.searchsorted(union, self.ids)] = self.hold
        loc = np.searchsorted(ids_all, union)
        present = loc < len(ids_all)
        if len(ids_all):
            present &= ids_all[np.minimum(loc, len(ids_all)-1)] == union
        else:
            present[:] = False
        forced = ((abs(h) > 1e-15) & ~present[None, :]).sum(axis=1)
        if rebalance:
            w = np.zeros_like(h)
            w[:, np.searchsorted(union, target_ids)] = weights
            w[self.dead] = 0
            x = np.ones(self.n_models)
            for _ in range(15):
                x = 1 - self.cost * abs(x[:, None] * w - h).sum(axis=1)
            position = x[:, None] * w
            turnover = abs(position - h).sum(axis=1)
            fees = self.cost * turnover
            residual = abs(x - (1-fees))
            self.max_accounting_error = max(self.max_accounting_error, float(residual.max()))
            if (residual > 1e-10).any() or (x < 0).any():
                raise ArithmeticError('Cost fixed point did not converge or is insolvent')
        else:
            position = h.copy()
            position[:, ~present] = 0
            turnover = abs(position-h).sum(axis=1)
            fees = self.cost * turnover
        r = np.zeros(len(union))
        rx = np.zeros(len(union))
        ismissing = np.zeros(len(union), bool)
        r[present] = np.where(np.isfinite(ret[loc[present]]), ret[loc[present]], 0.)
        rx[present] = np.where(np.isfinite(retx[loc[present]]), retx[loc[present]], r[present])
        ismissing[present] = ~np.isfinite(ret[loc[present]])
        short = -np.minimum(position, 0).sum(axis=1)
        borrow = self.borrow_rate / 252 * short
        gross = (position * r).sum(axis=1)
        pnl = gross - fees - borrow
        end_position = position * (1+rx)
        if terminal:
            liquidation = abs(end_position).sum(axis=1)
            turnover += liquidation
            fees += self.cost * liquidation
            pnl -= self.cost * liquidation
            end_position[:] = 0
        newly_dead = (~self.dead) & (1+pnl <= 0)
        pnl[self.dead] = 0
        pnl[newly_dead] = -1.
        self.dead |= newly_dead
        denominator = np.where(self.dead, 1., 1+pnl)
        end_position[self.dead] = 0
        self.nav *= 1+pnl
        keep = present & np.any(abs(end_position) > 1e-16, axis=0)
        self.ids = union[keep]
        self.hold = end_position[:, keep] / denominator[:, None]
        return dict(returns=pnl, gross_return=gross, turnover=turnover,
                    fees=fees, borrow=borrow,
                    gross_before=abs(h).sum(axis=1),
                    gross_exposure=abs(position).sum(axis=1),
                    net_exposure=position.sum(axis=1),
                    gross_after=abs(end_position).sum(axis=1)/denominator,
                    net_after=end_position.sum(axis=1)/denominator,
                    n_long=(position > 1e-15).sum(axis=1),
                    n_short=(position < -1e-15).sum(axis=1),
                    active=np.any(abs(position) > 1e-15, axis=1).astype(float),
                    missing=((abs(position) > 1e-15) & ismissing[None, :]).sum(axis=1),
                    forced_exit=forced, bankrupt=newly_dead.astype(int))


def validate():
    """Independent optimizer and absolute-dollar accounting comparisons."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(28471)
    errors, objectives = [], []
    for n in [5, 8, 31]:
        for cost in [0., .0005, .001]:
            mu = rng.normal(.0001, .001, n)
            sigma = rng.uniform(.01, .06, n)
            prev = rng.uniform(.2, 1.8, n) / n
            a = allocate_leg(mu, sigma, prev, cost)
            b = np.ones(n)/n
            # SLSQP uses a separately written nonsmooth objective.
            def f(x):
                return -5*mu.dot(x)+25*np.sum(sigma**2*(x-b)**2/b)+cost*abs(x-prev).sum()
            opt = minimize(f, b, method='SLSQP', bounds=[(.5/n, min(1.5/n, .2))]*n,
                           constraints=[{'type':'eq','fun':lambda x:x.sum()-1}],
                           options={'ftol':1e-13, 'maxiter':3000})
            if not opt.success:
                raise AssertionError(opt.message)
            errors.append(float(abs(a-opt.x).max()))
            objectives.append(float(abs(f(a)-f(opt.x))))
            assert abs(a.sum()-1)<1e-10
            assert f(a)-f(opt.x)<1e-9
    largest_ledger = 0.
    for cost in [0., .0005, .001]:
        led = Ledger(2, cost)
        nav = np.ones(2)
        held = [{}, {}]
        for day in range(17):
            ids = np.asarray([1,2,3,4] if day < 7 else [1,2,4], dtype=np.int64)
            r = rng.normal(0,.02,len(ids))
            rx = r - rng.uniform(0,.0005,len(ids))
            rebalance = day%5==0
            targets = None
            if rebalance:
                targets = np.zeros((2,len(ids)))
                targets[0,0]=1;targets[0,1]=-1
                targets[1,0]=.5;targets[1,-1]=.5;targets[1,2 if len(ids)==4 else 1]=-1
            out = led.step(ids,r,rx,ids if rebalance else None,targets,terminal=day==16)
            market = dict(zip(ids,zip(r,rx)))
            for j in range(2):
                old_nav=nav[j]
                old=held[j]
                if rebalance:
                    tw=dict(zip(ids,targets[j]))
                    keys=set(old)|set(tw)
                    lo,hi=0.,old_nav
                    for _ in range(70):
                        mid=(lo+hi)/2
                        equation=mid+cost*sum(abs(mid*tw.get(k,0)-old.get(k,0)) for k in keys)-old_nav
                        if equation>0:hi=mid
                        else:lo=mid
                    pos={k:(lo+hi)/2*w for k,w in tw.items()}
                else:
                    pos={k:v for k,v in old.items() if k in market}
                dollars=sum(abs(pos.get(k,0)-old.get(k,0)) for k in set(pos)|set(old))
                fee=cost*dollars
                borr=(.01/252 if cost else 0)*sum(-v for v in pos.values() if v<0)
                nav[j]-=fee+borr
                nav[j]+=sum(v*market[k][0] for k,v in pos.items())
                new={k:v*(1+market[k][1]) for k,v in pos.items()}
                if day==16:
                    liq=sum(abs(v) for v in new.values())
                    dollars+=liq;fee+=cost*liq;nav[j]-=cost*liq;new={}
                held[j]=new
                expected=[nav[j]/old_nav-1,dollars/old_nav,fee/old_nav,borr/old_nav]
                observed=[out[k][j] for k in ['returns','turnover','fees','borrow']]
                largest_ledger=max(largest_ledger,float(np.max(abs(np.asarray(expected)-observed))))
                assert np.allclose(led.nav[j],nav[j],rtol=1e-11,atol=1e-12)
    assert largest_ledger<1e-11
    return dict(allocator_cases=len(errors),allocator_max_position_difference=max(errors),
                allocator_max_objective_difference=max(objectives),
                independent_ledger_max_error=largest_ledger,
                ledger_scenarios=3,ledger_days_per_scenario=17,
                covers=['held-share drift','dividends retained as cash',
                        'forced missing exit','rebalance fee fixed point','terminal liquidation'])


if __name__ == '__main__':
    result=validate()
    target=Path(__file__).resolve().parent/'results'/'portfolio_validation.json'
    target.parent.mkdir(exist_ok=True,parents=True)
    target.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
