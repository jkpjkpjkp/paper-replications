"""Fixed decile and 2 x 3 characteristic/cycle portfolio constructions.

All input rows must be eligible at the previous close. Average ties are preserved;
no sorting or selection ever inspects contemporaneous returns.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

FACTORS = ['MS', 'CompEquIss', 'InvestPPEInv', 'Cash', 'RIO_Disp', 'RDIPO']
TIMINGS = ['none', 'H', 'L', 'P1', 'P2', 'P3']
CHAR_FIELDS = {'size': 'lag_cap', 'adv': 'lag_adv', 'vol': 'sigma63'}


def models():
    result = []
    for j, factor in enumerate(FACTORS):
        for family in ['D10', 'S30']:
            configs = [('none', 'all')] if family == 'D10' else []
            configs += [(c, g) for c in CHAR_FIELDS for g in ['low', 'high', 'balanced']]
            timings = [('none', 'none')]
            if family == 'D10':
                timings += [(a, b) for a in ['H', 'L'] for b in ['H', 'L']]
            timings += [(a, b) for a in ['P1', 'P2', 'P3'] for b in ['P1', 'P2', 'P3']]
            for char, group in configs:
                for weighting in ['EW', 'VW']:
                    prefix = '__'.join([family, factor, char, group, weighting])
                    baseline = prefix + '__none_none'
                    for a, b in timings:
                        result.append(dict(id=prefix + '__' + a + '_' + b,
                            factor=factor, factor_index=j, family=family,
                            characteristic=char, group=group, weighting=weighting,
                            long_timing=a, short_timing=b, baseline_id=baseline))
    for name, a, b in [('paper', 'P1', 'P3'), ('decile', 'H', 'L')]:
        result.append(dict(id='TIMING__' + name, factor='EARC', factor_index=-1,
            family='timing_only', characteristic='none', group='all', weighting='EW',
            long_timing=a, short_timing=b, baseline_id=None))
    assert len(result) == 2762
    assert len({s['id'] for s in result}) == len(result)
    return result


def percentile(values):
    x = np.asarray(values, float)
    good = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if good.any():
        out[good] = rankdata(x[good], method='average') / good.sum()
    return out


def masks(z):
    """Tail masks (25,n), characteristic cells (7,n), timing cells (6,n)."""
    n = len(z['age'])
    tail = np.zeros((25, n), bool)
    for j in range(6):
        r = np.asarray(z['rank'][:, j], dtype=float)
        for k, (hi, lo) in enumerate([(.9, .1), (.7, .3)]):
            # Cached ranks are float32: compare with identically represented
            # cutpoints so an exact cached .1 or .3 stays in the low group.
            tail[j * 4 + 2 * k] = r > float(np.float32(hi))
            tail[j * 4 + 2 * k + 1] = r <= float(np.float32(lo))
    tail[24] = True
    char = np.ones((7, n), bool)
    for j, field in enumerate(CHAR_FIELDS.values()):
        p = percentile(z[field])
        char[1 + 2*j] = p <= .5
        char[2 + 2*j] = p > .5
    age = np.asarray(z['age'], float)
    p = percentile(np.cos(2*np.pi*(age-13)/65))
    timing = np.vstack([np.ones(n, bool), p > .9, p <= .1,
                        (age >= 6) & (age <= 20), (age >= 21) & (age <= 35),
                        (age >= 36) & (age <= 50)])
    return tail, char, timing


def weights(z_day, specs, min_names=5):
    """Return model x stock targets, long +1 / short -1 after fees.

    Balanced models fund each characteristic half with +.5/-.5 and require the
    minimum in all four cells. Failed cell coverage makes the whole book cash.
    """
    tail, char, timing = masks(z_day)
    out = np.zeros((len(specs), tail.shape[1]))
    cap = np.asarray(z_day['lag_cap'], float)
    for i, s in enumerate(specs):
        if s['factor_index'] < 0:
            il = ir = 24
        else:
            il = s['factor_index']*4 + (2 if s['family'] == 'S30' else 0)
            ir = il+1
        if s['characteristic'] == 'none':
            groups = [0]
        else:
            lo = 1 + 2*list(CHAR_FIELDS).index(s['characteristic'])
            groups = [lo, lo+1] if s['group'] == 'balanced' else [lo+(s['group']=='high')]
        base = np.ones(len(cap)) if s['weighting'] == 'EW' else cap
        valid = np.isfinite(base) & (base > 0)
        good = True
        for g in groups:
            lm = tail[il] & char[g] & timing[TIMINGS.index(s['long_timing'])] & valid
            sm = tail[ir] & char[g] & timing[TIMINGS.index(s['short_timing'])] & valid
            if min(lm.sum(), sm.sum()) < min_names:
                good = False
                break
            out[i, lm] += base[lm] / base[lm].sum() / len(groups)
            out[i, sm] -= base[sm] / base[sm].sum() / len(groups)
        if not good:
            out[i] = 0
    return out


def compile_specs(specs):
    out = {k: [] for k in ['lt', 'st', 'lc', 'sc', 'group', 'other', 'balanced', 'vw']}
    for s in specs:
        t = 24 if s['factor_index'] < 0 else s['factor_index']*4 + (2 if s['family']=='S30' else 0)
        group = 0 if s['characteristic']=='none' else 1+2*list(CHAR_FIELDS).index(s['characteristic'])
        balanced = s['group']=='balanced'
        if s['group']=='high':
            group += 1
        values = dict(lt=t, st=t if t==24 else t+1,
            lc=TIMINGS.index(s['long_timing']), sc=TIMINGS.index(s['short_timing']),
            group=group, other=group+1 if balanced else group,
            balanced=balanced, vw=s['weighting']=='VW')
        for k, value in values.items():
            out[k].append(value)
    return {k: np.asarray(v) for k, v in out.items()}


def screen_day(z, compiled, min_names=5):
    """Exact daily gross portfolio returns using independently aggregated cells.

    This avoids materializing a 2,762 x stock target matrix on every date.
    It equals weights(z,specs) @ nan_to_num(ret), with full cash on sparse days.
    """
    tail, char, timing = masks(z)
    cells = (char[:, None, :] & timing[None, :, :]).reshape(42, -1).astype(float)
    tail = tail.astype(float)
    ret = np.where(np.isfinite(z['ret']), z['ret'], 0.)
    counts, denoms, payoffs = [], [], []
    for base in [np.ones(len(ret)), np.asarray(z['lag_cap'], float)]:
        valid = np.isfinite(base) & (base > 0)
        base = np.where(valid, base, 0.)
        counts.append(tail @ (cells*valid).T)
        denoms.append(tail @ (cells*base).T)
        payoffs.append(tail @ (cells*(base*ret)).T)
    counts = np.asarray(counts)
    denoms = np.asarray(denoms)
    payoffs = np.asarray(payoffs)
    c = compiled
    wi = c['vw'].astype(int)
    legs = []
    ns = []
    for t, tm in [(c['lt'], c['lc']), (c['st'], c['sc'])]:
        ca = c['group']*6+tm
        cb = c['other']*6+tm
        na, nb = counts[wi,t,ca], counts[wi,t,cb]
        da, db = denoms[wi,t,ca], denoms[wi,t,cb]
        ra = np.divide(payoffs[wi,t,ca], da, out=np.zeros(len(t)), where=da>0)
        rb = np.divide(payoffs[wi,t,cb], db, out=np.zeros(len(t)), where=db>0)
        legs.append((ra+rb)/2)
        ns.append((na, nb))
    active = (ns[0][0]>=min_names)&(ns[0][1]>=min_names)&(ns[1][0]>=min_names)&(ns[1][1]>=min_names)
    return dict(returns=np.where(active, legs[0]-legs[1], 0.), active=active,
        n_long=np.where(active, ns[0][0]+c['balanced']*ns[0][1], 0).astype(np.int16),
        n_short=np.where(active, ns[1][0]+c['balanced']*ns[1][1], 0).astype(np.int16))
