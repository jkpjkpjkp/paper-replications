"""Paper-stated AlphaQCM protocol, separate from the corrected-release adapter.

Implements Algorithm 1 and Appendices A, B, E. Unspecified missing-data and
numerical conventions are recorded in .agent/PAPER_PROTOCOL.txt; this is not an
assertion that the authors' unreleased execution has been recovered.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn

import implementation as release
from alphagen.data import expression as ex
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen_qlib.calculator import QLibStockDataCalculator
from fqf_iqn_qrdqn.memory.per import LazyPrioritizedMultiStepMemory
from fqf_iqn_qrdqn.utils import calculate_huber_loss, evaluate_quantile_at_action


class Sign(ex.UnaryOperator):
    def _apply(self, value):
        return torch.where(torch.isfinite(value), (value > 0).to(value.dtype), torch.nan)


def average_ranks(value):
    """One-based ascending ranks, average ties, preserving missing values."""
    finite = torch.isfinite(value)
    ordered, order = value.masked_fill(~finite, torch.inf).sort(dim=-1)
    n = value.shape[-1]
    positions = torch.arange(1, n+1, device=value.device).expand_as(order)
    first = torch.ones_like(order, dtype=torch.bool)
    first[..., 1:] = ordered[..., 1:] != ordered[..., :-1]
    last = torch.ones_like(first)
    last[..., :-1] = ordered[..., :-1] != ordered[..., 1:]
    lo = torch.where(first, positions, 0).cummax(dim=-1).values
    hi = torch.where(last, positions, n+1).flip([-1]).cummin(dim=-1).values.flip([-1])
    ranks = torch.empty_like(value).scatter_(-1, order, (lo+hi).to(value.dtype)/2)
    return ranks.masked_fill(~finite, torch.nan)


class Rank(ex.UnaryOperator):
    def _apply(self, value):
        return average_ranks(value)


def inclusive_rolling(self, data, period=slice(0, 1)):
    values = self._operand.evaluate(data, slice(period.start-self._delta_time, period.stop))
    return self._apply(values.unfold(0, self._delta_time+1, 1))


def inclusive_pair(self, data, period=slice(0, 1)):
    interval = slice(period.start-self._delta_time, period.stop)
    lhs = self._lhs.evaluate(data, interval).unfold(0, self._delta_time+1, 1)
    rhs = self._rhs.evaluate(data, interval).unfold(0, self._delta_time+1, 1)
    return self._apply(lhs, rhs)


class TsRank(ex.RollingOperator):
    evaluate = inclusive_rolling

    def _apply(self, value):
        last = value[..., -1:]
        ranks = (value < last).sum(-1) + ((value == last).sum(-1)+1)/2
        return ranks.to(value.dtype).masked_fill(~torch.isfinite(value).all(-1), torch.nan)


PAPER_OPERATORS = [Sign, ex.Abs, ex.Log, Rank, ex.Add, ex.Sub, ex.Mul, ex.Div,
                   ex.Greater, ex.Less, ex.Ref, TsRank]
PAPER_OPERATORS += [type(name, (getattr(ex, name),), {'evaluate': inclusive_rolling})
                    for name in ['Mean', 'Med', 'Sum', 'Std', 'Var', 'Max', 'Min', 'WMA', 'EMA']]
PAPER_OPERATORS += [type(name, (getattr(ex, name),), {'evaluate': inclusive_pair})
                    for name in ['Cov', 'Corr']]


def configure_paper_operators():
    """Install Table B.5 before constructing environments or parsing formulas."""
    from alphagen import config
    from alphagen.rl.env import wrapper
    config.OPERATORS[:] = PAPER_OPERATORS
    wrapper.SIZE_OP = len(PAPER_OPERATORS)
    wrapper.OFFSET_FEATURE = wrapper.OFFSET_OP + wrapper.SIZE_OP
    wrapper.OFFSET_DELTA_TIME = wrapper.OFFSET_FEATURE + wrapper.SIZE_FEATURE
    wrapper.OFFSET_CONSTANT = wrapper.OFFSET_DELTA_TIME + wrapper.SIZE_DELTA_TIME
    wrapper.OFFSET_SEP = wrapper.OFFSET_CONSTANT + wrapper.SIZE_CONSTANT
    wrapper.SIZE_ALL = wrapper.OFFSET_SEP + wrapper.SIZE_SEP
    wrapper.SIZE_ACTION = wrapper.SIZE_ALL - wrapper.SIZE_NULL
    release.SIZE_ACTION = wrapper.SIZE_ACTION
    release.state_mask.cache_clear()


def pearson_days(x, y, mask):
    """Appendix A Pearson IC, without a scale-dependent denominator cutoff."""
    x, y = x.double(), y.double()
    n = mask.sum(1)
    x = torch.where(mask, x, 0.)
    y = torch.where(mask, y, 0.)
    dx = torch.where(mask, x-x.sum(1, keepdim=True)/n.clamp_min(1)[:, None], 0.)
    dy = torch.where(mask, y-y.sum(1, keepdim=True)/n.clamp_min(1)[:, None], 0.)
    denom = (dx.square().sum(1)*dy.square().sum(1)).sqrt()
    cor = (dx*dy).sum(1)/denom
    return cor.masked_fill((n < 3) | (denom == 0), torch.nan)


class PaperCalculator(release.CachedCalculator):
    def __init__(self, data, target):
        self.target_expression = target
        self.raw_target = target.evaluate(data)
        super().__init__(data, target)

    def _calc_alpha(self, expr):
        raw = expr.evaluate(self.data)
        finite = torch.isfinite(raw)
        count = finite.sum(1, keepdim=True).clamp_min(1)
        value = torch.where(finite, raw, 0.)
        mean = value.sum(1, keepdim=True)/count
        centered = torch.where(finite, value-mean, 0.)
        std = (centered.square().sum(1, keepdim=True)/count).sqrt()
        return torch.where(finite & (std > 0), centered/std, torch.zeros_like(raw))

    def observation_mask(self, exprs):
        mask = torch.isfinite(self.raw_target)
        for expr in exprs:
            mask = mask & torch.isfinite(expr.evaluate(self.data))
        return mask

    def calc_pool_IC_ret(self, exprs, weights):
        if not exprs:
            return 0.
        prediction = self.make_ensemble_alpha(exprs, weights)
        mask = self.observation_mask(exprs) & torch.isfinite(prediction)
        daily = pearson_days(prediction, self.raw_target, mask)
        valid = torch.isfinite(daily)
        return daily[valid].mean().item() if valid.any() else 0.

    def calc_released_pool_IC_ret(self, exprs, weights):
        calc = QLibStockDataCalculator(self.data, self.target_expression)
        return calc.calc_pool_IC_ret(exprs, weights)

    def fit_ols(self, exprs):
        mask = self.observation_mask(exprs)
        if mask.sum().item() < len(exprs):
            raise ValueError('Insufficient observed samples for pool least squares')
        design = torch.stack([self._calc_alpha(e)[mask] for e in exprs]).double()
        target = self.raw_target[mask].double()
        # QR avoids squaring the condition number of nearly duplicate factors.
        # Solve the small R factor by SVD, including rank-deficient designs.
        q, r = torch.linalg.qr(design.T, mode='reduced')
        rhs = (q.T @ target).cpu().numpy()
        tolerance = np.finfo(np.float64).eps * max(design.shape)
        return np.linalg.lstsq(r.cpu().numpy(), rhs, rcond=tolerance)[0]


class PaperPool(AlphaPoolBase):
    """Algorithm 1: unpenalized fit, smallest-coefficient eviction, delta IC."""
    def __init__(self, capacity, calculator, device=torch.device('cpu')):
        super().__init__(capacity, calculator, device)
        self.exprs = []
        self.weights = np.empty(0)
        self.current_ic = 0.
        self.best_ic_ret = 0.
        self.eval_cnt = 0

    @property
    def size(self):
        return len(self.exprs)

    @property
    def state(self):
        return {'exprs': self.exprs, 'weights': self.weights, 'best_ic_ret': self.best_ic_ret}

    def to_dict(self):
        return {'exprs': [str(e) for e in self.exprs], 'weights': self.weights.tolist()}

    def test_ensemble(self, calculator):
        return calculator.calc_pool_IC_ret(self.exprs, self.weights)

    def try_new_expr(self, expr):
        if str(expr) in {str(e) for e in self.exprs}:
            return 0.
        exprs = self.exprs + [expr]
        try:
            weights = self.calculator.fit_ols(exprs)
        except (ValueError, np.linalg.LinAlgError):
            return -1.
        if not np.isfinite(weights).all():
            return -1.
        if len(exprs) > self.capacity:
            worst = int(np.argmin(np.abs(weights)))
            del exprs[worst]
            weights = np.delete(weights, worst)
        new_ic = self.calculator.calc_pool_IC_ret(exprs, weights)
        reward = new_ic-self.current_ic
        self.exprs, self.weights, self.current_ic = exprs, weights, new_ic
        self.best_ic_ret = max(self.best_ic_ret, new_ic)
        self.eval_cnt += 1
        return reward


def quantile_huber_elements(errors, taus, kappa):
    return (taus[..., None]-(errors.detach() < 0).float()).abs()*calculate_huber_loss(errors, kappa)/kappa


class PaperReplay(LazyPrioritizedMultiStepMemory):
    def _pa(self, priority):
        # A tiny floor keeps mathematically zero-loss transitions sampleable.
        return np.maximum(priority, 1e-12)**self.alpha

    def update_priority(self, errors):
        super().update_priority(errors)
        self.max_pa = max(self.max_pa, float(self._pa(errors.detach().cpu().numpy()).max()))


class PaperIQN(release.CorrectedIQN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        def head():
            return nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 64),
                                 nn.ReLU(), nn.Linear(64, self.num_actions)).to(self.device)
        self.online_net.quantile_net.net = head()
        self.online_mean_net.q_net = head()
        self.target_net.quantile_net.net = deepcopy(self.online_net.quantile_net.net)
        self.target_mean_net.q_net = deepcopy(self.online_mean_net.q_net)
        self.update_target(); self.update_mean_target()
        for net in [self.target_net, self.target_mean_net]:
            net.requires_grad_(False)
        lr = kwargs.get('lr', 5e-5)
        self.optim = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.mean_optim = torch.optim.Adam(self.online_mean_net.parameters(), lr=lr)
        self.memory = PaperReplay(self.memory.capacity, self.env.observation_space.shape,
                                  self.device, gamma=self.gamma_n, multi_step=1,
                                  alpha=0.5, beta_steps=max(1, self.num_steps-self.start_steps))
        original_sample = self.memory._sample
        def token_sample(indices, batch_size):
            s, a, r, ns, d = original_sample(indices, batch_size)
            return (s*255).round(), a, r, (ns*255).round(), d
        self.memory._sample = token_sample

    def is_random(self, eval=False):
        return self.steps < self.start_steps or np.random.rand() < 0.01

    def learn(self):
        self.learning_steps += 1
        self.online_net.train(); self.online_mean_net.train()
        self.target_net.eval(); self.target_mean_net.eval()
        (states, actions, rewards, next_states, dones), _ = self.memory.sample(self.batch_size)
        masks = torch.as_tensor(np.stack([release.state_mask(tuple(row))
                                         for row in next_states.long().cpu().tolist()]), device=self.device)
        masks[dones[:, 0].bool()] = True
        with torch.no_grad():
            # Appendix E.1 explicitly selects with the separate target Q network.
            next_q = self.target_mean_net.calculate_q(states=next_states).masked_fill(~masks, -torch.inf)
            next_actions = next_q.argmax(1, keepdim=True)
            taus_next = torch.rand(self.batch_size, self.N_dash, device=self.device)
            next_z = evaluate_quantile_at_action(self.target_net.calculate_quantiles(taus_next, states=next_states), next_actions).transpose(1, 2)
            target_z = rewards[..., None] + (1-dones[..., None])*self.gamma_n*next_z
            target_q = rewards + (1-dones)*self.gamma_n*next_q.max(1, keepdim=True).values
        taus = torch.rand(self.batch_size, self.N, device=self.device)
        current_z = evaluate_quantile_at_action(self.online_net.calculate_quantiles(taus, states=states), actions)
        errors = target_z-current_z
        elements = quantile_huber_elements(errors, taus, self.kappa)
        quantile_loss = elements.sum()
        current_q = self.online_mean_net.calculate_q(states=states).gather(1, actions)
        mean_loss = (current_q-target_q).square().sum()
        if not torch.isfinite(quantile_loss+mean_loss):
            raise FloatingPointError('Nonfinite paper-protocol loss')
        for optimizer, loss in [(self.optim, quantile_loss), (self.mean_optim, mean_loss)]:
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Appendix E: priority is the sum of quantile-Huber losses, not |TD|.
        self.memory.update_priority(elements.detach().sum((1, 2)).unsqueeze(1))
        self.last_losses = {'quantile': quantile_loss.item(), 'mean': mean_loss.item()}
