"""Explicit integration/correction layer over the pinned AlphaQCM release."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'upstream'))
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch import nn
from alphagen_qlib.stock_data import StockData, FeatureType
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen.rl.env.wrapper import AlphaEnvWrapper, action2token, SIZE_ACTION
from alphagen.rl.env.core import AlphaEnvCore
from alphagen.data.tokens import SequenceIndicatorToken
from alphagen.data.tree import ExpressionBuilder
from fqf_iqn_qrdqn.agent.iqcm_agent import IQCMAgent
from fqf_iqn_qrdqn.memory.base import LazyMemory
from fqf_iqn_qrdqn.utils import calculate_quantile_huber_loss, evaluate_quantile_at_action


class LocalStockData(StockData):
    """Qlib historical membership, explicit calendar, and NaN prehistory padding."""
    @classmethod
    def _init_qlib(cls):
        pass  # Caller initializes Qlib with an explicit provider.

    def _get_data(self):
        from qlib.data import D
        cal = pd.DatetimeIndex(D.calendar())
        lo = cal.searchsorted(pd.Timestamp(self._start_time))
        hi = cal.searchsorted(pd.Timestamp(self._end_time), side='right') - 1
        if hi + self.max_future_days >= len(cal):
            raise ValueError('Insufficient future calendar for return labels')
        left = max(0, lo - self.max_backtrack_days)
        dates = cal[left:hi + self.max_future_days + 1]
        fields = ['$' + f.name.lower() for f in self._features]
        instruments = D.instruments(self._instrument) if isinstance(self._instrument, str) else self._instrument
        frame = D.features(instruments, fields, start_time=dates[0], end_time=dates[-1])
        ids = pd.Index(sorted(frame.index.get_level_values('instrument').unique()))
        values = np.stack([frame[f].unstack('instrument').reindex(index=dates, columns=ids).to_numpy()
                           for f in fields], axis=1)
        pad = self.max_backtrack_days - (lo-left)
        if pad:
            values = np.concatenate([np.full((pad, len(fields), len(ids)), np.nan), values])
        # Padded dates are NaT; observation dates always remain true exchange dates.
        dates = pd.DatetimeIndex([pd.NaT] * pad).append(dates)
        self.eval_dates = cal[lo:hi+1]
        return torch.tensor(values, dtype=torch.float32, device=self.device), dates, ids


class CachedCalculator(QLibStockDataCalculator):
    """Bounded cache; retain the released zero-fill training metric explicitly."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._calc_alpha = lru_cache(maxsize=24)(self._calc_alpha)


class SnapshotEnv(AlphaEnvWrapper):
    def reset(self, **kwargs):
        state, info = super().reset(**kwargs)
        return state.copy(), info

    def step(self, action):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            # 0 is padding, action IDs are zero based. SEP never enters observations.
            self.state[self.counter] = action + 1
            self.counter += 1
        return self.state.copy(), self.reward(reward), done, truncated, info

    @property
    def pool(self):
        return self.env.pool


@lru_cache(maxsize=100000)
def state_mask(tokens):
    builder = ExpressionBuilder()
    for token in tokens:
        if token:
            builder.add_token(action2token(token-1))
    return np.array([builder.is_valid() if isinstance(action2token(a), SequenceIndicatorToken)
                     else builder.validate(action2token(a)) for a in range(SIZE_ACTION)])


def corrected_lstm_forward(self, obs):
    """Pool each sequence over its own real length, excluding batch padding."""
    bs, length = obs.shape
    beg = torch.full((bs, 1), self._n_actions, dtype=torch.long, device=obs.device)
    tokens = torch.cat((beg, obs.long()), dim=1)
    lengths = (tokens != 0).sum(1)
    values = self._lstm(self._pos_enc(self._token_emb(tokens)))[0]
    mask = torch.arange(length+1, device=obs.device)[None] < lengths[:, None]
    return (values * mask[..., None]).sum(1) / lengths[:, None]


def qcm_coefficients(taus, quantiles):
    # Stable least squares instead of explicitly inverting X'X.
    z = torch.distributions.Normal(0., 1.).icdf(taus.clamp(1e-6, 1-1e-6))
    design = torch.stack([torch.ones_like(z), z, z*z-1, z*z*z-3*z], dim=-1)
    return torch.linalg.lstsq(design, quantiles).solution


class CorrectedIQN(IQCMAgent):
    """IQN+DQN+QCM with audited replay, masking and target-network repairs."""
    def __init__(self, *args, **kwargs):
        from types import MethodType
        super().__init__(*args, **kwargs)
        if self.memory.multi_step != 1:
            raise ValueError('This runner supports the paper IQN one-step protocol only')
        original_sample = self.memory._sample
        def token_sample(indices, batch_size):
            s, a, r, ns, d = original_sample(indices, batch_size)
            return (s*255).round(), a, r, (ns*255).round(), d
        self.memory._sample = token_sample
        for net in self.networks:
            net.dqn_net.forward = MethodType(corrected_lstm_forward, net.dqn_net)
        self.last_losses = {}

    @property
    def networks(self):
        return [self.online_net, self.target_net, self.online_mean_net, self.target_mean_net]

    def exploit(self, state):
        self.online_net.eval()
        self.online_mean_net.eval()
        with torch.no_grad():
            s = torch.as_tensor(state, device=self.device).float()[None]
            taus = torch.rand(1, self.K, device=self.device)
            quantiles = self.online_net.calculate_quantiles(taus, states=s)
            # Paper: variance = slope^2, so sqrt(variance) = abs(slope).
            bonus = qcm_coefficients(taus, quantiles)[:, 1].abs()
            values = self.online_mean_net.calculate_q(states=s) + self.std_lam * bonus
            mask = torch.as_tensor(self.env.action_masks(), device=self.device)
            return values.masked_fill(~mask[None], -torch.inf).argmax().item()

    def learn(self):
        self.learning_steps += 1
        self.online_net.train()
        self.online_mean_net.train()
        self.target_net.eval()
        self.target_mean_net.eval()
        if self.use_per:
            (states, actions, rewards, next_states, dones), weights = self.memory.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None
        masks = torch.as_tensor(np.stack([state_mask(tuple(row)) for row in next_states.long().cpu().tolist()]), device=self.device)
        # Terminal transitions don't bootstrap; leave a valid numerical argmax.
        masks[dones[:, 0].bool()] = True
        with torch.no_grad():
            next_embeddings = self.target_net.calculate_state_embeddings(next_states)
            next_q = self.target_net.calculate_q(state_embeddings=next_embeddings).masked_fill(~masks, -torch.inf)
            next_actions = next_q.argmax(1, keepdim=True)
            tau_next = torch.rand(self.batch_size, self.N_dash, device=self.device)
            next_z = evaluate_quantile_at_action(self.target_net.calculate_quantiles(tau_next, state_embeddings=next_embeddings), next_actions).transpose(1, 2)
            target_z = rewards[..., None] + (1-dones[..., None]) * self.gamma_n * next_z
            q = self.target_mean_net.calculate_q(states=next_states).masked_fill(~masks, -torch.inf)
            target_q = rewards + (1-dones) * self.gamma_n * q.max(1, keepdim=True).values
        taus = torch.rand(self.batch_size, self.N, device=self.device)
        current_z = evaluate_quantile_at_action(self.online_net.calculate_quantiles(taus, states=states), actions)
        errors = target_z - current_z
        quantile_loss = calculate_quantile_huber_loss(errors, taus, weights, self.kappa)
        current_q = self.online_mean_net.calculate_q(states=states).gather(1, actions)
        # Keep released unweighted DQN MSE; PER weighting applies to quantile loss.
        mean_loss = nn.functional.mse_loss(current_q, target_q)
        if not torch.isfinite(quantile_loss + mean_loss):
            raise FloatingPointError('Nonfinite learning loss')
        for optimizer, loss in [(self.optim, quantile_loss), (self.mean_optim, mean_loss)]:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.use_per:
            self.memory.update_priority(errors.detach().abs().sum(1).mean(1, keepdim=True))
        self.last_losses = {'quantile': quantile_loss.item(), 'mean': mean_loss.item()}


def daily_metrics(data, calculator, exprs, weights):
    """Primary paired finite-mask IC plus the explicitly separate released IC."""
    from scipy.stats import rankdata
    prediction = calculator.make_ensemble_alpha(exprs, weights).cpu().numpy()
    close = data.data[data.max_backtrack_days:data.max_backtrack_days+data.n_days, FeatureType.CLOSE].cpu().numpy()
    future = data.data[data.max_backtrack_days+20:data.max_backtrack_days+20+data.n_days, FeatureType.CLOSE].cpu().numpy()
    target = future / close - 1
    # Each component must be observed; upstream normalized factors fill NaNs with 0.
    observed = np.ones_like(prediction, dtype=bool)
    for expr in exprs:
        observed &= torch.isfinite(expr.evaluate(data)).cpu().numpy()
    rows = []
    for day, x, y, valid in zip(data.eval_dates, prediction, target, observed):
        mask = valid & np.isfinite(x) & np.isfinite(y) & np.isfinite(close[len(rows)])
        x, y = x[mask].astype(float), y[mask].astype(float)
        if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
            ic = rank_ic = np.nan
        else:
            ic = np.corrcoef(x, y)[0, 1]
            rank_ic = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
        rows.append({'date': str(day.date()), 'n_stocks': int(mask.sum()), 'ic': ic, 'rank_ic': rank_ic})
    frame = pd.DataFrame(rows)
    summary = {'ic': float(frame.ic.mean()), 'rank_ic': float(frame.rank_ic.mean()),
               'icir': float(frame.ic.mean()/frame.ic.std(ddof=0)),
               'valid_days': int(frame.ic.notna().sum()), 'total_days': len(frame),
               'released_zero_fill_ic': getattr(calculator, 'calc_released_pool_IC_ret', calculator.calc_pool_IC_ret)(exprs, weights)}
    return summary, frame
