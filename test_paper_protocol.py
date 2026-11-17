import numpy as np
import pytest
import torch
from types import SimpleNamespace

import implementation
import paper_implementation as paper
from alphagen import config
from alphagen.rl.env import wrapper


@pytest.fixture
def paper_actions():
    operators = config.OPERATORS.copy()
    values = {k: v for k, v in vars(wrapper).items() if k.startswith(('SIZE_', 'OFFSET_'))}
    original_size = implementation.SIZE_ACTION
    paper.configure_paper_operators()
    yield
    config.OPERATORS[:] = operators
    for key, value in values.items():
        setattr(wrapper, key, value)
    implementation.SIZE_ACTION = original_size
    implementation.state_mask.cache_clear()


def test_paper_rank_and_sign_definitions():
    values = torch.tensor([[3., 1., 3., float('nan'), -2.]])
    assert torch.allclose(paper.average_ranks(values),
                          torch.tensor([[3.5, 2., 3.5, float('nan'), 1.]]), equal_nan=True)
    operand = SimpleNamespace()
    sign = paper.Sign.__new__(paper.Sign)
    assert torch.allclose(sign._apply(values), torch.tensor([[1., 1., 1., float('nan'), 0.]]), equal_nan=True)
    rank = paper.TsRank.__new__(paper.TsRank)
    assert rank._apply(torch.tensor([[[1., 3., 3.]]])).item() == 2.5
    assert torch.isnan(rank._apply(torch.tensor([[[1., float('nan'), 3.]]]))).all()


def test_table_b5_windows_and_serialization(paper_actions):
    from evaluate import parse_expression
    class Operand:
        def evaluate(self, data, interval):
            self.interval = interval
            return torch.arange(interval.start, interval.stop, dtype=torch.float)[:, None]
    operand = Operand()
    mean_class = next(c for c in config.OPERATORS if c.__name__ == 'Mean')
    op = mean_class.__new__(mean_class)
    op._operand, op._delta_time = operand, 10
    assert op.evaluate(None).item() == -5.
    assert operand.interval == slice(-10, 1)
    names = {c.__name__ for c in config.OPERATORS}
    assert {'Sign', 'Rank', 'TsRank'} <= names
    assert not {'Mad', 'Delta'} & names
    for text in ['Rank($close)', 'TsRank($volume,20)', 'Mean(Sign($close),10)']:
        assert str(parse_expression(text)) == text
    env = implementation.SnapshotEnv(implementation.AlphaEnvCore(None, device=torch.device('cpu')))
    state, _ = env.reset()
    state, _, _, _, _ = env.step(wrapper.OFFSET_FEATURE-1)
    assert np.array_equal(implementation.state_mask(tuple(state)), env.action_masks())


def test_ols_uses_raw_returns_and_matches_direct_design():
    rng = np.random.default_rng(2)
    factors = torch.tensor(rng.normal(size=(3, 7, 9)), dtype=torch.float64)
    target = torch.tensor(rng.normal(size=(7, 9))*np.arange(1, 8)[:, None], dtype=torch.float64)
    factors[1, 0, 0] = torch.nan
    class Expr:
        def __init__(self, n): self.n = n
        def evaluate(self, data): return factors[self.n]
    exprs = [Expr(i) for i in range(3)]
    calc = paper.PaperCalculator.__new__(paper.PaperCalculator)
    calc.data, calc.raw_target = None, target
    mask = calc.observation_mask(exprs)
    design = torch.stack([calc._calc_alpha(e)[mask] for e in exprs], 1).numpy()
    direct = np.linalg.lstsq(design, target[mask].numpy(), rcond=None)[0]
    assert np.allclose(calc.fit_ols(exprs), direct, rtol=1e-10, atol=1e-10)
    assert np.allclose(calc.fit_ols(exprs+[exprs[0]]),
                       np.linalg.lstsq(np.column_stack([design, design[:, 0]]), target[mask].numpy(), rcond=None)[0],
                       atol=1e-10)


def test_pool_rewards_previous_ic_and_evicts_after_fit():
    class Calc:
        scores = iter([.2, .1, .15])
        def fit_ols(self, exprs):
            return np.array([3., 2., .1][:len(exprs)])
        def calc_pool_IC_ret(self, exprs, weights):
            return next(self.scores)
    pool = paper.PaperPool(2, Calc())
    assert pool.try_new_expr('a') == pytest.approx(.2)
    assert pool.try_new_expr('b') == pytest.approx(-.1)
    assert pool.try_new_expr('c') == pytest.approx(.05)
    assert pool.exprs == ['a', 'b']
    assert np.array_equal(pool.weights, [3., 2.])
    assert pool.best_ic_ret == .2
    assert pool.try_new_expr('a') == 0


def test_nearly_collinear_ols_matches_direct_svd():
    rng = np.random.default_rng(14)
    a, noise = rng.normal(size=(2, 20, 30))
    factors = torch.tensor(np.stack([a, a+1e-7*noise]), dtype=torch.float64)
    class Expr:
        def __init__(self, n): self.n = n
        def evaluate(self, data): return factors[self.n]
    calc = paper.PaperCalculator.__new__(paper.PaperCalculator)
    calc.data, calc.raw_target = None, torch.tensor(noise, dtype=torch.float64)
    exprs = [Expr(0), Expr(1)]
    mask = calc.observation_mask(exprs)
    design = torch.stack([calc._calc_alpha(e)[mask] for e in exprs], 1).numpy()
    direct = np.linalg.lstsq(design, calc.raw_target[mask].numpy(), rcond=None)[0]
    assert np.allclose(calc.fit_ols(exprs), direct, rtol=1e-7, atol=1e-7)


def test_pearson_scale_invariance_and_missing_values():
    x = torch.tensor([[1., 3., 2., 4.], [2., 2., 2., 2.]], dtype=torch.float64)
    y = torch.tensor([[4., 1., 2., 3.], [1., 2., 3., 4.]], dtype=torch.float64)
    mask = torch.ones_like(x, dtype=torch.bool)
    mask[0, 3] = False
    expected = np.corrcoef(x[0, :3].numpy(), y[0, :3].numpy())[0, 1]
    assert paper.pearson_days(x*1e-10, y, mask)[0].item() == pytest.approx(expected)
    assert torch.isnan(paper.pearson_days(x, y, mask)[1])


def test_huber_priority_not_absolute_td_or_clipped():
    errors = torch.tensor([[[-2., 0.5], [1., 3.]]])
    taus = torch.tensor([[.25, .75]])
    expected = .75*1.5 + .25*.125 + .75*.5 + .75*2.5
    score = paper.quantile_huber_elements(errors, taus, 1).sum().item()
    assert score == pytest.approx(expected)
    replay = paper.PaperReplay(4, (20,), torch.device('cpu'))
    assert replay._pa(np.array([score]))[0] == pytest.approx(np.sqrt(expected))
    assert replay._pa(np.array([100.]))[0] == 10.


def test_paper_network_updates_and_q_target_selection(tmp_path, paper_actions, monkeypatch):
    env = implementation.SnapshotEnv(implementation.AlphaEnvCore(None, device=torch.device('cpu')))
    agent = paper.PaperIQN(env, None, None, str(tmp_path), cuda=False, num_steps=30,
                          start_steps=8, batch_size=4, N=8, N_dash=8, K=8,
                          memory_size=30, use_per=True)
    for i in range(8):
        s, _ = env.reset()
        a = wrapper.OFFSET_FEATURE-1+i%6
        ns, _, _, _, _ = env.step(a)
        agent.memory.append(s.copy(), a, .01+i/100, ns.copy(), False)
    old_q = [p.clone() for p in agent.online_mean_net.parameters()]
    old_z = [p.clone() for p in agent.online_net.parameters()]
    def forbid(*args, **kwargs):
        raise AssertionError('Quantile mean must not select Bellman target actions')
    monkeypatch.setattr(agent.target_net, 'calculate_q', forbid)
    agent.learn()
    assert any(not torch.equal(a,b) for a,b in zip(old_q, agent.online_mean_net.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(old_z, agent.online_net.parameters()))
    agent.update_target(); agent.update_mean_target()
    for net in [agent.online_net.quantile_net.net, agent.online_mean_net.q_net]:
        assert [layer.out_features for layer in net if isinstance(layer, torch.nn.Linear)] == [64,64,env.action_space.n]
    for online,target in [(agent.online_net, agent.target_net),(agent.online_mean_net, agent.target_mean_net)]:
        assert all(torch.equal(a,b) for a,b in zip(online.parameters(), target.parameters()))
    agent.steps = 20
    monkeypatch.setattr(np.random, 'rand', lambda: .02)
    assert not agent.is_random()
    monkeypatch.setattr(np.random, 'rand', lambda: .001)
    assert agent.is_random()
    assert all(np.isfinite(v) for v in agent.last_losses.values())
    agent.writer.close()
