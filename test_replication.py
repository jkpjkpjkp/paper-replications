import json
from pathlib import Path
import numpy as np
import torch
from types import MethodType, SimpleNamespace
from implementation import (LazyMemory, SnapshotEnv, AlphaEnvCore, state_mask,
                            corrected_lstm_forward, qcm_coefficients, CorrectedIQN)
from alphagen.rl.env.wrapper import OFFSET_FEATURE
from fqf_iqn_qrdqn.network import LSTMBase

def test_released_replay_collapses_tokens():
    mem = LazyMemory(3, (20,), torch.device('cpu'))
    s = np.arange(1, 21, dtype=np.uint8)
    mem.append(s, 1, 0., s+1, False)
    batch = mem._sample([0], 1)
    assert batch[0].long().count_nonzero() == 0
    assert torch.equal((batch[0]*255).round().long()[0], torch.from_numpy(s.astype('int64')))

def test_snapshot_transition_and_masks():
    env = SnapshotEnv(AlphaEnvCore(None, device=torch.device('cpu')))
    state, _ = env.reset()
    action = OFFSET_FEATURE-1
    ns, _, done, _, _ = env.step(action)
    assert not done
    assert state.sum() == 0
    assert ns[0] == action+1
    assert np.array_equal(state_mask(tuple(ns)), env.action_masks())
    env.step(0)  # Abs, zero-based action 0 must not disappear as padding.
    assert ns[1] == 0
    assert env.state[1] == 1

def test_lstm_batch_invariance():
    torch.manual_seed(4)
    net = LSTMBase(n_actions=48).eval()
    net.forward = MethodType(corrected_lstm_forward, net)
    short = torch.tensor([[24, 0, 0, 0]])
    long = torch.tensor([[24, 25, 3, 1]])
    with torch.no_grad():
        assert torch.allclose(net(short), net(torch.cat([short, long]))[:1], atol=1e-6)

def test_qcm_recovers_scale_with_location_bias():
    taus = torch.linspace(.01, .99, 100, dtype=torch.float64)[None]
    z = torch.distributions.Normal(0., 1.).icdf(taus)
    y = (8 + 2.5*z + .3*(z*z-1) + .05*(z*z*z-3*z))[..., None]
    coef = qcm_coefficients(taus, y)
    assert torch.allclose(coef[0, :, 0], torch.tensor([8., 2.5, .3, .05], dtype=torch.float64), atol=1e-8)
    assert torch.allclose(qcm_coefficients(taus, y+50)[:, 1], coef[:, 1], atol=1e-8)

def test_both_networks_learn_and_targets_refresh(tmp_path):
    env = SnapshotEnv(AlphaEnvCore(None, device=torch.device('cpu')))
    agent = CorrectedIQN(env, None, None, str(tmp_path), cuda=False, num_steps=30,
                         start_steps=8, batch_size=4, N=8, N_dash=8, K=8,
                         memory_size=30, use_per=True)
    states = []
    for i in range(8):
        s, _ = env.reset()
        a = OFFSET_FEATURE-1+i%6
        ns, _, _, _, _ = env.step(a)
        agent.memory.append(s.copy(), a, .1+i/100, ns.copy(), False)
    old_q = [p.clone() for p in agent.online_mean_net.parameters()]
    old_z = [p.clone() for p in agent.online_net.parameters()]
    agent.learn()
    assert any(not torch.equal(a,b) for a,b in zip(old_q, agent.online_mean_net.parameters()))
    assert any(not torch.equal(a,b) for a,b in zip(old_z, agent.online_net.parameters()))
    agent.update_target(); agent.update_mean_target()
    for online, target in [(agent.online_net, agent.target_net), (agent.online_mean_net, agent.target_mean_net)]:
        assert all(torch.equal(a,b) for a,b in zip(online.parameters(), target.parameters()))
    assert all(np.isfinite(v) for v in agent.last_losses.values())
    agent.writer.close()

def test_purge_dates():
    import pandas as pd
    cal = pd.DatetimeIndex(pd.read_csv(Path(__file__).resolve().parent.parent / 'csi/qlib/cn_data/calendars/day.txt', header=None)[0])
    for end, next_start in [('2019-12-31', '2020-01-01'), ('2020-12-31', '2021-01-01')]:
        i = cal.searchsorted(pd.Timestamp(end), side='right')-1
        last_origin = i-20
        assert cal[last_origin+20] < pd.Timestamp(next_start)

def test_expression_serialization_and_rejection():
    import pytest
    from evaluate import parse_expression
    expression = 'Mean(Div($close,Constant(-2.0)),20)'
    assert str(parse_expression(expression)) == expression
    with pytest.raises(ValueError):
        parse_expression('__import__("os").system("false")')

def test_replay_wraparound_alignment():
    mem = LazyMemory(3, (1,), torch.device('cpu'))
    for token in range(1, 8):
        mem.append(np.array([token], dtype='uint8'), token, token/10,
                   np.array([token+1], dtype='uint8'), False)
    s,a,r,ns,d=mem._sample([0,1,2],3)
    assert torch.equal((s*255).round().long(), a)
    assert torch.equal((ns*255).round().long(), a+1)
