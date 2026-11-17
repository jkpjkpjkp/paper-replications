import json
import numpy as np
import pandas as pd
import pytest
import torch

from usdata import sha
from limt.model import LiMT
from experiments.market_adjusted_monthly.data import month_targets, Panel
from experiments.market_adjusted_monthly.model import Model
from experiments.market_adjusted_monthly.sizing import physical_weights


def test_monthly_volatility_is_next_month_daily_sample_std_and_masks_missing():
    rng = np.random.default_rng(3)
    dates = np.array(pd.bdate_range("2014-11-03", "2015-02-27").strftime("%Y-%m-%d"))
    market = rng.normal(scale=.01, size=len(dates))
    returns = rng.normal(scale=.02, size=(3, len(dates)))
    december = np.flatnonzero(dates.astype("U7") == "2014-12")
    returns[1, december[:8]] = np.nan
    sigma, market_sigma, end = month_targets(returns, market, dates)
    t = np.flatnonzero(dates == "2014-11-10")[0]
    np.testing.assert_allclose(sigma[0, t], returns[0, december].std(ddof=1), rtol=1e-6)
    np.testing.assert_allclose(market_sigma[t], market[december].std(ddof=1))
    assert np.isnan(sigma[1, t])
    assert dates[end[t]] == dates[december[-1]]
    assert np.isnan(sigma[:, dates.astype("U7") == "2015-02"]).all()
    adjusted_sigma = sigma[0, t] - market_sigma[t]
    assert not np.isclose(adjusted_sigma, (returns[0, december] - market[december]).std(ddof=1))
    panel = object.__new__(Panel)
    panel.dates, panel.lookback, panel.eligible, panel.label_end = dates, 1, np.ones((60, len(dates)), bool), end
    train = panel.split("train")
    assert dates[train[-1]][:7] == "2014-11"
    assert (dates[end[train]] <= "2014-12-31").all()


def test_daily_cs_scaling_cancels_market_but_fixed_training_scaling_preserves_it():
    raw = np.array([-.03, -.01, .01, .07])
    market = .025
    adjusted = raw - market
    z = lambda y: (y - y.mean()) / y.std(ddof=1)
    np.testing.assert_allclose(z(raw), z(adjusted), atol=1e-14)
    mean, scale = .001, .02
    np.testing.assert_allclose((raw - mean) / scale - (adjusted - mean) / scale, market / scale)


def test_joint_reproduces_all_three_released_outputs_and_specialists_initialize_equally():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    original = LiMT(d_model=32, experts=2, dropout=0.)
    torch.manual_seed(8)
    joint = Model("joint", d_model=32, experts=2, dropout=0.)
    x = torch.randn(7, 5, 158)
    torch.testing.assert_close(joint(x), original(x), atol=1e-7, rtol=1e-6)
    for task in ["return", "volume", "volatility"]:
        torch.manual_seed(8)
        single = Model(task, d_model=32, experts=2, dropout=0.)
        assert single.tasks == (task,) and not hasattr(single, "fusion_gate")
        for name, value in single.state_dict().items():
            torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)


def test_physical_monthly_sizing_uses_only_known_inputs_and_common_risk_shift_cancels():
    rng = np.random.default_rng(5)
    market = np.ones((5, 10, 7)) * 100
    market[:, :, 1] = rng.uniform(1e9, 3e9, size=(5, 10))
    pred = rng.normal(size=(5, 3))
    pred[:, 2] *= .02
    ids, member = np.arange(5), np.ones((5, 10), bool)
    first = physical_weights(pred, ids, market, member, 5)
    shifted = pred.copy()
    shifted[:, 2] -= .03
    np.testing.assert_allclose(first, physical_weights(shifted, ids, market, member, 5), atol=1e-15)
    market[:, 6:] = 999999.
    np.testing.assert_array_equal(first, physical_weights(pred, ids, market, member, 5))


def test_global_scaling_trainer_resumes_identical_weights_and_predictions(tmp_path, monkeypatch):
    from experiments.market_adjusted_monthly import train

    class Fixture:
        dates = np.array(["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07",
            "2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07",
            "2017-01-03", "2017-01-04", "2017-01-05", "2017-01-06"])
        symbols = np.array([f"P{i}" for i in range(7)])
        mean, scale = np.array([.001, .1, .02]), np.array([.03, .8, .01])
        label_end = np.minimum(np.arange(12) + 2, 11)

        def __init__(self, _):
            pass

        def split(self, split):
            return np.array({"train": [0, 1], "valid": [4, 5], "test": [8, 9]}[split])

        def day(self, t):
            rng = np.random.default_rng(t + 15)
            x = rng.normal(size=(7, 5, 158)).astype(np.float32)
            y = rng.normal(size=(7, 3)).astype(np.float32)
            raw = (y * self.scale + self.mean).astype(np.float32)
            return np.arange(7), x, ((raw - self.mean) / self.scale).astype(np.float32), raw

    def cpu_setup(seed):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        return torch.device("cpu")

    monkeypatch.setattr(train, "Panel", Fixture)
    monkeypatch.setattr(train, "setup", cpu_setup)
    monkeypatch.setattr(train, "build_model", lambda task: Model(task, d_model=32, experts=2, dropout=.1))
    monkeypatch.setattr(train, "input_hashes", lambda: {"fixture": "fixed"})
    monkeypatch.setattr(train, "protocol", lambda: {"training": dict(epochs=3, patience=21, learning_rate=1e-5, gradient_value_clip=3., weights=[.65, .3, .05])})
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    original_save = train.save

    def prepare(name):
        here = tmp_path / name
        (here / "data").mkdir(parents=True)
        (here / "data/manifest.json").write_text("{}")
        (here / "locked_before_training.json").write_text(json.dumps(dict(inputs_sha256={"fixture": "fixed"},
            data_manifest_sha256=sha(here / "data/manifest.json"))))
        monkeypatch.setattr(train, "HERE", here)
        out = here / "output"
        monkeypatch.setattr(train, "folder", lambda *_: out)
        return out

    uninterrupted = prepare("uninterrupted")
    train.run("raw_monthly", "return", 0)
    expected = torch.load(uninterrupted / "latest.pt", weights_only=False)
    resumed = prepare("resumed")
    interrupted = False

    def save_and_interrupt(value, path):
        nonlocal interrupted
        original_save(value, path)
        if path.name == "latest.pt" and not interrupted:
            interrupted = True
            raise RuntimeError("interrupted fixture")

    monkeypatch.setattr(train, "save", save_and_interrupt)
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        train.run("raw_monthly", "return", 0)
    monkeypatch.setattr(train, "save", original_save)
    train.run("raw_monthly", "return", 0)
    actual = torch.load(resumed / "latest.pt", weights_only=False)
    for key, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][key], value, atol=0, rtol=0)
    for split in ["valid", "test"]:
        pd.testing.assert_frame_equal(pd.read_parquet(uninterrupted / f"{split}_predictions.parquet"),
                                      pd.read_parquet(resumed / f"{split}_predictions.parquet"))
