import numpy as np
import pytest
import torch
import json

from experiments.joint_price_volatility.model import PriceVolatilityModel, task_loss
from experiments.joint_price_volatility.train import selection_score


def small_model(variant):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    return PriceVolatilityModel(variant, d_model=32, experts=2, dropout=0.)


def test_same_seed_common_initialization_and_true_specialists():
    joint = small_model("joint")
    for variant, task in [("return_only", "return"), ("volatility_only", "volatility")]:
        specialist = small_model(variant)
        for name, value in specialist.state_dict().items():
            torch.testing.assert_close(value, joint.state_dict()[name], rtol=0, atol=0)
        assert set(specialist.heads) == {task}
        assert set(specialist.gates) == {task}
        assert not hasattr(specialist, "fusion_gate")
        assert not any("volume" in name or "feature_gate" in name for name, _ in specialist.named_parameters())
    x = torch.randn(7, 5, 158)
    with torch.no_grad():
        # Volatility has no incoming fusion, so it matches exactly at init.
        torch.testing.assert_close(joint(x)[:, 1], small_model("volatility_only")(x)[:, 0])


def test_each_joint_task_supervises_shared_encoder():
    model = small_model("joint")
    x = torch.randn(7, 5, 158)
    for task_index in [0, 1]:
        model.zero_grad(set_to_none=True)
        model(x)[:, task_index].square().mean().backward()
        assert model.feature_layer[0].weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.experts.parameters())
    assert set(model.heads) == {"return", "volatility"}
    assert model(x).shape == (7, 2)


def test_loss_equal_task_weighting_and_missing_labels():
    prediction = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    target = torch.tensor([[0., float("nan")], [1., 2.]])
    loss = task_loss(prediction, target)
    assert loss.item() == pytest.approx(((1 + 4) / 2 + 4) / 2)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 1] == 0
    with pytest.raises(ValueError, match="observed label"):
        task_loss(prediction, torch.full_like(target, float("nan")))


def test_joint_checkpoint_balances_both_targets_and_specialists_use_their_own():
    metrics = {"return": {"IC": .02}, "volatility": {"IC": .6}}
    assert selection_score(metrics, ("return", "volatility")) == pytest.approx(.31)
    assert selection_score(metrics, ("return",)) == .02
    assert selection_score(metrics, ("volatility",)) == .6


def test_checkpoint_round_trip_preserves_active_predictions(tmp_path):
    model = small_model("joint").eval()
    x = torch.randn(7, 5, 158)
    expected = model(x)
    torch.save(model.state_dict(), tmp_path / "model.pt")
    actual = small_model("joint").eval()
    actual.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True))
    torch.testing.assert_close(actual(x), expected, rtol=0, atol=0)


def test_interrupted_training_resumes_exact_trajectory_and_finishes_forecasts(tmp_path, monkeypatch):
    """Exercise the actual trainer, optimizer/RNG restore and forecast finalization."""
    from experiments.joint_price_volatility import train

    class TinyPanel:
        dates = np.array(["2014-12-24", "2014-12-26", "2014-12-29", "2014-12-30",
                          "2015-01-02", "2015-01-05", "2015-01-06", "2015-01-07",
                          "2017-01-03", "2017-01-04", "2017-01-05", "2017-01-06"])
        symbols = np.array([f"P{i}" for i in range(7)])

        def split(self, name):
            return np.array({"train": [0, 1], "valid": [4, 5], "test": [8, 9]}[name])

        def day(self, t):
            rng = np.random.default_rng(t + 23)
            x = rng.normal(size=(7, 5, 158)).astype(np.float32)
            raw = rng.normal(size=(7, 3)).astype(np.float32)
            normalized = (raw - raw.mean(0)) / raw.std(0, ddof=1)
            return np.arange(7), x, normalized, raw

    def setup_cpu(seed):
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        return torch.device("cpu")

    def build(variant):
        return PriceVolatilityModel(variant, d_model=32, experts=2, dropout=.1)

    monkeypatch.setattr(train, "Panel", TinyPanel)
    monkeypatch.setattr(train, "setup", setup_cpu)
    monkeypatch.setattr(train, "build_model", build)
    settings = dict(epochs=3, patience=21, learning_rate=1e-5, gradient_value_clip=3.)
    monkeypatch.setattr(train, "protocol", lambda: {"training": settings})
    monkeypatch.setattr(train, "input_hashes", lambda: {"fixture": "fixed"})
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: torch.get_rng_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU test fixture")
    original_save = train.atomic_save

    def prepare(name):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "locked_before_training.json").write_text(json.dumps({"inputs_sha256": {"fixture": "fixed"}}))
        monkeypatch.setattr(train, "HERE", folder)
        return folder / "results/joint/seed0"

    uninterrupted = prepare("uninterrupted")
    train.run("joint", 0)
    expected = torch.load(uninterrupted / "latest.pt", weights_only=False)
    resumed = prepare("resumed")
    interrupted = False

    def interrupt_after_durable_checkpoint(value, path):
        nonlocal interrupted
        original_save(value, path)
        if path.name == "latest.pt" and not interrupted:
            interrupted = True
            raise RuntimeError("simulated process interruption")

    monkeypatch.setattr(train, "atomic_save", interrupt_after_durable_checkpoint)
    with pytest.raises(RuntimeError, match="simulated"):
        train.run("joint", 0)
    monkeypatch.setattr(train, "atomic_save", original_save)
    train.run("joint", 0)
    actual = torch.load(resumed / "latest.pt", weights_only=False)
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, rtol=0, atol=0)
    assert actual["epoch"] == expected["epoch"] == 3
    assert actual["best_epoch"] == expected["best_epoch"]
    assert actual["best"] == expected["best"]
    assert json.loads((resumed / "status.json").read_text())["phase"] == "complete"
    for filename in ["valid_predictions.parquet", "test_predictions.parquet", "test_task_selected_predictions.parquet"]:
        import pandas as pd
        pd.testing.assert_frame_equal(pd.read_parquet(resumed / filename), pd.read_parquet(uninterrupted / filename))
