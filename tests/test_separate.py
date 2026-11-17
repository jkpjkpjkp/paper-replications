import numpy as np
import pandas as pd
import pytest
import torch

from limt.model import LiMT, SingleTaskLiMT
from scripts.compare_separate import merge_predictions
from scripts.train import masked_loss
from scripts.train_separate import TASKS


@pytest.mark.parametrize("task", TASKS)
def test_single_task_matches_seeded_author_branch_and_gradients(task):
    torch.set_num_threads(1)
    torch.manual_seed(13)
    reference = LiMT(d_model=32, dropout=0., optimized=False).double()
    torch.manual_seed(13)
    single = SingleTaskLiMT(task, d_model=32, dropout=0.).double()
    # Disable only cross-task fusion in the author return branch.
    if task == "return":
        with torch.no_grad():
            reference.net.mmoe_fusion.fusion_gate[0].weight.zero_()
            reference.net.mmoe_fusion.fusion_gate[0].bias.fill_(-1000.)
    x = torch.randn(7, 5, 158, dtype=torch.float64, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    expected, actual = reference(x)[:, TASKS.index(task)], single(z)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(z.grad, x.grad, atol=1e-10, rtol=1e-8)
    modules = [(single.feature_layer, reference.net.feature_layer),
               (single.s_attn, reference.net.s_attn), (single.t_attn, reference.net.t_attn),
               (single.experts, reference.net.mmoe_fusion.experts),
               (single.gate, reference.net.mmoe_fusion.gates[task]),
               (single.projection, getattr(reference.net.mmoe_fusion, f"{task}_proj"))]
    for a, b in modules:
        for pa, pb in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(pa.grad, pb.grad, atol=1e-10, rtol=1e-8)


def test_separate_tasks_share_no_parameter_storage():
    a = SingleTaskLiMT("return", d_model=32)
    b = SingleTaskLiMT("volume", d_model=32)
    assert not ({p.data_ptr() for p in a.parameters()} & {p.data_ptr() for p in b.parameters()})


def test_single_task_loss_masks_missing_labels_and_retains_weight():
    pred = torch.tensor([[2.], [4.], [99.]], requires_grad=True)
    target = torch.tensor([[1.], [2.], [float("nan")]])
    loss = masked_loss(pred, target, torch.tensor([.3]))
    torch.testing.assert_close(loss, torch.tensor(.75))
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.tensor([[.3], [.6], [0.]]))


def prediction_frames():
    return {task: pd.DataFrame(dict(date=["2020-01-02", "2020-01-02"],
        realized_date=["2020-01-06", "2020-01-06"], symbol=["B", "A"],
        **{f"pred_{task}": [1., 2.], f"label_{task}": [0., np.nan]})) for task in TASKS}


def test_merge_aligns_all_tasks_without_dropping_missing_labels():
    frames = prediction_frames()
    frames["volume"] = frames["volume"].iloc[::-1]
    merged = merge_predictions(frames)
    assert merged.symbol.tolist() == ["A", "B"]
    assert len(merged) == 2 and merged.label_return.isna().sum() == 1


@pytest.mark.parametrize("failure", ["missing", "duplicate", "realization", "nonfinite"])
def test_merge_rejects_incompatible_predictions(failure):
    frames = prediction_frames()
    if failure == "missing":
        frames["volume"] = frames["volume"].iloc[:1]
    elif failure == "duplicate":
        frames["volume"] = pd.concat([frames["volume"], frames["volume"].iloc[:1]])
    elif failure == "realization":
        frames["volume"].loc[0, "realized_date"] = "2020-01-07"
    else:
        frames["volume"].loc[0, "pred_volume"] = np.inf
    with pytest.raises(ValueError):
        merge_predictions(frames)
