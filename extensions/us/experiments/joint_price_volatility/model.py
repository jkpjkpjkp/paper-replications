"""Two active tasks or a genuine specialist, without dormant auxiliary branches."""
import torch
from torch import nn

from limt.model import LiMT
from experiments.joint_price_volatility.common import TASK_INDEX, VARIANTS


class PriceVolatilityModel(nn.Module):
    def __init__(self, variant, **kwargs):
        super().__init__()
        self.tasks = VARIANTS[variant]
        # Initialize the full released core first: every shared weight and every
        # corresponding task head starts identically across same-seed variants.
        base = LiMT(**kwargs).net
        self.feature_layer = base.feature_layer
        self.s_attn = base.s_attn
        self.t_attn = base.t_attn
        old = base.mmoe_fusion
        self.experts = old.experts
        self.gates = nn.ModuleDict({task: old.gates[task] for task in self.tasks})
        self.heads = nn.ModuleDict({task: getattr(old, f"{task}_proj") for task in self.tasks})
        if variant == "joint":
            # In the released core row 0 gates volume and row 1 volatility.
            original = old.fusion_gate[0]
            self.fusion_gate = nn.Linear(original.in_features, 1)
            with torch.no_grad():
                self.fusion_gate.weight.copy_(original.weight[1:2])
                self.fusion_gate.bias.copy_(original.bias[1:2])

    def forward(self, x):
        h = self.t_attn(self.s_attn(self.feature_layer(x)))[:, -1]
        experts = torch.stack([expert(h) for expert in self.experts], dim=1)
        hidden = {}
        for task in self.tasks:
            gates = self.gates[task](h).softmax(dim=1)
            hidden[task] = self.heads[task][:2](torch.einsum("bi,bij->bj", gates, experts))
        if hasattr(self, "fusion_gate"):
            hidden["return"] = hidden["return"] + self.fusion_gate(h).sigmoid() * hidden["volatility"]
        return torch.stack([self.heads[task][2](hidden[task]).squeeze(-1) for task in self.tasks], dim=-1)


def task_loss(prediction, targets):
    """Equal task weights; inactive targets must already have been excluded."""
    valid = torch.isfinite(targets)
    if not bool(valid.any(dim=0).all()):
        raise ValueError("Every active target needs at least one observed label")
    residual = prediction - torch.where(valid, targets, torch.zeros_like(targets))
    return ((residual.square() * valid).sum(0) / valid.sum(0)).mean()
