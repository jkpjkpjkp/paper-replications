import torch
from torch import nn
from limt.model import LiMT
from experiments.market_adjusted_monthly.common import TASKS


class Model(nn.Module):
    def __init__(self, task="joint", **settings):
        super().__init__()
        self.tasks = TASKS if task == "joint" else (task,)
        base = LiMT(**settings).net
        self.feature_layer, self.s_attn, self.t_attn = base.feature_layer, base.s_attn, base.t_attn
        fusion = base.mmoe_fusion
        self.experts = fusion.experts
        self.gates = nn.ModuleDict({name: fusion.gates[name] for name in self.tasks})
        self.heads = nn.ModuleDict({name: getattr(fusion, f"{name}_proj") for name in self.tasks})
        if task == "joint":
            self.fusion_gate = fusion.fusion_gate

    def forward(self, x):
        h = self.t_attn(self.s_attn(self.feature_layer(x)))[:, -1]
        experts = torch.stack([expert(h) for expert in self.experts], dim=1)
        hidden = {name: self.heads[name][:2](torch.einsum("bi,bij->bj", self.gates[name](h).softmax(1), experts))
                  for name in self.tasks}
        if hasattr(self, "fusion_gate"):
            beta = self.fusion_gate(h)
            hidden["return"] = hidden["return"] + beta[:, :1] * hidden["volume"] + beta[:, 1:] * hidden["volatility"]
        return torch.stack([self.heads[name][2](hidden[name]).squeeze(-1) for name in self.tasks], dim=-1)
