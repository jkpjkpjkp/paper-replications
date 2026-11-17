"""Use the checked resumable specialist trainer without editing its frozen code."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn
from limt.model import LiMT
from usdata import config
from experiments.joint_price_volatility import train as engine
from experiments.separate_three_task_portfolio.common import HERE, TASK_INDEX, input_hashes, protocol


class VolumeOnlyModel(nn.Module):
    tasks = ("volume",)

    def __init__(self, d_model=256, cross_heads=4, temporal_heads=2, experts=8, dropout=.5):
        super().__init__()
        base = LiMT(d_model=d_model, cross_heads=cross_heads, temporal_heads=temporal_heads,
                    experts=experts, dropout=dropout).net
        self.feature_layer, self.s_attn, self.t_attn = base.feature_layer, base.s_attn, base.t_attn
        self.experts = base.mmoe_fusion.experts
        self.gates = nn.ModuleDict({"volume": base.mmoe_fusion.gates["volume"]})
        self.heads = nn.ModuleDict({"volume": base.mmoe_fusion.volume_proj})

    def forward(self, x):
        h = self.t_attn(self.s_attn(self.feature_layer(x)))[:, -1]
        experts = torch.stack([expert(h) for expert in self.experts], dim=1)
        gates = self.gates["volume"](h).softmax(dim=1)
        return self.heads["volume"](torch.einsum("bi,bij->bj", gates, experts))


def build_model(_):
    m = config()["model"]
    return VolumeOnlyModel(**{key: m[key] for key in ["d_model", "cross_heads", "temporal_heads", "experts", "dropout"]})


def configure_engine():
    engine.HERE = HERE
    engine.protocol = protocol
    engine.input_hashes = input_hashes
    engine.TASK_INDEX = TASK_INDEX
    engine.VARIANTS = {"volume_only": ("volume",)}
    engine.build_model = build_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[0, 1, 2])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    configure_engine()
    if args.smoke:
        engine.smoke()
    elif args.seed is None:
        parser.error("--seed is required")
    else:
        engine.run("volume_only", args.seed)
