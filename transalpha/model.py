"""Section 3.2 and Algorithm 1. Stock and time axes are never conflated.

Inputs: [stocks, time, features] or [cross_sections, stocks, time, features].
The optional stock mask supports padded cross sections, not missing time steps.
"""
import torch
from torch import nn


class CrossSectionalDenoising(nn.Module):
    def __init__(self, features, threshold=1e-3):
        super().__init__()
        self.alpha_scale = nn.Parameter(torch.ones(features))
        self.beta_scale = nn.Parameter(torch.ones(features))
        self.threshold = threshold

    def forward(self, x, mask=None):
        if mask is None:
            beta = x.mean(dim=-3, keepdim=True)
        else:
            valid = mask[..., None, None].to(x.dtype)
            beta = (x * valid).sum(-3, keepdim=True) / valid.sum(-3, keepdim=True).clamp_min(1)
        alpha = x - beta
        scale = torch.where(alpha.abs() > self.threshold, self.alpha_scale, 1.0)
        return alpha * scale + beta * self.beta_scale


class TransAlpha(nn.Module):
    def __init__(self, features=400, hidden=64, dropout=0.1, threshold=1e-3,
                 variant="transalpha"):
        super().__init__()
        if variant not in {"transalpha", "no_csd", "no_tag", "no_sp"}:
            raise ValueError(variant)
        self.variant = variant
        self.csd = CrossSectionalDenoising(features, threshold) if variant != "no_csd" else None
        self.projection = nn.Sequential(nn.Linear(features, hidden), nn.LayerNorm(hidden),
                                        nn.ReLU(), nn.Dropout(dropout))
        self.gate = nn.Linear(hidden, 1) if variant != "no_tag" else None
        # Head width/activation and dropout rate are not specified by the paper.
        self.head = nn.Sequential(nn.Linear(hidden if variant == "no_sp" else 2 * hidden, hidden),
                                  nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x, mask=None, return_attention=False):
        if x.ndim not in (3, 4):
            raise ValueError("Expected [N,L,D] or [B,N,L,D]")
        z = self.projection(self.csd(x, mask) if self.csd is not None else x)
        if self.gate is None:
            # Removal ablation: identity gating, explicitly not multi-head attention.
            weights = torch.ones_like(z[..., :1])
        else:
            weights = self.gate(z).softmax(dim=-2)
        gated = z * weights
        pooled = gated.mean(-2) if self.variant == "no_sp" else torch.cat(
            [gated[..., -1, :], gated.amax(dim=-2)], dim=-1)
        prediction = self.head(pooled).squeeze(-1)
        return (prediction, weights) if return_attention else prediction


class LinearControl(nn.Module):
    """Simple last-observation linear control; not the paper's DLinear baseline."""
    def __init__(self, features):
        super().__init__()
        self.linear = nn.Linear(features, 1)

    def forward(self, x, mask=None):
        return self.linear(x[..., -1, :]).squeeze(-1)
