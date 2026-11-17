"""Execute selected author classes, with a numerically checked SDPA adapter.

AST selection only removes unavailable, unused imports and unrelated models.
The baseline author definitions and weights are not rewritten on disk.
"""
from pathlib import Path
import ast
import math
import types
import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "Gate", "PositionalEncoding", "SAttention", "TAttention",
    "MMoE_TaskFusion_RVol2Vola2", "MMoE_TaskFusion_RV1", "LiMT_v2_MMoE_RV1",
    "LiMT_v2_MMoE_RVol2Vola2",
}


def author_namespace():
    path = ROOT / "upstream/limt.py"
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASSES]
    scope = dict(torch=torch, nn=nn, Linear=Linear, Dropout=Dropout,
                 LayerNorm=LayerNorm, math=math)
    exec(compile(tree, str(path), "exec"), scope)
    return scope


def fast_attention(self, x):
    """Equivalent to the released attention, including unscaled temporal logits."""
    x = self.norm1(x)
    z = x.transpose(0, 1) if self.cross_stock else x
    b, length, width = z.shape
    def split(layer):
        return layer(z).reshape(b, length, self.nhead, width // self.nhead).transpose(1, 2)
    q, k, v = [split(layer) for layer in (self.qtrans, self.ktrans, self.vtrans)]
    scale = 1 / math.sqrt(width // self.nhead) if self.cross_stock else 1.0
    drop = self.attn_dropout[0].p if len(self.attn_dropout) and self.training else 0.0
    h = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, scale=scale)
    h = h.transpose(1, 2).reshape(b, length, width)
    if self.cross_stock:
        h = h.transpose(0, 1)
    h = self.norm2(x + h)
    return h + self.ffn(h)


def fast_mmoe(self, shared_feat):
    # Released experts are deterministic linear/ReLU functions. Reusing their
    # outputs across task gates preserves both values and summed gradients.
    experts = torch.stack([e(shared_feat) for e in self.experts], dim=1)
    h = []
    for task, proj in [("return", self.return_proj), ("volume", self.volume_proj),
                       ("volatility", self.volatility_proj)]:
        gate = self.gates[task](shared_feat).softmax(dim=1)
        h.append(proj[:2](torch.einsum("bi,bij->bj", gate, experts)))
    beta = self.fusion_gate(shared_feat)
    fused = h[0] + beta[:, :1] * h[1] + beta[:, 1:] * h[2]
    return tuple(p[2](z).squeeze(-1) for p, z in zip(
        [self.return_proj, self.volume_proj, self.volatility_proj], [fused, h[1], h[2]]))


class LiMT(nn.Module):
    def __init__(self, d_model=256, cross_heads=4, temporal_heads=2, experts=8,
                 dropout=0.5, optimized=True, variant="full"):
        super().__init__()
        if d_model % cross_heads or d_model % temporal_heads:
            raise ValueError("Hidden dimension must be divisible by both head counts")
        scope = author_namespace()
        self.net = scope["LiMT_v2_MMoE_RVol2Vola2"](
            d_feat=158, d_model=d_model, t_nhead=temporal_heads,
            s_nhead=cross_heads, T_dropout_rate=dropout,
            S_dropout_rate=dropout, gate_input_start_index=158,
            gate_input_end_index=221, beta=1, n_experts=experts)
        self.variant = variant
        if optimized:
            for module, cross in [(self.net.s_attn, True), (self.net.t_attn, False)]:
                module.cross_stock = cross
                module.forward = types.MethodType(fast_attention, module)
            self.net.mmoe_fusion.forward = types.MethodType(fast_mmoe, self.net.mmoe_fusion)
        if variant == "no_cross":
            self.net.s_attn = nn.Identity()
        if variant == "no_ldl":
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model // 2),
                nn.ReLU(), nn.Linear(d_model // 2, 1)) for _ in range(3)])

    def forward(self, x):
        if self.variant == "no_ldl":
            h = self.net.t_attn(self.net.s_attn(self.net.feature_layer(x)))[:, -1]
            return torch.stack([p(h).squeeze(-1) for p in self.heads], dim=-1)
        return torch.stack(self.net(x), dim=-1)
