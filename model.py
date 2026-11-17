"""Explicit reconstruction of the incomplete PDU release; not an exact replica.

Input is [stock, time=4, indicator=9]. Graph edges index stocks.
The backbone is the released PDUArch with a documented dimension/device patch.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from repaired.PDUArch import PDU, Predictor


def indicator_edges(edges, stocks, indicators=9):
    return torch.cat([edges + i * stocks for i in range(indicators)], dim=1)


class PDUModel(nn.Module):
    def __init__(self, stocks, hidden=64, noise=0.01, mask=0.01, device="cpu", memory_efficient=True):
        super().__init__()
        self.stocks, self.hidden = stocks, hidden
        self.noise, self.mask = noise, mask
        self.memory_efficient = memory_efficient
        self.online = PDU(torch.device(device), 1, 2, 1, 32, 32, 1,
                          hidden, 4, stocks, 512, 64, 64, 8)
        self.projector = nn.Sequential(nn.Linear(hidden, 2 * hidden),
            nn.BatchNorm1d(2 * hidden), nn.ReLU(), nn.Linear(2 * hidden, hidden))
        self.predictor = Predictor(hidden, 1, stocks, hidden, headers=3, block_num=2)
        self.readout = nn.Sequential(nn.Linear(9, 18), nn.LeakyReLU(), nn.Linear(18, 1))
        self.target = copy.deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.target.eval()
        return self

    @torch.no_grad()
    def update_moving_average(self, beta=0.99):
        for t, o in zip(self.target.parameters(), self.online.parameters()):
            t.lerp_(o, 1 - beta)
        for t, o in zip(self.target.buffers(), self.online.buffers()):
            t.copy_(o)

    def forward(self, x, edges, contrast=True):
        if x.shape != (self.stocks, 4, 9):
            raise ValueError(f"Expected {(self.stocks, 4, 9)}, got {tuple(x.shape)}")
        # Stock/indicator axes must be explicit, not reshaped out of time order.
        data = x.permute(2, 0, 1).contiguous().reshape(9 * self.stocks, 4)
        def graph_call(network, values):
            if not self.memory_efficient:
                return network(values, indicator_edges(edges, self.stocks))
            outputs = []
            for part in values.split(self.stocks):
                if self.training and torch.is_grad_enabled() and network is not self.target:
                    outputs.append(checkpoint(network, part, edges, use_reentrant=False))
                else:
                    outputs.append(network(part, edges))
            return torch.cat(outputs, dim=0)
        h = graph_call(self.online, data)
        price = self.readout(graph_call(self.predictor, h).view(9, self.stocks).T)
        if not contrast:
            return price, price.new_zeros(())
        augmented = (data + torch.randn_like(data) * self.noise)
        augmented = augmented * (torch.rand_like(data) >= self.mask)
        ha = graph_call(self.online, augmented)
        with torch.no_grad():
            target_h, target_a = graph_call(self.target, data), graph_call(self.target, augmented)
        def normalize(z):
            z = z.reshape(9, self.stocks, self.hidden).transpose(1, 2)
            return F.normalize(z.reshape(9 * self.hidden, self.stocks), dim=1)
        loss = (2 - 2 * (normalize(self.projector(h)) * normalize(target_a)).sum(1)
              + 2 - 2 * (normalize(self.projector(ha)) * normalize(target_h)).sum(1)).mean()
        return price, loss


def objective(pred_price, base, truth, mask, contrast, weight=0.5):
    pred = (pred_price.squeeze(-1) - base) / base.clamp_min(1e-8)
    mask = mask.to(pred.dtype)
    regression = (mask * (pred - truth).square()).mean()
    pair = (pred[:, None] - pred[None, :]) * (truth[None, :] - truth[:, None])
    ranking = (F.relu(pair) * mask[:, None] * mask[None, :]).mean()
    return regression + ranking + weight * contrast, pred
