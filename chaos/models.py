import torch
from torch import nn


def mlp(in_dim, out_dim, width=128, depth=3):
    layers = [nn.Linear(in_dim, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(width, width), nn.Tanh()])
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


class SDENO(nn.Module):
    """Appendix E: deterministic propagators contracted with Wick features."""
    def __init__(self, feature_dim, state_dim=1, width=128, encoded=False, latent=64):
        super().__init__()
        self.state_dim = state_dim
        self.feature_dim = latent if encoded else feature_dim
        self.encoder = mlp(feature_dim, latent, width, 2) if encoded else nn.Identity()
        self.propagator = mlp(1, state_dim * self.feature_dim, width, 3)

    def forward(self, features, times):
        p = self.propagator(times[:, None]).reshape(len(times), self.state_dim, self.feature_dim)
        return torch.einsum("bf,tdf->btd", self.encoder(features), p)


class SpectralConv1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(torch.randn(width, width, modes, dtype=torch.cfloat) / width)

    def forward(self, x):
        xf = torch.fft.rfft(x)
        out = torch.zeros(x.shape[0], x.shape[1], xf.shape[-1], dtype=xf.dtype, device=x.device)
        m = min(self.modes, xf.shape[-1])
        out[:, :, :m] = torch.einsum("bim,iom->bom", xf[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out, n=x.shape[-1])


class FSPDENO(nn.Module):
    """Appendix D Eq 49-50: spatial FNO -> stochastic temporal coefficients."""
    def __init__(self, in_channels, temporal_basis, width=24, modes=12, layers=6):
        super().__init__()
        self.register_buffer("temporal_basis", torch.as_tensor(temporal_basis, dtype=torch.float32))
        self.lift = nn.Conv1d(in_channels, width, 1)
        self.spectral = nn.ModuleList([SpectralConv1d(width, modes) for _ in range(layers)])
        self.local = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(layers)])
        self.project = nn.Sequential(nn.Conv1d(width, width * 2, 1), nn.GELU(),
                                     nn.Conv1d(width * 2, len(temporal_basis), 1))

    def forward(self, x):
        x = self.lift(x)
        for spectral, local in zip(self.spectral, self.local):
            x = torch.nn.functional.gelu(spectral(x) + local(x))
        return torch.einsum("bkx,kt->btx", self.project(x), self.temporal_basis)


def real_parameter_count(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
