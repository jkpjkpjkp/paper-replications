import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class GaussianMLP(nn.Module):
    def __init__(self, x_dim, hidden_dims, n):
        super().__init__()
        layers = []
        prev = x_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
            prev = h
        self.feature = nn.Sequential(*layers)
        self.mu_head = nn.Linear(prev, n)
        self.logvar_head = nn.Linear(prev, n)

    def forward(self, x):
        h = self.feature(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

class PortfolioPolicyNet(nn.Module):
    def __init__(self, x_dim: int, n: int, hidden=(1024, 1024)):
        super().__init__()
        layers = []
        in_dim = x_dim
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
            ]
            in_dim = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.backbone(x))
        z = F.softmax(logits, dim=-1)
        return z
