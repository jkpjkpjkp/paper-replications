import numpy as np
import operator
from functools import reduce

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from constants import *


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.bias_only = nn.Parameter(torch.zeros(output_dim))
        for p in self.net.parameters():
            p.requires_grad = False
            p.data.zero_()

    def forward(self, x):
        return self.bias_only.expand(x.size(0), -1)


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


class Net(nn.Module):
    def __init__(self, X, Y, hidden_layer_sizes):
        super(Net, self).__init__()


        X_ = np.hstack([X, np.ones((X.shape[0], 1))])
        Theta = np.linalg.solve(X_.T.dot(X_), X_.T.dot(Y))

        self.lin = nn.Linear(X.shape[1], Y.shape[1])
        W, b = self.lin.parameters()
        W.data = torch.Tensor(Theta[:-1, :].T)
        b.data = torch.Tensor(Theta[-1, :])


        layer_sizes = [X.shape[1]] + hidden_layer_sizes
        layers = reduce(operator.add,
            [[nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(p=0.2)]
                for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], Y.shape[1])]
        self.net = nn.Sequential(*layers)
        self.sig = Parameter(torch.ones(1, Y.shape[1], device=DEVICE))

    def forward(self, x):
        return self.lin(x) + self.net(x),\
            self.sig.expand(x.size(0), self.sig.size(1))

    def set_sig(self, X, Y):
        Y_pred = self.lin(X) + self.net(X)
        var = torch.mean((Y_pred - Y) ** 2, 0)
        self.sig.data = torch.sqrt(var).data.unsqueeze(0)


class PolicyNet(nn.Module):
    def __init__(self, x_dim, params):
        super().__init__()
        layer_sizes = [x_dim, 1024, 1024, params['n']]
        layers = reduce(operator.add,
            [[nn.Linear(a, b), nn.ReLU()] for a, b in zip(layer_sizes[0:-2], layer_sizes[1:-1])])
        layers += [nn.Linear(layer_sizes[-2], layer_sizes[-1])]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
