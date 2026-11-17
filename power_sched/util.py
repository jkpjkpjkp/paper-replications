import torch
import numpy as np
from collections import OrderedDict, namedtuple
import random


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    def __len__(self):
        return len(self.xs)
    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], idx


CacheEntryW = namedtuple(
    "CacheEntry",
    ["y_pred", "x", "weight", "z_star", "info", "elbo"]
)

CacheEntry = namedtuple(
    "CacheEntry",
    ["x", "y", "elbo"]
)


class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self._buffer = OrderedDict()

    @torch.no_grad()
    def put(self, idx_tensor, x_rep_bmc, y_rep_bmc, elbo_bmc):
        for i, k in enumerate(idx_tensor):
            k = int(k)
            if k in self._buffer:
                self._buffer.move_to_end(k)
            self._buffer[k] = CacheEntry(
                x_rep_bmc[i].cpu(),
                y_rep_bmc[i].cpu(),
                elbo_bmc[i].cpu()
            )
            if len(self._buffer) > self.capacity:
                self._buffer.popitem(last=False)

    def get(self, idx: int):
        entry = self._buffer.get(int(idx))
        if entry is not None:
            self._buffer.move_to_end(int(idx))
        return entry

    def get_many(self, idx_list):
        entries = []
        for k in idx_list:
            k_int = int(k)
            e = self._buffer.get(k_int)
            if e is not None:
                self._buffer.move_to_end(k_int)
            entries.append(e)
        if len([e.x for e in entries if e is not None]) == 0:
            return None, None, None
        x_ls = torch.stack([e.x for e in entries if e is not None], dim=0).to(self.device)
        y_ls = torch.stack([e.y for e in entries if e is not None], dim=0).to(self.device)
        elbo_ls = torch.stack([e.elbo for e in entries if e is not None], dim=0).to(self.device)
        return y_ls, x_ls, elbo_ls

    def sample(self, k):
        ids = random.sample(list(self._buffer.keys()), k)
        batch = [self._buffer[i] for i in ids]
        return ids, batch


class ReplayBufferW:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self._buffer = OrderedDict()

    def get_many(self, idx_tensor):
        y_ls, x_ls, w_ls, z_ls, info_ls, elbo_ls = [], [], [], [], [], []

        for t in idx_tensor:
            entry = self.get(int(t))
            if entry is None:
                return None
            y_ls.append(entry.y_pred)
            x_ls.append(entry.x)
            w_ls.append(entry.weight)
            z_ls.append(entry.z_star)
            info_ls.append(torch.tensor(entry.info))
            elbo_ls.append(entry.elbo)

        y_batch = torch.stack(y_ls, dim=0).to(self.device)
        x_batch = torch.stack(x_ls, dim=0).to(self.device)
        w_batch = torch.stack(w_ls, dim=1).to(self.device)
        z_batch = torch.stack(z_ls, dim=0).to(self.device)
        info_batch = torch.stack(info_ls, dim=0).to(self.device)
        elbo_batch = torch.stack(elbo_ls, dim=1).to(self.device)

        return y_batch, x_batch, w_batch, z_batch, info_batch, elbo_batch

    def put(self, idx_list, **fields):
        for i, t in enumerate(idx_list):
            key = int(t)
            self.put_single(
                key,
                y_pred=fields["y_pred"][i],
                x=fields["x"][i],
                weight=fields["weight"][:, i],
                z_star=fields["z_star"][i],
                info=fields["info"]['KKT_matrices'][i],
                elbo=fields["elbo"][:, i],
            )

    def get(self, idx):
        entry = self._buffer.get(idx)
        if entry is not None:
            self._buffer.move_to_end(idx)
        return entry

    def put_single(self, idx, *, y_pred, x, weight, z_star, info, elbo):
        if idx in self._buffer:
            self._buffer.move_to_end(idx)
        self._buffer[idx] = CacheEntryW(
            y_pred.detach().cpu(),
            x.detach().cpu(),
            weight.detach().cpu(),
            z_star.detach().cpu(),
            info,
            elbo.detach().cpu()
        )
        if len(self._buffer) > self.capacity:
            self._buffer.popitem(last=False)
            print("POP")

    def sample(self, k):
        ids = random.sample(list(self._buffer.keys()), k)
        batch = [self._buffer[i] for i in ids]
        return batch


def bger(x, y):
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def expandParam(X, nBatch, nDim):
    if X.ndimension() in (0, nDim) or X.nelement() == 0:
        return X, False
    elif X.ndimension() == nDim - 1:
        return X.unsqueeze(0).expand(*([nBatch] + list(X.size()))), True
    else:
        raise RuntimeError("Unexpected number of dimensions.")


def extract_nBatch(Q, p, G, h, A, b):
    dims = [3, 2, 3, 2, 3, 2]
    params = [Q, p, G, h, A, b]
    for param, dim in zip(params, dims):
        if param.ndimension() == dim:
            return param.size(0)
    return 1
