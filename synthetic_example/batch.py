#!/usr/bin/env python3

import torch
from constants import *


def get_vars(batch_sz, X_test_t, Y_test_t):
    device = X_test_t.device
    batch_data_ = torch.empty(batch_sz, X_test_t.size(1), device=device)
    batch_targets_ = torch.empty(batch_sz, Y_test_t.size(1), device=device)
    return batch_data_, batch_targets_


def get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t,
    loss_fn, var_getter_fn, params=None):
    test_cost = 0
    batch_data_, batch_targets_ = var_getter_fn(batch_sz, X_test_t, Y_test_t)
    size = batch_sz

    for i in range(0, X_test_t.size(0), batch_sz):
        if i + batch_sz > X_test_t.size(0):
            size = X_test_t.size(0) - i
            batch_data_, batch_targets_ = var_getter_fn(size, X_test_t, Y_test_t)

        batch_data_.data[:] = X_test_t[i:i+size]
        batch_targets_.data[:] = Y_test_t[i:i+size]

        preds = model(batch_data_)
        batch_cost = loss_fn(preds, batch_targets_, params).sum()
        test_cost += (batch_cost.item() - test_cost) * size / (i + size)

    return test_cost


def get_cost(batch_sz, epoch, model, X_test_t, Y_test_t, loss_fn, params):
    return get_cost_helper(batch_sz, epoch, model, X_test_t, Y_test_t,
        loss_fn, get_vars, params=params)
