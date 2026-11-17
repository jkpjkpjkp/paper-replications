import os
import time
import tqdm
import torch
import numpy as np
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt

def rmse_loss(mu_pred, Y_actual):
    if mu_pred.ndim == 3:
        return ((mu_pred - Y_actual)**2).mean(dim=(0,1)).sqrt().data.cpu().numpy()
    else:
        return ((mu_pred - Y_actual)**2).mean(dim=0).sqrt().data.cpu().numpy()

def _dfl_loss(y, z, alpha):
    y_z = (y * z).sum(dim=-1)
    return (0.5 * alpha * y_z**2 - y_z).sum()

def _dfl_loss_linear(y, z):
    y_z = (y * z).sum(dim=-1)
    return y_z.mean()

def get_grad_norm(loss, params):
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=False)
    grads = [g for g in grads if g is not None]
    return torch.sqrt(sum(g.pow(2).sum() for g in grads) + 1e-12)

def train_portfolio_policy(model, optimizer, epoch, dataset, params, device='cpu', evaluate=False):
    model.train()
    alpha = params["alpha"]
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()

            features, covariance_mat, labels = features.to(device), covariance_mat.to(device), labels.to(device).float()
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])
            n = len(covariance_mat)

            y_preds = model(features.float())
            dfl_loss = _dfl_loss(labels, y_preds.float(), alpha)

            optimizer.zero_grad()
            backward_start_time = time.time()
            dfl_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            optimizer.step()
            backward_time += time.time() - backward_start_time

            train_objs.append(dfl_loss.mean().detach().item())

            tqdm_loader.set_postfix(obj=f'{dfl_loss.mean().detach().item():.2f}', grad_norm=f'{grad_norm:.2e}')

    average_loss    = 0
    average_obj     = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

@torch.no_grad()
def validate_portfolio_policy(model, epoch, dataset, params, device='cpu', evaluate=False):
    model.eval()
    alpha = params["alpha"]

    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()

            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_preds = model(features.float())
            dfl_loss = _dfl_loss_linear(labels, y_preds.float())

            validate_objs.append(dfl_loss.item())

    average_loss    = 0
    average_obj     = np.mean(validate_objs)

    return average_loss, average_obj

@torch.no_grad()
def test_portfolio_policy(model, epoch, dataset, params, device='cpu', evaluate=False):
    model.eval()
    alpha = params["alpha"]
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()

            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_preds = model(features.float())
            dfl_loss = _dfl_loss_linear(labels, y_preds.float())

            test_objs.append(dfl_loss.item())

    average_loss    = 0
    average_obj     = np.mean(test_objs)
    return average_loss, average_obj

def eval_net(which, model, train_loader, test_loader, params, save_folder, mc_samples=None):
    mc_samples = 1
    solver = cvxpy_portfolio_parallel_kkt(params, 1)
    device = next(model.parameters()).device

    with torch.no_grad():
        model.eval()
        train_rmse_list = []
        train_dfl_list = []
        for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_preds_train = model(features.repeat(mc_samples, 1))
            y_preds_train = y_preds_train.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            train_rmse = rmse_loss(y_preds_train, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            train_rmse_list.append(train_rmse)

            z_star_train, _ = solver(y_preds_train.double())
            z_star_train = z_star_train[0]
            train_loss_task = _dfl_loss_linear(
                labels, z_star_train)
            train_dfl_list.append(train_loss_task.item())

        train_rmse = np.mean(train_rmse_list)
        train_dfl = np.mean(train_dfl_list)

        test_rmse_list = []
        test_dfl_list = []
        for batch_idx, (features, covariance_mat, labels) in enumerate(test_loader):
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_preds_test = model(features.repeat(mc_samples, 1))
            y_preds_test = y_preds_test.view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            test_rmse = rmse_loss(y_preds_test, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            test_rmse_list.append(test_rmse)

            z_star_test, _ = solver(y_preds_test.double())
            z_star_test = z_star_test[0]
            test_loss_task = _dfl_loss_linear(
                labels, z_star_test)
            test_dfl_list.append(test_loss_task.item())

        test_rmse = np.mean(test_rmse_list)
        test_dfl = np.mean(test_dfl_list)

        with open(
            os.path.join(save_folder, '{}_train_rmse'.format(which)), 'wb') as f:
            np.save(f, train_rmse)

        with open(
            os.path.join(save_folder, '{}_test_rmse'.format(which)), 'wb') as f:
            np.save(f, test_rmse)

        torch.save(train_dfl,
            os.path.join(save_folder, '{}_train_task'.format(which)))
        torch.save(test_dfl,
            os.path.join(save_folder, '{}_test_task'.format(which)))

        print(f"train_dfl = {train_dfl:.4f}, train_rmse = {train_rmse:.4f}, test_dfl = {test_dfl:.4f}, test_rmse = {test_rmse:.4f}")
