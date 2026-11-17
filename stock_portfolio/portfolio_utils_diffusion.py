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

def train_portfolio_diffusion(model, layer, optimizer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.train()
    alpha = params["alpha"]
    train_losses, train_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()

            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)
            z_star = layer(features.float())
            z_star = z_star.squeeze(1)
            dfl_loss = _dfl_loss(labels, z_star, alpha)

            diffusion_loss = 0.0
            num_samples = 10
            for _ in range(num_samples):
                t = torch.randint(0, model.timesteps, (features.shape[0],), device=device).long()
                diffusion_loss += model.diffusion_loss(labels, t, features)
            diffusion_loss /= num_samples

            dfl_grad_norm = get_grad_norm(dfl_loss.mean(), model.model_net.parameters())
            mse_grad_norm = get_grad_norm(diffusion_loss, model.model_net.parameters())
            print(f"dfl_grad_norm = {dfl_grad_norm.item():.4f}, mse_grad_norm = {mse_grad_norm.item():.4f}")
            gamma = 0.5
            loss = gamma * (mse_grad_norm / dfl_grad_norm + 1e-8).detach() * dfl_loss.mean() + (1 - gamma) * diffusion_loss

            if not evaluate:
                optimizer.zero_grad()
                backward_start_time = time.time()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.model_net.parameters(), float('inf'))
                optimizer.step()
                backward_time += time.time() - backward_start_time
            else:
                grad_norm = 0

            train_objs.append(dfl_loss.mean().detach().item())

            tqdm_loader.set_postfix(obj=f'{loss.detach().item():.2f}', grad_norm=f'{grad_norm:.2e}')

    average_loss = 0
    average_obj = np.mean(train_objs)
    return average_loss, average_obj, (forward_time, inference_time, qp_time, backward_time)

@torch.no_grad()
def validate_portfolio_diffusion(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.eval()
    alpha = params["alpha"]

    loss_fn = torch.nn.MSELoss()
    validate_losses, validate_objs = [], []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)

            z_star = layer(features.float())
            z_star = z_star.squeeze(1)

            obj = _dfl_loss_linear(labels, z_star)

            validate_objs.append(obj.item())

    average_loss = 0
    average_obj = np.mean(validate_objs)

    return average_loss, average_obj

@torch.no_grad()
def test_portfolio_diffusion(model, layer, epoch, dataset, params, device='cpu', evaluate=False):
    model.model_net.eval()
    alpha = params["alpha"]
    loss_fn = torch.nn.MSELoss()
    test_losses, test_objs = [], []
    test_opts = []

    forward_time, inference_time, qp_time, backward_time = 0, 0, 0, 0
    x_dim = params["x_dim"]
    y_dim = params["n"]

    with tqdm.tqdm(dataset) as tqdm_loader:
        for batch_idx, (features, covariance_mat, labels) in enumerate(tqdm_loader):
            forward_start_time = time.time()
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, x_dim)
            labels = labels.reshape(-1, y_dim)

            if evaluate:
                layer = cvxpy_portfolio_parallel_kkt(params, mc_samples=1)
                _res = layer(labels)
                z_star = _res[0][0]
                obj = _dfl_loss_linear(labels, z_star.float())
            else:
                z_star = layer(features.float())
                z_star = z_star.squeeze(1)

                obj = _dfl_loss_linear(labels, z_star)

            test_objs.append(obj.item())

    average_loss = 0
    average_obj = np.mean(test_objs)
    return average_loss, average_obj

def eval_diffusion(which, diffusion, train_loader, test_loader, params, mc_samples, save_folder):
    layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
    device = diffusion.device
    with torch.no_grad():
        diffusion.model_net.eval()
        train_rmse_list = []
        train_dfl_list = []
        for batch_idx, (features, covariance_mat, labels) in enumerate(train_loader):
            features, covariance_mat, labels = features.float().to(device), covariance_mat.float().to(device), labels.float().to(device)
            features = features.reshape(-1, params["x_dim"])
            labels = labels.reshape(-1, params["n"])

            y_pred_train = diffusion.sample_elbo(features.repeat(mc_samples, 1), test_mode=True)
            y_pred_train = y_pred_train.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            train_rmse = rmse_loss(y_pred_train, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            train_rmse_list.append(train_rmse)

            z_star_train, _ = layer(y_pred_train.double())
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

            y_pred_test = diffusion.sample_elbo(features.repeat(mc_samples, 1), test_mode=True)
            y_pred_test = y_pred_test.contiguous().view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous()

            test_rmse = rmse_loss(y_pred_test, labels.repeat(mc_samples, 1).view(mc_samples, -1, params['n']).permute(1, 0, 2).contiguous())
            test_rmse_list.append(test_rmse)
            z_star_test, _ = layer(y_pred_test.double())
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

        print(f"test_dfl = {test_dfl:.4f}")
