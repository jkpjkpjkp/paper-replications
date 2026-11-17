import torch
import torch.nn.functional as F
import time
import wandb
from util import ReplayBufferW
from cvxpy_powers_sched import dg_dz, cvxpy_ps_parallel

class DiffusionCvxpyModuleReplay(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, interval, distr_est=True, buffer_size=5000):
        super().__init__()
        self.diffusion = diffusion

        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_ps_parallel(params, mc_samples)
        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.total_samples = mc_samples
        self.replay_buffer = ReplayBufferW(capacity=buffer_size, device=diffusion.device)
        self.cnt = -1
        self.interval = interval
        self.old_elbo_losses = None

    def _sample(self, x, test_mode=False):
        batch_size = x.shape[0]
        x = x.repeat(self.total_samples, 1)
        time_start = time.time()
        with torch.no_grad():
            y_preds = self.diffusion.sample_elbo(x, test_mode=True)
        time_end = time.time()

        wandb.log({
            f"time/time_for_sampling_{self.total_samples}_{batch_size}" : time_end - time_start,
        })

        return y_preds

    def forward(self, _x, idx, epoch, test_mode=False):
        self.cnt += 1
        if epoch % self.interval == 0 or epoch == -1:
            _y_preds = self._sample(_x, test_mode=test_mode)
            ori_y_preds = _y_preds.clone()

            if self.total_samples > 1:
                _y_preds = _y_preds.contiguous().view(self.total_samples, -1, self.n)
                _y_preds = _y_preds.permute(1, 0, 2).contiguous()
            else:
                _y_preds = _y_preds.unsqueeze(0).permute(1, 0, 2).contiguous()


            if self.mc_samples > 1:
                _y_preds_input = _y_preds[:, :self.mc_samples, :]
            else:
                _y_preds_input = _y_preds

            time_start = time.time()
            with torch.no_grad():
                z_star_tuple, info = self.cvxpy_layer(_y_preds_input)
            time_end = time.time()

            z_star_tensor = z_star_tuple[0]
        else:


            _y_preds_sample, _x_sample, _w_sample, _z_star_sample, _info_sample, _elbo_sample = self.replay_buffer.get_many(idx)


        diffusion = self.diffusion
        device = diffusion.device
        distr_est = self.distr_est

        _diffusion_params = tuple(p for p in diffusion.model_net.parameters() if p.requires_grad)

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, x, idx, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, x, idx, *diffusion_params)
                return z_star

            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, x, idx, *diffusion_params = ctx.saved_tensors
                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]

                assert distr_est
                if epoch % self.interval == 0 or epoch == -1:
                    inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device)
                    B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                    n = dl_dz_star.shape[1]
                    dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))
                    if y_preds.ndim == 3:
                        y_preds_mc = y_preds.permute(1, 0, 2).contiguous()
                    else:
                        y_preds_mc = y_preds.unsqueeze(0)
                    dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)
                    dg_dz_all = dg_dz_flat.contiguous().view(self.total_samples, B, n)
                    dg_dz_pad = F.pad(dg_dz_all, (0, m - n))
                    dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).expand(self.total_samples, -1, -1)
                    inv_matrices_exp   = inv_matrices.unsqueeze(0).expand(self.total_samples, -1, -1, -1)
                    w = - torch.einsum('sbi,sbij,sbj->sb',
                            dl_dz_star_pad_exp,
                            inv_matrices_exp,
                            dg_dz_pad)


                    _y_preds_sample, _x_sample, _w_sample = ori_y_preds, x.repeat(self.total_samples, 1), w.detach()
                else:
                    _y_preds_sample, _x_sample, _w_sample, _, _, _elbo_sample = self.replay_buffer.get_many(idx)

                with torch.enable_grad():
                    time_start = time.time()
                    if epoch % self.interval == 0 or epoch == -1:
                        _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample, _x_sample)
                    else:
                        _elbo_losses = self.diffusion.compute_elbo_grad_vec(_y_preds_sample.reshape(-1, self.n), _x_sample.repeat_interleave(self.total_samples, dim=0))

                    _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1)


                    time_end = time.time()


                    if epoch % self.interval == 0:
                        weighted_elbo = (_w_sample.detach() * _elbo_losses).mean()
                        self.replay_buffer.put(idx, y_pred=y_preds, x=x, weight=w.detach(), z_star=z_star, info=info, elbo=_elbo_losses.detach())
                    elif epoch == -1:
                        weighted_elbo = (_w_sample.detach() * _elbo_losses).mean()
                    else:
                        is_weight = torch.exp(_elbo_losses - _elbo_sample).detach()
                        is_weight = is_weight.clamp(min=0, max=10)

                        weighted_elbo = (is_weight * _w_sample.detach() * _elbo_losses).mean()
                        print(f"is_weight = {is_weight.mean()}")


                    weighted_elbo_grads = torch.autograd.grad(
                        weighted_elbo,
                        diffusion_params,
                        retain_graph=False,
                        allow_unused=False,
                        create_graph=False
                    )

                time_end = time.time()


                return None, None, None, None, *tuple(weighted_elbo_grads)

        if epoch % self.interval == 0 or epoch == -1:
            return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, _x, idx, *_diffusion_params)
        else:
            return DiffusionCvxpyFn.apply(_y_preds_sample, _z_star_sample, _x_sample, idx, *_diffusion_params)
