import torch
import torch.nn.functional as F
import time
import wandb
from cvxpy_powers_sched import dg_dz, cvxpy_ps_parallel


class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion

        self.mc_samples = mc_samples
        self.total_samples = mc_samples

        self.cvxpy_layer =  cvxpy_ps_parallel(params, self.mc_samples)

        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est
        self.resample = resample

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.total_samples, 1)
        time_start = time.time()

        y_preds = self.diffusion.sample_elbo(x, test_mode=True)
        time_end = time.time()


        time_start = time.time()
        elbo_losses = self.diffusion.compute_elbo_grad_vec(y_preds, x)
        elbo_losses = elbo_losses.contiguous().view(self.total_samples, -1)
        time_end = time.time()


        return y_preds, elbo_losses

    def forward(self, x, test_mode=False):
        _y_preds, _elbo_losses = self._sample(x, test_mode=True)
        _elbo_losses = _elbo_losses.contiguous().view(self.total_samples, -1)

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

        diffusion = self.diffusion
        device = diffusion.device


        _diffusion_params = tuple(
            p
            for name, p in diffusion.model_net.named_parameters()
            if p.requires_grad and not name.startswith("v_head")
        )

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star, *diffusion_params):
                ctx.save_for_backward(y_preds, z_star, *diffusion_params)
                return z_star

            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star, *diffusion_params = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                if y_preds.ndim == 3:
                    y_preds_mc = y_preds.permute(1, 0, 2).contiguous()
                else:
                    y_preds_mc = y_preds.unsqueeze(0)


                dg_dz_flat = dg_dz(y_preds_mc.reshape(-1, n), z_star.repeat(self.total_samples, 1), self.params)
                dg_dz_all = dg_dz_flat.reshape(self.total_samples, B, n)
                dg_dz_pad = F.pad(dg_dz_all, (0, m - n))

                weighted_elbo_grads = [torch.zeros_like(p) for p in diffusion_params]

                dl_dz_star_pad_exp = dl_dz_star_padded.unsqueeze(0).repeat(self.total_samples, 1, 1)
                inv_matrices_exp   = inv_matrices.unsqueeze(0).repeat(self.total_samples, 1, 1, 1)
                w = - torch.einsum('sbi,sbij,sbj->sb',
                        dl_dz_star_pad_exp,
                        inv_matrices_exp,
                        dg_dz_pad)

                with torch.enable_grad():
                    wandb.log({"_elbo_losses_max": _elbo_losses.max(), "_elbo_losses_min": _elbo_losses.min()})
                    weighted_elbo = (w.detach() * _elbo_losses).sum() / self.total_samples

                weighted_elbo_grads = torch.autograd.grad(
                    weighted_elbo,
                    diffusion_params,
                    retain_graph=False,
                    allow_unused=False,
                    create_graph=False
                )

                time_end = time.time()

                return None, None, *tuple(weighted_elbo_grads)

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor, *_diffusion_params)
