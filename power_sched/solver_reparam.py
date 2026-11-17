import torch
import torch.nn.functional as F
import time
from cvxpy_powers_sched import cvxpy_ps_parallel


class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion

        self.mc_samples = mc_samples

        self.cvxpy_layer = cvxpy_ps_parallel(params, mc_samples)
        self.gs = params["gamma_under"]
        self.ge = params["gamma_over"]
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.mc_samples, 1)

        y_preds = self.diffusion.sample_elbo(x, test_mode=test_mode)

        return y_preds

    def forward(self, x, test_mode=False):
        time_start = time.time()
        try:
            y_preds = self._sample(x, test_mode=test_mode)
        except RuntimeError as e:
            print(f"Error: {e}")
            torch.cuda.empty_cache()
            raise e
        time_end = time.time()

        if self.mc_samples > 1:
            y_preds = y_preds.contiguous().view(self.mc_samples, -1, self.n)
            y_preds = y_preds.permute(1, 0, 2).contiguous()

        y_preds_copy = y_preds.clone().detach()

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(y_preds_copy)
        time_end = time.time()

        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device
        distr_est = self.distr_est

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star):
                ctx.save_for_backward(y_preds, z_star)
                return z_star

            @staticmethod
            def backward(ctx, dl_dz_star):
                y_preds, z_star = ctx.saved_tensors
                inv_matrices = torch.from_numpy(info["inv_matrices"]).float().to(device)

                B, m = inv_matrices.shape[0], inv_matrices.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))

                dg_dy = - torch.eye(n, device=device)
                dg_dy = dg_dy.unsqueeze(0).expand(B, -1, -1)
                dg_dy_padded = F.pad(dg_dy, (0, 0, 0, m - n))

                weighted_grad_y_preds_list = []
                for i in range(self.mc_samples):

                    dl_dy = - torch.einsum('bi,bij,bjk->bk',
                                dl_dz_star_padded,
                                inv_matrices,
                                dg_dy_padded)
                    weighted_grad_y_preds_list.append(dl_dy)

                if self.mc_samples > 1:
                    weighted_grad_y_preds = torch.stack(weighted_grad_y_preds_list, dim=1)
                    weighted_grad_y_preds = weighted_grad_y_preds / float(self.mc_samples)
                else:
                    weighted_grad_y_preds = weighted_grad_y_preds_list[0]


                return weighted_grad_y_preds, None

        return DiffusionCvxpyFn.apply(y_preds, z_star_tensor)
