import torch
import torch.nn.functional as F
import time
from cvxpy_toy_kkt import cvxpy_toy_parallel_kkt, d2g_dzdy
from model_classes import MLP, GaussianMLP

class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion

        self.mc_samples = mc_samples

        self.params = params
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.mc_samples, 1)

        y_preds = self.diffusion.sample_elbo(x, test_mode=test_mode)

        return y_preds

    def forward(self, x, _y=None, test_mode=False):
        if _y is not None:
            _y_preds = _y
        else:
            time_start = time.time()
            _y_preds = self._sample(x, test_mode=test_mode)
            time_end = time.time()


        if self.mc_samples > 1:
            _y_preds = _y_preds.view(self.mc_samples, -1, self.n)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous()

        _y_preds_copy = _y_preds.clone().detach()

        time_start = time.time()
        with torch.no_grad():
            cvxpy_layer = cvxpy_toy_parallel_kkt(self.params, x.shape[0] * self.mc_samples)
            z_star_tuple, info = cvxpy_layer(_y_preds_copy.reshape(1, -1, self.n))
        time_end = time.time()

        z_star_tensor = z_star_tuple[0]

        diffusion = self.diffusion
        device = diffusion.device

        class DiffusionCvxpyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y_preds, z_star):
                ctx.save_for_backward(y_preds, z_star)
                return z_star

            @staticmethod
            def backward(ctx, dl_dz_star):
                time_start = time.time()
                y_preds, z_star = ctx.saved_tensors

                kkt = torch.from_numpy(info["KKT_matrices"]).float().to(x.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))


                _d2g_dzy = d2g_dzdy(y_preds, z_star, self.params)
                d2g_dzy_padded = F.pad(_d2g_dzy, (0, 0, 0, m - n))

                k_inv_dg = torch.linalg.solve(kkt, d2g_dzy_padded)

                weighted_grad_y_preds_list = []
                for _ in range(self.mc_samples):

                    dl_dy = - torch.einsum('bi,bij->bj', dl_dz_star_padded, k_inv_dg)
                    weighted_grad_y_preds_list.append(dl_dy)

                if self.mc_samples > 1:
                    weighted_grad_y_preds = torch.stack(weighted_grad_y_preds_list, dim=1)
                    weighted_grad_y_preds = weighted_grad_y_preds / float(self.mc_samples)
                else:
                    weighted_grad_y_preds = weighted_grad_y_preds_list[0]

                time_end = time.time()


                return weighted_grad_y_preds, None

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor)
