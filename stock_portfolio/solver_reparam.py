import torch
import torch.nn.functional as F
import time
from cvxpy_stock_portfolio import cvxpy_portfolio_parallel_kkt

class DiffusionCvxpyModule(torch.nn.Module):
    def __init__(self, diffusion, params, mc_samples, distr_est=True, resample=False):
        super().__init__()
        self.diffusion = diffusion

        self.mc_samples = mc_samples
        self.cvxpy_layer = cvxpy_portfolio_parallel_kkt(params, mc_samples)
        self.params = params
        self.alpha = params["alpha"]
        self.n = params["n"]
        self.eps = 1e-12
        self.distr_est = distr_est

    def _sample(self, x, test_mode=False):
        x = x.repeat(self.mc_samples, 1)

        y_preds = self.diffusion.sample_elbo(x, test_mode=test_mode)
        print(f"y_preds.max = {y_preds.max()}, y_preds.min = {y_preds.min()}")


        return y_preds

    def forward(self, x, test_mode=False):
        time_start = time.time()
        _y_preds = self._sample(x, test_mode=test_mode)
        time_end = time.time()
        print(f"Time taken for sampling: {time_end - time_start}")
        if self.mc_samples > 1:
            _y_preds = _y_preds.view(self.mc_samples, -1, self.n)
            _y_preds = _y_preds.permute(1, 0, 2).contiguous()

        _y_preds_copy = _y_preds.clone().detach()

        time_start = time.time()
        with torch.no_grad():
            z_star_tuple, info = self.cvxpy_layer(_y_preds_copy)
        time_end = time.time()
        print(f"Time taken for cvxpy layer: {time_end - time_start}")
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
                y_preds, z_star = ctx.saved_tensors

                kkt = torch.from_numpy(info["KKT_matrices"]).to(x.device)

                B, m = kkt.shape[0], kkt.shape[1]
                n = dl_dz_star.shape[1]

                dl_dz_star_padded = F.pad(dl_dz_star, (0, m - n))
                z_star_unsq = z_star.unsqueeze(1)

                d2f_dzy = 2 * self.alpha * y_preds * z_star_unsq
                d2f_dzy = torch.diag_embed(d2f_dzy)
                d2f_dzy_padded = F.pad(d2f_dzy, (0, 0, 0, m - n))

                kkt_exp = kkt.unsqueeze(1).repeat(1, self.mc_samples, 1, 1)
                k_inv_dg = torch.linalg.solve(kkt_exp, d2f_dzy_padded)

                for _ in range(self.mc_samples):

                    dl_dy = - torch.einsum('bi,bsij->bsj', dl_dz_star_padded, k_inv_dg)

                dl_dy = dl_dy / float(self.mc_samples)

                return dl_dy, None

        return DiffusionCvxpyFn.apply(_y_preds, z_star_tensor)
