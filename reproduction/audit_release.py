"""Produce concrete evidence of released-code issues without editing upstream files."""
import ast
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils import data as data_utils
from torch.utils.data.sampler import SubsetRandomSampler

ROOT = Path(__file__).resolve().parents[1]


def main():
    tree = ast.parse((ROOT/'stock_portfolio/data_utils.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='generateDataset')
    ns = dict(np=np, torch=torch, data_utils=data_utils, SubsetRandomSampler=SubsetRandomSampler)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'upstream_generateDataset','exec'),ns)
    class Dummy:
        def load_pytorch_data(self):
            return (torch.zeros(30,5,28),torch.zeros(30,5,1),list(range(28)),
                    torch.zeros(30,5,9),['y'],list(range(30)),list(range(5)))
    try:
        ns['generateDataset'](Dummy(),n=3,num_samples=30)
        reshape = 'No error'
    except Exception as e:
        reshape = f'{type(e).__name__}: {e}'
    torch.manual_seed(7)
    y = torch.randn(2,4,3,dtype=torch.double,requires_grad=True)
    z = torch.randn(2,3,dtype=torch.double)
    alpha=1e-6
    jac = torch.autograd.functional.jacobian(lambda a:alpha*a*(a*z[:,None]).sum(-1,keepdim=True)-a,y)
    exact = torch.stack([torch.stack([jac[b,m,:,b,m,:] for m in range(4)]) for b in range(2)])
    released = torch.diag_embed(2*alpha*y*z[:,None])
    evidence = {
        'upstream_commit':(ROOT/'sources/upstream_commit.txt').read_text().strip(),
        'generateDataset_runtime_exception':reshape,
        'diffusion_reparameterization_cross_derivative_max_error':float((exact-released).abs().max().detach()),
        'issues':[
            {'file':'stock_portfolio/main.py','issue':'wandb used without import; time, sys also missing.'},
            {'file':'stock_portfolio/data_utils.py','issue':'Features normalized on entire sample before chronological splitting.'},
            {'file':'stock_portfolio/main.py','issue':'generateDataset selects stocks before per-run seeding; called again for pretraining, selecting a new basket.'},
            {'file':'stock_portfolio/diffusion_opt.py','issue':'Diffusion pretrained checkpoint filename omits run seed and stock IDs.'},
            {'file':'stock_portfolio/solver_reparam.py','issue':'Cross derivative is implemented as diag(2*alpha*y*z), missing alpha*y*z.T plus alpha*(y.T*z)*I-I.'},
            {'file':'stock_portfolio/solver_mlp.py','issue':'Reparameterization uses sample-averaged mixed derivative for every sample rather than each sample derivative.'},
            {'file':'stock_portfolio/solver_mlp.py','issue':'Gaussian score loss clamps log-density to <=0; valid positive log-densities can have zero gradient. Also has extra batch scaling.'},
            {'file':'stock_portfolio/portfolio_utils_diffusion.py','issue':'Validation/test means average batch means with equal weight, rather than weighting by batch size.'},
            {'file':'stock_portfolio/data_utils.py','issue':'28 features per asset; paper appendix describes 28 total including PCA and volume. Volume is excluded in release.'},
            {'file':'stock_portfolio/diffusion_opt.py','issue':'Three 1024-unit hidden layers; Table 2 describes two. Denoising score surrogate uses unweighted MSE in released code.'},
            {'file':'stock_portfolio/cvxpy_stock_portfolio.py','issue':'alpha=1e-6 in main and redundant active box bounds make nearly linear, often vertex-optimal portfolios; ridge=1e-6 creates approximate backward gradients.'}
        ]}
    (ROOT/'results/release_audit.json').write_text(json.dumps(evidence,indent=2))
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':
    main()
