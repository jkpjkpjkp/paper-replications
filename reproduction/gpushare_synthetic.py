"""Bounded synthetic pilot using the released model, sampler, KKT and training loop.

Run from an isolated copy of this repository. No portfolio data are needed.
The JSON receipt explicitly records differences from the released launcher.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'synthetic_example'))
os.environ.update(WANDB_MODE='disabled', WANDB_DISABLED='true')

import numpy as np
import torch
import torch.nn.functional as F
from diffusion_opt import Diffsion


def chunked_elbo(self, x_t, s, num_iter=1000, requires_grad=True, resample=False):
    """Same time/noise draws and mean as release, recomputed in bounded chunks.

    Only parameter gradients are required by the released score estimator.
    Samples are intentionally detached, just as in compute_elbo_grad_vec.
    """
    if resample or not requires_grad:
        raise ValueError('This pilot implements the ordinary score estimator only')
    b, d = x_t.shape
    times = torch.randint(0, self.timesteps, (b, num_iter), device=x_t.device)
    noise = torch.randn(b, num_iter, d, device=x_t.device)
    params = tuple(self.model_net.parameters())
    chunk = 2048

    def values(x, context, start, end):
        rows = torch.arange(start, end, device=x.device) // num_iter
        t = times.reshape(-1)[start:end]
        eps = noise.reshape(-1, d)[start:end]
        pred = self.model_net(self.q_sample(x[rows], t, eps), t, context[rows])
        return F.mse_loss(pred, eps, reduction='none').sum(1), rows

    class Recompute(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, context, *weights):
            ctx.save_for_backward(x.detach(), context.detach())
            out = torch.zeros(b, device=x.device)
            for start in range(0, b * num_iter, chunk):
                loss, rows = values(x, context, start, min(start + chunk, b * num_iter))
                out.index_add_(0, rows, -loss / num_iter)
            return out

        @staticmethod
        def backward(ctx, upstream):
            x, context = ctx.saved_tensors
            accum = [torch.zeros_like(p) for p in params]
            for start in range(0, b * num_iter, chunk):
                with torch.enable_grad():
                    loss, rows = values(x, context, start, min(start + chunk, b * num_iter))
                    objective = (-loss * upstream[rows] / num_iter).sum()
                    grads = torch.autograd.grad(objective, params)
                for total, grad in zip(accum, grads):
                    total.add_(grad)
            return (None, None, *accum)

    return Recompute.apply(x_t.detach(), s.detach(), *params)


def verify_chunking():
    model = Diffsion(20, 10, device='cuda:0')
    model.model_net.eval()
    x = torch.randn(3, 10, device='cuda:0')
    context = torch.randn(3, 20, device='cuda:0')
    weight = torch.tensor([.3, -.7, 1.2], device='cuda:0')
    state = torch.cuda.get_rng_state()
    expected = model.compute_elbo_grad_vec(x, context, num_iter=700)
    eg = torch.autograd.grad((expected * weight).sum(), model.model_net.parameters())
    torch.cuda.set_rng_state(state)
    actual = chunked_elbo(model, x, context, num_iter=700)
    ag = torch.autograd.grad((actual * weight).sum(), model.model_net.parameters())
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=3e-4, atol=3e-5)
    return {'passed': True, 'max_loss_error': float((actual-expected).abs().max()),
            'max_parameter_gradient_error': max(float((a-e).abs().max()) for a,e in zip(ag,eg))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', choices=['release', 'paper'], default='release')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--output', required=True)
    ap.add_argument('--pretrain', type=int, default=1000)
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        raise RuntimeError('Choose an unused output directory')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    verified = verify_chunking()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    Diffsion.compute_elbo_grad_vec = chunked_elbo

    import main_toy
    import nets_toy
    import data_gen
    import constants
    constants.set_device(0)
    original_generate = data_gen.generate_data_multi_dim

    def generate(**kw):
        kw.update(p_pos=.8 if args.protocol == 'paper' else .9,
                  y_scale=1. if args.protocol == 'paper' else .5)
        x, y, info = original_generate(**kw)
        if args.protocol == 'paper':
            x.zero_()  # No contextual information, as stated in section 6.1.
        return x, y, info

    main_toy.generate_data_multi_dim = generate
    # Retain the training computation, but never select a model with test loss.
    source = inspect.getsource(nets_toy.run_task_net_diffusion)
    assert 'if test_loss < best_val - min_delta:' in source
    source = source.replace('if test_loss < best_val - min_delta:',
                            'if val_loss < best_val - min_delta:')
    source = source.replace('best_val     = test_loss', 'best_val     = val_loss')
    source = source.replace('epochs = 100 if args.pretrain_epochs == 0 else 10',
                            f'epochs = {args.epochs}')
    marker = '            total_training_time_list.append(total_training_time)'
    assert source.count(marker) == 1
    source = source.replace(marker, marker + '''
            torch.save({'epoch': epoch, 'model': diffusion.model_net.state_dict(),
                        'optimizer': opt.state_dict(), 'training_seconds': total_training_time,
                        'validation_loss': val_loss, 'test_loss': test_loss},
                       os.path.join(save_folder, 'latest_epoch.pt'))
''')
    exec(compile(source, '<validation-selected-release-loop>', 'exec'), nets_toy.__dict__)
    original_train = nets_toy.run_task_net_diffusion
    original_eval = nets_toy.eval_diffusion
    receipt = {
        'protocol': args.protocol, 'seed': args.seed, 'pretrain_epochs': args.pretrain,
        'dfl_epochs': args.epochs, 'batch_size': 32, 'mc_samples': 10,
        'sampler_steps': 1000, 'score_time_draws': 1000, 'learning_rate': 8e-6,
        'p_positive': .8 if args.protocol == 'paper' else .9,
        'return_scale': 1. if args.protocol == 'paper' else .5,
        'x': 'constant zero (no context)' if args.protocol == 'paper' else 'independent Gaussian noise',
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'chunking_test': verified, 'started_unix': time.time(), 'status': 'running',
        'paper_targets': {'diffusion_ts_task': .393, 'diffusion_score_task': .362},
        'deviations': ['Validation-selected checkpoint instead of test-selected checkpoint.',
                       'Seed initialized before split, separate checkpoint path per seed/protocol.',
                       'Exact score-gradient recomputation in chunks; all 1000 time draws retained.',
                       'Paper/released-code distribution conflict is tested explicitly; no exact-match claim.',
                       'Released three-hidden-layer architecture retained; paper table specifies two.'],
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT/'synthetic_example').rglob('*.py'))},
        'results': {},
    }

    def save_receipt():
        receipt['elapsed_seconds'] = time.time() - receipt['started_unix']
        receipt['peak_gpu_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        temp = out / 'receipt.tmp'
        temp.write_text(json.dumps(receipt, indent=2))
        temp.replace(out/'receipt.json')

    def evaluate(which, diffusion, variables, params, mc_samples, save_folder, layer=None):
        decisions = []
        original_factory = nets_toy.cvxpy_toy_parallel_kkt

        def capture_factory(*a, **kw):
            solve = original_factory(*a, **kw)

            def capture(y):
                result = solve(y)
                decisions.append(result[0][0].detach().cpu().numpy())
                return result

            return capture

        nets_toy.cvxpy_toy_parallel_kkt = capture_factory
        try:
            original_eval(which, diffusion, variables, params, mc_samples, save_folder, layer)
        finally:
            nets_toy.cvxpy_toy_parallel_kkt = original_factory
        assert len(decisions) == 3
        decision_map = dict(zip(['train', 'test', 'hold'], decisions))
        result = {}
        for split in ['train', 'hold', 'test']:
            result[f'{split}_task'] = float(torch.load(Path(save_folder)/f'{which}_{split}_task', weights_only=False))
            with open(Path(save_folder)/f'{which}_{split}_rmse', 'rb') as f:
                result[f'{split}_rmse'] = float(np.load(f))
            z = decision_map[split].reshape(-1).astype(np.float64)
            actual_y = variables[f'Y_{split}_'].detach().cpu().numpy().astype(np.float64)
            independent_loss = float(np.exp(-actual_y @ z).mean())
            np.testing.assert_allclose(independent_loss, result[f'{split}_task'], rtol=2e-5, atol=2e-6)
            scaled_z = receipt['return_scale'] * z
            p = receipt['p_positive']
            result[f'{split}_population_task'] = float(np.exp(np.sum(
                .5 * .15**2 * scaled_z**2 +
                np.logaddexp(np.log(p)-scaled_z, np.log1p(-p)+3*scaled_z))))
            result[f'{split}_independent_empirical_task'] = independent_loss
            result[f'{split}_max_bound_violation'] = float(max(0., -z.min(), z.max()-2.))
        np.savez(out/f'{which}_decisions.npz', **decision_map)
        receipt['results'][which] = result
        save_receipt()
        print('PILOT_METRICS ' + json.dumps({which: result}), flush=True)

    def train(model, train_loader, hold_loader, test_loader, layer, params, original_args,
              which, save_folder, other_layers=None):
        variables = {}
        for split, loader in [('train', train_loader), ('hold', hold_loader), ('test', test_loader)]:
            variables[f'X_{split}_'], variables[f'Y_{split}_'] = loader.dataset.tensors
        torch.save({k: v.cpu() for k,v in variables.items()}, out/'data_split.pt')
        cuda_state, cpu_state = torch.cuda.get_rng_state(), torch.get_rng_state()
        evaluate('two_stage_diffusion', model, variables, params, original_args.mc_samples, save_folder)
        torch.cuda.set_rng_state(cuda_state)
        torch.set_rng_state(cpu_state)
        return original_train(model, train_loader, hold_loader, test_loader, layer, params,
                              original_args, which, save_folder, other_layers)

    nets_toy.run_task_net_diffusion = train
    nets_toy.eval_diffusion = evaluate
    # Numerical verification must not perturb the experiment seed.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    save_receipt()
    sys.argv = ['main_toy.py', '--task', 'diffusion', '--use_distr_est', '--cuda_device', '0',
                '--nRuns', '1', '--mc_samples', '10', '--batch_size', '32', '--lr', '8e-6',
                '--seed', str(args.seed), '--pretrain_epochs', str(args.pretrain),
                '--save', str(out/'synthetic')]
    try:
        main_toy.main()
        receipt['status'] = 'completed_one_seed_protocol_check'
    except BaseException as error:
        receipt.update(status='failed', error=repr(error))
        raise
    finally:
        save_receipt()


if __name__ == '__main__':
    main()
