"""Read-only diagnostics of fixed checkpoints; no training or model selection."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes.model import Hermes, ranking_loss


def main():
    torch.set_num_threads(4)
    data = np.load(ROOT / 'data/sp500.npz')
    x = torch.tensor(data['features'])
    y = torch.tensor(data['returns'])
    output = {'scope': 'CPU float32 probes on first 32 training target dates, fixed CUDA-trained best checkpoints; no tuning',
              'stock_pair_count': 474**2,
              'sector_sizes': data['incidence'].sum(0).tolist(), 'runs': []}
    for suffix in ['', '-meanrank']:
        for seed in [0, 1, 2]:
            run = ROOT / f'results/sp500-full-seed{seed}{suffix}'
            ck = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            c = ck['config']
            model = Hermes(data['incidence'], hidden=c['hidden'],
                           scales=c['scales'], windows=c['windows'])
            model.load_state_dict(ck['model'])
            captured = {'initial': [], 'graph': []}
            handles = []
            for name, modules in [('initial', model.initial_heads), ('graph', model.graph_heads)]:
                for module in modules:
                    handles.append(module.register_forward_hook(
                        lambda mod, args, val, key=name: captured[key].append(val)))
            original_fuse = model.fuse

            def capture_fuse(edges):
                b, s, k, t, d = edges.shape
                flat = edges.detach().permute(0, 3, 1, 2, 4).reshape(b, t, s*k, d)
                z = flat @ model.metric_factor.detach().T
                norm = z.square().sum(-1, keepdim=True)
                dist = norm + norm.transpose(-1, -2) - 2*z @ z.transpose(-1, -2)
                diag = torch.eye(s*k, dtype=torch.bool)
                logits = dist.clamp_min(model.distance_eps).reciprocal().masked_fill(diag, 0.)
                adj = logits.softmax(-2)
                direct_dist = (z.double().unsqueeze(-2)-z.double().unsqueeze(-3)).square().sum(-1)
                captured['fusion'] = {
                    'columns_max_weight_gt_0999_fraction': float((adj.max(-2).values > .999).float().mean()),
                    'mean_column_entropy': float(-(adj * adj.clamp_min(1e-30).log()).sum(-2).mean()),
                    'maximum_entropy': math.log(s*k),
                    'off_diagonal_distance_floor_fraction': float((dist[..., ~diag] <= model.distance_eps).float().mean()),
                    'float64_direct_off_diagonal_distance_floor_fraction': float((direct_dist[..., ~diag] <= model.distance_eps).float().mean()),
                    'off_diagonal_negative_float32_distance_fraction': float((dist[..., ~diag] < 0).float().mean()),
                    'row_sum_min': float(adj.sum(-1).min()),
                    'row_sum_max': float(adj.sum(-1).max()),
                }
                return original_fuse(edges)

            model.fuse = capture_fuse
            days = torch.arange(16, 48)
            inputs = x[days[:, None] + torch.arange(-16, 0)].permute(0, 2, 1, 3)
            pred = model(inputs)
            target = y[days]
            loss, mse, rank = ranking_loss(pred, target, reduction=c['rank_reduction'])
            named = list(model.named_parameters())
            mse_live = (pred-target).square().mean()
            rank_live = loss - mse_live
            gradients = {}
            for name, objective in [('mse', mse_live), ('ranking', rank_live)]:
                grads = torch.autograd.grad(objective, [p for _, p in named], retain_graph=True, allow_unused=True)
                groups = {}
                for (key, _), grad in zip(named, grads):
                    if grad is not None:
                        group = key.split('.')[0]
                        groups[group] = groups.get(group, 0.) + float(grad.square().sum())
                gradients[name] = {key: math.sqrt(value) for key, value in groups.items()}
            initial = torch.stack(captured['initial']).sum(0).squeeze(-1)
            graph = torch.stack(captured['graph']).sum(0).squeeze(-1)
            torch.testing.assert_close(initial + graph, pred)
            assert data['valid'][days.numpy()].all()
            assert all(math.isfinite(v) for groups in gradients.values() for v in groups.values())
            hist = [json.loads(line) for line in (run/'history.jsonl').read_text().splitlines()]
            chosen = min(hist, key=lambda r:r['val_loss'])
            max_ic = max(hist, key=lambda r:r['val_IC'])
            output['runs'].append({
                'run':run.name, 'best_epoch':ck['epoch'],
                'train_probe_mse': float(mse), 'train_probe_ranking': float(rank),
                'ranking_to_mse_loss_ratio':float(rank/mse), 'gradient_norms':gradients,
                'initial_branch_daily_cross_section_std_mean':float(initial.detach().std(1, correction=0).mean()),
                'graph_branch_daily_cross_section_std_mean':float(graph.detach().std(1, correction=0).mean()),
                'target_daily_cross_section_std_mean':float(target.std(1, correction=0).mean()),
                'fusion':captured['fusion'], 'selected_epoch_validation_IC':chosen['val_IC'],
                'max_validation_IC':max_ic['val_IC'], 'max_validation_IC_epoch':max_ic['epoch'],
                'train_history_ranking_to_mse_at_selected_epoch': chosen['train_rank']/chosen['train_mse'],
            })
            for h in handles:
                h.remove()
    output['feature_shift'] = {
        name: {'max_abs_zscore':float(np.abs(data['features'][a:b]).max()),
               'rms_zscore':float(np.sqrt(np.mean(data['features'][a:b].astype(float)**2)))}
        for name, a, b in [('train',0,1006),('validation',1006,1259),('test',1259,1611)]}
    dest = ROOT/'results/exact-audit-2026-11-11/training-diagnostics.json'
    dest.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
