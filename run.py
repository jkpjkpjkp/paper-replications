"""Run a fixed-budget, validation-selected AlphaQCM IQN experiment."""
import argparse
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
import random
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import yaml
from implementation import (LocalStockData, CachedCalculator, SnapshotEnv, AlphaEnvCore,
                            CorrectedIQN, daily_metrics)
from alphagen.data.expression import Feature, FeatureType, Ref
from alphagen.models.alpha_pool import AlphaPool

ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    temporary = Path(str(path)+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
    temporary.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/pilot-csi300.json')
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    paper_method = cfg.get('method') == 'paper'
    calculator_class, agent_class = CachedCalculator, CorrectedIQN
    if paper_method:
        from paper_implementation import configure_paper_operators, PaperCalculator, PaperIQN, PaperPool
        configure_paper_operators()
        calculator_class, agent_class = PaperCalculator, PaperIQN
    provider = Path(cfg['provider']).expanduser()
    cfg['provider'] = str((ROOT / provider).resolve() if not provider.is_absolute() else provider)
    for key in ['steps', 'seed']:
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = out/'source'
    source.mkdir()
    for name in ['run.py', 'implementation.py', 'evaluate.py']:
        shutil.copy2(ROOT/name, source/name)
    if paper_method:
        for name in ['paper_implementation.py', '.agent/PAPER_PROTOCOL.txt']:
            (source/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, source/name)
    write_json(source/'config.json', cfg)
    started = time.time()
    torch.set_num_threads(cfg.get('threads', 2))
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(cfg['device'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    import qlib
    qlib.init(provider_uri=cfg['provider'], region='cn', kernels=1)
    from qlib.data import D
    calendar = pd.DatetimeIndex(D.calendar())
    datasets, calculators, manifest = {}, {}, {}
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20)/close-1
    for split, (start, end) in cfg['splits'].items():
        if cfg.get('purge_splits', True) and split in ['train', 'valid']:
            # Last label endpoint stays strictly before the next split.
            idx = calendar.searchsorted(pd.Timestamp(end), side='right')-1
            end = str(calendar[idx-20].date())
        data = LocalStockData(cfg['universe'], start, end, device=device)
        datasets[split] = data
        calculators[split] = calculator_class(data, target)
        manifest[split] = {'start': str(data.eval_dates[0].date()), 'end': end,
                           'days': data.n_days, 'stocks_union': data.n_stocks,
                           'tensor_shape': list(data.data.shape),
                           'finite_fraction': torch.isfinite(data.data).float().mean().item()}
    hyper = yaml.safe_load((ROOT/'upstream/qcm_config/iqn.yaml').read_text())
    hyper.update(num_steps=cfg['steps'], eval_interval=cfg['eval_interval'])
    hyper.update(cfg.get('hyper_overrides', {}))
    if hyper['num_steps'] <= hyper['start_steps']:
        raise ValueError('Budget must exceed replay warmup to train networks')
    if paper_method:
        pool = PaperPool(capacity=cfg['pool'], calculator=calculators['train'])
    else:
        pool = AlphaPool(capacity=cfg['pool'], calculator=calculators['train'], l1_alpha=0.005, device=torch.device('cpu'))
    env = SnapshotEnv(AlphaEnvCore(pool, device=device))
    env.action_space.seed(cfg['seed'])
    agent = agent_class(env, calculators['valid'], None, str(out), std_lam=cfg['lambda'],
                         cuda=device.type=='cuda', seed=cfg['seed'], **hyper)
    provenance = {'config': cfg, 'resolved_hyperparameters': hyper, 'data': manifest,
                  'upstream_commit': subprocess.check_output(['git', '-C', str(ROOT/'upstream'), 'rev-parse', 'HEAD'], text=True).strip(),
                  'paper_sha256': hashlib.sha256((ROOT/'sources/paper.pdf').read_bytes()).hexdigest(),
                  'torch': torch.__version__, 'cuda': torch.version.cuda,
                  'device_name': torch.cuda.get_device_name() if device.type=='cuda' else 'cpu',
                  'implementation_sha256': hashlib.sha256((ROOT/'implementation.py').read_bytes()).hexdigest(),
                  'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'selection': ('highest validation finite-pair IC' if paper_method else 'highest validation released-zero-fill IC') + '; no test access until frozen',
                  'status': 'running'}
    if paper_method:
        provenance['paper_implementation_sha256'] = hashlib.sha256((ROOT/'paper_implementation.py').read_bytes()).hexdigest()
        provenance['trainable_parameters'] = sum(p.numel() for net in [agent.online_net, agent.online_mean_net] for p in net.parameters())
        provenance['action_count'] = int(agent.num_actions)
        provenance['operators'] = [op.__name__ for op in __import__('alphagen.config', fromlist=['OPERATORS']).OPERATORS]
    write_json(out/'manifest.json', provenance)
    best, selected, history = -float('inf'), None, []
    state, _ = env.reset()
    for step in range(1, cfg['steps']+1):
        action = agent.explore() if agent.is_random() else agent.exploit(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.memory.append(state.copy(), action, reward, next_state.copy(), done or truncated)
        agent.steps = step
        agent.epsilon_train.step()
        if step % agent.target_update_interval == 0:
            agent.update_target(); agent.update_mean_target()
        if agent.is_update():
            agent.learn()
        state = next_state
        if done or truncated:
            agent.episodes += 1
            state, _ = env.reset()
        if step % cfg['eval_interval'] == 0 or step == cfg['steps']:
            valid_ic = pool.test_ensemble(calculators['valid']) if pool.size else None
            record = {'step': step, 'episodes': agent.episodes, 'updates': agent.learning_steps,
                      'pool_size': pool.size, 'valid_ic': valid_ic,
                      'elapsed_seconds': time.time()-started, 'losses': agent.last_losses}
            if valid_ic is not None and np.isfinite(valid_ic) and valid_ic > best:
                best = valid_ic
                selected = {'exprs': copy.copy(pool.exprs[:pool.size]), 'weights': pool.weights[:pool.size].copy(), 'step': step}
                agent.save_models(str(out/'model'/'validation_best'))
                write_json(out/'selected_pool.json', {'step': step, 'valid_ic': best, 'exprs': [str(e) for e in selected['exprs']], 'weights': selected['weights'].tolist()})
            history.append(record)
            write_json(out/'progress.json', record)
            print(json.dumps(record), flush=True)
    if selected is None:
        raise RuntimeError('No finite validation-selected alpha pool')
    agent.save_models(str(out/'model'/'final'))
    summaries = {}
    for split in ['train', 'valid', 'test']:
        summary, daily = daily_metrics(datasets[split], calculators[split], selected['exprs'], selected['weights'])
        summaries[split] = summary
        daily.to_csv(out/f'{split}_daily_metrics.csv', index=False)
    final_summary, final_daily = daily_metrics(datasets['test'], calculators['test'], pool.exprs[:pool.size], pool.weights[:pool.size])
    final_daily.to_csv(out/'final_pool_test_daily_metrics.csv', index=False)
    write_json(out/'final_pool.json', pool.to_dict())
    report = {'status': 'completed', 'scope': cfg['scope'], 'steps': agent.steps,
              'learning_updates': agent.learning_steps, 'selected_step': selected['step'],
              'elapsed_seconds': time.time()-started, 'metrics': summaries,
              'final_pool_test_metrics': final_summary,
              'selected_during_random_warmup': selected['step'] < agent.start_steps,
              'method': cfg.get('method', 'corrected_release'),
              'paper_csi300_iqn_ic': 0.0849, 'paper_csi300_iqn_seed_std': 0.0103,
              'numerical_replication_established': False}
    if device.type == 'cuda':
        report['gpu_memory'] = {
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device),
            'total_device_bytes': torch.cuda.get_device_properties(device).total_memory,
            'scope': 'PyTorch allocator peaks; excludes CUDA context and other processes'}
    write_json(out/'results.json', report)
    write_json(out/'history.json', history)
    provenance['status'] = 'completed'
    write_json(out/'manifest.json', provenance)
    agent.writer.close()
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
