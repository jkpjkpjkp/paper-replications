"""One independently seeded fit, using the pinned public SIT implementation.

The separate controller performs the complete paper grid and freezes validation
selection before invoking evaluation. Test data are not loaded during fitting.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'source'))
import numpy as np
import torch
from exp.exp_main import EXP_main, decision_focused_cvar_loss
from utils.tools import EarlyStopping


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source():
    manifest = json.loads((HERE / 'source_manifest.json').read_text())
    for item in manifest['files']:
        assert sha(HERE / 'source' / item['path']) == item['sha256'], item['path']


def allocator(mode):
    if mode == 'native':
        return None
    library = HERE / 'managed_allocator.so'
    lib = ctypes.CDLL(str(library))
    lib.sit_managed_peak_bytes.restype = ctypes.c_size_t
    assert lib.sit_managed_supported() == 1
    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        str(library), 'sit_managed_malloc', 'sit_managed_free')
    torch.cuda.memory.change_current_allocator(alloc)
    torch.cuda.empty_cache = lambda: None
    return lib


def config(protocol, pool, index):
    params = dict(protocol['configurations'][index])
    params.update(protocol['fixed_model'])
    params.update(data_pool=pool, model='Signature', data='FULL',
                  root_path=str(ROOT / 'asset_data'), data_path='full_dataset.csv',
                  precomp_root=str(ROOT / 'signature_cache_6020'),
                  use_gpu=True, use_multi_gpu=False, gpu=0,
                  batch_size=64, num_workers=10, train_epochs=100,
                  patience=10, learning_rate=0.001, checkpoints='',
                  lradj='constant_paper', cvar_alpha=0.95)
    return SimpleNamespace(**params)


def environment():
    import pandas, scipy, sklearn, matplotlib
    return dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                numpy=np.__version__, pandas=pandas.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, matplotlib=matplotlib.__version__,
                gpu=torch.cuda.get_device_name(0),
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def reset_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args, protocol):
    cfg = config(protocol, args.pool, args.index)
    reset_seed(args.seed)
    experiment = EXP_main(cfg)
    _, train_loader = experiment._get_data('train', True)
    _, val_loader = experiment._get_data('val', True)
    optimizer = experiment._select_optimizer()
    stopping = EarlyStopping(patience=10, verbose=True)
    best = float('inf')
    best_epoch = None
    started = time.time()
    history = []
    for epoch in range(1, 101):
        tick = time.time()
        experiment.model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss, _ = decision_focused_cvar_loss(
                experiment.model,
                batch['x_sigs'].to(experiment.device),
                batch['cross_sigs'].to(experiment.device),
                batch['date_feats'].to(experiment.device),
                batch['future_return_unscaled'].to(experiment.device),
                alpha=cfg.cvar_alpha, temperature=cfg.temperature)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; grid cannot silently omit this fit')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_loss = experiment._vali(val_loader)
        if not np.isfinite(val_loss):
            raise RuntimeError('Non-finite validation loss')
        if val_loss <= best:
            best, best_epoch = val_loss, epoch
        stopping(val_loss, experiment.model, str(args.output))
        record = dict(epoch=epoch, train_cvar=float(np.mean(losses)),
                      validation_cvar=val_loss, best_validation_cvar=best,
                      best_epoch=best_epoch, learning_rate=optimizer.param_groups[0]['lr'],
                      seconds=time.time()-tick, patience_counter=stopping.counter)
        history.append(record)
        print(json.dumps(record), flush=True)
        with (args.output / 'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if stopping.early_stop:
            break
    result = dict(status='completed', pool=args.pool, config_index=args.index,
                  configuration=protocol['configurations'][args.index], seed=args.seed,
                  best_validation_cvar=best, best_epoch=best_epoch,
                  stopped_epoch=epoch, early_stopped=stopping.early_stop,
                  elapsed_seconds=time.time()-started, train_samples=len(train_loader.dataset),
                  validation_samples=len(val_loader.dataset),
                  checkpoint=str(args.output / 'checkpoint.pth'),
                  checkpoint_sha256=sha(args.output / 'checkpoint.pth'),
                  protocol_sha256=sha(HERE/'protocol.json'), environment=environment(),
                  test_data_loaded=False, memory_allocator=args.allocator)
    write_json(args.output / 'result.json', result)


def evaluate(args, protocol):
    selection = json.loads((HERE / 'selection.json').read_text())
    assert selection['protocol_sha256'] == sha(HERE/'protocol.json')
    chosen = selection['pools'][str(args.pool)]
    assert chosen['config_index'] == args.index
    record = chosen['runs'][str(args.seed)]
    checkpoint = Path(record['checkpoint'])
    assert sha(checkpoint) == record['checkpoint_sha256']
    cfg = config(protocol, args.pool, args.index)
    reset_seed(args.seed)
    experiment = EXP_main(cfg)
    experiment.model.load_state_dict(torch.load(checkpoint, map_location=experiment.device,
                                               weights_only=False))
    os.chdir(args.output)
    setting = f'dp{args.pool}_paper_cfg{args.index:03d}_seed{args.seed}'
    experiment.eval(setting, load=False)
    write_json(args.output / 'evaluation.json', dict(status='completed', pool=args.pool,
               seed=args.seed, config_index=args.index, selection_sha256=sha(HERE/'selection.json'),
               checkpoint_sha256=sha(checkpoint), setting=setting,
               protocol_sha256=sha(HERE/'protocol.json')))


def probe(args, protocol):
    """Small native/managed equivalence check, never a financial result."""
    cfg = config(protocol, 40, 0)
    cfg.batch_size = 2
    cfg.num_workers = 0
    reset_seed(7191)
    experiment = EXP_main(cfg)
    _, loader = experiment._get_data('train', True)
    batch = next(iter(loader))
    optimizer = experiment._select_optimizer()
    for _ in range(3):
        optimizer.zero_grad()
        loss, _ = decision_focused_cvar_loss(experiment.model,
            batch['x_sigs'].to(experiment.device), batch['cross_sigs'].to(experiment.device),
            batch['date_feats'].to(experiment.device),
            batch['future_return_unscaled'].to(experiment.device), temperature=cfg.temperature)
        loss.backward()
        optimizer.step()
    torch.save({k:v.detach().cpu() for k,v in experiment.model.state_dict().items()},
               args.output / 'probe.pth')
    write_json(args.output/'probe.json', dict(loss=float(loss.item()), environment=environment(),
               allocator=args.allocator, note='Three small-batch engineering steps, not a result run.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','evaluate','probe'])
    parser.add_argument('--pool', type=int, choices=[40,50], default=40)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allocator', choices=['native','managed'], default='managed')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    verify_source()
    protocol = json.loads((HERE/'protocol.json').read_text())
    assert torch.__version__ == protocol['required_torch'], torch.__version__
    assert np.__version__ == '1.26.4', np.__version__
    assert torch.cuda.is_available()
    assert args.seed in protocol['seeds'] or args.mode == 'probe'
    torch.set_num_threads(4)
    lib = allocator(args.allocator)
    if args.mode == 'train':
        train(args, protocol)
    elif args.mode == 'evaluate':
        evaluate(args, protocol)
    else:
        probe(args, protocol)
    if lib:
        write_json(args.output/'memory.json', dict(managed_peak_mib=lib.sit_managed_peak_bytes()/2**20))


if __name__ == '__main__':
    main()
