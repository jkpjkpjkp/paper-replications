"""Bounded one-seed NASDAQ run, hardware benchmark, and artifact audit."""
import argparse
import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np
import torch
from model import PDUModel, objective

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def benchmark(mode, data, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tensors = {k: torch.as_tensor(data[k].astype(np.float32), device='cuda')
               for k in ['features', 'price', 'truth', 'mask']}
    edges = torch.as_tensor(data['edges'], device='cuda')
    model = PDUModel(1026, device='cuda', memory_efficient=mode == 'checkpointed')
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                  lr=.001, weight_decay=.001)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    durations = []
    for i, day in enumerate(np.random.RandomState(42).permutation(np.arange(4, 756))[:7]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pred, contrast = model(tensors['features'][:, day-4:day], edges)
        loss, _ = objective(pred, tensors['price'][:, day-1], tensors['truth'][:, day],
                             tensors['mask'][:, day-4:day+1].amin(1), contrast)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        model.update_moving_average()
        torch.cuda.synchronize()
        if i >= 2:
            durations.append(time.perf_counter() - start)
    result = {'mode': mode, 'mean_update_seconds': float(np.mean(durations)),
              'update_seconds': durations,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
    return result


def main(args):
    os.chdir(ROOT)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    started = time.time()
    deadline = started + args.max_hours * 3600
    receipt = {'scope': 'one full-universe NASDAQ seed; reconstructed protocol',
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'seed': 42, 'max_training_epochs': 300, 'patience': 20,
               'max_process_hours': args.max_hours,
               'paper_screen': {'sharpe5': {'mean': 2.2112, 'seed_sd': .1283},
                                'mrr': {'mean': .0407, 'seed_sd': .0005}},
               'screen_rule': 'Both metrics within two published seed SDs; descriptive only.',
               'exact_paper_replication': False}
    def status(stage, **extra):
        receipt.update(stage=stage, elapsed_seconds=time.time()-started, **extra)
        write_json(ROOT / 'case-receipt.json', receipt)
        print(json.dumps({'stage': stage, **extra}), flush=True)
    def command(cmd, log):
        with (logs / log).open('x') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT,
                           check=True, timeout=max(1, deadline-time.time()))
    try:
        assert torch.cuda.is_available(), 'This rented case requires a CUDA GPU.'
        receipt['hardware'] = {'gpu': torch.cuda.get_device_name(0),
                               'torch': torch.__version__, 'cuda': torch.version.cuda,
                               'python': sys.version, 'cpu_threads': args.cpu_threads}
        with (ROOT / 'environment.txt').open('w') as stream:
            subprocess.run(['nvidia-smi'], stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=stream, check=True)
        receipt['code_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(list(ROOT.glob('*.py')) + list((ROOT/'repaired').glob('*.py')))}
        status('tests')
        command([sys.executable, '-m', 'unittest', '-v', 'test_reproduction'], 'tests.log')
        status('benchmark')
        data = np.load(ROOT / 'data/NASDAQ.npz', allow_pickle=False)
        benchmarks = []
        for mode in ['checkpointed', 'batched']:
            try:
                result = benchmark(mode, data, args.cpu_threads)
            except torch.OutOfMemoryError:
                result = {'mode': mode, 'error': 'CUDA out of memory; excluded'}
            benchmarks.append(result)
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        write_json(ROOT / 'benchmark.json', benchmarks)
        valid = [r for r in benchmarks if 'mean_update_seconds' in r]
        graph_mode = min(valid, key=lambda r: r['mean_update_seconds'])['mode']
        status('training', graph_mode=graph_mode)
        command([sys.executable, '-u', 'train.py', '--market', 'NASDAQ', '--tag', args.tag,
                 '--seed', '42', '--epochs', '300', '--patience', '20', '--device', 'cuda',
                 '--graph-mode', graph_mode, '--cpu-threads', str(args.cpu_threads)], 'train.log')
        status('verification')
        command([sys.executable, 'verify_case.py', 'results/' + args.tag], 'verification.log')
        status('complete', result_directory='results/' + args.tag)
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
    finally:
        with tarfile.open(ROOT / 'case-artifacts.tar.gz', 'w:gz') as tar:
            for name in ['case-receipt.json', 'benchmark.json', 'environment.txt', 'logs', 'results']:
                path = ROOT / name
                if path.exists():
                    tar.add(path, arcname=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-hours', type=float, default=24)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--tag', default='nasdaq-seed42-gpushare-20261118')
    main(parser.parse_args())
