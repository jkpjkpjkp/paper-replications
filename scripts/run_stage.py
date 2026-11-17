"""Run one unmodified Qlib 0.7.1 TRA stage, in its own fresh process."""
import os
EXECUTION_DEVICE = os.environ.get('TRA_DEVICE', 'cuda')
assert EXECUTION_DEVICE in ('cuda', 'cpu')
if EXECUTION_DEVICE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(key, '2')
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT/'sources/examples/benchmarks/TRA'
DATA = ROOT.parent/'ordered-reproductions/data/TRA/kdd21'
sys.path.insert(0, str(ROOT/'sources'))
import numpy as np
import pandas as pd
import torch
import qlib
import yaml
from qlib.utils import init_instance_by_config
from evaluate import evaluate


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False)+'\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=['initialization', 'tra'], required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    runtime_dir = 'paper-runtime' if EXECUTION_DEVICE == 'cuda' else 'paper-runtime-cpu'
    preflight_dir = 'preflight' if EXECUTION_DEVICE == 'cuda' else 'preflight-cpu'
    out = ROOT/'results'/(preflight_dir if args.preflight else runtime_dir)/str(args.seed)/args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        raise RuntimeError('Completed stage exists; refusing overwrite')
    if (out/'model.bin').exists():
        raise RuntimeError('Existing trained artifacts require review; refusing overwrite')
    started = time.monotonic()
    def status(**kwargs):
        save(out/'status.json', dict(seed=args.seed, stage=args.stage,
             updated_utc=datetime.now(timezone.utc).isoformat(),
             elapsed_seconds=time.monotonic()-started, **kwargs))
    try:
        assert qlib.__version__ == '0.7.1'
        assert torch.__version__ == '1.7.0+cu110'
        assert np.__version__ == '1.17.4'
        assert pd.__version__ == '1.1.2'
        assert torch.cuda.is_available() == (EXECUTION_DEVICE == 'cuda')
        locked = json.loads((ROOT/'protocol.json').read_text())
        for name, expected in locked['data_sha256'].items():
            assert sha(DATA/name) == expected, ('data changed', name)
        released = json.loads((ROOT/'sources/source-verification.json').read_text())
        for relative, expected in released['files'].items():
            assert sha(ROOT/'sources'/relative) == expected, ('source changed', relative)
        torch.set_num_threads(2)
        filename = 'config_alstm_tra_init.yaml' if args.stage == 'initialization' else 'config_alstm_tra.yaml'
        config = yaml.safe_load((CODE/'configs'/filename).read_text())
        task = config['task']
        loader = task['dataset']['kwargs']['handler']['kwargs']['data_loader']['kwargs']['config']
        loader.update(feature=str(DATA/'feature.pkl'), label=str(DATA/'label.pkl'))
        for part in ('model', 'dataset'):
            task[part]['module_path'] = str(CODE/task[part]['module_path'])
        task['model']['kwargs'].update(seed=args.seed, logdir=str(out))
        if args.stage == 'tra':
            init = ROOT/'results'/runtime_dir/str(args.seed)/'initialization'
            if args.preflight:
                # Only a capacity test; no training result uses this random backbone.
                task['model']['kwargs']['model_init_state'] = None
            else:
                assert (init/'complete.json').exists()
                task['model']['kwargs']['model_init_state'] = str(init/'model.bin')
        save(out/'config.json', config)
        save(out/'provenance.json', dict(
            source_commit=(ROOT/'source-commit.txt').read_text().strip(),
            runner_sha256=sha(__file__),
            source_sha256={str(p.relative_to(CODE)):sha(p) for p in
                           [CODE/'src/model.py', CODE/'src/dataset.py', CODE/'configs'/filename]},
            data_sha256={p.name:sha(p) for p in DATA.glob('*.pkl')},
            versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__,
                          torch=torch.__version__, qlib=qlib.__version__, cuda=torch.version.cuda,
                          cudnn=torch.backends.cudnn.version()),
            device=EXECUTION_DEVICE,
            gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            changes=['Absolute input/output paths', 'Declared run.sh seed', 'Two CPU threads',
                     'Status wrappers that do not change tensors or RNG'],
            original_sequence_bug_preserved=True, original_stochastic_evaluation_preserved=True))
        os.chdir(CODE)
        qlib.init(provider_uri=str(DATA), region='cn', kernels=2,
                  expression_cache=None, dataset_cache=None)
        status(phase='loading')
        dataset = init_instance_by_config(task['dataset'])
        model = init_instance_by_config(task['model'])
        if args.preflight:
            batch = next(iter(dataset.prepare('train')))
            step_started = time.monotonic()
            features = batch['data'][:, :, :-model.tra.num_states]
            memory = batch['data'][:, :-dataset.horizon, -model.tra.num_states:]
            hidden = model.model(features)
            pred, experts, prob = model.tra(hidden, memory)
            loss = (pred-batch['label']).pow(2).mean()
            sinkhorn = sys.modules[type(model).__module__].sinkhorn
            errors = (experts.detach()-batch['label'][:, None]).pow(2)
            errors -= errors.min(dim=-1, keepdim=True).values
            assignment = sinkhorn(-errors, epsilon=.01)
            loss = loss-model.lamb*prob.log().mul(assignment).sum(dim=-1).mean()
            loss.backward()
            model.optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            assert torch.isfinite(loss)
            save(out/'complete.json', dict(kind='preflight_only', loss=float(loss),
                 rows=len(dataset._index), shape=list(batch['data'].shape),
                 device=EXECUTION_DEVICE, elapsed_seconds=time.monotonic()-started,
                 training_step_seconds=time.monotonic()-step_started,
                 cuda_peak_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))
            status(phase='preflight_passed')
            return
        epoch = [-1]
        original_train, original_test = model.train_epoch, model.test_epoch
        def train(data):
            epoch[0] += 1
            status(phase='training', epoch=epoch[0])
            return original_train(data)
        def test(data, return_pred=False):
            status(phase='final_test' if return_pred else 'refresh_or_validation', epoch=epoch[0])
            return original_test(data, return_pred=return_pred)
        model.train_epoch, model.test_epoch = train, test
        model.fit(dataset)
        status(phase='independent_verification', epoch=epoch[0])
        del model, dataset
        torch.cuda.empty_cache()
        result = evaluate(out/'pred.pkl', out)
        save(out/'complete.json', dict(kind='full_stage', seed=args.seed, stage=args.stage,
             epochs=epoch[0]+1, elapsed_seconds=time.monotonic()-started,
             files={name:sha(out/name) for name in ['model.bin','pred.pkl','info.json','config.json']},
             evaluation=result))
        status(phase='complete', epochs=epoch[0]+1)
    except BaseException as error:
        status(phase='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
