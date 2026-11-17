"""Persistently run and report one locked full-period CZ PRISM experiment."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import HERE, ROOT, DATA, FACTORS, PROTOCOL, settings, training_sources, check_lock, sha, save_json, now


def update(phase, **details):
    state = dict(status=phase, updated_utc=now(), pid=os.getpid(),
        experiment='cz_vw', protocol=str(PROTOCOL), protocol_sha256=sha(PROTOCOL),
        experiment_status=str(HERE / 'STATUS.json'), exact_published_results_reproduced=False, **details)
    save_json(HERE / 'STATUS.json', state)
    save_json(ROOT / 'STATUS.json', state)
    print(json.dumps(state), flush=True)


def call(script, args, phase):
    update(phase, log=str(HERE / f'{phase}.log'))
    env = os.environ | dict(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', WANDB_MODE='disabled',
        MPLBACKEND='Agg', CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    with (HERE / f'{phase}.log').open('a') as log:
        subprocess.run([sys.executable, '-u', str(HERE / script), *args], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    lock = (HERE / 'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (HERE / '.complete').exists():
        print('Already complete; see REPORT.txt', flush=True)
        return
    for name in ['STATUS.json', 'RESULTS.txt']:
        legacy = HERE / f'legacy_{name}'
        if not legacy.exists():
            shutil.copy2(ROOT / name, legacy)
    try:
        call('build.py', [], 'preparing')
        call('checks.py', [], 'preflight')
        frozen = HERE / 'locked_before_training.json'
        if frozen.exists():
            check_lock()
        else:
            import torch
            import statsmodels
            import numpy
            import pandas
            save_json(frozen, dict(locked_utc=now(), protocol=settings(),
                training_sources_sha256=training_sources(), data_manifest_sha256=sha(DATA / 'manifest.json'),
                factor_sha256=sha(FACTORS), user_protocol_sha256=sha(PROTOCOL),
                versions=dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    statsmodels=statsmodels.__version__, numpy=numpy.__version__, pandas=pandas.__version__),
                test_outcomes_not_used_for_selection=True))
        call('training.py', ['--stage', '1'], 'stage1_training')
        call('training.py', ['--stage', '2'], 'stage2_training')
        call('analyze.py', [], 'evaluating')
        call('verify.py', [], 'verifying')
        assert json.loads((HERE / 'analysis/verification.json').read_text())['passed']
        update('protocol_run_complete', report=str(HERE / 'REPORT.txt'),
            verification=str(HERE / 'analysis/verification.json'))
        save_json(HERE / '.complete', dict(completed_utc=now(), report_sha256=sha(HERE / 'REPORT.txt')))
        (ROOT / 'RESULTS.txt').write_text((HERE / 'REPORT.txt').read_text())
    except Exception as error:
        update('failed', error=str(error), resumable=True)
        raise


if __name__ == '__main__':
    main()
