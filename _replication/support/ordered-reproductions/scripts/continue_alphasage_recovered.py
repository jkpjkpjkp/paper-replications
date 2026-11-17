"""Finite continuation: wait for this collection, verify it, run one experiment.

No recurring scheduler, automatic fidelity approval, or CZ training is created.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / 'data/alphasage-baostock-20261115'
OUT = ROOT / 'results/alphasage-recovered-20261115'
SMOKE = ROOT / 'results/alphasage-recovered-smoke-20261115/complete.json'


def save(value):
    OUT.mkdir(exist_ok=True)
    path = OUT / 'pipeline-status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    assert json.loads(SMOKE.read_text())['state'] == 'smoke_complete'
    scripts = ['rebuild_alphasage_panel.py', 'verify_alphasage_recovery.py', 'run_alphasage_recovered.py']
    hashes = {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() for name in scripts}
    started = time.monotonic()
    save(dict(state='waiting_for_current_data_recovery', max_wait_hours=4, source_sha256=hashes,
        original_then_cz=True, new_cz_training_started=False))
    while not (PANEL / 'verification.json').exists():
        if (PANEL / 'status.json').exists():
            status = json.loads((PANEL / 'status.json').read_text())
            if status['state'] == 'failed':
                raise RuntimeError('Data recovery failed; inspect its failure list. Training not launched.')
        if time.monotonic() - started > 4 * 3600:
            raise TimeoutError('Finite four-hour data wait expired; training not launched.')
        time.sleep(30)
    for name, expected in hashes.items():
        if hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Runner changed after launch: {name}')
    subprocess.run([sys.executable, str(ROOT / 'scripts/verify_alphasage_recovery.py'),
        '--panel', str(PANEL)], check=True, cwd=ROOT)
    save(dict(state='original_method_training_and_evaluation', source_sha256=hashes))
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/run_alphasage_recovered.py'),
        '--panel', str(PANEL), '--output', str(OUT)], check=True, cwd=ROOT)
    save(dict(state='original_method_run_complete_fidelity_review_required',
        result=str(OUT / 'complete.json'), cz_training_started=False,
        reason='Source/data ambiguities and native proxy backtest remain explicit; no automatic original-paper gate promotion.'))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        save(dict(state='failed', error=repr(error), cz_training_started=False))
        raise
