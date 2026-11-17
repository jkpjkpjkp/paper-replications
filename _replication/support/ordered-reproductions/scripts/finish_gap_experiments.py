"""One-shot completion chain for the already declared experiments. No timer or recurring service."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/gap-resolution-20261113'
SCRIPTS = ROOT / 'scripts'
CPU = ROOT / '.venv-doubleadapt/bin/python'
CUDA = ROOT / '.venv-tra/bin/python'
jobs = {}
finished = {}
handles = []


def save(name, value):
    p = OUT / name
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, indent=2, default=str) + '\n')
    t.replace(p)


def launch(name, python, script, args=()):
    assert name not in jobs and name not in finished
    f = (OUT / (name + '.log')).open('w')
    handles.append(f)
    env = os.environ.copy()
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    proc = subprocess.Popen([str(python), str(SCRIPTS / script), *args], cwd=ROOT.parent,
                            env=env, stdout=f, stderr=subprocess.STDOUT)
    jobs[name] = proc
    save(name + '-launch.json', dict(pid=proc.pid, script=script, args=list(args),
        script_sha256=hashlib.sha256((SCRIPTS / script).read_bytes()).hexdigest()))
    print(json.dumps(dict(event='launched', name=name)), flush=True)


def status(name):
    p = OUT / name / 'status.json'
    return json.loads(p.read_text()) if p.exists() else {}


def done(name):
    return (OUT / name / 'summary.json').exists()


assert not (OUT / 'completion-chain.json').exists(), 'One owner only; do not silently restart'
save('completion-chain.json', dict(stage='running', pid=os.getpid(), one_shot=True))
probe_decided = False
step5_unavailable = False
started = time.monotonic()
while True:
    for name, proc in list(jobs.items()):
        code = proc.poll()
        if code is not None:
            finished[name] = code
            del jobs[name]
            print(json.dumps(dict(event='finished', name=name, exit_code=code)), flush=True)
            if code:
                save('completion-chain.json', dict(stage='failed_followup', job=name, exit_code=code,
                                                 still_running=list(jobs), one_shot=True))
                raise RuntimeError((name, code))

    for parent in ['available-inputs', 'tra-calendar-corrected']:
        if status(parent).get('stage') == 'failed':
            save('completion-chain.json', dict(stage='parent_failed', parent=parent, still_running=list(jobs)))
            raise RuntimeError(parent)

    # Wait for TRA to release the GPU before the final capacity attempt.
    if done('tra-calendar-corrected') and not probe_decided:
        probe_decided = True
        launch('cuda-step5-capacity-after-tra', CUDA, 'probe_doubleadapt_step5_cuda.py', ['--after-tra'])
    if 'cuda-step5-capacity-after-tra' in finished and 'available-inputs-step5-training' not in jobs and 'available-inputs-step5-training' not in finished and not step5_unavailable:
        result = json.loads((OUT / 'cuda-step5-probe.json').read_text())
        if result['passed']:
            launch('available-inputs-step5-training', CUDA, 'run_doubleadapt_available_inputs_step5.py')
        else:
            step5_unavailable = True
            save('step5-not-run.json', dict(reason='Native full-task capacity check failed after TRA completion',
                                          outcome='No five-session model trained; no model setting altered to force a pass.'))

    if done('available-inputs') and 'rates-clean' not in jobs and 'rates-clean' not in finished:
        launch('rates-clean', CPU, 'validate_doubleadapt_online_rates.py', ['--clean', '--evaluate'])
    if 'rates-clean' in finished and 'rates-coherent' not in jobs and 'rates-coherent' not in finished:
        launch('rates-coherent', CPU, 'validate_doubleadapt_online_rates.py', ['--clean', '--coherent', '--evaluate'])
    if done('available-inputs-step5') and 'rates-step5' not in jobs and 'rates-step5' not in finished:
        launch('rates-step5', CUDA, 'validate_doubleadapt_online_rates.py', ['--clean', '--step', '5', '--evaluate'])

    if done('tra-calendar-corrected') and done('available-inputs') and 'rates-coherent' in finished and (step5_unavailable or 'rates-step5' in finished):
        candidates = []
        for name in ['available-inputs', 'available-inputs-coherent', 'available-inputs-step5']:
            path = OUT / ('validation-rates-' + name) / 'summary.json'
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            candidates.append(dict(name=name, validation_ic=result['selection']['validation_ic'],
                                   metrics=result['metrics'], artifact=str(path)))
        winner = max(candidates, key=lambda row: row['validation_ic'])
        save('validation-selected-clean-model.json', dict(selection_rule='Maximum common-date validation IC, never held-out metrics',
             winner=winner, all_candidates=candidates, single_seed=True, full_cz_authorized_by_gate=False))
        save('completion-chain.json', dict(stage='experiments_complete_pending_review', seconds=time.monotonic()-started,
                                          finished=finished, step5_unavailable=step5_unavailable, one_shot=True))
        print('All declared follow-ups complete; independent review still required.', flush=True)
        break
    save('completion-chain-live.json', dict(active=list(jobs), finished=finished,
                                           step5_unavailable=step5_unavailable, seconds=time.monotonic()-started))
    time.sleep(10)
