"""Resumable complete-grid controller; failed fits are never silently dropped."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def select_configs(protocol, results):
    """All five seeds of every candidate must be present before selection."""
    pools = {}
    for pool in protocol['asset_pools']:
        candidates = []
        for index in range(len(protocol['configurations'])):
            runs = {}
            for seed in protocol['seeds']:
                r = results[(pool, index, seed)]
                assert r['status'] == 'completed' and not r['test_data_loaded']
                assert (r['pool'], r['config_index'], r['seed']) == (pool, index, seed)
                runs[str(seed)] = r
            score = sum(r['best_validation_cvar'] for r in runs.values()) / len(runs)
            candidates.append((score, index, runs))
        score, index, runs = min(candidates, key=lambda value: (value[0], value[1]))
        pools[str(pool)] = dict(config_index=index, mean_validation_cvar=score,
                               configuration=protocol['configurations'][index], runs=runs)
    return pools


def resources():
    memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    available = int(memory['MemAvailable'].split()[0]) / 2**20
    try:
        output = subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'], text=True)
        gpu_mib = float(output.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu_mib = 0
    return dict(available_ram_gib=available, free_gpu_mib=gpu_mib)


def validate_result(path, key, protocol_hash):
    result = json.loads(path.read_text())
    assert result['protocol_sha256'] == protocol_hash
    assert (result['pool'], result['config_index'], result['seed']) == key
    assert result['status'] == 'completed' and not result['test_data_loaded']
    assert digest(Path(result['checkpoint'])) == result['checkpoint_sha256']
    return result


def call_worker(mode, pool, index, seed, output, allocator='managed'):
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, '-u', str(HERE/'worker.py'), mode,
               '--pool',str(pool),'--index',str(index),'--seed',str(seed),
               '--output',str(output),'--allocator',allocator]
    write_json(output/'command.json', command)
    with (output/'worker.log').open('w') as stream:
        return subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode


def main():
    lock = (HERE/'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = json.loads((HERE/'protocol.json').read_text())
    protocol_hash = digest(HERE/'protocol.json')
    stamp = HERE/'execution_manifest.json'
    script_hashes = {name:digest(HERE/name) for name in
                     ['worker.py','controller.py','managed_allocator.cpp','source_manifest.json','protocol.json']}
    if stamp.exists():
        assert json.loads(stamp.read_text())['hashes'] == script_hashes, 'Code/protocol changed after launch'
    else:
        write_json(stamp, dict(started_at=now(), hashes=script_hashes,
                   container=protocol['container'], note='Recorded before any production fit'))
    assert json.loads((HERE/'preflight.json').read_text())['passed']
    results = {}
    status = dict(status='running_grid', started_at=now(), completed_fits=0,
                  planned_fits=protocol['planned_fits'], protocol_sha256=protocol_hash)
    try:
        for pool in protocol['asset_pools']:
            for index in range(len(protocol['configurations'])):
                for seed in protocol['seeds']:
                    key = (pool,index,seed)
                    job = HERE/'fits'/f'pool{pool}'/f'cfg{index:03d}'/f'seed{seed}'
                    complete = job/'result.json'
                    if complete.exists():
                        results[key] = validate_result(complete,key,protocol_hash)
                        continue
                    while True:
                        resource = resources()
                        status.update(current_job=dict(pool=pool,config_index=index,seed=seed),
                                      completed_fits=len(results), resources=resource, updated_at=now())
                        if resource['available_ram_gib'] >= 10 and resource['free_gpu_mib'] >= 512:
                            break
                        status['status'] = 'waiting_for_local_resources'
                        write_json(HERE/'progress.json', status)
                        time.sleep(30)
                    status['status'] = 'running_grid'
                    job.mkdir(parents=True,exist_ok=True)
                    attempt_index = 1
                    while (job/f'attempt_{attempt_index:03d}').exists():
                        attempt_index += 1
                    attempt = job/f'attempt_{attempt_index:03d}'
                    status.update(current_log=str(attempt/'worker.log'), updated_at=now())
                    write_json(HERE/'progress.json',status)
                    code = call_worker('train',pool,index,seed,attempt)
                    if code != 0:
                        raise RuntimeError(f'Fit {key} exited {code}; inspect {attempt}/worker.log. No fit omitted.')
                    result = validate_result(attempt/'result.json',key,protocol_hash)
                    write_json(complete,result)
                    results[key] = result
                    status.update(completed_fits=len(results),updated_at=now())
                    write_json(HERE/'progress.json',status)
                    print(json.dumps(dict(completed=len(results), planned=protocol['planned_fits'],
                          pool=pool,index=index,seed=seed,best_epoch=result['best_epoch'],
                          stopped_epoch=result['stopped_epoch'],elapsed=result['elapsed_seconds'])),flush=True)
        selection = dict(protocol_sha256=protocol_hash, frozen_at=now(),
                         completed_fits=len(results), pools=select_configs(protocol,results))
        selection_path = HERE/'selection.json'
        if selection_path.exists():
            assert json.loads(selection_path.read_text())['pools'] == selection['pools']
        else:
            write_json(selection_path,selection)
        status['status']='evaluating_frozen_selection'
        write_json(HERE/'progress.json',status)
        for pool in protocol['asset_pools']:
            index=selection['pools'][str(pool)]['config_index']
            for seed in protocol['seeds']:
                output=HERE/'evaluation'/f'pool{pool}_seed{seed}'
                if not (output/'evaluation.json').exists():
                    code=call_worker('evaluate',pool,index,seed,output)
                    if code != 0:
                        raise RuntimeError(f'Evaluation failed for pool {pool}, seed {seed}')
                for path in (output/'results').glob('*'):
                    dest=HERE/'results'/path.name
                    dest.parent.mkdir(exist_ok=True)
                    shutil.copy2(path,dest)
        code=subprocess.run([sys.executable,str(HERE.parents[0]/'audit_results.py'),
             '--results',str(HERE/'results'),'--label','paper_protocol_20261117/audit']).returncode
        if code:
            raise RuntimeError('Independent return audit failed')
        audit=json.loads((HERE/'audit/summary.json').read_text())
        assert audit['runs_audited']==10 and audit['cells']==16
        assert audit['max_daily_return_error'] < 1e-6
        status.update(status='completed',completed_fits=len(results),updated_at=now(),
                      completed_test_runs=10,audit=audit)
        write_json(HERE/'progress.json',status)
    except Exception as exc:
        status.update(status='failed_requires_inspection',error=str(exc),updated_at=now(),
                      completed_fits=len(results))
        write_json(HERE/'progress.json',status)
        raise


if __name__=='__main__':
    main()
