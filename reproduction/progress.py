"""Read-only, receipt-verified progress for the frozen five-seed experiments."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live_training(pid, root, name):
    try:
        proc = Path('/proc') / str(int(pid))
        if proc.joinpath('stat').read_text().split(') ', 1)[1].startswith('Z '):
            return False
        argv = proc.joinpath('cmdline').read_bytes().split(b'\0')
        script = str(root / 'reproduction/train.py').encode()
        return (script in argv and b'--backbone' in argv
                and argv[argv.index(b'--backbone') + 1] == name.encode())
    except (FileNotFoundError, TypeError, ValueError, IndexError, PermissionError, ProcessLookupError):
        return False


def collect_progress(root):
    root = Path(root).resolve()
    here = root / 'reproduction'
    config = read(here / 'protocol.json')
    result = {}
    for name in ['LSTM', 'GRU', 'TCN']:
        folder = here / 'results' if name == 'LSTM' else here / 'results/backbones' / name.lower()
        manifest = read(folder / 'manifest.json')
        state = read(folder / 'status.json')
        source = manifest.get('inputs', {}).get('source', {})
        mismatches = [f for f, expected in source.items()
                      if not (here / f).is_file() or digest(here / f) != expected]
        completed, invalid = [], []
        seeds = config.get('seeds', [])
        for seed in seeds:
            for stage in ['standard', 'selection', 'selected']:
                path = folder / f'{stage}-seed{seed}.pt'
                receipt = read(path.with_suffix('.receipt.json'))
                if not path.exists() and not receipt:
                    continue
                if (not manifest or not path.exists() or not receipt
                        or receipt.get('signature') != manifest.get('signature')
                        or receipt.get('bytes') != path.stat().st_size
                        or receipt.get('sha256') != digest(path)):
                    invalid.append(path.name)
                else:
                    completed.append(path.name)
        live = live_training(manifest.get('pid'), root, name)
        state_matches = bool(manifest and state.get('signature') == manifest.get('signature'))
        summary = read(folder / 'summary.json')
        financial_complete = bool(summary.get('complete') and manifest
                                  and summary.get('signature') == manifest.get('signature')
                                  and len(completed) == 3 * len(seeds)
                                  and not invalid and not mismatches)
        result[name] = {'live': live, 'pid': manifest.get('pid') if live else None,
                        'completed_fits': len(completed), 'required_fits': 3 * len(seeds),
                        'completed_checkpoints': completed, 'invalid_checkpoints': invalid,
                        'source_mismatches': mismatches, 'manifest_present': bool(manifest),
                        'state_matches_manifest': state_matches,
                        'state': state if state_matches else {},
                        'financial_complete': financial_complete}
    return result


def describe(entry):
    count = f"{entry['completed_fits']}/{entry['required_fits']} fits"
    if entry['source_mismatches'] or entry['invalid_checkpoints']:
        return count + '; integrity mismatch; inspect STATUS.json'
    if entry['financial_complete']:
        return count + '; financial evaluation complete'
    state = entry['state']
    if entry['live']:
        detail = f"running {state.get('phase', 'initializing')}"
        if 'seed' in state:
            detail += f", seed {state['seed']}"
        if 'epoch' in state:
            detail += f", epoch {state['epoch']}"
        return count + '; ' + detail
    return count + ('; stopped (saved state is stale)' if state else '; not started')
