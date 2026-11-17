#!/usr/bin/env python3
"""Recheck completed CSI summaries against frozen code, tensors, and checkpoints."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    result = {'status': 'passed', 'checked_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'scope': 'Audit existing financial artifacts; no new fits and no GPU access', 'universes': {}}
    for universe in ['csi300', 'csi500', 'csi1000']:
        folder = HERE / 'results' / universe
        summary = json.loads((folder / 'summary.json').read_text())
        frozen = json.loads((folder / 'manifest.json').read_text())
        assert summary['complete'] and summary['signature'] == frozen['signature']
        inputs = frozen['inputs']
        assert all(sha(HERE / name) == digest for name, digest in inputs['source'].items())
        assert sha(HERE.parent / 'reproduction/core.py') == inputs['core_sha256']
        tensor_manifest = json.loads((HERE / 'cache' / universe / 'manifest.json').read_text())
        assert tensor_manifest['complete']
        records = {record['date']: record for record in tensor_manifest['records']}
        assert set(records) == set(inputs['tensor_receipts'])
        for date, digest in inputs['tensor_receipts'].items():
            record = records[date]
            path = HERE / 'cache' / universe / record['file']
            assert record['sha256'] == digest and sha(path) == digest
            assert path.stat().st_size == record['bytes']
        receipts = list(folder.glob('*.receipt.json'))
        assert len(receipts) >= 55  # ten horizons and one BLO selection per seed
        for receipt_path in receipts:
            receipt = json.loads(receipt_path.read_text())
            checkpoint = receipt_path.with_suffix('').with_suffix('.pt')
            assert sha(checkpoint) == receipt['sha256']
            assert checkpoint.stat().st_size == receipt['bytes']
            assert receipt['signature'] == summary['signature']
        per_seed = pd.read_csv(folder / 'per_seed_metrics.csv')
        assert set(per_seed.seed) == set(range(5))
        for method, saved in summary['primary'].items():
            rows = per_seed[per_seed.method == ('h10' if method == 'standard' else method)]
            assert len(rows) == 5
            for metric, values in saved.items():
                assert np.isclose(rows[metric].mean(), values['mean'], atol=1e-12, rtol=1e-12)
                assert np.isclose(rows[metric].std(ddof=1), values['seed_std'], atol=1e-12, rtol=1e-12)
        result['universes'][universe] = {'complete': True, 'tensor_hashes_verified': len(records),
                                        'checkpoint_hashes_verified': len(receipts),
                                        'summary_matches_per_seed_metrics': True,
                                        'primary': summary['primary'],
                                        'paired_ic_bootstrap': summary['paired_ic_bootstrap']}
    (HERE / 'completed_result_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'universes': {
        universe: {key: value for key, value in audit.items() if key not in ['primary', 'paired_ic_bootstrap']}
        for universe, audit in result['universes'].items()}}, indent=2))


if __name__ == '__main__':
    main()
