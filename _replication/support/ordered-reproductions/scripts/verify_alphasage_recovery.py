"""Independent identities and failure checks for a recovered vendor panel."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rebuild_alphasage_panel import adjust


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', type=Path, required=True)
    args = parser.parse_args()
    # A two-for-one split with no economic return must not become a -50% label.
    raw = pd.DataFrame(dict(date=['2024-12-30', '2024-12-31'], code=['x', 'x'],
        open=[100., 50.], high=[100., 50.], low=[100., 50.], close=[100., 50.],
        volume=[10., 20.], amount=[1000., 1000.], tradestatus=['1', '1']))
    vendor = raw.copy()
    vendor[['open', 'high', 'low', 'close']] = 50.
    result, _ = adjust(raw, vendor)
    assert np.isclose(result.close.iloc[1] / result.close.iloc[0] - 1, 0)
    assert np.allclose(result.volume, [20., 20.])
    assert np.allclose(result.vwap, 50.)
    # A later corporate action rescales all pre-anchor vendor prices equally.
    newer = vendor.copy()
    newer[['open', 'high', 'low', 'close']] *= .4
    again, _ = adjust(raw, newer)
    assert np.allclose(result[['open', 'high', 'low', 'close', 'vwap', 'volume', 'factor']],
                       again[['open', 'high', 'low', 'close', 'vwap', 'volume', 'factor']])
    try:
        adjust(raw, vendor.iloc[::-1].reset_index(drop=True))
    except ValueError:
        pass
    else:
        raise AssertionError('Mismatched dates accepted')
    root = args.panel
    completed = json.loads((root / 'complete.json').read_text())
    protocol = json.loads((root / 'protocol.json').read_text())
    assert hashlib.sha256((root / 'protocol.json').read_bytes()).hexdigest() == completed['protocol_sha256']
    receipts = sorted((root / 'receipts').glob('*.json'))
    assert len(receipts) == completed['stocks'] == protocol['stock_count']
    rows, max_return_error = 0, 0.
    calendar = pd.DatetimeIndex((root / 'qlib/calendars/day.txt').read_text().splitlines())
    assert calendar.is_monotonic_increasing and not calendar.has_duplicates
    for file in receipts:
        receipt = json.loads(file.read_text())
        for name, expected in receipt['sha256'].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
        stock = receipt['stock']
        raw = pd.read_parquet(root / 'raw' / (stock + '.parquet'))
        vendor = pd.read_parquet(root / 'adjusted' / (stock + '.parquet'))
        derived = pd.read_parquet(root / 'derived' / (stock + '.parquet'))
        a = vendor.close.to_numpy()
        b = derived.close.to_numpy()
        good = np.isfinite(a[1:] / a[:-1]) & np.isfinite(b[1:] / b[:-1])
        error = np.abs(a[1:] / a[:-1] - b[1:] / b[:-1])[good]
        assert len(error) and error.max() < 1e-10
        max_return_error = max(max_return_error, float(error.max()))
        assert np.allclose((derived.volume * derived.factor), raw.volume, equal_nan=True)
        for name in ['open', 'high', 'low', 'close', 'volume', 'vwap', 'factor']:
            binary = np.fromfile(root / 'qlib/features' / stock.lower() / (name + '.day.bin'), dtype='<f4')
            assert binary[0] == 0 and len(binary) == 1 + len(calendar)
            expected = derived.set_index(pd.to_datetime(derived.date))[name].reindex(calendar)
            assert np.allclose(binary[1:], expected, rtol=1e-6, equal_nan=True)
        rows += len(raw)
    report = dict(passed=True, stocks=len(receipts), rows=rows,
        max_adjusted_return_ratio_error=max_return_error,
        checks=['Two-for-one split produces zero adjusted return',
            'Events after frozen anchor do not change earlier normalized prices',
            'Mismatched vendor date keys rejected', 'Raw and adjusted receipts hash-verified',
            'Derived return ratios agree with vendor adjusted returns',
            'Volume adjustment identity', 'Every exported Qlib field reconstructed independently'],
        interpretation='Input implementation checks only; not original-paper fidelity or a trained model',
        original_fidelity_passed=False, limitations=protocol['limitations'])
    (root / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
