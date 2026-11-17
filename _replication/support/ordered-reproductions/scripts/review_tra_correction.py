"""Verify corrected stock boundaries on the entire released feature index."""
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/tra-boundary-corrected'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'sources/tra_dataset_boundary_corrected.py'
    original_source = ROOT/'sources/TRA-qlib/examples/benchmarks/TRA/src/dataset.py'
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert sha(source) == protocol['dataset_patch']['patched_sha256']
    assert sha(original_source) == protocol['dataset_patch']['base_sha256']
    assert source.read_text() == original_source.read_text().replace(
        'start = max(end - seq_len, 0)', 'start = max(end - seq_len, cur_loc)')
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == '_create_ts_slices')
    scope = dict(np=np, pd=pd)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    frame = pd.read_pickle(ROOT/'data/TRA/kdd21/feature.pkl')
    index = frame.index.reorder_levels(['instrument', 'datetime']).sort_values()
    assert not index.has_duplicates
    config = json.loads((OUT/'tra-config.json').read_text())['task']['dataset']['kwargs']
    slices = scope['_create_ts_slices'](index, config['seq_len'])
    starts = np.fromiter((s.start for s in slices), dtype=np.int64)
    targets = np.fromiter((s.stop-1 for s in slices), dtype=np.int64)
    assert np.array_equal(targets, np.arange(len(index)))
    codes = index.get_level_values('instrument').to_numpy()
    dates = index.get_level_values('datetime').to_numpy()
    cross_stock = codes[starts] != codes[targets]
    # The index is sorted within each stock, so no interior row can be later
    # than its target once start/end stock boundaries agree.
    assert not cross_stock.any()
    assert not (dates[starts] > dates[targets]).any()
    latest_memory = targets-config['horizon']
    has_memory = latest_memory >= starts
    calendar_positions = np.searchsorted(np.unique(dates), dates)
    observed_lags = calendar_positions[targets[has_memory]]-calendar_positions[latest_memory[has_memory]]
    assert (observed_lags >= config['horizon']).all()
    counts = {}
    for name, (begin, end) in config['segments'].items():
        mask = (dates >= np.datetime64(begin)) & (dates <= np.datetime64(end))
        counts[name] = dict(rows=int(mask.sum()), cross_instrument_windows=int(cross_stock[mask].sum()),
                            future_dated_input_windows=0)
    native = json.loads((ROOT/'results/tra-original/summary.json').read_text())
    corrected = json.loads((OUT/'summary.json').read_text())
    deltas = {k: dict(native=native['independently_verified'][k], corrected=v,
                     delta=v-native['independently_verified'][k], paper=corrected['comparison'][k]['paper'])
              for k, v in corrected['independently_verified'].items()}
    review = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),
                  status='boundary_correction_verified_paper_fidelity_unresolved',
                  summary_sha256=sha(OUT/'summary.json'), protocol_sha256=sha(OUT/'protocol.json'),
                  exact_one_line_patch_verified=True, raw_panel_rows=len(index),
                  window_audit=counts, metrics=deltas,
                  memory_check=dict(horizon=config['horizon'], minimum_observed_calendar_lag=int(observed_lags.min()),
                      source_review='test_epoch emits predictions before assigning same-day losses; eval uses chronological daily batches. The last horizon observations are masked and excluded from the router.',
                      limitation='Checks indexing against the configured 21-observation horizon; does not recover or independently validate how the supplied labels were originally constructed.'),
                  limitations=['One seed and a changed validity condition do not reproduce the paper multi-seed aggregate.',
                               'Paper-era Qlib 0.7.1 versus the compatibility runtime remains unquantified.'],
                  full_cz_authorized_by_gate=False)
    (OUT/'fidelity-review.json').write_text(json.dumps(review, indent=2)+'\n')
    print(json.dumps(review, indent=2))


if __name__ == '__main__':
    main()
