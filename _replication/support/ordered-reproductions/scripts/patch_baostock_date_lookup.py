"""Preserve upstream collectors and prepare narrowly corrected copies for data rebuilds."""
import ast
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    'alphasage': ROOT.parent/'alphasage-reproduction/upstream/src/data_collection/fetch_baostock_data.py',
    'alphagen': ROOT/'sources/AlphaGen/data_collection/fetch_baostock_data.py',
    'alphaqcm': ROOT/'sources/AlphaQCM/data_collection/fetch_baostock_data.py',
}
OLD = '''        def as_of_date(df: pd.DataFrame, date: str) -> pd.Series:
            index: int = df.index.searchsorted(date, side="right") - 1   # type: ignore
            return df.iloc[index]
'''
NEW = '''        def as_of_date(df: pd.DataFrame, date: str) -> pd.Series:
            dates = pd.DatetimeIndex(pd.to_datetime(df["date"]))
            if not dates.is_monotonic_increasing or dates.has_duplicates:
                raise ValueError("Adjustment dates must be sorted and unique")
            index = dates.searchsorted(pd.Timestamp(date), side="right") - 1
            if index < 0:
                raise ValueError("Adjustment anchor predates available history")
            return df.iloc[index]
'''


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    receipts = {}
    for name, source in SOURCES.items():
        text = source.read_text()
        assert text.count(OLD) == 1, name
        patched = text.replace(OLD, NEW)
        target = ROOT/'sources'/f'{name}_baostock_date_corrected.py'
        node = next(n for n in ast.walk(ast.parse(patched))
                    if isinstance(n, ast.FunctionDef) and n.name == 'as_of_date')
        scope = dict(pd=pd)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(target), 'exec'), scope)
        lookup = scope['as_of_date']
        sample = pd.DataFrame({'date': ['2010-01-04', '2010-01-05', '2010-01-08'],
                               'foreAdjustFactor': [0.5, 0.75, 1.0]})
        assert lookup(sample, '2010-01-05')['foreAdjustFactor'] == 0.75
        assert lookup(sample, '2010-01-07')['foreAdjustFactor'] == 0.75
        assert lookup(sample, '2010-01-20')['foreAdjustFactor'] == 1.0
        for frame, anchor in [(sample, '2009-12-31'), (sample.iloc[::-1], '2010-01-05'),
                              (pd.concat([sample, sample.iloc[[-1]]]), '2010-01-08')]:
            try:
                lookup(frame, anchor)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid date history must fail explicitly')
        target.write_text(patched)
        receipts[name] = dict(original=str(source), original_sha256=sha(text),
                              corrected=str(target), corrected_sha256=sha(patched),
                              verified_cases=['exact date', 'between trading dates', 'after final date',
                                              'before available history rejected', 'unsorted rejected', 'duplicates rejected'])
    receipt = dict(status='date_lookup_corrected_and_checked', collectors=receipts,
                   interpretation='Source-level fix only. Full data panels are not rebuilt and no effect on model results is established.',
                   remaining='Choose documented anchor dates and historical universes before collection. Pre-IPO anchors fail explicitly; do not silently use the last row. Copies require the original repository import paths/dependencies.')
    out = ROOT/'results/baostock-date-correction.json'
    out.write_text(json.dumps(receipt, indent=2)+'\n')
    print(str(out))


if __name__ == '__main__':
    main()
