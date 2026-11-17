"""Verify every CSI300 OHLCV/VWAP input file against the existing archive manifest."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent.parent / 'csi'
provider = SOURCE/'qlib/cn_data'
published = json.loads((SOURCE/'download_manifest.json').read_text())
symbols = sorted({line.split()[0].lower() for line in (provider/'instruments/csi300.txt').read_text().splitlines()})
files = ['calendars/day.txt', 'instruments/csi300.txt']
files += [f'features/{symbol}/{field}.day.bin' for symbol in symbols
          for field in ['open', 'close', 'high', 'low', 'volume', 'vwap']]
records, mismatches = {}, []
for name in files:
    file = provider/name
    digest = hashlib.sha256(file.read_bytes()).hexdigest() if file.exists() else None
    expected = published['files'].get(name, {}).get('sha256')
    records[name] = digest
    if digest is None or expected != digest:
        mismatches.append(name)
result = {k: published[k] for k in ['source', 'source_page', 'revision', 'archive_sha256', 'is_author_snapshot']}
result.update(verified_at=datetime.now(timezone.utc).isoformat(), verified_file_count=len(files),
              csi300_historical_symbols=len(symbols), mismatches=mismatches, sha256=records)
(ROOT/'sources/data-provenance.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'sha256'}, indent=2))
if mismatches:
    raise SystemExit(1)
