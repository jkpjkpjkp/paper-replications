"""Validate and unpack the dated author-linked Qlib archive without overwriting data."""
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT/'data'
archive = folder/'qlib-2023-06-01.tar.gz'
part = archive.with_suffix(archive.suffix+'.part')
target = folder/'crowd_data'
receipt = folder/'doubleadapt-data-receipt.json'
if receipt.exists():
    raise SystemExit('Already extracted; preserve existing receipt')
src = archive if archive.exists() else part
if not src.exists():
    raise SystemExit('Download missing')
h = hashlib.sha256()
with open(src, 'rb') as f:
    for chunk in iter(lambda: f.read(1<<20), b''):
        h.update(chunk)
with tarfile.open(src, 'r:gz') as tar:
    members = tar.getmembers()  # Also validates the complete compressed stream.
    out = []
    for member in members:
        parts = member.name.split('/')  # Keep the literal leading '.' for tar --strip-components=2.
        # Author command strips the two leading components.
        if len(parts) <= 2:
            continue
        relative = Path(*parts[2:])
        if relative.is_absolute() or '..' in relative.parts or member.issym() or member.islnk():
            raise ValueError(f'Unsafe archive member: {member.name}')
        member.name = str(relative)
        out.append(member)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('Extraction target is not empty')
    target.mkdir(parents=True, exist_ok=True)
    tar.extractall(target, members=out, filter='data')
if src == part:
    part.rename(archive)
cal = (target/'calendars/day.txt').read_text().splitlines()
members = (target/'instruments/csi300.txt').read_text().splitlines()
assert cal[0] <= '2008-01-01' and cal[-1] >= '2020-08-01'
assert len(members) > 300
receipt.write_text(json.dumps(dict(
    url='https://github.com/chenditc/investment_data/releases/download/2023-06-01/qlib_bin.tar.gz',
    author_reference='sources/DoubleAdapt-qlib/examples/benchmarks_dynamic/incremental/README.md',
    downloaded_utc=datetime.now(timezone.utc).isoformat(), archive_bytes=archive.stat().st_size,
    archive_sha256=h.hexdigest(), transformations='strip-components=2 exactly as author README',
    calendar=[cal[0],cal[-1]], calendar_days=len(cal), csi300_membership_intervals=len(members),
    extracted_files=sum(p.is_file() for p in target.rglob('*'))), indent=2))
print(receipt.read_text())
