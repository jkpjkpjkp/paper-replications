#!/usr/bin/env python3
"""Package exactly the source selection used by the paper upload branches."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tarfile
from prepare_upload import selected, inspect

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path,
                   default=ROOT / '.replication-local/release/replication-source.tar.gz')
    a = p.parse_args()
    catalogue = json.loads((ROOT / '_replication/upload_projects.json').read_text())
    files = set()
    for base in [ROOT] + [ROOT / p['path'] for p in catalogue['projects']]:
        if not (base / '.git').exists():
            raise SystemExit('Run python _replication/prepare_upload.py first.')
        names, _, _ = selected(base)
        inspect(base, names)
        files.update(base / name for name in names)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    with a.output.open('wb') as handle, gzip.GzipFile(fileobj=handle, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for f in sorted(files):
                rel = f.relative_to(ROOT)
                raw = f.read_bytes()
                manifest.append({'path': str(rel), 'bytes': len(raw),
                                 'sha256': hashlib.sha256(raw).hexdigest()})
                info = tar.gettarinfo(str(f), arcname=str(rel))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                with f.open('rb') as stream:
                    tar.addfile(info, stream)
    record = {'file_count': len(manifest), 'source_bytes': sum(x['bytes'] for x in manifest),
              'archive_sha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
              'files': manifest}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, indent=2))
    print(a.output)


if __name__ == '__main__':
    main()
