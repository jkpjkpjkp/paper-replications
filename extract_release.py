#!/usr/bin/env python3
"""Guarded streaming extraction and checksummed inventory of official archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parent


def atomic_json(path, obj):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-extracted-bytes', type=int, required=True)
    args = ap.parse_args()
    archive = ROOT / 'data.tar.xz'
    dest = ROOT / 'release'
    manifest = ROOT / 'sources' / 'extraction.json'
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior.get('status') == 'complete':
            print(json.dumps({'status': 'reused_complete_extraction', 'manifest': str(manifest)}), flush=True)
            return
    if shutil.disk_usage(ROOT).free < args.max_extracted_bytes + 21 * 1024**3:
        raise RuntimeError('Insufficient extraction storage plus20GiB reserve')
    dest.mkdir(exist_ok=True)
    info = {'status': 'running', 'started_at_unix': time.time(),
            'archive': str(archive), 'destination': str(dest),
            'max_extracted_bytes': args.max_extracted_bytes}
    atomic_json(manifest, info)
    total = count = 0
    dirs = {}
    index = ROOT / 'sources' / 'archive-members.jsonl'
    last = time.monotonic()
    with tarfile.open(archive, mode='r|xz') as tar, index.open('w') as listing:
        for member in tar:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or '..' in rel.parts:
                raise RuntimeError('Unsafe archive member path: ' + member.name)
            path = dest / str(rel)
            entry = {'name': member.name, 'bytes': member.size,
                     'type': member.type.decode(errors='replace')}
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                total += member.size
                count += 1
                if total > args.max_extracted_bytes:
                    raise RuntimeError('Archive file bytes exceed XZ size/admitted extraction estimate')
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + '.partial')
                # Existing extracted files are preserved and byte-hash checked
                # against archive content during a resumed extraction.
                existing = path.exists()
                old_digest = None
                if existing:
                    h = hashlib.sha256()
                    with path.open('rb') as old:
                        for chunk in iter(lambda: old.read(1024 * 1024), b''):
                            h.update(chunk)
                    old_digest = h.hexdigest()
                digest = hashlib.sha256()
                source = tar.extractfile(member)
                output = None if existing else temp.open('wb')
                try:
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(chunk)
                        if output is not None:
                            output.write(chunk)
                finally:
                    source.close()
                    if output is not None:
                        output.close()
                entry['sha256'] = digest.hexdigest()
                if existing:
                    if old_digest != entry['sha256'] or path.stat().st_size != member.size:
                        raise RuntimeError('Pre-existing extracted content differs: ' + str(path))
                    entry['reused'] = True
                else:
                    temp.replace(path)
                top = rel.parts[0] if rel.parts else '.'
                dirs[top] = dirs.get(top, 0) + member.size
            else:
                raise RuntimeError('Unsupported special/link archive member: ' + member.name)
            listing.write(json.dumps(entry) + '\n')
            if time.monotonic() - last > 20:
                print(json.dumps({'files_extracted_or_verified': count, 'file_bytes': total,
                                  'last_member': member.name}), flush=True)
                last = time.monotonic()
    info.update({'status': 'complete', 'finished_at_unix': time.time(),
                 'file_count': count, 'file_bytes': total,
                 'top_level_bytes': dirs, 'member_index': str(index)})
    atomic_json(manifest, info)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    main()
