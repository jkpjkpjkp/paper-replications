#!/usr/bin/env python3
"""Acquire only the paper's TFB data from TFB's own published archive.

Run from the staged workspace described in README.md. No model is trained and no returns are inferred.
Archive acquisition is streamed; only NASDAQ/NYSE and metadata are extracted.
"""
import csv
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULT = ROOT / 'results'
FILE_ID = '1vgpOmAygokoUt235piWKUjfwao6KwLv7'
SOURCE = f'https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t'
EXPECTED_ARCHIVE_BYTES = 380653886
EXPECTED_SAMPLES = {'NASDAQ.csv': 1240, 'NYSE.csv': 1239}


def utc():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inspect_csv(path, samples):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        raise ValueError(f'{path.name}: empty CSV')
    if set(['date', 'data', 'cols']).issubset(fields):
        dates = sorted({row['date'] for row in rows}, key=float)
        channels = list(dict.fromkeys(row['cols'] for row in rows))
        pairs = {(row['date'], row['cols']) for row in rows}
        if len(pairs) != len(rows):
            raise ValueError(f'{path.name}: duplicate date/channel pair')
        by_channel = {key: [r for r in rows if r['cols'] == key] for key in channels}
        counts = {key: len(value) for key, value in by_channel.items()}
        if len(set(counts.values())) != 1:
            raise ValueError(f'{path.name}: unbalanced channel lengths; official block loader assumptions fail')
        native_n = next(iter(counts.values()))
        # Matches read_data in official TFB ts_benchmark/data/utils.py.
        # It uses row blocks and the first channel's dates, not a pivot by date.
        native = [[rows[i]['date']] + [by_channel[key][i]['data'] for key in channels]
                  for i in range(native_n)]
        by_pair = {(r['date'], r['cols']): r['data'] for r in rows}
        common = set.intersection(*(set(r['date'] for r in by_channel[key]) for key in channels))
        common = sorted(common, key=float)
        complete = [[day] + [by_pair[day, key] for key in channels] for day in common]
        if len(complete) != samples:
            raise ValueError(f'{path.name}: date-complete count {len(complete)} differs from paper {samples}')
        interpretations = []
        for name, table, reason in [
            ('tfb_native_blocks', native, 'Official TFB loader groups contiguous channel blocks; per-channel date labels are ignored.'),
            ('date_complete', complete, 'Inner join on numeric date labels; matches paper count, but original RISE preprocessing is unavailable. Explicit adaptation, not proven original preprocessing.'),
        ]:
            destination = DATA / (path.stem + '.' + name + '.csv')
            with destination.open('w', newline='') as out:
                writer = csv.writer(out); writer.writerow(['ordinal'] + channels); writer.writerows(table)
            interpretations.append({'name': name, 'samples': len(table), 'path': str(destination),
                                    'sha256': digest(destination), 'reason': reason,
                                    'genuine_calendar_dates': False})
        numeric = (row['data'] for row in rows)
        schema = 'TFB long-form date,data,cols'
    elif 'date' in fields:
        dates = [row['date'] for row in rows]
        channels = [key for key in fields if key != 'date']
        if dates != sorted(dates) or len(set(dates)) != len(dates):
            raise ValueError(f'{path.name}: unordered or duplicate dates')
        numeric = (row[key] for row in rows for key in channels)
        schema = 'wide date plus numeric channels'
    else:
        raise ValueError(f'{path.name}: unknown schema {fields}; no reshaping guessed')
    if len(channels) != 5:
        raise ValueError(f'{path.name}: expected five channels, found {len(channels)}')
    if any(not math.isfinite(float(value)) for value in numeric):
        raise ValueError(f'{path.name}: missing/nonfinite values; no imputation authorized')
    return {
        'path': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
        'schema': schema, 'csv_rows': len(rows), 'samples': len(dates),
        'channels': channels, 'first_date': dates[0], 'last_date': dates[-1],
        'paper_reported_samples': samples,
        'raw_grid_complete': len(rows) == len(dates) * len(channels),
        'channel_counts': counts,
        'interpretations': interpretations,
        'paper_shape_matches': 'date_complete only; native official loader length differs',
        'price_and_investable_instrument_semantics': 'Requires source metadata inspection; channels are not assumed to be distinct investable assets.',
    }


def main():
    DATA.mkdir(exist_ok=True)
    RESULT.mkdir(exist_ok=True)
    manifest = {
        'paper_id': 405, 'started_at': utc(), 'status': 'acquiring',
        'official_code_url': 'https://github.com/decisionintelligence/TFB',
        'official_data_page': f'https://drive.google.com/file/d/{FILE_ID}/view',
        'source_url': SOURCE, 'expected_archive_bytes': EXPECTED_ARCHIVE_BYTES,
        'expected_archive_bytes_evidence': str(ROOT / 'sources/tfb-drive-metadata-summary.json'),
        'estimated_peak_additional_disk_gib': 1.0,
        'scope': 'TFB NASDAQ and NYSE only; no curated Yahoo replacement, no synthetic data',
        'license': 'TFB repository is MIT; dataset-specific upstream terms require source metadata review.',
        'datasets': [],
    }
    manifest_path = RESULT / 'tfb-data-manifest.json'
    try:
        if shutil.disk_usage(ROOT).free < 21 * 1024 ** 3:
            raise RuntimeError('20 GiB disk reserve plus declared 1 GiB peak would be breached')
        archive = DATA / 'tfb-forecasting.zip'
        if not archive.exists():
            part = DATA / 'tfb-forecasting.zip.part'
            if part.exists():
                raise RuntimeError('Prior partial archive exists; inspect provenance before resuming or removing it')
            request = urllib.request.Request(SOURCE, headers={'User-Agent': 'paper405-reproduction-data-audit/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, part.open('xb') as output:
                kind = response.headers.get('Content-Type', '')
                manifest['response_content_type'] = kind
                if 'text/html' in kind.lower():
                    raise RuntimeError('Official public download returned an HTML access/confirmation page; no login or bypass attempted')
                count = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b''):
                    count += len(chunk)
                    if count > 512 * 1024 ** 2:
                        raise RuntimeError('Archive exceeded the declared 512 MiB safety bound')
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if count != EXPECTED_ARCHIVE_BYTES:
                raise RuntimeError(f'Official archive changed or was incomplete: received {count}, advertised {EXPECTED_ARCHIVE_BYTES}; inspect before use')
            if not zipfile.is_zipfile(part):
                raise RuntimeError('Response is not a valid ZIP archive')
            os.replace(part, archive)
        if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise RuntimeError('Cached archive size does not match saved official metadata')
        manifest['archive'] = {'path': str(archive), 'bytes': archive.stat().st_size, 'sha256': digest(archive)}
        with zipfile.ZipFile(archive) as z:
            members = z.infolist()
            atomic_json(RESULT / 'tfb-archive-members.json', [
                {'name': item.filename, 'bytes': item.file_size, 'compressed_bytes': item.compress_size, 'crc32': f'{item.CRC:08x}'} for item in members
            ])
            for name, expected in EXPECTED_SAMPLES.items():
                matches = [item for item in members if Path(item.filename).name.lower() == name.lower()]
                if len(matches) != 1:
                    raise ValueError(f'Expected one {name} in official archive, found {len(matches)}')
                member = matches[0]
                if member.file_size > 10 * 1024 ** 2:
                    raise ValueError('Individual expected small TFB dataset unexpectedly exceeds 10 MiB')
                output = DATA / name
                blob = z.read(member)
                if output.exists() and output.read_bytes() != blob:
                    raise RuntimeError(f'Existing {name} differs; preserve it for investigation')
                if not output.exists():
                    output.write_bytes(blob)
                result = inspect_csv(output, expected)
                result['archive_member'] = member.filename
                manifest['datasets'].append(result)
            for item in members:
                if Path(item.filename).name.upper() == 'FORECAST_META.CSV' and item.file_size < 10 * 1024 ** 2:
                    output = DATA / 'FORECAST_META.csv'
                    blob = z.read(item)
                    if not output.exists():
                        output.write_bytes(blob)
                    elif output.read_bytes() != blob:
                        raise RuntimeError('Existing source metadata differs; preserve it for investigation')
                    manifest['source_metadata'] = {'path': str(output), 'sha256': digest(output)}
        manifest['status'] = 'data_verified_not_reproduced'
        manifest['completed_at'] = utc()
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return 0
    except Exception as exc:
        manifest['status'] = 'blocked_data'
        manifest['error_type'] = type(exc).__name__
        manifest['error'] = str(exc)
        manifest['completed_at'] = utc()
        atomic_json(manifest_path, manifest)
        print(json.dumps(manifest, indent=2), flush=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
