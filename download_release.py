#!/usr/bin/env python3
"""Stream the official RURA data archive and inspect XZ size; run via the guard.

No dependencies, extraction, model downloads or experiment execution. A complete
archive is reused only with its saved size and SHA256. Interrupted downloads are
kept as .part evidence and can be retried; never used as valid data.
"""
import hashlib
import http.cookiejar
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from html.parser import HTMLParser
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import build_opener, HTTPCookieProcessor, Request

ROOT = Path(__file__).resolve().parent
FILE_ID = '1KA23UTBJF8iwG2fE7q1iIIruOmYdCZbP'
EXPECTED_BYTES = 3866306020
URL = 'https://drive.google.com/uc?' + urlencode({'export': 'download', 'id': FILE_ID})
DEST = ROOT / 'data.tar.xz'
MANIFEST = ROOT / 'sources' / 'data-download.json'


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


class ConfirmationForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.forms = []
        self.form = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.form = {'action': attrs.get('action', ''), 'fields': {}}
            self.forms.append(self.form)
        elif tag == 'input' and self.form is not None and attrs.get('name'):
            self.form['fields'][attrs['name']] = attrs.get('value', '')

    def handle_endtag(self, tag):
        if tag == 'form':
            self.form = None


def main():
    info = {'paper_id': 388, 'source_url': URL,
            'source_folder': 'https://drive.google.com/drive/folders/1jl9YmSnbkQoSaZ2rMWKILBfHjOCnQtTA',
            'expected_bytes': EXPECTED_BYTES,
            'expected_size_evidence': str(ROOT / 'sources' / 'drive-folder.html'),
            'started_at_unix': time.time(), 'status': 'started',
            'archive_extracted': False, 'dataset_license': 'Not specified in code repository; inspect archive before use.'}
    if DEST.exists():
        if not MANIFEST.exists():
            raise RuntimeError('Archive exists without download manifest: inspect before reusing')
        prior = json.loads(MANIFEST.read_text())
        digest = sha256_file(DEST)
        if DEST.stat().st_size != EXPECTED_BYTES or digest != prior.get('sha256'):
            raise RuntimeError('Existing archive failed saved hash/size verification; not overwritten')
        info.update(prior)
        info['reused_at_unix'] = time.time()
    else:
        # Only compressed data is downloaded in this stage. Extraction is a
        # separate guarded operation after XZ uncompressed bytes are known.
        if shutil.disk_usage(ROOT).free < EXPECTED_BYTES + 21 * 1024**3:
            raise RuntimeError('Insufficient free space to preserve 20 GiB reserve')
        part = DEST.with_suffix(DEST.suffix + '.part')
        if part.exists():
            # Do not reserve disk twice or silently overwrite prior partial work.
            raise RuntimeError('Partial archive exists; inspect/remove explicitly before retry')
        atomic_json(MANIFEST, info)
        opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))
        url = URL
        response = None
        for attempt in range(3):
            response = opener.open(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=60)
            if 'text/html' not in response.headers.get('Content-Type', ''):
                break
            text = response.read(1024 * 1024).decode('utf-8', errors='replace')
            response.close()
            page = ROOT / 'sources' / ('download-response-%d.html' % attempt)
            page.write_text(text)
            parser = ConfirmationForm()
            parser.feed(text)
            candidates = [f for f in parser.forms if 'download' in f['action']]
            if not candidates:
                raise RuntimeError('Drive did not supply a download confirmation form; see ' + str(page))
            form = candidates[0]
            action = urljoin(url, form['action'])
            if urlparse(action).hostname not in {'drive.google.com', 'drive.usercontent.google.com'}:
                raise RuntimeError('Unexpected confirmation host; inspect without following')
            url = action + ('&' if '?' in action else '?') + urlencode(form['fields'])
        else:
            raise RuntimeError('Drive confirmation did not resolve')
        if response is None:
            raise RuntimeError('No archive response')
        digest = hashlib.sha256()
        count = 0
        progress = time.monotonic()
        with response, part.open('xb') as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                if count > EXPECTED_BYTES:
                    raise RuntimeError('Response exceeds published bytes; stopping without acceptance')
                digest.update(chunk)
                output.write(chunk)
                if time.monotonic() - progress >= 20:
                    print(json.dumps({'bytes_received': count, 'expected_bytes': EXPECTED_BYTES}), flush=True)
                    progress = time.monotonic()
            output.flush()
            os.fsync(output.fileno())
        if count != EXPECTED_BYTES:
            raise RuntimeError('Archive size differs from official folder metadata: %d' % count)
        with part.open('rb') as stream:
            if stream.read(6) != b'\xfd7zXZ\x00':
                raise RuntimeError('Response is not an XZ archive')
        part.replace(DEST)
        info.update({'downloaded_bytes': count, 'sha256': digest.hexdigest(), 'status': 'download_complete'})
        atomic_json(MANIFEST, info)
    listing = subprocess.run(['xz', '--robot', '--list', '--verbose', str(DEST)],
                             text=True, capture_output=True, check=True)
    (ROOT / 'sources' / 'archive-xz-metadata.txt').write_text(listing.stdout)
    info.update({'xz_metadata_path': str(ROOT / 'sources' / 'archive-xz-metadata.txt'),
                 'status': 'download_complete_unextracted', 'finished_at_unix': time.time()})
    atomic_json(MANIFEST, info)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'status': 'failed', 'error': type(exc).__name__ + ': ' + str(exc)}), flush=True)
        raise
