#!/usr/bin/env python3
"""Optional retrieval of the two public source PDFs; not needed to run experiments."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'sources/source_manifest.json').read_text())
for source in manifest['sources']:
    target=ROOT/source['local_path']
    if target.exists():
        payload=target.read_bytes()
    else:
        with urlopen(source['download_url'],timeout=90) as response:
            payload=response.read()
        if not payload.startswith(b'%PDF'):
            raise ValueError(f'Non-PDF response for {source["id"]}')
        target.write_bytes(payload)
    actual=hashlib.sha256(payload).hexdigest()
    print(source['id'],len(payload),actual)
    if actual!=source['sha256']:
        raise ValueError('Source bytes differ from archived reproduction; inspect before replacing manifest')
