"""Acquire only the two pinned aggregate CSVs; never execute upstream code."""

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from urllib.request import Request, urlopen


def specification():
    return json.loads(files("vision_to_value").joinpath("inputs.json").read_text())


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def verify(content: bytes, entry: dict) -> str:
    digest = sha256(content)
    if len(content) != entry["bytes"] or digest != entry["sha256"]:
        raise ValueError(f"Checksum/size mismatch for {entry['path']}; expected {entry['sha256']}, got {digest}")
    return digest


def acquire(cache_dir: Path, upstream_dir: Path | None = None, offline: bool = False):
    """Return validated bytes and provenance, before any result files are written.

    An explicit local checkout takes precedence and is read-only. Otherwise a
    valid cached CSV is reused; missing inputs are fetched unless offline.
    A corrupt cache is an error, never silently repaired or used.
    """
    spec = specification()
    inputs, receipts = {}, []
    for entry in spec["files"]:
        name = Path(entry["path"]).name
        url = f"https://raw.githubusercontent.com/finxlab/vit-sdf-lasso/{spec['commit']}/{entry['path']}"
        local = upstream_dir / entry["path"] if upstream_dir is not None else cache_dir / name
        if local.is_file():
            content = local.read_bytes()
            origin = "local_checkout" if upstream_dir is not None else "cache"
        elif upstream_dir is not None or offline:
            raise FileNotFoundError(f"Missing input: {local}. Supply the pinned checkout or run once without --offline.")
        else:
            request = Request(url, headers={"User-Agent": "vision-to-value-replication/0.1.0"})
            with urlopen(request, timeout=60) as response:
                # A bounded read catches oversized/incorrect responses.
                content = response.read(entry["bytes"] + 1)
            verify(content, entry)
            cache_dir.mkdir(parents=True, exist_ok=True)
            local.write_bytes(content)
            origin = "download"
        digest = verify(content, entry)
        inputs[entry["model"]] = content
        receipts.append({**entry, "url": url, "sha256": digest, "origin": origin})
    return inputs, receipts
