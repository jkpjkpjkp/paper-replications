#!/usr/bin/env python3
"""Retrieve the canonical sources without confusing a challenge page with a PDF.

Standard library only. No credentials, browser cookies, or login automation.
Run from any directory; all output stays under this reproduction directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
NOTE_ID = "NZHuWIlAUx"
TITLE = (
    "Stable Alpha: Adversarial Invariant Representation Learning for Nonlinear "
    "Asset Pricing under Temporal Distribution Shifts"
)
URLS = {
    "neurips-metadata": "https://neurips.cc/static/virtual/data/neurips-2026-orals-posters.json",
    "openreview-note-api2": f"https://api2.openreview.net/notes?id={NOTE_ID}",
    "openreview-pdf": f"https://openreview.net/pdf?id={NOTE_ID}",
}


def save_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def paper_records(data: object) -> list[dict]:
    """Search structured conference metadata, including nested event lists."""
    if isinstance(data, dict):
        title = data.get("name") or data.get("title") or ""
        if normalize(str(title)) == normalize(TITLE):
            return [data]
        return [record for value in data.values() for record in paper_records(value)]
    if isinstance(data, list):
        return [record for value in data for record in paper_records(value)]
    return []


def inspect_pdf(path: Path) -> dict:
    body = path.read_bytes()
    result = {
        "file": str(path.relative_to(ROOT)),
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "pdf_magic_valid": body.startswith(b"%PDF-"),
        "text_extractable": False,
        "canonical_title_found": False,
    }
    if not result["pdf_magic_valid"]:
        return result
    text_path = path.with_suffix(".txt")
    try:
        run = subprocess.run(
            ["pdftotext", "-layout", str(path), str(text_path)],
            capture_output=True, text=True, timeout=45,
        )
        if run.returncode == 0:
            extracted = text_path.read_text(errors="replace")
            result["text_extractable"] = bool(extracted.strip())
            # A conference programme or bibliography may mention this title.
            # Only a title on the first physical page identifies a candidate.
            first_page = extracted.split("\f", 1)[0]
            result["canonical_title_mentioned_anywhere"] = normalize(TITLE) in normalize(extracted)
            result["canonical_title_found"] = normalize(TITLE) in normalize(first_page)
            result["identity_check"] = "canonical title on first physical page; manual method/version review still required"
            result["text_file"] = str(text_path.relative_to(ROOT))
        else:
            result["extraction_error"] = run.stderr[:500]
    except (OSError, subprocess.TimeoutExpired) as error:
        result["extraction_error"] = str(error)
    return result


def fetch(item: tuple[str, str]) -> dict:
    name, url = item
    record = {"name": name, "url": url}
    request = urllib.request.Request(url, headers={"User-Agent": "StableAlphaReproduction/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read()
            record.update(status=response.status, content_type=response.headers.get("Content-Type", ""))
    except urllib.error.HTTPError as error:
        body = error.read()
        record.update(status=error.code, content_type=error.headers.get("Content-Type", ""))
    except (OSError, TimeoutError) as error:
        return {**record, "error": str(error)}
    is_pdf = body.startswith(b"%PDF-")
    suffix = ".pdf" if is_pdf else ".json" if "json" in record["content_type"] else ".html"
    path = SOURCES / (name + suffix)
    path.write_bytes(body)
    record.update(file=str(path.relative_to(ROOT)), bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    record["verified_pdf"] = bool(record["status"] == 200 and is_pdf)
    if record["verified_pdf"]:
        record["inspection"] = inspect_pdf(path)
    if name == "neurips-metadata" and record["status"] == 200:
        try:
            matches = paper_records(json.loads(body))
            record["matched_papers"] = len(matches)
            if len(matches) == 1:
                save_json(SOURCES / "paper-metadata.json", matches[0])
        except (ValueError, TypeError) as error:
            record["parse_error"] = str(error)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Inspect saved files without making network requests.")
    parser.add_argument("--paper", type=Path, help="Import a supplied manuscript PDF for identity verification.")
    args = parser.parse_args()
    SOURCES.mkdir(parents=True, exist_ok=True)
    if args.paper:
        body = args.paper.expanduser().resolve().read_bytes()
        if not body.startswith(b"%PDF-"):
            parser.error("The supplied file is not a PDF; HTML challenge pages are not manuscripts.")
        destination = SOURCES / "manuscript.pdf"
        if destination.exists() and destination.read_bytes() != body:
            destination = SOURCES / ("manuscript-" + hashlib.sha256(body).hexdigest()[:12] + ".pdf")
        destination.write_bytes(body)
    records = []
    if not args.offline:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            records = list(pool.map(fetch, URLS.items()))
        save_json(SOURCES / "retrieval-latest.json", records)
    candidate_paths = {path for pattern in ("manuscript*.pdf", "openreview-pdf.pdf")
                       for path in SOURCES.glob(pattern)}
    manuscripts = [inspect_pdf(path) for path in sorted(candidate_paths)]
    identified = [x for x in manuscripts if x["pdf_magic_valid"] and x["canonical_title_found"]]
    audit = {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "title": TITLE,
        "openreview_id": NOTE_ID,
        "offline": args.offline,
        "manuscripts": manuscripts,
        "identified_manuscript_available": bool(identified),
        "status": "manuscript_requires_method_review" if identified else "manuscript_unavailable",
        "reproduction_complete": False,
        "training_run": False,
        "network_results": records,
        "next_step": "Read the manuscript, supplement, and any code before specifying CASH or comparing results.",
    }
    save_json(ROOT / "SOURCE_AUDIT.json", audit)
    print(json.dumps({key: audit[key] for key in ["status", "identified_manuscript_available", "reproduction_complete", "training_run"]}, indent=2))
    return 0 if identified else 2


if __name__ == "__main__":
    raise SystemExit(main())
