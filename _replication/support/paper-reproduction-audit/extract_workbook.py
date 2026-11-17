#!/usr/bin/env python3
"""Extract the supplied paper list without non-stdlib spreadsheet dependencies.

Run explicitly, choosing a destination directory:
    python paper-reproduction-audit/extract_workbook.py --output-dir /tmp/paper-list

Existing output files are protected unless --overwrite is explicitly supplied.
Importing this module performs no extraction or writes.
"""

import argparse
import csv
import hashlib
import io
import json
import posixpath
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path


MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS = {"m": MAIN_NS}
WORKBOOK_PART = "xl/workbook.xml"
RELATIONSHIPS_PART = "xl/_rels/workbook.xml.rels"
FIELD_COLUMNS = (
    ("id", "A"),
    ("conference", "B"),
    ("track", "D"),
    ("category", "E"),
    ("title", "F"),
    ("topic", "G"),
    ("authors", "H"),
    ("url", "I"),
    ("evidence_url", "J"),
    ("notes", "K"),
)
EXPECTED_HEADERS = {
    "A": "编号", "B": "会议", "C": "届次", "D": "正式赛道",
    "E": "关联类型", "F": "英文题名", "G": "中文主题", "H": "作者",
    "I": "论文链接", "J": "收录依据", "K": "说明",
}
DEFAULT_WORKBOOK = Path(__file__).resolve().parent.parent / "2026_AI_Finance_Papers.xlsx"


def relationship_part(target):
    """Resolve an internal package relationship relative to workbook.xml."""
    if "\\" in target or ":" in target:
        raise ValueError("Unsupported workbook relationship target: " + target)
    part = posixpath.normpath(
        target.lstrip("/") if target.startswith("/")
        else posixpath.join(posixpath.dirname(WORKBOOK_PART), target)
    )
    if part in (".", "..") or part.startswith("../"):
        raise ValueError("Workbook relationship escapes the package: " + target)
    return part


def text_value(element):
    """Join rich text runs, excluding phonetic annotations."""
    if element is None:
        return None
    parts = []
    for child in element:
        if child.tag == "{" + MAIN_NS + "}t":
            parts.append(child.text or "")
        elif child.tag == "{" + MAIN_NS + "}r":
            parts.extend(run.text or "" for run in child.findall("m:t", NS))
    return "".join(parts)


def cell_value(cell, shared_strings):
    kind = cell.get("t")
    if kind == "inlineStr":
        return text_value(cell.find("m:is", NS))
    value = cell.find("m:v", NS)
    if value is None:
        return None
    raw = value.text or ""
    if kind == "s":
        index = int(raw)
        if not 0 <= index < len(shared_strings):
            raise ValueError("Invalid shared-string index: " + raw)
        return shared_strings[index]
    return raw


def row_values(row, shared_strings):
    values = {}
    for cell in row.findall("m:c", NS):
        reference = cell.get("r", "")
        match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", reference)
        if match is None:
            raise ValueError("Invalid cell reference: " + reference)
        value = cell_value(cell, shared_strings)
        if value is not None:
            values[match.group(1)] = value
    return values


def extract(workbook_path, sheet_name="论文清单", expected_count=437):
    """Return paper records and descriptive extraction provenance; write nothing."""
    workbook_path = Path(workbook_path).resolve()
    workbook_bytes = workbook_path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(workbook_bytes)) as package:
        workbook = ET.fromstring(package.read(WORKBOOK_PART))
        relationships = ET.fromstring(package.read(RELATIONSHIPS_PART))
        targets = {}
        strings_part = None
        for relationship in relationships:
            if relationship.get("TargetMode") == "External":
                continue
            part = relationship_part(relationship.attrib["Target"])
            targets[relationship.attrib["Id"]] = part
            if relationship.get("Type", "").endswith("/sharedStrings"):
                strings_part = part
        shared_strings = []
        if strings_part is not None:
            shared_strings = [
                text_value(item)
                for item in ET.fromstring(package.read(strings_part))
            ]
        sheets = workbook.find("m:sheets", NS)
        if sheets is None:
            raise ValueError("Workbook contains no sheets")
        metadata = []
        selected_rows = None
        for sheet in sheets:
            name = sheet.attrib["name"]
            relation_id = sheet.attrib["{" + REL_NS + "}id"]
            if relation_id not in targets:
                raise ValueError("Sheet has no internal relationship: " + name)
            part = targets[relation_id]
            xml = ET.fromstring(package.read(part))
            rows = xml.findall("m:sheetData/m:row", NS)
            metadata.append({
                "name": name,
                "part": part,
                "xml_row_count": len(rows),
                "nonempty_row_count": sum(bool(row_values(row, shared_strings)) for row in rows),
            })
            if name == sheet_name:
                if selected_rows is not None:
                    raise ValueError("Duplicate target sheet: " + name)
                selected_rows = rows
        if selected_rows is None:
            raise ValueError("Missing target sheet: " + sheet_name)
        paper_records = []
        header_found = False
        source_row_numbers = []
        for row in selected_rows:
            values = row_values(row, shared_strings)
            if values.get("A") == "编号":
                if any(values.get(column) != label for column, label in EXPECTED_HEADERS.items()):
                    raise ValueError("Paper-list columns no longer match the expected schema")
                header_found = True
                continue
            identifier = values.get("A", "")
            if not re.fullmatch(r"[0-9]+", identifier):
                continue
            if not header_found:
                raise ValueError("Found a paper before the expected header")
            record = {field: values.get(column) for field, column in FIELD_COLUMNS}
            record["id"] = int(identifier)
            if not record["title"]:
                raise ValueError("Missing title for paper " + identifier)
            paper_records.append(record)
            source_row_numbers.append(int(row.attrib["r"]))
        identifiers = [record["id"] for record in paper_records]
        if not header_found:
            raise ValueError("Paper-list header not found")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Duplicate paper IDs")
        if len(paper_records) != expected_count:
            raise ValueError(f"Expected {expected_count} papers; extracted {len(paper_records)}")
        if identifiers != list(range(1, expected_count + 1)):
            raise ValueError("Paper IDs must be ordered and contiguous from 1")
    provenance = {
        "source_workbook": str(workbook_path),
        "source_workbook_sha256": hashlib.sha256(workbook_bytes).hexdigest(),
        "source_workbook_size_bytes": len(workbook_bytes),
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "extractor": "extract_workbook.py",
        "dependencies": "Python standard library only",
        "sheet_count": len(metadata),
        "sheets": metadata,
        "selected_sheet": sheet_name,
        "paper_record_count": len(paper_records),
        "selected_source_row_numbers": source_row_numbers,
        "output_fields": [field for field, _ in FIELD_COLUMNS],
        "scope": "Faithful extraction of all indexed paper-list entries, including missing optional fields; no eligibility, bibliographic identity, source availability, or reproduction claims are verified by extraction.",
        "excluded_source_column": "C (conference edition/year), matching the existing papers.json/tsv schema",
        "provenance_limit": "The workbook's own verification notes are copied as source data and are not independently confirmed by this script.",
    }
    return paper_records, provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--sheet", default="论文清单")
    parser.add_argument("--expected-count", type=int, default=437)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_count < 1:
        parser.error("--expected-count must be positive")
    destination = args.output_dir.resolve()
    output_paths = [destination / filename for filename in (
        "papers.json", "papers.tsv", "SOURCE_PROVENANCE.json"
    )]
    if not args.overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            parser.error("Refusing to overwrite existing files without --overwrite: " + ", ".join(existing))
    records, provenance = extract(args.workbook, args.sheet, args.expected_count)
    destination.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    with output_paths[1].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[field for field, _ in FIELD_COLUMNS], delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    output_paths[2].write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"paper_count": len(records), "output_directory": str(destination), "source_sha256": provenance["source_workbook_sha256"]}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, ET.ParseError) as error:
        print("Extraction failed: " + str(error), file=sys.stderr)
        sys.exit(1)
