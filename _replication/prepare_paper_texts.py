#!/usr/bin/env python3
"""Build searchable paper text and line/page indexes from curated local sources."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = '_replication/prepare_paper_texts.py'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def normalized_lines(text):
    return [line.rstrip() for line in text.replace('\r\n', '\n').splitlines()]


def pdf_text(path, layout=False):
    command = ['pdftotext', '-enc', 'UTF-8']
    if layout:
        command.append('-layout')
    return subprocess.run(command + [str(path), '-'], check=True,
                          capture_output=True, text=True).stdout


def split_pages(text):
    pages = text.split('\f')
    if len(pages) > 1 and not pages[-1].strip():
        pages.pop()
    return pages


def clean_browser_text(text, kind):
    """Remove known site chrome; do not reconstruct prose, formulas or tables."""
    if kind == 'abstract':
        return text.split('\nShow more', 1)[0].strip(), 'Official abstract only'
    match = re.search(r'^# [^\n]+', text, re.M)
    if not match:
        raise ValueError('Expected a verified article title heading')
    body = text[match.start():]
    ends = [body.find(marker) for marker in (
        '\nExperimental support, please',
        '\n## Instructions for reporting errors',
        '\n## ACTIONS',
    )]
    ends = [end for end in ends if end >= 0]
    if ends:
        body = body[:min(ends)]
    return body.strip(), 'Article body; navigation and known site footer removed'


def render(title, coverage, note, pages, page_kind, source, source_hash, layout=False):
    lines = [
        title, '',
        'AGENT-READABLE PAPER REFERENCE',
        'Coverage: ' + coverage,
        'Original source: ' + source,
        'Source SHA-256: ' + source_hash,
        'Index and provenance: PAPER_INDEX.json',
        'This is a text reference, not a replication assessment.',
        'For experiment scope, read the project README and STATUS.txt when present.',
        'Equations, tables and figures require checking against the original source.',
    ]
    if note:
        lines.extend(['Note: ' + note])
    if layout:
        lines.append('Layout extraction: spacing retains approximate PDF columns and tables.')
    lines.extend(['', '=== BEGIN SOURCE TEXT ===', ''])
    body_start = len(lines) + 1
    locations = []
    for number, page in enumerate(pages, 1):
        start = len(lines) + 1
        marker = f'=== {page_kind} {number} ==='
        lines.append(marker)
        lines.extend(normalized_lines(page.strip('\n')))
        lines.append('')
        locations.append({'page': number if page_kind == 'PDF PAGE' else None,
                          'segment': number, 'line_start': start, 'line_end': len(lines)})
    lines.append('=== END SOURCE TEXT ===')
    return '\n'.join(lines) + '\n', locations, body_start


def write_atomic(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def build(key, spec, folder):
    # Keep top-level project references in their agent-only directory.
    reference_folder = folder / '.agent' if folder.parent == ROOT else folder
    source = ROOT / spec['source']
    raw = source.read_bytes()
    source_hash = digest(raw)
    relative_source = os.path.relpath(source, folder)
    abstract = folder / 'ABSTRACT.txt'
    title = spec.get('title') or (abstract.read_text().splitlines()[0] if abstract.is_file() else key)
    kind = spec['kind']
    note = spec.get('note', '')
    layout_pages = None
    original_pages = None
    if kind == 'pdf':
        content = pdf_text(source)
        pages = split_pages(content)
        layout_pages = split_pages(pdf_text(source, layout=True))
        info = subprocess.run(['pdfinfo', str(source)], check=True,
                              capture_output=True, text=True).stdout
        original_pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
        if len(pages) != original_pages or len(layout_pages) != original_pages:
            raise ValueError(f'{key}: PDF page count mismatch')
        method = 'Poppler pdftotext UTF-8, reading order plus separate layout extraction'
        page_kind = 'PDF PAGE'
    elif kind == 'layout_text':
        content = raw.decode('utf-8')
        pages = split_pages(content)
        original_pages = len(pages)
        method = 'Existing PDF text extraction; preserved form-feed page boundaries and spacing'
        page_kind = 'PDF PAGE'
    elif kind == 'summary':
        content = spec['summary']
        pages = [content]
        method = 'Concise paper guide; original full text linked, not reproduced'
        page_kind = 'TEXT SEGMENT'
    else:
        content = raw.decode('utf-8')
        if kind in {'browser_text', 'abstract'}:
            content, method = clean_browser_text(content, kind)
        else:
            method = 'Existing local study overview; no single target paper'
        pages = [content]
        page_kind = 'TEXT SEGMENT'
    if not content.strip():
        raise ValueError(f'{key}: empty source')
    if kind not in {'abstract', 'study', 'summary'} and len(content.strip()) < 5000:
        raise ValueError(f'{key}: unexpectedly short paper extraction')

    text, page_index, body_start = render(title, spec['coverage'], note, pages,
                                        page_kind, relative_source, source_hash)
    output = {'PAPER_TEXT.txt': text.encode('utf-8')}
    layout_index = None
    if layout_pages is not None:
        layout, layout_index, _ = render(title, spec['coverage'], note, layout_pages,
                                         page_kind, relative_source, source_hash, layout=True)
        output['PAPER_LAYOUT.txt'] = layout.encode('utf-8')
    lines = text.splitlines()
    chunks = [{'id': i + 1, 'line_start': start, 'line_end': min(start + 79, len(lines))}
              for i, start in enumerate(range(body_start, len(lines) + 1, 72))]
    heading_candidates = []
    for number, line in enumerate(lines[body_start - 1:], body_start):
        stripped = line.strip()
        if re.match(r'^#{1,6}\s+\S', stripped) or (
            len(stripped) < 110 and re.match(r'^(?:\d+(?:\.\d+)*\.?|[A-Z](?:\.\d+)*)\s+[A-Z][A-Za-z]', stripped)
        ) or stripped.lower() in {'abstract', 'references', 'bibliography', 'conclusion', 'appendix'}:
            heading_candidates.append({'line': number, 'text': stripped})
    urls = [spec['url']] if spec.get('url') else []
    header_text = ''
    if kind in {'pdf', 'layout_text'}:
        header_text = pages[0]
    elif kind == 'browser_text':
        original = raw.decode('utf-8')
        heading = re.search(r'^# [^\n]+', original, re.M)
        header_text = original[:heading.start()] if heading else ''
    id_match = re.search(r'arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)', header_text)
    if id_match:
        urls.append('https://arxiv.org/abs/' + id_match.group(1))
    metadata = {
        'schema_version': 1, 'generated_by': GENERATOR, 'generated_on': str(date.today()),
        'project': key, 'title': title, 'coverage': spec['coverage'],
        'source': {'path_relative_to_project': relative_source,
                   'path_relative_to_collection': spec['source'],
                   'sha256': source_hash, 'bytes': len(raw), 'urls': sorted(set(urls))},
        'method': method, 'note': note,
        'limitations': ['No OCR or figure interpretation is synthesized.',
                       'Mathematical symbols, two-column reading order, and table alignment may be lossy.',
                       'HTML captures have no verified PDF page numbers; appendix conversion may be incomplete.'],
        'text_file': 'PAPER_TEXT.txt', 'line_numbering': '1-based, inclusive',
        'page_count': original_pages, 'pages': page_index,
        'layout_file': 'PAPER_LAYOUT.txt' if layout_pages is not None else None,
        'layout_pages': layout_index, 'heading_candidates': heading_candidates,
        'chunks': chunks, 'line_count': len(lines),
        'files': {name: {'sha256': digest(data), 'bytes': len(data)} for name, data in output.items()},
    }
    index_path = reference_folder / 'PAPER_INDEX.json'
    if index_path.exists():
        previous = read_json(index_path)
        if previous.get('generated_by') != GENERATOR:
            raise ValueError(f'Refusing to overwrite an existing unmanaged index: {index_path}')
        for name, record in previous['files'].items():
            if digest((reference_folder / name).read_bytes()) != record['sha256']:
                raise ValueError(f'Generated file edited since last build: {reference_folder / name}')
    elif any((reference_folder / name).exists() for name in output):
        raise ValueError(f'Refusing to overwrite unmanaged paper text in {reference_folder}')
    reference_folder.mkdir(parents=True, exist_ok=True)
    for name, data in output.items():
        write_atomic(reference_folder / name, data)
    write_atomic(index_path, (json.dumps(metadata, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    return {'project': key, 'folder': str(reference_folder.relative_to(ROOT)), 'coverage': spec['coverage'],
            'pages': original_pages, 'lines': len(lines), 'source_sha256': source_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', action='append', help='Project key; repeat to select several')
    args = parser.parse_args()
    catalogue = read_json(ROOT / '_replication/projects.json')
    sources = read_json(ROOT / '_replication/paper_sources.json')['projects']
    selected = args.project or list(sources)
    rows = []
    for key in selected:
        spec = sources[key]
        folder = ROOT / (catalogue[key]['source'] if key in catalogue else spec['folder'])
        rows.append(build(key, spec, folder))
        print(f'{key}: {rows[-1]["coverage"]}, {rows[-1]["lines"]} lines')
    if not args.project:
        index = ['AGENT-READABLE PAPER TEXT INDEX', 'Generated: ' + str(date.today()), '',
                 'Each listed folder has PAPER_TEXT.txt and PAPER_INDEX.json.',
                 'PDF sources also have PAPER_LAYOUT.txt for table/column inspection.',
                 'Top-level projects keep these files in .agent/; nested projects retain their existing locations.',
                 'Extract source paths and index path_relative_to_project are relative to the project root.',
                 'Indexes record source hashes, relative paths, pages, line chunks and heading candidates.',
                 'Coverage describes the source representation, not scientific replication success.',
                 'Use STATUS.txt for replication evidence. Figures/equations still require the original.', '',
                 'Search: rg -n "search phrase" --glob PAPER_TEXT.txt --glob "!.replication-local/**" --hidden --no-ignore .',
                 'Rebuild: python _replication/prepare_paper_texts.py [--project NAME]', '',
                 'FOLDER | COVERAGE | PDF PAGES | TEXT LINES']
        index += [f'{r["folder"]} | {r["coverage"]} | {r["pages"] if r["pages"] is not None else "-"} | {r["lines"]}' for r in rows]
        index += ['', 'Coverage counts: ' + json.dumps(dict(Counter(r['coverage'] for r in rows)), sort_keys=True),
                  '', 'Known gaps:',
                  '- stable-alpha: only the official abstract is available.',
                  '- Kronos: linked paper guide; full paper text is not duplicated here.',
                  '- corporate-graph-anomalies, multihorizon-forecast: independent studies without one target paper.',
                  '- monthly-coforecast: full NGBoost paper available for a scoped component benchmark; the CRSP study remains independent.',
                  '- alphaprobe: archived HTML has an empty final appendix table; see PAPER_INDEX.json.']
        write_atomic(ROOT / 'PAPER_TEXT_INDEX.txt', ('\n'.join(index) + '\n').encode('utf-8'))
        record = {'generated_on': str(date.today()), 'folders': rows}
        write_atomic(ROOT / '.replication-local/paper-text-build.json',
                     (json.dumps(record, indent=2) + '\n').encode('utf-8'))
    print('Prepared', len(rows), 'paper references.')


if __name__ == '__main__':
    main()
