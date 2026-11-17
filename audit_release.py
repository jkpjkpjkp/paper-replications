#!/usr/bin/env python3
"""Guarded source-data schema and chronology audit, without fitting or tuning."""
import collections
import importlib.metadata
import json
from pathlib import Path
import time
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'release' / 'data'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def inspect_parquet(path):
    f = pq.ParquetFile(path)
    meta = f.metadata
    result = {'path': str(path.relative_to(DATA)), 'bytes': path.stat().st_size,
              'rows': meta.num_rows, 'row_groups': meta.num_row_groups,
              'schema': str(f.schema_arrow), 'columns': f.schema_arrow.names}
    columns = f.schema_arrow.names
    # Tiny source preview only; not performance/return exploration.
    batch = next(f.iter_batches(batch_size=5, use_threads=False), None)
    result['first_rows'] = batch.to_pylist() if batch is not None else []
    return result


def main():
    started = time.time()
    out = ROOT / 'evidence' / 'release-audit.json'
    assert json.loads((ROOT / 'sources' / 'extraction.json').read_text())['status'] == 'complete'
    slices = [json.loads(line) for line in (DATA / 'splits' / 'slices.jsonl').read_text().splitlines() if line]
    records = []
    # Inspect canonical per-slice aggregate input files and small metadata,
    # rather than opening hundreds of thousands of duplicate per-user shards.
    for directory in ['clean', 'features', 'profiles', 'profiles_riskmap_cat', 'calib', 'splits']:
        parent = DATA / directory
        if parent.exists():
            for path in sorted(parent.rglob('*')):
                if path.is_file() and path.suffix == '.parquet':
                    records.append(inspect_parquet(path))
    for path in sorted((DATA / 'candidates').glob('*_all.parquet')):
        records.append(inspect_parquet(path))
    docs = []
    # Use the checksummed archive index for complete path/size counts.
    dirs = collections.Counter()
    files = []
    with (ROOT / 'sources' / 'archive-members.jsonl').open() as stream:
        for line in stream:
            x = json.loads(line)
            if x['type'] != '0':
                continue
            p = Path(x['name'])
            parts = p.parts
            dirs['/'.join(parts[:2])] += 1
            # Retain full non-candidate path inventory. Per-user candidate
            # shards are indexed completely in archive-members.jsonl already.
            if len(parts) < 4 or parts[1] != 'candidates':
                files.append({'path':x['name'], 'bytes':x['bytes']})
            if p.suffix.lower() in {'.json', '.jsonl', '.yaml', '.yml', '.txt', '.md', '.csv'} and x['bytes'] < 200000:
                path = ROOT / 'release' / x['name']
                docs.append({'path':x['name'], 'text':path.read_text(errors='replace')[:200000]})
    result = {'paper_id':388, 'status':'complete_schema_audit', 'started_at_unix':started,
              'finished_at_unix':time.time(), 'slices':slices,
              'file_counts':dict(dirs), 'nonshard_inventory':files,
              'canonical_parquet':records, 'small_documents':docs,
              'versions':{n:importlib.metadata.version(n) for n in ['numpy','pandas','pyarrow','scipy','PyYAML']},
              'scope':'Schema/source inputs only; no training,performance search or financial evaluation.'}
    temp = out.with_name(out.name + '.tmp')
    temp.write_text(json.dumps(result,indent=2,default=str)+'\n')
    temp.replace(out)
    print(json.dumps({'status':result['status'],'slices':len(slices),'canonical_parquet_files':len(records),
                      'file_counts':dict(dirs),'results_path':str(out)}),flush=True)


if __name__ == '__main__':
    main()
