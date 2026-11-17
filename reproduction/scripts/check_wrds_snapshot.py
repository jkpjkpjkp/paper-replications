"""Read-only WRDS catalog and actual SELECT probes; never emit credentials."""
import json
import sys
from pathlib import Path
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'ibes'))
from bridge import Bridge
from download_ibes import remote
ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'evidence').mkdir(exist_ok=True)
result = {}
try:
    with Bridge(BASE / 'wrds-crsp-download/wrds-jupyter.token') as b:
        worker = (BASE / 'ibes/stream_worker.py').read_text()
        b.execute('WRDS_USERNAME=' + repr(b.username) + '\n' + worker, timeout=180)
        try:
            result['schemas'] = remote(b, 'query', "SELECT nspname, has_schema_privilege(oid,'USAGE') AS usage FROM pg_namespace WHERE nspname ~ '(comp|snapshot)' ORDER BY 1")
            result['tables'] = remote(b, 'query', "SELECT n.nspname AS schema, c.relname AS table, has_table_privilege(c.oid,'SELECT') AS can_select, obj_description(c.oid,'pg_class') AS description FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid WHERE c.relkind IN ('r','v','m','p','f') AND (n.nspname ~ '^comp' AND (n.nspname ~ '(snap|pit)' OR c.relname ~ '(snap|pit|short|fundq|funda)')) ORDER BY 1,2")
            result['probes'] = []
            for t in result['tables']:
                if any(x in t['schema'] for x in ('snap', 'pit')):
                    name = '"' + t['schema'] + '"."' + t['table'] + '"'
                    try:
                        remote(b, 'query', 'SELECT * FROM ' + name + ' LIMIT 1')
                        result['probes'].append({'table': name, 'accessible': True})
                    except Exception as e:
                        result['probes'].append({'table': name, 'accessible': False, 'error': str(e).replace(b.token, '[REDACTED]')})
            result['status'] = 'catalog_and_select_probes_completed'
        finally:
            remote(b, 'cleanup')
except Exception as e:
    result['status'] = 'probe_failed'
    result['error_type'] = type(e).__name__
(ROOT / 'evidence/wrds-snapshot-access.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
