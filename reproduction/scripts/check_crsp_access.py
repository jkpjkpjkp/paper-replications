"""Fresh read-only CRSP queries using the existing WRDS credential."""
from pathlib import Path
import datetime, json, sys
BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE / 'ibes'))
from bridge import Bridge
ROOT = Path(__file__).resolve().parents[1]
(ROOT / 'evidence').mkdir(exist_ok=True)
result = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'authentication_succeeded': False}
try:
    with Bridge(BASE / 'wrds-crsp-download/wrds-jupyter.token') as b:
        result['authentication_succeeded'] = True
        print('Fresh WRDS authentication succeeded; querying CRSP.', flush=True)
        code = '''
import json, wrds
from psycopg2 import sql
db = wrds.Connection(wrds_username=WRDS_USERNAME)
results = []
try:
    for schema, table in [('crsp_a_stock','dsf_v2'), ('crsp_a_stock','msf_v2'), ('crsp','dsf'), ('crsp','msf')]:
        conn = db.engine.raw_connection()
        entry = {'table': schema + '.' + table}
        try:
            with conn.cursor() as cur:
                cur.execute('SET TRANSACTION READ ONLY')
                cur.execute("SET LOCAL statement_timeout = '30s'")
                cur.execute(sql.SQL('SELECT * FROM {}.{} LIMIT 1').format(sql.Identifier(schema),sql.Identifier(table)))
                rows = cur.fetchall()
                entry.update(accessible=True, rows_returned=len(rows), column_count=len(cur.description))
        except Exception as exc:
            entry.update(accessible=False, error=str(exc))
        finally:
            conn.rollback()
            conn.close()
        results.append(entry)
finally:
    db.close()
print('CRSP_RESULT=' + json.dumps(results))
'''
        output = b.execute('WRDS_USERNAME=' + repr(b.username) + '\n' + code, timeout=180)
        result['queries'] = json.loads(next(line.split('=',1)[1] for line in output.splitlines() if line.startswith('CRSP_RESULT=')))
        result['status'] = 'completed'
except Exception as exc:
    result['status'] = 'failed'
    result['error_type'] = type(exc).__name__
path = ROOT / 'evidence/crsp-access-current.json'
path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
