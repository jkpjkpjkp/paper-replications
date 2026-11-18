"""Inspect live WRDS intraday and S&P 500 data access without exposing credentials."""
import json
import sys
from pathlib import Path

sys.path.insert(0, '/home/jkp/data/ibes')
from bridge import Bridge
from download_ibes import remote, save_json

ROOT = Path(__file__).resolve().parent
TOKEN = Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')

with Bridge(TOKEN) as b:
    b.execute('WRDS_USERNAME=' + repr(b.username) + '\n' + Path('/home/jkp/data/ibes/stream_worker.py').read_text(), timeout=180)
    try:
        schemas = remote(b, 'query', """
            SELECT nspname AS schema, has_schema_privilege(oid,'USAGE') AS accessible
            FROM pg_namespace WHERE nspname ~* '(taq|intraday|tick|minute|algoseek|nyse|polygon|factset|iqfeed)'
            ORDER BY 1
        """)
        save_json(ROOT / 'wrds-intraday-schemas.json', schemas)
        print(json.dumps(schemas), flush=True)
        inventory = remote(b, 'query', """
            SELECT n.nspname AS schema,c.relname AS table,c.relkind AS kind,
                obj_description(c.oid,'pg_class') AS description,
                c.reltuples::bigint AS estimated_rows,
                has_schema_privilege(n.oid,'USAGE') AS schema_access,
                has_table_privilege(c.oid,'SELECT') AS table_select
            FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE c.relkind IN ('r','v','m','p','f') AND
              ((n.nspname ~* '(taq|intraday|minute|algoseek|nyse)' AND
                (c.relname !~ '[0-9]{8}' OR c.relname ~ '20250102|20190102|20250731'))
               OR (n.nspname IN ('crsp_a_indexes','crsp_a_stock') AND c.relname IN
                    ('dsp500list_v2','dsp500list','stocknames_v2','stocknames','dsenames','dsf_v2','dsedist')))
            ORDER BY 1,2
        """)
        save_json(ROOT / 'wrds-source-inventory.json', inventory)
        print(json.dumps(inventory), flush=True)
        checked = []
        for schema, table in [('crsp_a_indexes','dsp500list_v2'), ('crsp_a_stock','stocknames_v2')]:
            try:
                columns = remote(b, 'table_schema', schema, table)
                sample = remote(b, 'query', f'SELECT * FROM {schema}.{table} LIMIT 3')
                item = {'schema':schema, 'table':table, 'columns':columns, 'sample':sample}
            except Exception as e:
                item = {'schema':schema, 'table':table, 'error':str(e).replace(b.token,'[REDACTED]')}
            checked.append(item)
        save_json(ROOT / 'wrds-source-check.json', checked)
        print(json.dumps(checked, default=str), flush=True)
    finally:
        remote(b, 'cleanup')
