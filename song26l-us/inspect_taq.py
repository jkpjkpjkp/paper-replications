import json
import sys
from pathlib import Path
sys.path.insert(0, '/home/jkp/data/ibes')
from bridge import Bridge
from download_ibes import remote, save_json
ROOT = Path(__file__).resolve().parent
with Bridge(Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')) as b:
    b.execute('WRDS_USERNAME=' + repr(b.username) + '\n' + Path('/home/jkp/data/ibes/stream_worker.py').read_text(), timeout=180)
    try:
        b.execute("def table_schema(schema,table):\n    return query('SELECT a.attname AS name,format_type(a.atttypid,a.atttypmod) AS type,col_description(c.oid,a.attnum) AS description FROM pg_attribute a JOIN pg_class c ON a.attrelid=c.oid JOIN pg_namespace n ON c.relnamespace=n.oid WHERE n.nspname=%s AND c.relname=%s AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum',(schema,table))")
        checked=[]
        for schema, table in [('taqm_2019','ctm_20190102'),('taqm_2025','ctm_20250731'),
                              ('wrdsapps_link_crsp_taqm','taqmclink'),('taqm_2025','mastm_20250731')]:
            item={'schema':schema,'table':table,'columns':remote(b,'table_schema',schema,table),
                  'sample':remote(b,'query',f'SELECT * FROM {schema}.{table} LIMIT 3')}
            checked.append(item)
            save_json(ROOT/'taq-schema-check.json',checked)
            print(json.dumps(item,default=str),flush=True)
        idx=remote(b,'query',"SELECT schemaname,tablename,indexdef FROM pg_indexes WHERE schemaname='taqm_2025' AND tablename='ctm_20250731'")
        save_json(ROOT/'taq-indexes.json',idx)
        print(json.dumps(idx),flush=True)
    finally:
        remote(b,'cleanup')
