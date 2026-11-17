#!/usr/bin/env python3
"""Read-only entitlement/schema discovery through the existing WRDS connection."""
from pathlib import Path
import sys,json,datetime
sys.path.insert(0,'/home/jkp/data/ibes')
from bridge import Bridge
ROOT=Path(__file__).resolve().parent
REMOTE='''
import json,wrds
_db=wrds.Connection(wrds_username=WRDS_USERNAME)
_db.connection.close()
def q(sql):
 conn=_db.engine.raw_connection()
 try:
  with conn.cursor() as cur:
   cur.execute('SET TRANSACTION READ ONLY');cur.execute("SET statement_timeout='45s'")
   cur.execute(sql)
   return [dict(zip([c.name for c in cur.description],r)) for r in cur.fetchall()]
 finally:conn.rollback();conn.close()
'''
SQL="""SELECT n.nspname AS schema,c.relname AS table,c.relkind AS kind,
obj_description(c.oid,'pg_class') AS description,c.reltuples::bigint AS estimated_rows,
has_schema_privilege(n.oid,'USAGE') AND has_table_privilege(c.oid,'SELECT') AS advertised_access
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE c.relkind IN ('r','v','m','p','f') AND (
(n.nspname ~ '^comp' AND (c.relname ~ '(seg|customer)' OR n.nspname ~ 'seg')) OR
(n.nspname ~ '^factset.*(revere|common|sym)' AND c.relname ~ '(rel|sup|cus|sym|entity|isin|link|security)') OR
(n.nspname ~ '^wrdsapps.*factset')) ORDER BY 1,2"""
try:
 with Bridge(next(p for p in [Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token'),Path('/home/jkp/wrds-crsp-download/wrds-jupyter.token')] if p.is_file())) as b:
  b.execute('WRDS_USERNAME='+repr(b.username)+'\n'+REMOTE,timeout=120)
  rows=json.loads(b.execute('print(json.dumps(q('+repr(SQL)+'),default=str))',timeout=60).strip())
  (ROOT/'sources/wrds_relation_catalog.json').write_text(json.dumps(rows,indent=2)+'\n')
  # Verify actual view access for focused supply-chain/customer candidates, not just ACLs.
  selected=[x for x in rows if ('supply' in x['schema'] or ('comp' in x['schema'] and any(k in x['table'] for k in ['customer','cusseg']))) and x['advertised_access']]
  if not selected:selected=[x for x in rows if x['schema']=='comp_segments' and x['advertised_access']]
  checks=[]
  for item in selected[:30]:
   name=item['schema']+'.'+item['table'];entry={'source':name}
   cols="SELECT a.attname AS name,format_type(a.atttypid,a.atttypmod) AS type,col_description(c.oid,a.attnum) AS description FROM pg_attribute a JOIN pg_class c ON a.attrelid=c.oid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname="+"'"+item['schema']+"' AND c.relname='"+item['table']+"' AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum"
   try:
    entry['columns']=json.loads(b.execute('print(json.dumps(q('+repr(cols)+'),default=str))',timeout=60).strip())
    entry['sample']=json.loads(b.execute('print(json.dumps(q('+repr('SELECT * FROM '+name+' LIMIT 3')+'),default=str))',timeout=60).strip());entry['accessible']=True
   except Exception as e:entry.update(accessible=False,error=str(e).replace(b.token,'[REDACTED]'))
   checks.append(entry)
   print(name,entry['accessible'],len(entry.get('columns',[])),flush=True)
   (ROOT/'sources/wrds_relation_access.json').write_text(json.dumps(checks,indent=2)+'\n')
  b.execute('_db.close()')
 print('Listed',len(rows),'candidate tables; focused checks',len(checks))
except Exception as e:
 (ROOT/'sources/wrds_access_error.json').write_text(json.dumps({'error_type':type(e).__name__,'error':str(e),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n');raise
