#!/usr/bin/env python3
"""Guarded, synchronous full Compustat customer-history export; no remote model work."""
from pathlib import Path
import sys,json,gzip,hashlib,datetime
from urllib.parse import quote
sys.path.insert(0,'/home/jkp/data/ibes')
from bridge import Bridge
import ast
tree=ast.parse((Path(__file__).with_name("inspect_wrds.py")).read_text())
REMOTE=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="REMOTE" for t in n.targets)))
ROOT=Path(__file__).resolve().parent
# Do not import inspect_wrds because it has an entrypoint; read only its REMOTE constant via AST.
RAW=ROOT/'raw';RAW.mkdir(exist_ok=True)
with Bridge(Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')) as b:
 b.execute('WRDS_USERNAME='+repr(b.username)+'\n'+REMOTE,timeout=120)
 def query(sql):return json.loads(b.execute('print(json.dumps(q('+repr(sql)+'),default=str))',timeout=90).strip())
 checks=[]
 for name in ['factset_revere_supply_chain.wrds_relationship','factset_revere_supply_chain.supply_chain','compseg.wrds_seg_customer','compseg.names_seg']:
  try:entry={'table':name,'accessible':True,'sample':query('SELECT * FROM '+name+' LIMIT 3')}
  except Exception as e:entry={'table':name,'accessible':False,'error':str(e).replace(b.token,'[REDACTED]')}
  checks.append(entry);print(name,entry['accessible'],flush=True)
 (ROOT/'sources/verified_access.json').write_text(json.dumps(checks,indent=2)+'\n')
 cat=query("SELECT n.nspname AS schema,c.relname AS table,has_schema_privilege(n.oid,'USAGE') AND has_table_privilege(c.oid,'SELECT') AS access FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname ~ '^crsp' AND c.relname IN ('stocknames','dsenames','stksecurityinfohist','stksecurityinfo','stocknames_v2') ORDER BY1,2".replace('BY1','BY 1'))
 (ROOT/'sources/crsp_names_catalog.json').write_text(json.dumps(cat,indent=2)+'\n')
 stats=query("SELECT count(*) AS rows,min(srcdate) AS first_date,max(srcdate) AS last_date,count(DISTINCT gvkey) AS suppliers,count(DISTINCT cnms) AS customer_names FROM compseg.wrds_seg_customer")
 (ROOT/'sources/customer_source_counts.json').write_text(json.dumps(stats,indent=2)+'\n');print('CUSTOMERS',json.dumps(stats),flush=True)
 (ROOT/'sources/customer_type_counts.json').write_text(json.dumps(query('SELECT ctype,count(*) AS rows,count(cnms) AS named_rows FROM compseg.wrds_seg_customer GROUP BY ctype ORDER BY count(*) DESC'),indent=2)+'\n')
 b.execute("import os,gzip,hashlib,uuid,shutil\nfrom pathlib import Path\n_export_root=Path.cwd()/('wrds_corpgraph_'+uuid.uuid4().hex)\n_export_root.mkdir(mode=0o700)")
 exports=[('customer_history','SELECT * FROM compseg.wrds_seg_customer'),('segment_names','SELECT * FROM compseg.names_seg')]
 # Export one canonical historical-name table if entitled, preserving full historical intervals.
 for preferred in [('crsp_a_stock','stocknames'),('crsp','stocknames'),('crsp_a_stock','dsenames'),('crsp','dsenames')]:
  if any(x['schema']==preferred[0] and x['table']==preferred[1] and x['access'] for x in cat):
   exports.append(('crsp_stocknames','SELECT * FROM '+'.'.join(preferred)));break
 manifest=[]
 try:
  for label,sql in exports:
   dest=RAW/(label+'.csv.gz')
   if dest.exists() and (RAW/(label+'.receipt.json')).exists():
    receipt=json.loads((RAW/(label+'.receipt.json')).read_text())
    with dest.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
    manifest.append(receipt);print('REUSED',label,flush=True);continue
   code="""_path=_export_root/%r
_conn=_db.engine.raw_connection()
try:
 with _conn.cursor() as _cur:
  _cur.execute('SET TRANSACTION READ ONLY')
  _cur.execute("SET statement_timeout='120s'")
  with gzip.open(_path,'wt',newline='') as _f:
   _cur.copy_expert(%r,_f)
finally:
 _conn.rollback();_conn.close()
with _path.open('rb') as _f:_hash=hashlib.file_digest(_f,'sha256').hexdigest()
print(json.dumps({'path':_export_root.name+'/'+_path.name,'bytes':_path.stat().st_size,'sha256':_hash}))
"""%(label+'.csv.gz','COPY ('+sql+') TO STDOUT WITH (FORMAT CSV,HEADER TRUE)')
   info=json.loads(b.execute(code,timeout=150).strip());print('EXPORT',label,info['bytes'],flush=True)
   part=dest.with_suffix('.gz.part');digest=hashlib.sha256();size=0
   with b.request('GET',b.server_path+'/files/'+quote(info['path'],safe='/'),stream=True,timeout=(30,120)) as response,part.open('wb') as f:
    for chunk in response.iter_content(1024*1024):
     if chunk:f.write(chunk);digest.update(chunk);size+=len(chunk)
   assert size==info['bytes'] and digest.hexdigest()==info['sha256'];part.replace(dest)
   record={**info,'file':str(dest),'sql':sql,'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source':'WRDS authenticated entitled read-only COPY; full table no sample limit'}
   (RAW/(label+'.receipt.json')).write_text(json.dumps(record,indent=2)+'\n');manifest.append(record)
   b.execute('_path.unlink()');(ROOT/'sources/download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 finally:b.execute('shutil.rmtree(_export_root);_db.close()')
print('DOWNLOAD_COMPLETE',len(manifest),flush=True)
