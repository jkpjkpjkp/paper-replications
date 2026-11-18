"""Download historical S&P membership and dated TAQ mappings using existing WRDS access."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, '/home/jkp/data/ibes')
from bridge import Bridge
from download_ibes import remote, download, save_json, verified_file

ROOT = Path(__file__).resolve().parent
META = ROOT / 'metadata'
META.mkdir(exist_ok=True)
WORKER = '''
import datetime, hashlib, json, shutil, uuid
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import wrds
db=wrds.Connection(wrds_username=WRDS_USERNAME)
db.connection.close()
root=Path.cwd()/('song26l_metadata_'+uuid.uuid4().hex)
root.mkdir(mode=0o700)
def export_query(statement,name):
    conn=db.engine.raw_connection()
    try:
        with conn.cursor() as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute(statement)
            types={1082:pa.date32(),20:pa.int64(),21:pa.int16(),23:pa.int32(),701:pa.float64(),700:pa.float32(),16:pa.bool_()}
            schema=pa.schema([pa.field(c.name,types.get(c.type_code,pa.string())) for c in cur.description])
            path=root/(name+'.parquet')
            rows=0
            with pq.ParquetWriter(path,schema,compression='zstd') as writer:
                while True:
                    data=cur.fetchmany(50000)
                    if not data: break
                    columns=[pa.array([r[i] for r in data],type=f.type) for i,f in enumerate(schema)]
                    writer.write_table(pa.Table.from_arrays(columns,schema=schema))
                    rows+=len(data)
        with path.open('rb') as f: sha=hashlib.file_digest(f,'sha256').hexdigest()
        return {'path':root.name+'/'+path.name,'rows':rows,'bytes':path.stat().st_size,'sha256':sha,'source_sql':statement,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    finally:
        conn.rollback();conn.close()
def cleanup():
    db.close();shutil.rmtree(root);return {'cleaned':True}
'''

queries = {
    'sp500_membership': "SELECT * FROM crsp_a_indexes.dsp500list_v2 WHERE mbrstartdt<=DATE '2025-08-01' AND mbrenddt>=DATE '2019-01-01' ORDER BY permno,mbrstartdt",
    'sp500_security_names': "SELECT * FROM crsp_a_stock.stocknames_v2 WHERE nameenddt>=DATE '2019-01-01' AND namedt<=DATE '2025-08-01' AND permno IN (SELECT permno FROM crsp_a_indexes.dsp500list_v2 WHERE mbrstartdt<=DATE '2025-08-01' AND mbrenddt>=DATE '2019-01-01') ORDER BY permno,namedt",
    'taq_trade_partitions': "SELECT n.nspname AS schema,c.relname AS table,c.reltuples::bigint AS estimated_rows FROM pg_class c JOIN pg_namespace n ON c.relnamespace=n.oid WHERE n.nspname ~ '^taqm_20(19|20|21|22|23|24|25)$' AND c.relname ~ '^ctm_[0-9]{8}$' AND substring(c.relname from 5)>= '20190101' AND substring(c.relname from 5)<='20250801' ORDER BY 2",
    'taq_master_2025': "SELECT date,symbol_15,cusip,sec_desc,sec_type,listed_exchange,tape FROM taqm_2025.mastm_2025 WHERE date<=DATE '2025-08-01' ORDER BY date,symbol_15",
}
for year in range(2019,2026):
    queries[f'taq_master_{year}'] = f"SELECT date,symbol_15,cusip,sec_desc,sec_type,listed_exchange,tape FROM taqm_{year}.mastm_{year} WHERE date<=DATE '2025-08-01' ORDER BY date,symbol_15"
    queries[f'taq_crsp_link_{year}'] = f"SELECT * FROM wrdsapps_link_crsp_taqm.taqmclink WHERE date>=DATE '{year}-01-01' AND date<=LEAST(DATE '{year}-12-31',DATE '2025-08-01') AND permno IN (SELECT permno FROM crsp_a_indexes.dsp500list_v2 WHERE mbrstartdt<=DATE '2025-08-01' AND mbrenddt>=DATE '2019-01-01') ORDER BY date,permno,sym_root,sym_suffix"

with Bridge(Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')) as b:
    b.execute('WRDS_USERNAME='+repr(b.username)+'\n'+WORKER,timeout=180)
    try:
        for name, statement in queries.items():
            target=META/(name+'.parquet')
            receipt=META/(name+'.receipt.json')
            if receipt.exists() and verified_file(target,json.loads(receipt.read_text())):
                print('Verified existing '+name,flush=True);continue
            info=remote(b,'export_query',statement,name)
            download(b,info,target)
            save_json(receipt,info)
            b.execute('(root/'+repr(target.name)+').unlink()')
            print(f"Downloaded {name}: {info['rows']:,} rows, {info['bytes']:,} bytes",flush=True)
    finally:
        remote(b,'cleanup')
