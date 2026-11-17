#!/usr/bin/env python3
"""Audit full disclosed relationships and strict dated CRSP identity matches; no returns read."""
import json,hashlib,gzip,csv,re,datetime
from pathlib import Path
import duckdb
ROOT=Path(__file__).resolve().parent;RAW=ROOT/'raw';OUT=ROOT/'data';OUT.mkdir(exist_ok=True)
db=duckdb.connect();db.execute("SET threads=2;SET memory_limit='3GB'")
for label in ['customer_history','segment_names','crsp_stocknames']:
 path=RAW/(label+'.csv.gz');receipt=json.loads((RAW/(label+'.receipt.json')).read_text())
 with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==receipt['sha256']
 # Every column starts as varchar, retaining leading-zeroGVKEYs; cast explicitly below.
 db.read_csv(str(path),header=True,all_varchar=True).create_view(label)
db.read_parquet('/home/jkp/data/compustat/all-parquet/comp/co_filedate/part-*.parquet').create_view('filings_raw')
db.read_parquet('/home/jkp/data/compustat/ccm/primary_links.parquet').create_view('links')
print('LINK_COLUMNS',db.sql('DESCRIBE links').fetchall(),flush=True)
summary={'raw_customer_rows':db.sql('SELECT count(*) FROM customer_history').fetchall()[0][0],'filing_types':db.sql('SELECT srctype,count(*) FROM filings_raw GROUP BY srctype ORDER BY count(*) DESC').fetchall()}
print('FILINGS',summary['filing_types'],flush=True)
def norm(s):
 if s is None:return None
 s=re.sub(r'[^A-Z0-9 ]',' ',s.upper().replace('&',' AND '))
 words=[x for x in s.split() if x not in {'INC','INCORPORATED','CORP','CORPORATION','CO','COMPANY','PLC','LTD','LIMITED','LLC','LP'}]
 if words and words[0]=='THE':words=words[1:]
 return ''.join(words)
db.create_function('normname',norm,[str],str,null_handling='special')
db.execute('''CREATE TABLE customers AS SELECT row_number() OVER() AS row_id,gvkey,cid,cnms,ctype,sid,stype,
TRY_CAST(salecs AS DOUBLE) AS salecs,TRY_CAST(srcdate AS DATE) AS period_end,normname(cnms) AS name_key
FROM customer_history''')
db.execute('''CREATE TABLE names AS SELECT DISTINCT TRY_CAST(permno AS BIGINT) AS permno,
TRY_CAST(permco AS BIGINT) AS permco,TRY_CAST(namedt AS DATE) AS namedt,TRY_CAST(nameenddt AS DATE) AS nameenddt,
comnam,ticker,normname(comnam) AS name_key FROM crsp_stocknames WHERE TRY_CAST(shrcd AS INTEGER) IN(10,11)''')
# Availability uses annual10-K filing dates. Period/srcdate is never treated as publication.
db.execute('''CREATE TABLE filings AS SELECT gvkey,datadate AS period_end,min(filedate) AS filing_date
FROM filings_raw WHERE (upper(srctype) LIKE '%10-K%' OR upper(srctype) LIKE '%10K%')
AND upper(srctype) NOT LIKE '%/A%' AND filedate>datadate AND filedate<=datadate+INTERVAL'400 days'
GROUP BY gvkey,datadate''')
db.execute('''CREATE TABLE prepared AS SELECT c.*,f.filing_date,l.permno AS supplier_permno
FROM customers c LEFT JOIN filings f USING(gvkey,period_end)
LEFT JOIN links l ON c.gvkey=l.gvkey AND f.filing_date>=coalesce(l.ccm_linkdt,DATE'1900-01-01')
AND f.filing_date<=coalesce(l.ccm_linkenddt,DATE'9999-12-31')''')
# Exact normalized historical names; accept only a unique company at that actual filing date.
db.execute('''CREATE TABLE candidates AS SELECT DISTINCT p.row_id,n.permno,n.permco,n.comnam,n.ticker
FROM prepared p JOIN names n ON p.name_key=n.name_key AND p.filing_date>=n.namedt
AND p.filing_date<=n.nameenddt WHERE p.ctype='COMPANY' AND length(p.name_key)>=4''')
db.execute('''CREATE TABLE unique_match AS SELECT row_id,min(permno) AS customer_permno,min(permco) AS customer_permco,
min(comnam) AS matched_customer_name,min(ticker) AS customer_ticker,count(DISTINCT permno) AS share_classes
FROM candidates GROUP BY row_id HAVING count(DISTINCT permco)=1''')
db.execute('''CREATE TABLE matched AS SELECT p.*,m.customer_permno,m.customer_permco,m.matched_customer_name,
m.customer_ticker,m.share_classes FROM prepared p LEFT JOIN unique_match m USING(row_id)''')
summary.update({'company_rows':db.sql("SELECT count(*) FROM customers WHERE ctype='COMPANY'").fetchall()[0][0],
'rows_with_annual_filing':db.sql('SELECT count(*) FROM prepared WHERE filing_date IS NOT NULL').fetchall()[0][0],
'company_rows_with_annual_filing':db.sql("SELECT count(*) FROM prepared WHERE ctype='COMPANY' AND filing_date IS NOT NULL").fetchall()[0][0],
'uniquely_matched_customer_rows':db.sql('SELECT count(*) FROM unique_match').fetchall()[0][0],
'fully_linked_rows':db.sql("SELECT count(*) FROM matched WHERE supplier_permno IS NOT NULL AND customer_permno IS NOT NULL AND supplier_permno<>customer_permno").fetchall()[0][0],
'unmatched_frequent_names':db.sql("SELECT cnms,count(*) AS rows FROM matched WHERE ctype='COMPANY' AND filing_date IS NOT NULL AND customer_permno IS NULL GROUP BY cnms ORDER BY rows DESC LIMIT30".replace('LIMIT30','LIMIT 30')).fetchall()})
# A customer can recur across disclosed segments: preserve raw rows and max reported sales per pair/date.
db.execute('''CREATE TABLE edges AS SELECT supplier_permno,customer_permno,gvkey,period_end,filing_date,
max(salecs) AS max_reported_customer_sales,count(*) AS source_rows,min(cnms) AS disclosed_customer_name,
min(matched_customer_name) AS matched_customer_name,min(customer_ticker) AS customer_ticker
FROM matched WHERE supplier_permno IS NOT NULL AND customer_permno IS NOT NULL AND supplier_permno<>customer_permno
GROUP BY supplier_permno,customer_permno,gvkey,period_end,filing_date''')
for label in ['matched','edges']:
 path=OUT/(label+'.parquet');db.execute('COPY '+label+' TO ?(FORMAT PARQUET,COMPRESSION ZSTD)',[str(path)])
summary['edges']=db.sql('SELECT count(*),count(DISTINCT supplier_permno),count(DISTINCT customer_permno),min(filing_date),max(filing_date) FROM edges').fetchall()
summary['edge_year_counts']=db.sql('SELECT year(filing_date) AS year,count(*) AS edges,count(DISTINCT supplier_permno) AS suppliers,count(DISTINCT customer_permno) AS customers FROM edges GROUP BY1 ORDER BY1'.replace('BY1','BY 1')).fetchall()
summary['date_contract']='filedate from10-K matching fiscalperiod; postpone tradable availability through at least next trading session. Historical customer records and names are retrospective current release, not certified vintage snapshots.'
summary['identity_contract']='Strict unique normalized historicalCRSP company name at filingdate; government/market/geographic records excluded; no fuzzy matches. Multiple shareclasses within samePERMCO use smallestPERMNO fixed without return selection.'
summary['restatement_limit']='Historic Compustat disclosures can contain later revisions. Filed-date reconstruction is not a certified point-in-time vendor archive.'
summary['no_price_outcomes_read']=True
(ROOT/'results/relation_audit.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
print(json.dumps(summary,indent=2,default=str),flush=True)
