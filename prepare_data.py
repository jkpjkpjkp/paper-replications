#!/usr/bin/env python3
"""Guard-only selective official ZIP member download, CRC/size/hash verification."""
import hashlib,json,os,struct,urllib.request,zlib
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parent
m=json.loads((R/'sources/gadbench-zip-members.json').read_text());f=next(x for x in m['members'] if x['filename']=='datasets/tfinance')
URL=m['url'];TOTAL=m['total_archive_bytes'];OUT=R/'data';OUT.mkdir(exist_ok=True);dest=OUT/'tfinance';tmp=OUT/'tfinance.partial'
if dest.exists():
 print('Existing verified member preserved:',dest);raise SystemExit(0)
def ranged(a,b):
 r=urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}'}),timeout=60)
 if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{TOTAL}':r.close();raise RuntimeError('Exact Range not honored')
 return r
start=f['local_header_offset']
with ranged(start,start+29) as h:head=h.read()
fields=struct.unpack('<IHHHHHIIIHH',head)
assert fields[0]==0x04034b50 and fields[3]==8
start+=30+fields[-2]+fields[-1];end=start+f['compressed_bytes']-1
crc=0;size=0;compressed=0;sha=hashlib.sha256();decomp=zlib.decompressobj(-15)
try:
 with ranged(start,end) as source,tmp.open('wb') as target:
  while True:
   chunk=source.read(1024*1024)
   if not chunk:break
   compressed+=len(chunk);data=decomp.decompress(chunk)
   size+=len(data)
   if size>f['uncompressed_bytes']:raise RuntimeError('Unexpected decompressed size')
   target.write(data);crc=zlib.crc32(data,crc);sha.update(data)
  last=decomp.flush();target.write(last);size+=len(last);crc=zlib.crc32(last,crc);sha.update(last)
 assert decomp.eof and size==f['uncompressed_bytes'] and compressed==f['compressed_bytes'] and crc==f['crc32']
 tmp.replace(dest)
except BaseException:
 tmp.unlink(missing_ok=True);raise
info={'paper_id':429,'url':URL,'retrieved_at':datetime.now(timezone.utc).isoformat(),'archive_total_bytes':TOTAL,'zip_member':f,'range_start':start,'range_end':end,'compressed_bytes_downloaded':compressed,'uncompressed_bytes':size,'crc32_verified':crc,'sha256':sha.hexdigest(),'output':str(dest),'origin':'Exact GADBench processed bundle linked by FraudGNAM official README; no raw-producer substitution','other_datasets_downloaded':False}
(R/'sources/data-manifest.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2),flush=True)
