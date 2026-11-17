#!/usr/bin/env python3
import io,json,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URL='https://drive.usercontent.google.com/download?id=1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1&export=download&confirm=t'
TOTAL=757193110
class Remote(io.RawIOBase):
 def __init__(self):self.pos=0;self.fetched=0
 def seekable(self):return True
 def seek(self,n,w=0):
  self.pos=n if w==0 else self.pos+n if w==1 else TOTAL+n;return self.pos
 def tell(self):return self.pos
 def read(self,n=-1):
  if n<0:n=TOTAL-self.pos
  n=min(n,TOTAL-self.pos)
  if n==0:return b''
  if n>2*1024**2:raise ValueError('Metadata-only read budget exceeded')
  a=self.pos;b=a+n-1
  with urllib.request.urlopen(urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}'}),timeout=45) as r:
   if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {a}-{b}/{TOTAL}':raise ValueError('Server did not honor exact bounded range')
   data=r.read(n+1)
  if len(data)!=n:raise ValueError('Unexpected range size')
  self.pos+=n;self.fetched+=n;return data
remote=Remote()
with zipfile.ZipFile(remote) as z:
 rows=[{'filename':x.filename,'uncompressed_bytes':x.file_size,'compressed_bytes':x.compress_size,'compression':x.compress_type,'crc32':x.CRC,'local_header_offset':x.header_offset} for x in z.infolist()]
d={'url':URL,'total_archive_bytes':TOTAL,'metadata_downloaded_bytes':remote.fetched,'members':rows}
(ROOT/'sources/gadbench-zip-members.json').write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(d,indent=2))
