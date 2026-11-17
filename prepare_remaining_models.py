#!/usr/bin/env python3
"""Fetch only pinned compact GTR/RAFT/DUET model source and licenses."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request

ROOT=Path(__file__).resolve().parent
RELEASES={
 'GTR':('macovaseas/GTR','d94161906151bdc0c94f4d21c2b0ca356aeb3135',['models/GTR.py','run.py','data_provider/data_loader.py','LICENSE']),
 'RAFT':('archon159/RAFT','bfe2320f5d719d1e2a024330de3410b370bbb2a9',['models/RAFT.py','layers/Retrieval.py','LICENSE']),
 'DUET':('decisionintelligence/DUET','dcc6e6780a9138731b64b9b5398a94a1d97033f0',[
  'ts_benchmark/baselines/duet/models/duet_model.py','ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py',
  'ts_benchmark/baselines/duet/layers/linear_pattern_extractor.py','ts_benchmark/baselines/duet/layers/distributional_router_encoder.py',
  'ts_benchmark/baselines/duet/layers/RevIN.py','ts_benchmark/baselines/duet/layers/Autoformer_EncDec.py',
  'ts_benchmark/baselines/duet/utils/masked_attention.py','ts_benchmark/baselines/duet/duet.py','LICENSE'])}
records=[]
for name,(repo,commit,files) in RELEASES.items():
 for relative in files:
  url=f'https://raw.githubusercontent.com/{repo}/{commit}/{relative}'
  destination=ROOT/'official-extra'/name/relative;destination.parent.mkdir(parents=True,exist_ok=True)
  if destination.exists():blob=destination.read_bytes()
  else:
   with urllib.request.urlopen(url,timeout=30) as response:blob=response.read(1024*1024+1)
   if len(blob)>1024*1024:raise RuntimeError('Source file exceeds predeclared1MiB cap')
   destination.write_bytes(blob)
  records.append({'method':name,'repository':repo,'commit':commit,'url':url,'path':str(destination),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),'numerical_source_modified':False})
 for file in files:
  p=Path(file).parent
  while str(p)!='.':
   directory=ROOT/'official-extra'/name/p
   if directory.exists():(directory/'__init__.py').touch(exist_ok=True)
   p=p.parent
result={'status':'source_ready','files':records,'namespace_adaptation':'Empty namespace initializers avoid unrelated framework imports. All selected model/retriever numerical sources byte-identical to pinned upstream.','licenses':{'GTR':'Apache2.0','RAFT':'MIT','DUET':'MIT'},'source_download_bytes':sum(x['bytes'] for x in records)}
tmp=ROOT/'sources/remaining-model-source-manifest.tmp';tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,ROOT/'sources/remaining-model-source-manifest.json')
print(json.dumps({'status':'source_ready','files':len(records),'bytes':result['source_download_bytes']}),flush=True)
