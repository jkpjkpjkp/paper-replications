#!/usr/bin/env python3
"""Fetch a small, pinned subset of official TFB source; no environments altered."""
from pathlib import Path
import hashlib
import json
import os
import urllib.request

ROOT=Path(__file__).resolve().parent
COMMIT='889459188ec89c9cb3c7e5ec40f30b8c0c5c2529'
FILES=['models/'+name+'.py' for name in ['DLinear','PatchTST','iTransformer','TimeMixer']]+[
    'layers/Autoformer_EncDec.py','layers/Embed.py','layers/SelfAttention_Family.py',
    'layers/Transformer_EncDec.py','layers/StandardNorm.py','utils/masking.py']
base='ts_benchmark/baselines/time_series_library'
records=[]
for folder in ['ts_benchmark','ts_benchmark/baselines',base,base+'/models',base+'/layers',base+'/utils']:
    path=ROOT/'official-tfb'/folder;path.mkdir(parents=True,exist_ok=True)
    (path/'__init__.py').touch(exist_ok=True)
for file in FILES+['LICENSE']:
    rel='LICENSE' if file=='LICENSE' else base+'/'+file
    url=f'https://raw.githubusercontent.com/decisionintelligence/TFB/{COMMIT}/{rel}'
    destination=ROOT/'official-tfb'/rel
    original=ROOT/'sources/tfb-pinned-originals'/rel
    original.parent.mkdir(parents=True,exist_ok=True)
    if original.exists():blob=original.read_bytes()
    else:
        with urllib.request.urlopen(url,timeout=30) as response:blob=response.read(1024*1024+1)
        if len(blob)>1024*1024:raise RuntimeError('Unexpected source file size exceeds1MiB')
        original.write_bytes(blob)
    text=blob.decode()
    adaptation=None
    if file=='layers/SelfAttention_Family.py':
        exact='from reformer_pytorch import LSHSelfAttention\n'
        if text.count(exact)!=1:raise RuntimeError('Optional import source changed')
        text=text.replace(exact,'# Unused Reformer-only optional import omitted for selected FullAttention models.\n')
        adaptation='Removed only the unused Reformer import; selected FullAttention/AttentionLayer and four model numerical code unchanged. Reformer model is not executed.'
    destination.write_text(text)
    records.append({'url':url,'commit':COMMIT,'original_path':str(original),'original_sha256':hashlib.sha256(blob).hexdigest(),
                    'runtime_path':str(destination),'runtime_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'adaptation':adaptation})
manifest={'status':'source_ready','repo':'https://github.com/decisionintelligence/TFB','commit':COMMIT,'license':'MIT; retained LICENSE',
          'namespace_init_adaptation':'Empty namespace initializers prevent importing unrelated benchmark/model plugins. No selected model code changed.',
          'read_only_einops':'einops0.8.1 installed in the active runtime environment; ordinary package import.',
          'files':records}
tmp=ROOT/'sources/tfb-model-source-manifest.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');os.replace(tmp,ROOT/'sources/tfb-model-source-manifest.json')
print(json.dumps({'status':'source_ready','source_files':len(records),'total_original_bytes':sum(Path(x['original_path']).stat().st_size for x in records)}),flush=True)
