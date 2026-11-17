"""Download the two public FRED series used by the frozen reconstruction."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
import requests

out=Path(__file__).resolve().parent/'data'
out.mkdir(exist_ok=True)
receipts=[]
for series in ['T10Y3M','DTWEXBGS']:
    url=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd=2012-01-01&coed=2025-03-31'
    response=requests.get(url,timeout=60)
    response.raise_for_status()
    if not response.text.startswith('observation_date,'):raise ValueError('Unexpected FRED response')
    (out/f'{series}.csv').write_bytes(response.content)
    receipts.append({'series':series,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                     'sha256':hashlib.sha256(response.content).hexdigest(),'bytes':len(response.content)})
(out/'fred_download_receipt.json').write_text(json.dumps(receipts,indent=2))
print(json.dumps(receipts,indent=2))
