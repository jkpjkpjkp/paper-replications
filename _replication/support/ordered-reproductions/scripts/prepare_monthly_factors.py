"""Archive official monthly FF5 and momentum; reject daily/annual rows."""
import csv
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/factors-monthly'
OUT.mkdir(parents=True, exist_ok=True)
if (OUT/'receipt.json').exists():
    raise SystemExit('Existing factor vintage is preserved')
base = 'https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
frames, receipts = [], []
for filename, columns in [
    ('F-F_Research_Data_5_Factors_2x3_CSV.zip', ['Mkt-RF','SMB','HML','RMW','CMA','RF']),
    ('F-F_Momentum_Factor_CSV.zip', ['UMD']),
]:
    url = base+filename
    archive = OUT/filename
    if not archive.exists():
        with urllib.request.urlopen(url, timeout=90) as response:
            archive.write_bytes(response.read())
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        files = [n for n in z.namelist() if n.lower().endswith('.csv')]
        assert len(files) == 1
        content = z.read(files[0]).decode('utf-8-sig')
    rows = []
    for row in csv.reader(io.StringIO(content)):
        if row and re.fullmatch(r'\d{6}', row[0].strip()):
            assert len(row) == len(columns)+1
            values = [float(v) for v in row[1:]]
            assert all(v not in (-99.99, -999) for v in values)
            rows.append([int(row[0])] + [v/100 for v in values])
    frame = pd.DataFrame(rows, columns=['return_month']+columns).set_index('return_month')
    assert not frame.index.duplicated().any()
    assert frame.loc[202201:202512].shape[0] == 48
    frames.append(frame)
    receipts.append(dict(url=url, archive=filename, sha256=hashlib.sha256(raw).hexdigest(),
                         rows=len(frame), first=int(frame.index.min()), last=int(frame.index.max())))
factors = pd.concat(frames, axis=1, join='inner').sort_index()
assert factors.loc[200901:202512].notna().all().all()
factors.to_csv(OUT/'ff5-umd.csv')
(OUT/'receipt.json').write_text(json.dumps(dict(downloaded_utc=datetime.now(timezone.utc).isoformat(),
    sources=receipts, units='decimal monthly returns', momentum='Ken French Mom is UMD',
    limitation='Current downloaded historical vintage, not a historical publication vintage.'), indent=2))
print(factors.loc[202201:202512].agg(['count','min','max']).to_string())
