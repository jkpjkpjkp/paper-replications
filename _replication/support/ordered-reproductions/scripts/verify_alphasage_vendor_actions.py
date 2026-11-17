"""Read-only source-vendor cross-check of the three largest observed corporate-action discontinuities."""
from pathlib import Path
import json,socket,time
from datetime import datetime,timezone
import numpy as np,pandas as pd,baostock as bs
socket.setdefaulttimeout(20)
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/gap-resolution-20261113';BASE=ROOT.parent/'csi/qlib/cn_data'
cal=pd.DatetimeIndex(pd.to_datetime((BASE/'calendars/day.txt').read_text().splitlines()))
examples=json.loads((ROOT/'results/data-audit-20261113/alphasage-adjustment-diagnostic.json').read_text())['examples'][:3]
def read(stock,field,date):
 a=np.fromfile(BASE/'features'/stock.lower()/(field+'.day.bin'),dtype='<f4');return float(a[1+cal.get_loc(date)-int(a[0])])
def query(code,start,end,flag):
 q=bs.query_history_k_data_plus(code,'date,code,close,volume,amount',start_date=start,end_date=end,frequency='d',adjustflag=flag)
 assert q.error_code=='0',q.error_msg
 rows=[]
 while q.next():rows.append(q.get_row_data())
 f=pd.DataFrame(rows,columns=q.fields);f['date']=pd.to_datetime(f.date);f=f.set_index('date')
 for c in ['close','volume','amount']:f[c]=pd.to_numeric(f[c])
 return f
login=bs.login();assert login.error_code=='0'
results=[]
try:
 for ex in examples:
  day=pd.Timestamp(ex['date']);stock=ex['stock'];code=stock[-2:].lower()+'.'+stock[:6];start=str((day-pd.Timedelta(days=8)).date());end=str(day.date())
  raw=query(code,start,end,'3');adj=query(code,start,end,'2');assert raw.index.equals(adj.index)
  loc=raw.index.get_loc(day);previous=raw.index[loc-1]
  own_raw=read(stock,'close',day)/read(stock,'close',previous)-1
  own_adj=(read(stock,'close',day)*read(stock,'factor',day))/(read(stock,'close',previous)*read(stock,'factor',previous))-1
  vendor_raw=raw.loc[day,'close']/raw.loc[previous,'close']-1;vendor_adj=adj.loc[day,'close']/adj.loc[previous,'close']-1
  vol_ratio=raw.loc[day,'volume']/read(stock,'volume',day)
  vwap=raw.loc[day,'amount']/raw.loc[day,'volume'];own_vwap=read(stock,'vwap',day)
  assert abs(own_raw-vendor_raw)<1e-5
  assert abs(own_adj-vendor_adj)<.001
  assert abs(vol_ratio-100)<.01
  assert abs(vwap/own_vwap-1)<1e-5
  results.append(dict(stock=stock,date=end,vendor_raw_return=float(vendor_raw),local_raw_return=own_raw,
      vendor_adjusted_return=float(vendor_adj),local_factor_adjusted_return=own_adj,
      vendor_shares_per_local_volume_unit=float(vol_ratio),vendor_raw_vwap=float(vwap),local_raw_vwap=own_vwap))
finally:bs.logout()
result=dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),baostock_version=bs.__version__,examples=results,
    interpretation='Direct Baostock samples corroborate the demonstrated raw-price corporate-action error and confirm a 100-share unit per local volume unit. This validates these three actions and units, not the full original Baostock snapshot or an adjustment anchor.')
(OUT/'alphasage-direct-vendor-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
