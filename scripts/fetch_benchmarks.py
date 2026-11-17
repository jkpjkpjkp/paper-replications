"""Archive public CSI index closes from Eastmoney's documented AkShare route."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import requests
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.build_data import sha, write_json
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--refresh',action='store_true',help='Explicitly replace the archived benchmark inputs')
    args=parser.parse_args()
    folder=ROOT/'sources/benchmarks';folder.mkdir(exist_ok=True)
    receipt_file=folder/'receipts.json'
    if receipt_file.exists() and not args.refresh:
        saved=json.loads(receipt_file.read_text())
        for universe in ['csi300','csi500']:
            assert sha(folder/f'{universe}.json')==saved[universe]['response_sha256']
            assert sha(folder/f'{universe}.csv')==saved[universe]['csv_sha256']
        print('Reused verified archived CSI benchmark inputs.',flush=True)
        return
    receipts={}
    for universe,code,name in [('csi300','000300','沪深300'),('csi500','000905','中证500')]:
        params=dict(secid=f'1.{code}',fields1='f1,f2,f3,f4,f5',
            fields2='f51,f52,f53,f54,f55,f56,f57,f58',klt='101',fqt='0',beg='20100101',end='20200805')
        response=requests.get('https://push2his.eastmoney.com/api/qt/stock/kline/get',params=params,timeout=30)
        response.raise_for_status();raw=response.json()
        assert raw['data']['code']==code and raw['data']['name']==name
        (folder/f'{universe}.json').write_text(response.text)
        frame=pd.DataFrame([line.split(',') for line in raw['data']['klines']],
            columns=['date','open','close','high','low','volume','amount','amplitude'])
        assert not frame.date.duplicated().any()
        assert frame.date.min()<='2010-01-04' and frame.date.max()>='2020-08-05'
        frame['close']=pd.to_numeric(frame.close)
        frame['return']=frame.close.pct_change()
        frame.to_csv(folder/f'{universe}.csv',index=False)
        receipts[universe]=dict(url=response.url,retrieved_utc=datetime.now(timezone.utc).isoformat(),
            response_sha256=sha(folder/f'{universe}.json'),csv_sha256=sha(folder/f'{universe}.csv'),
            rows=len(frame),first=str(frame.date.min()),last=str(frame.date.max()))
    write_json(receipt_file,receipts)


if __name__=='__main__':main()
