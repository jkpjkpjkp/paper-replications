"""Bounded public-data connectivity probe; no credentials or proxy values logged."""
import argparse,json,os,socket,time
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import urlparse,unquote
ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['direct','proxy']);args=ap.parse_args()
socket.setdefaulttimeout(15)
if args.mode=='proxy':
    import socks
    value=os.environ.get('ALL_PROXY') or os.environ.get('HTTPS_PROXY')
    if not value:raise SystemExit('No configured proxy available')
    proxy=urlparse(value)
    kind=socks.SOCKS5 if proxy.scheme.startswith('socks5') else socks.HTTP
    socks.set_default_proxy(kind,proxy.hostname,proxy.port,rdns=True,
        username=unquote(proxy.username) if proxy.username else None,
        password=unquote(proxy.password) if proxy.password else None)
    socket.socket=socks.socksocket
import baostock as bs
started=time.monotonic()
result=dict(mode=args.mode,checked_utc=datetime.now(timezone.utc).isoformat(),baostock_version=bs.__version__)
try:
    login=bs.login();result.update(login_code=login.error_code,login_message=login.error_msg)
    if login.error_code=='0':
        data=bs.query_history_k_data_plus('sh.600000','date,code,close',start_date='2022-01-04',end_date='2022-01-05',frequency='d',adjustflag='2')
        rows=[]
        while data.error_code=='0' and data.next():rows.append(data.get_row_data())
        result.update(query_code=data.error_code,rows=rows)
        bs.logout()
except Exception as exc:result.update(exception_type=type(exc).__name__)
result['seconds']=time.monotonic()-started
out=Path(__file__).resolve().parents[1]/'results'/f'baostock-probe-{args.mode}-20261113.json'
out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
