#!/usr/bin/env python3
"""Resume checksummed 2019-2025 TAQ minute data extraction in dedicated WRDS kernels."""
from __future__ import annotations
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import time

import duckdb
import pyarrow.parquet as pq
sys.path.insert(0, '/home/jkp/data/ibes')
from bridge import AuthenticationError, Bridge
from download_ibes import download, save_json, verified_file

ROOT = Path(__file__).resolve().parent
TOKEN = Path('/home/jkp/data/wrds-crsp-download/wrds-jupyter.token')
PRICE_FILTER = "coalesce(t.tr_scond,'') ~ '^[ @EFKX56O]*$'"
VOLUME_FILTER = "coalesce(t.tr_scond,'') !~ '[MQ9]'"
print_lock = threading.Lock()
stop_event = threading.Event()

def remote(bridge,function,*args):
    code='print(json.dumps('+function+'(*json.loads('+repr(json.dumps(args))+')),default=str))'
    return json.loads(bridge.execute(code,timeout=45).strip())

def log(message):
    with print_lock:
        print(dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')+' '+message,flush=True)

def mappings_and_calendar(dates):
    db=duckdb.connect()
    calendar=db.sql(f"SELECT session_date,market_close_new_york FROM read_parquet('{ROOT}/support/trading_calendar.parquet') ORDER BY session_date DESC").fetchall()
    map_path=ROOT/'support/taq_daily_mappings.parquet'
    if map_path.exists():
        data=db.sql(f"SELECT session_date,permno,sym_root,sym_suffix FROM read_parquet('{map_path}')").fetchall()
    else:
        # The official link can benchmark dates <=2024 while the 2025 master fallback builds.
        data=db.sql(f"""SELECT l.date,l.permno,l.sym_root,l.sym_suffix
          FROM read_parquet('{ROOT}/metadata/taq_crsp_link_*.parquet') l
          JOIN read_parquet('{ROOT}/support/sp500_daily_membership.parquet') u
          ON l.date=u.session_date AND l.permno=u.permno""").fetchall()
    grouped={}
    for date,p,r,s in data:
        grouped.setdefault(date,[]).append({'permno':p,'sym_root':r,'sym_suffix':s})
    for mapping in grouped.values():
        mapping.sort(key=lambda m:(m['permno'],m['sym_root'],m['sym_suffix'] or ''))
    wanted=set(dt.date.fromisoformat(x) for x in dates.split(',')) if dates else None
    tasks=[]
    for day, close in calendar:
        if wanted is not None and day not in wanted:continue
        if not grouped.get(day):raise ValueError('No date-specific TAQ mappings for '+str(day))
        tasks.append((day,close,grouped[day]))
    if wanted and len(tasks)!=len(wanted):raise ValueError('Requested date is not in the trading calendar')
    return tasks

def verify_local(target, info):
    if not verified_file(target,info):raise RuntimeError('Local transfer hash/size mismatch')
    table=pq.read_table(target)
    if table.num_rows!=info['totals']['rows']:raise RuntimeError('Parquet count differs from source bars')
    db=duckdb.connect()
    db.register('bars',table)
    rows,vol,trades,amount,bad=db.execute("""SELECT count(*),sum(volume),sum(trade_count),sum(amount),
      count(*) FILTER(WHERE minute_index<0 OR minute_index>=390 OR volume<=0 OR trade_count<=0)
      FROM bars""").fetchone()
    t=info['totals']
    if vol!=t['volume'] or trades!=t['trades'] or trades!=info['copy_rows'] or bad:
        raise RuntimeError('Local data totals or minute range invalid')
    if abs(amount-t['amount'])>max(0.02,abs(t['amount'])*1e-10):
        raise RuntimeError('Local dollar amount differs from remote total')
    duplicates=db.execute('SELECT count(*) FROM (SELECT permno,minute_index FROM bars GROUP BY ALL HAVING count(*)>1)').fetchone()[0]
    if duplicates:raise RuntimeError('Duplicate security/minute keys')
    db.close()

def close_bridge(bridge):
    if bridge is None:return
    # A lost socket must not hide the export/transfer error or block all workers.
    try:remote(bridge,'cleanup')
    except Exception:pass
    if bridge.ws is not None:
        try:bridge.ws.close(timeout=1)
        except Exception:pass
        bridge.ws=None
    if bridge.kernel_id is not None:
        try:bridge.request('DELETE',bridge.server_path+'/api/kernels/'+bridge.kernel_id,timeout=10)
        except Exception:pass
        bridge.kernel_id=None
    bridge.http.close()

def run_worker(number,tasks,progress,total,worker_hash,worker_source):
    bridge=None
    connected_at=0
    try:
        while not stop_event.is_set():
            try:day,close,mappings=tasks.get_nowait()
            except queue.Empty:break
            try:
                folder=ROOT/'bars'/str(day.year)
                folder.mkdir(parents=True,exist_ok=True)
                target=folder/(str(day)+'.parquet')
                receipt=folder/(str(day)+'.receipt.json')
                mapping_hash=hashlib.sha256(json.dumps(mappings,sort_keys=True).encode()).hexdigest()
                old=json.loads(receipt.read_text()) if receipt.exists() else {}
                if (old.get('worker_sha256')==worker_hash and old.get('mappings_sha256')==mapping_hash
                    and old.get('price_condition_sql')==PRICE_FILTER and old.get('volume_condition_sql')==VOLUME_FILTER
                    and verified_file(target,old)):
                    with print_lock:progress['completed']+=1
                    (folder/(str(day)+'.error.json')).unlink(missing_ok=True)
                    continue
                for attempt in range(1,9):
                    try:
                        if bridge is not None and time.monotonic()-connected_at>2700:
                            close_bridge(bridge);bridge=None
                        if bridge is None:
                            bridge=Bridge(TOKEN)
                            bridge.connect()
                            bridge.execute('import json\nWRDS_USERNAME='+repr(bridge.username)+'\n'+worker_source.read_text(),timeout=120)
                            connected_at=time.monotonic()
                            log(f'worker {number} connected a dedicated WRDS kernel')
                        log(f'worker {number} extracting {day}, attempt {attempt}')
                        remote(bridge,'start_day',str(day),mappings,close,PRICE_FILTER,VOLUME_FILTER)
                        started=time.monotonic();next_log=started+60
                        while True:
                            info=remote(bridge,'status')
                            if info['state']=='error':raise RuntimeError(info.get('error',str(info)))
                            if info['state']=='complete':break
                            if time.monotonic()-started>1800:raise TimeoutError('Daily export exceeded 30 minutes')
                            if time.monotonic()>next_log:
                                log(f'worker {number} {day}: {round(time.monotonic()-started)}s, state={info["state"]}')
                                next_log=time.monotonic()+60
                            if stop_event.wait(3):raise RuntimeError('Another worker failed; stopping safely')
                        download(bridge,info,target)
                        verify_local(target,info)
                        info.update(worker_sha256=worker_hash,mappings_sha256=mapping_hash,local_file=str(target),
                                    verified_utc=dt.datetime.now(dt.timezone.utc).isoformat())
                        save_json(receipt,info)
                        (folder/(str(day)+'.error.json')).unlink(missing_ok=True)
                        # Receipt is durable before acknowledgement. A lost ack only
                        # requires retiring this kernel, never re-exporting valid data.
                        try:remote(bridge,'acknowledge',info['sha256'])
                        except Exception:
                            close_bridge(bridge);bridge=None
                        with print_lock:
                            progress['completed']+=1;complete=progress['completed']
                            save_json(ROOT/'status.json',{'state':'running','planned_sessions':total,'verified_sessions':complete,
                              'latest_date':str(day),'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
                        log(f'{complete}/{total} saved {day}: {info["totals"]["rows"]:,} bars, {info["elapsed_seconds"]}s')
                        break
                    except Exception as exc:
                        message=str(exc).replace(bridge.token if bridge else TOKEN.read_text().strip(),'[REDACTED]')
                        save_json(folder/(str(day)+'.error.json'),{'date':str(day),'attempt':attempt,'error':message,
                                  'utc':dt.datetime.now(dt.timezone.utc).isoformat()})
                        close_bridge(bridge);bridge=None
                        if isinstance(exc,AuthenticationError) or attempt==8 or stop_event.is_set():raise
                        log(f'worker {number} reconnecting after {type(exc).__name__}: {message[:180]}')
                        stop_event.wait(min(60,5*attempt))
            finally:tasks.task_done()
    except BaseException:
        stop_event.set()
        raise
    finally:close_bridge(bridge)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dates',help='comma-separated ISO dates for a benchmark')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--worker-source',type=Path,default=ROOT/'taq_worker.py')
    args=parser.parse_args()
    if not 1<=args.workers<=4:parser.error('Use 1 to 4 workers')
    lock=(ROOT/'.download.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=mappings_and_calendar(args.dates)
    worker_hash=hashlib.sha256(args.worker_source.read_bytes()).hexdigest()
    save_json(ROOT/('benchmark-plan.json' if args.dates else 'download-plan.json'),{'first_date':str(min(x[0] for x in plan)),'last_date':str(max(x[0] for x in plan)),
      'sessions':len(plan),'workers':args.workers,'worker_sha256':worker_hash,'worker_source':str(args.worker_source.resolve()),'price_condition_sql':PRICE_FILTER,'volume_condition_sql':VOLUME_FILTER,
      'dates':[str(x[0]) for x in plan]})
    save_json(ROOT/'status.json',{'state':'running','planned_sessions':len(plan),'verified_sessions':0,
      'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    tasks=queue.Queue()
    for task in plan:tasks.put(task)
    progress={'completed':0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(run_worker,i+1,tasks,progress,len(plan),worker_hash,args.worker_source) for i in range(args.workers)]
        for f in concurrent.futures.as_completed(futures):f.result()
    save_json(ROOT/'status.json',{'state':'benchmark_complete' if args.dates else 'download_complete','planned_sessions':len(plan),'verified_sessions':progress['completed'],
      'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat()})
    log('All requested sessions downloaded and transfer-verified')

if __name__=='__main__':main()
