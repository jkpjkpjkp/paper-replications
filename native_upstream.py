#!/usr/bin/env python3
"""Exact frozen upstream order, uniform slate, native author evaluation helpers."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parent;DATA=ROOT/'release/data';OUT=ROOT/'results/native-upstream'
sys.path.insert(0,str(ROOT/'upstream'))
from src.groundtruth_utils import build_newbuy_labels,load_prices_at_dates
from src.metrics_core_fast import roi_from_prices,ndcg_at_k_binary
pa.set_cpu_count(1);pa.set_io_thread_count(1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    slices=[json.loads(x) for x in (DATA/'splits/slices.jsonl').read_text().splitlines()]
    aggregates=[];checks=[]
    for meta in slices:
        sid=meta['slice_id'];began=time.time()
        frame=pq.read_table(ROOT/'inputs'/(sid+'.parquet')).to_pandas()
        frame=frame[frame.in_pit30].copy()
        selected=frame.groupby('user_id',sort=False).head(10)
        sizes=selected.groupby('user_id',sort=False).size()
        if not sizes.eq(10).all():raise RuntimeError('Native upstream has fewer than10 eligible recommendations')
        labels=build_newbuy_labels(DATA/'clean/transactions.parquet',pd.Timestamp(meta['test_start']).date(),pd.Timestamp(meta['test_end']).date(),chunk_rows=200000)
        first,last=load_prices_at_dates(DATA/'clean/prices.parquet',selected.item_id.unique().tolist(),pd.Timestamp(meta['test_start']).date(),pd.Timestamp(meta['test_end']).date(),chunk_rows=200000)
        rows=[];rankrows=[]
        for uid,g in selected.groupby('user_id',sort=False):
            items=g.item_id.astype(str).to_numpy();weights=np.full(10,.1)
            ndcg=ndcg_at_k_binary(items,labels.get(str(uid),set()),10)
            roi=roi_from_prices(items,weights,first,last,10.,10.,10.)
            missing=sum(not(np.isfinite(first.get(a,np.nan)) and first.get(a,0)>0 and np.isfinite(last.get(a,np.nan))) for a in items)
            rows.append({'user_id':str(uid),'ndcg':ndcg,'native_period_roi':roi,'missing_exact_price_slots':missing})
            rankrows.append({'user_id':str(uid),'items':items.tolist()})
        results=pd.DataFrame(rows);results.to_parquet(OUT/(sid+'-user-metrics.parquet'),index=False,compression='zstd')
        selected[['user_id','item_id','source_order']].to_parquet(OUT/(sid+'-rankings.parquet'),index=False,compression='zstd')
        agg={'slice_id':sid,'users':len(results),'ndcg_percent':float(results.ndcg.mean()*100),'roi_percent':float(results.native_period_roi.mean()*100),'missing_exact_price_slot_fraction':float(results.missing_exact_price_slots.sum()/(10*len(results))),'seconds':time.time()-began}
        aggregates.append(agg);print(json.dumps(agg),flush=True)
    table=pd.DataFrame(aggregates);table.to_csv(OUT/'slice-metrics.csv',index=False)
    result={'status':'complete','method_slice_runs':16,'user_slices':int(table.users.sum()),'ndcg_percent':float(np.average(table.ndcg_percent,weights=table.users)),'roi_percent':float(np.average(table.roi_percent,weights=table.users)),'source':'Frozen released upstream row order,eligible and unheld filter,uniform10slate,unchanged author label/nDCG/ROI helpers; exact native calendar-date pricing','paper_reported':{'ndcg_percent':.7604,'roi_percent':-.3897},'test_selection':'None; native conventions fixed before execution,not tuned to target cells','daily_market_neutral':False}
    result['differences']={m:result[m]-result['paper_reported'][m] for m in ['ndcg_percent','roi_percent']}
    result['matches_printed_rounding']={m:abs(result['differences'][m])<=.00005 for m in ['ndcg_percent','roi_percent']}
    p=OUT/'status.json';tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
