#!/usr/bin/env python3
"""Guarded, streaming paper-shortlist preparation from immutable released data."""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'release' / 'data'
OUT = ROOT / 'inputs'
pa.set_cpu_count(1)
pa.set_io_thread_count(1)


def atomic_json(p, v):
    t = p.with_name(p.name + '.tmp')
    t.write_text(json.dumps(v, indent=2, default=str) + '\n')
    t.replace(p)


def user_groups(path):
    columns = ['user_id','item_id','score_blend','p_i','eligible','in_holdings_flag']
    tail = None
    previous = None
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=65536, columns=columns, use_threads=False):
        frame = batch.to_pandas()
        if tail is not None:
            frame = pd.concat([tail,frame],ignore_index=True)
        u = frame.user_id.to_numpy()
        starts = np.r_[0,np.flatnonzero(u[1:] != u[:-1])+1]
        for a,b in zip(starts[:-1],starts[1:]):
            uid = str(u[a])
            if previous is not None and uid <= previous:
                raise RuntimeError('Candidate users are not globally contiguous/strictly ordered')
            previous = uid
            yield uid,frame.iloc[a:b].copy()
        tail = frame.iloc[starts[-1]:].copy()
    if tail is not None:
        uid = str(tail.user_id.iloc[0])
        if previous is not None and uid <= previous:
            raise RuntimeError('Last candidate user is not strictly ordered')
        yield uid,tail


def main():
    OUT.mkdir(exist_ok=True)
    slices = [json.loads(x) for x in (DATA/'splits/slices.jsonl').read_text().splitlines()]
    price = pq.read_table(DATA/'clean/prices.parquet').to_pandas()
    tx = pq.read_table(DATA/'clean/transactions.parquet').to_pandas()
    customers = pq.read_table(DATA/'clean/customers.parquet').to_pandas()
    assets = pq.read_table(DATA/'clean/assets.parquet').to_pandas()
    px = price.pivot(index='date',columns='asset_id',values='close_eur').sort_index()
    assert not price.duplicated(['date','asset_id']).any()
    quality = {'dataset_counts':{'prices':len(price),'transactions':len(tx),
                                'customers':len(customers),'assets':len(assets)},
               'transaction_synthetic_flag_counts':tx.synthetic.value_counts(dropna=False).to_dict(),
               'asset_types':assets.asset_type.value_counts(dropna=False).to_dict(),
               'price_columns':list(price.columns),'transaction_columns':list(tx.columns),
               'cashflow_or_total_return_fields_present':False,
               'price_min_date':str(price.date.min()),'price_max_date':str(price.date.max()),
               'candidate_order_rule':'Use stored frozen upstream row order; validate score is nonincreasing; preserve input tie order.',
               'paper_shortlist_rule':'eligible & not held & >=2 positive observations within half-open test window,then first30 in frozen upstream order; all methods share this subset.',
               'price_endpoint_convention':'First and last observed positive closes inside [test_start,test_end), per asset. Freeze before any return analysis. Record actual dates/gaps. This resolves calendar/weekend boundaries but is an explicit interpretation of unspecified paper endpoints,not exact release-evaluator date equality.',
               'pit_shortlist_rule':'eligible & not held,then first30,without future endpoint filter; retained separately,not a claim of fully point-in-time provenance.',
               'adaptation':'Inputs are released artifacts; upstream/calibrator construction code and point-in-time profile history are not available.'}
    summaries=[]
    for meta in slices:
        sid=meta['slice_id']; target=OUT/(sid+'.parquet'); summarypath=OUT/(sid+'.json')
        if target.exists() and summarypath.exists():
            existing=json.loads(summarypath.read_text())
            if existing.get('status')=='complete':
                summaries.append(existing);print(json.dumps({'slice':sid,'status':'reuse_completed_inputs'}),flush=True);continue
        begin=time.time()
        profiles=pq.read_table(DATA/'profiles_riskmap_cat'/sid/'user_params.parquet').to_pandas().set_index('user_id')
        stats=pq.read_table(DATA/'features'/sid/'market_stats.parquet').to_pandas().set_index('asset_id')
        var_default=max(1e-8,float(np.nanmedian(stats.sigma2)))
        start=pd.Timestamp(meta['test_start']);end=pd.Timestamp(meta['test_end'])
        window=price[(price.date>=start)&(price.date<end)&price.close_eur.gt(0)&np.isfinite(price.close_eur)].sort_values(['asset_id','date'],kind='stable')
        grouped=window.groupby('asset_id',sort=True)
        endpoints=grouped.agg(start_date=('date','first'),end_date=('date','last'),start_close=('close_eur','first'),end_close=('close_eur','last'),n_obs=('close_eur','size'))
        valid=(endpoints.n_obs>=2)&(endpoints.start_close>0)&(endpoints.end_close>0)
        endpoints['valid']=valid
        endpoints['start_gap_days']=(endpoints.start_date-start).dt.days
        endpoints['end_gap_days']=(end-endpoints.end_date).dt.days
        endpoints.to_parquet(OUT/(sid+'-endpoints.parquet'))
        eligible_assets=set(valid[valid].index)
        summary={'slice_id':sid,'status':'running','source_rows':0,'users':0,'cache_rows':0,
                 'users_with_paper30':0,'users_with_pit30':0,'users_with_less_than10_paper':0,
                 'order_violations':0,'duplicate_items':0,'missing_profiles':0,'feature_imputations':0,
                 'valid_within_test_window_price_assets':len(eligible_assets),'test_start':str(start),'test_end':str(end),
                 'endpoint_path':str(OUT/(sid+'-endpoints.parquet')),
                 'start_gap_days_max':int(endpoints.start_gap_days.max()),'end_gap_days_max':int(endpoints.end_gap_days.max()),
                 'rho_median_by_category':profiles.groupby('category').gamma_or_eta.median().to_dict(),
                 'feature_asof_min':str(stats.asof_date.min()),'feature_asof_max':str(stats.asof_date.max()),
                 'category_scale_applied':'profiles_riskmap_cat already includes scale; do not apply twice',
                 'source_score':'score_blend','variance_source':'features/sigma2; missing slice median',
                 'started_at_unix':begin}
        pending=[];writer=None;temp=target.with_name(target.name+'.tmp')
        def flush():
            nonlocal writer,pending
            if not pending:return
            frame=pd.concat(pending,ignore_index=True)
            table=pa.Table.from_pandas(frame,preserve_index=False)
            if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
            writer.write_table(table,row_group_size=32768);pending=[]
        for uid,g in user_groups(DATA/'candidates'/(sid+'_all.parquet')):
            summary['source_rows']+=len(g);summary['users']+=1
            if g.item_id.duplicated().any():summary['duplicate_items']+=int(g.item_id.duplicated().sum())
            if uid not in profiles.index:
                summary['missing_profiles']+=1;continue
            g=g[g.eligible & ~g.in_holdings_flag].copy()
            if len(g)==0:continue
            if np.any(np.diff(g.score_blend.to_numpy())>1e-12):
                summary['order_violations']+=1
                raise RuntimeError('Frozen source candidate order does not match decreasing score')
            g['source_order']=np.arange(len(g),dtype=np.int32)
            pit=g.head(30);paper=g[g.item_id.isin(eligible_assets)].head(30)
            summary['users_with_pit30']+=int(len(pit)==30);summary['users_with_paper30']+=int(len(paper)==30)
            summary['users_with_less_than10_paper']+=int(len(paper)<10)
            keep=g.iloc[np.sort(np.unique(np.r_[pit.source_order.to_numpy(),paper.source_order.to_numpy()]))].copy()
            keep['in_pit30']=keep.source_order.isin(pit.source_order)
            keep['in_paper30']=keep.source_order.isin(paper.source_order)
            keep['price_valid_endpoints']=keep.item_id.isin(eligible_assets)
            keep['r_feature']=keep.item_id.map(stats.r).fillna(0.).astype(float)
            mapped=keep.item_id.map(stats.sigma2)
            summary['feature_imputations']+=int(mapped.isna().sum())
            keep['sigma2']=mapped.fillna(var_default).clip(lower=0.).astype(float)
            profile=profiles.loc[uid]
            keep['rho']=float(profile.gamma_or_eta);keep['B']=float(profile.B)
            keep['category']=str(profile.category)
            keep=keep.drop(columns=['eligible','in_holdings_flag'])
            pending.append(keep);summary['cache_rows']+=len(keep)
            if len(pending)>=256:flush()
            if summary['users']%5000==0:print(json.dumps({'slice':sid,'users':summary['users'],'source_rows':summary['source_rows']}),flush=True)
        flush()
        if writer is None:raise RuntimeError('No shortlist rows generated')
        writer.close();temp.replace(target)
        summary.update({'status':'complete','finished_at_unix':time.time(),'output':str(target),'output_bytes':target.stat().st_size})
        atomic_json(summarypath,summary);summaries.append(summary)
        print(json.dumps({'slice':sid,'status':'prepared','cache_rows':summary['cache_rows'],'seconds':time.time()-begin}),flush=True)
    quality['slices']=summaries
    quality['risk_order_discrepancy']='Released aggressive rho exceeds conservative rho; preserve original artifact values and flag conflict with paper risk-order claim.'
    quality['extension_blockers']=['No dividend/coupon/total-return field in any released clean data schema.','No upstream-builder,calibrator-code or timestamped customer-profile source history to establish full point-in-time availability.','Published original common universe uses test-end price validity,which cannot define a prospective investable universe.']
    atomic_json(ROOT/'evidence/input-quality.json',quality)
    print(json.dumps({'status':'complete','slices':len(summaries),'quality':str(ROOT/'evidence/input-quality.json')}),flush=True)


if __name__=='__main__':main()
