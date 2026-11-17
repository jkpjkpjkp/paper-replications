#!/usr/bin/env python3
"""Build patch-local inputs and 390 next-session cumulative-price labels.

No feature depends on another patch. Missing feature observations are left
for training-only preprocessing. Label prices carry forward causally.
Full-session input and label days are required; split-boundary labels are purged.
"""
from __future__ import annotations
import argparse
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
FIELDS=['open','high','low','close','amount','volume','trade_count']
DESCRIPTORS=['mean','change_rate','normalized_range','std','skew','excess_kurtosis']
FEATURES=[f'{field}_{stat}' for field in FIELDS for stat in DESCRIPTORS]
FEATURES += [f'corr_{FIELDS[i]}_{FIELDS[j]}' for i in range(7) for j in range(i+1,7)]
FEATURES += ['vwap_to_mean_close','shares_per_trade','dollars_per_trade']

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def save_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)

def ratio(a,b):
    a,b=np.broadcast_arrays(a,b)
    out=np.full(a.shape,np.nan,dtype=np.float64)
    np.divide(a,b,out=out,where=np.isfinite(a)&np.isfinite(b)&(b!=0))
    return out

def moments(values):
    """Six descriptors over minute axis 2, with undefined statistics masked."""
    finite=np.isfinite(values);count=finite.sum(axis=2)
    mean=ratio(np.where(finite,values,0).sum(axis=2),count)
    delta=np.where(finite,values-mean[:,:,None,:],0)
    variance=ratio((delta*delta).sum(axis=2),count)
    std=np.sqrt(variance)
    skew=ratio(ratio((delta**3).sum(axis=2),count),std**3)
    kurt=ratio(ratio((delta**4).sum(axis=2),count),variance**2)-3
    constant=(std==0)&(count>0)
    skew=np.where(constant,0,skew);kurt=np.where(constant,0,kurt)
    skew=np.where(count>=3,skew,np.nan);kurt=np.where(count>=4,kurt,np.nan)
    hi=np.where(count>0,np.where(finite,values,-np.inf).max(axis=2),np.nan)
    lo=np.where(count>0,np.where(finite,values,np.inf).min(axis=2),np.nan)
    first_idx=finite.argmax(axis=2)
    last_idx=values.shape[2]-1-finite[:,:,::-1,:].argmax(axis=2)
    first=np.take_along_axis(values,first_idx[:,:,None,:],axis=2).squeeze(2)
    last=np.take_along_axis(values,last_idx[:,:,None,:],axis=2).squeeze(2)
    first=np.where(count>0,first,np.nan);last=np.where(count>0,last,np.nan)
    stats=[mean,ratio(last,first)-1,ratio(hi-lo,np.abs(mean)),std,skew,kurt]
    return np.stack(stats,axis=-1).reshape(values.shape[0],values.shape[1],42)

def features_from_grid(grid):
    """grid: stocks x 390 minutes x seven nominal fields, NaNs preserved."""
    if grid.shape[1:]!=(390,7):raise ValueError('Feature input must be N x 390 x 7')
    n=grid.shape[0]
    raw=grid.reshape(n,26,15,7)
    parts=[moments(raw)]
    correlations=[]
    for i in range(7):
        for j in range(i+1,7):
            x,y=raw[:,:,:,i],raw[:,:,:,j]
            ok=np.isfinite(x)&np.isfinite(y);count=ok.sum(2)
            mx=ratio(np.where(ok,x,0).sum(2),count)
            my=ratio(np.where(ok,y,0).sum(2),count)
            dx=np.where(ok,x-mx[:,:,None],0);dy=np.where(ok,y-my[:,:,None],0)
            sx=(dx*dx).sum(2);sy=(dy*dy).sum(2)
            corr=ratio((dx*dy).sum(2),np.sqrt(sx*sy))
            corr=np.where((sx==0)|(sy==0),0,corr)
            correlations.append(np.where(count>=3,corr,np.nan))
    parts.append(np.stack(correlations,axis=-1))
    totals=np.nansum(raw,axis=2)
    close_count=np.isfinite(raw[:,:,:,3]).sum(2)
    mean_close=ratio(np.nansum(raw[:,:,:,3],axis=2),close_count)
    interactions=np.stack([ratio(ratio(totals[:,:,4],totals[:,:,5]),mean_close)-1,
                           ratio(totals[:,:,5],totals[:,:,6]),
                           ratio(totals[:,:,4],totals[:,:,6])],axis=-1)
    parts.append(interactions)
    result=np.concatenate(parts,axis=-1)
    result[~np.isfinite(result)]=np.nan
    if result.shape!=(n,26,len(FEATURES)):raise RuntimeError('Feature dimension mismatch')
    return result

def cumulative_labels(current_close, future_close, factor0, factor1):
    """Adjusted last-observed price / decision price - 1; never backward fill.

    The decision close seeds the path before the next day's first eligible
    print. The terminal minute must actually be observed to enter the sample.
    This market-data convention is explicit; the paper does not supply one.
    """
    base=ratio(current_close,factor0)
    adjusted=ratio(future_close,factor1[:,None])
    path=np.column_stack([base,adjusted])
    valid=np.isfinite(path)&(path>0)
    indices=np.maximum.accumulate(np.where(valid,np.arange(path.shape[1]),0),axis=1)
    carried=np.take_along_axis(path,indices,axis=1)[:,1:]
    labels=ratio(carried,base[:,None])-1
    keep=(np.isfinite(base)&(base>0)&np.isfinite(adjusted[:,-1])
          &(adjusted[:,-1]>0)&(factor0>0)&(factor1>0))
    return labels,keep,~np.isfinite(adjusted)

def grid_for(table,permnos):
    grid=np.full((len(permnos),390,7),np.nan,dtype=np.float64)
    p=np.asarray(table['permno']);minute=np.asarray(table['minute_index'])
    indices=np.searchsorted(permnos,p)
    valid=(indices<len(permnos))&(minute>=0)&(minute<390)
    # Avoid indexing beyond sorted identifiers while checking membership.
    valid &= permnos[np.minimum(indices,len(permnos)-1)]==p
    for channel,name in enumerate(FIELDS):
        grid[indices[valid],minute[valid],channel]=np.asarray(table[name],dtype=np.float64)[valid]
    return grid

class BarStore:
    def __init__(self,root):self.root=root;self.cache={};self.verified={}
    def get(self,date):
        path=self.root/'bars'/date[:4]/(date+'.parquet')
        receipt=path.with_suffix('.receipt.json')
        if not path.exists() or not receipt.exists():return None
        info=json.loads(receipt.read_text())
        if self.verified.get(str(path))!=info['sha256']:
            if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
                raise ValueError('Raw bar receipt mismatch: '+str(path))
            self.verified[str(path)]=info['sha256']
        if date not in self.cache:
            self.cache[date]=pq.read_table(path,columns=['permno','minute_index']+FIELDS)
            while len(self.cache)>3:self.cache.pop(next(iter(self.cache)))
        return self.cache[date],info['sha256']

def input_indices(root):
    calendar=pq.read_table(root/'support/trading_calendar.parquet').to_pylist()
    calendar={str(row['session_date']):row for row in calendar}
    m=pq.read_table(root/'support/sp500_daily_membership.parquet',columns=['session_date','permno','membership_active']).to_pydict()
    members={}
    for day,p,active in zip(m['session_date'],m['permno'],m['membership_active']):
        if active:members.setdefault(str(day),[]).append(p)
    a=pq.read_table(root/'support/crsp_daily_auxiliary.parquet',columns=['dlycaldt','permno','dlycumfacpr']).to_pydict()
    factors={}
    for day,p,factor in zip(a['dlycaldt'],a['permno'],a['dlycumfacpr']):
        factors.setdefault(str(day),{})[p]=float(factor) if factor is not None else np.nan
    return calendar,members,factors

def eligible_date(day,calendar):
    row=calendar[day];following=str(row['next_session'])
    if row['session_split']=='label_extension':return False,'label_extension'
    if following not in calendar:return False,'no_label_calendar'
    nextrow=calendar[following]
    if row['expected_minutes']!=390 or nextrow['expected_minutes']!=390:return False,'short_input_or_label_session'
    same_split=row['session_split']==nextrow['session_split']
    allowed_terminal=row['session_split']=='test' and nextrow['session_split']=='label_extension'
    if not (same_split or allowed_terminal):return False,'split_boundary_purge'
    return True,following

def build(args):
    root=args.root;out=root/'reproduction/cache';out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'reproduction/protocol.json').read_text())
    config={'protocol':protocol,'builder_sha256':sha(__file__),'features':FEATURES,
            'auxiliary_sha256':sha(root/'support/crsp_daily_auxiliary.parquet')}
    config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    lock=(out/'.build.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    calendar,members,factors=input_indices(root);bars=BarStore(root)
    report={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'config_sha256':config_hash,
            'config':config,'cached':[],'pending':[],'excluded':[],'complete':False}
    wanted=set(args.dates.split(',')) if args.dates else None
    for day in sorted(calendar):
        if wanted and day not in wanted:continue
        allowed,nextday=eligible_date(day,calendar)
        if not allowed:
            report['excluded'].append({'date':day,'reason':nextday});continue
        first=bars.get(day);second=bars.get(nextday)
        if first is None or second is None:
            report['pending'].append(day);continue
        target=out/(day+'.npz');receipt=out/(day+'.receipt.json')
        source_hashes=[first[1],second[1]]
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if (saved.get('config_sha256')==config_hash and saved.get('source_bar_sha256')==source_hashes
                and target.exists() and sha(target)==saved.get('sha256')):
                report['cached'].append(saved);continue
        permnos=np.array(sorted(members[day]),dtype=np.int32)
        current=grid_for(first[0],permnos);future=grid_for(second[0],permnos)
        factor0=np.array([factors[day].get(int(p),np.nan) for p in permnos])
        factor1=np.array([factors[nextday].get(int(p),np.nan) for p in permnos])
        labels,keep,carried=cumulative_labels(current[:,-1,3],future[:,:,3],factor0,factor1)
        if keep.sum()<protocol['minimum_stocks']:
            report['excluded'].append({'date':day,'reason':'insufficient_final_label_stocks','stocks':int(keep.sum())});continue
        X=features_from_grid(current[keep]);Y=labels[keep]
        if not np.isfinite(Y).all():raise ValueError('Dense candidate labels required')
        temp=target.with_suffix('.npz.tmp')
        with temp.open('wb') as handle:
            np.savez_compressed(handle,X=X,Y=Y,permno=permnos[keep],date=np.datetime64(day,'D'),label_date=np.datetime64(nextday,'D'))
        temp.replace(target)
        saved={'date':day,'label_date':nextday,'split':calendar[day]['session_split'],'stocks':int(keep.sum()),
               'excluded_stocks':int(len(permnos)-keep.sum()),'missing_feature_values':int(np.isnan(X).sum()),
               'missing_label_values':int(np.isnan(Y).sum()),'source_bar_sha256':source_hashes,
               'carried_label_prices':int(carried[keep].sum()),
               'config_sha256':config_hash,'sha256':sha(target),'bytes':target.stat().st_size}
        save_json(receipt,saved);report['cached'].append(saved)
        print(f'Cached {day}: {keep.sum()} stocks, X={X.shape}, Y={Y.shape}',flush=True)
    report['complete']=not report['pending'] and not args.dates
    report['summary']={'cached_days':len(report['cached']),'pending_days':len(report['pending']),
                       'excluded_days':len(report['excluded']),'stocks_days':sum(x['stocks'] for x in report['cached'])}
    save_json(root/'reproduction/tensor_status.json',report)
    print(json.dumps(report['summary']),flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--dates',help='optional comma-separated dates for an engineering check')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();report=build(args)
    sys.exit(1 if args.require_complete and not report['complete'] else 0)
