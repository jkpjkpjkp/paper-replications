"""Verify every annual input contract before expensive training."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import digest


def audit(data):
    raw=np.load(data/'returns.npy',mmap_mode='r')
    dates=np.load(data/'dates.npy');ids=np.load(data/'permnos.npy')
    ff=pd.read_parquet(data/'factors.parquet')
    cohorts=pd.read_csv(data/'cohorts/candidate_equity_cohorts.csv')
    output=[]
    for year in range(2002,2022):
        p=data/'panels'/str(year)
        m=json.loads((p/'manifest.json').read_text())
        x=np.load(p/'x.npy',mmap_mode='r');lab=np.load(p/'labels.npz')
        assert list(x.shape)==m['shape'] and x.shape[1:]==(500,79)
        assert digest(p/'x.npy')==m['x_sha256'] and digest(p/'labels.npz')==m['labels_sha256']
        assert np.isfinite(x).all() and np.isfinite(lab['y']).all()
        train=lab['train'];n=int(train.sum())
        assert train[:n].all() and not train[n:].any()
        assert np.all(lab['dates'][train]>=np.datetime64(f'{year-8}-01-01'))
        assert np.all(lab['dates'][train]<np.datetime64(f'{year}-01-01'))
        assert np.all(lab['dates'][~train]>=np.datetime64(f'{year}-01-01'))
        assert np.all(lab['dates'][~train]<np.datetime64(f'{year+1}-01-01'))
        assert np.all(lab['input_dates']<lab['dates'])
        positions=np.searchsorted(dates,lab['dates'])
        np.testing.assert_array_equal(dates[positions-1],lab['input_dates'])
        columns=np.searchsorted(ids,lab['permnos'])
        source=raw[positions][:,columns]-ff.rf.to_numpy(dtype=np.float32)[positions,None]
        np.testing.assert_array_equal(source,lab['y'])
        chosen=cohorts[(cohorts.test_year==year)&(cohorts.cohort=='paper_test_complete')].sort_values('rank')
        np.testing.assert_array_equal(chosen.permno,lab['permnos'])
        constant=[]
        for j,name in enumerate(m['features']):
            # Inspect a deterministic spread across training dates, all 500 units.
            # Median differences can change only on sparse accounting-release
            # dates, so inspect every date for common aggregate columns.
            v=x[:n,0,j] if j>=40 else x[np.linspace(0,n-1,20).astype(int),:,j]
            if np.ptp(v)<1e-10:constant.append(name)
            if j<40 and name!='RF':assert v.min()>=-.500001 and v.max()<=.500001
        if constant:
            raise ValueError(f'{year}: constant inputs require inspection: {constant}')
        # In particular the cash-flow input must survive modern-data mapping.
        assert m['missing_raw']['CF']['train']<.5
        output.append({'year':year,'shape':list(x.shape),'train_days':n,'test_days':int((~train).sum()),
            'max_train_missing':max(v['train'] for v in m['missing_raw'].values()),'CF_missing_train':m['missing_raw']['CF']['train'],
            'source_labels_match':True,'cohort_match':True,'strict_next_session_alignment':True,'constant_features':constant})
        print(f'audit {year}: passed',flush=True)
    record={'passed':True,'annual_panels':len(output),'years':output,'raw_manifest_sha256':digest(data/'raw_manifest.json'),
        'note':'Checks reconstruction integrity; does not certify author-equivalent inputs or historical point-in-time accounting.'}
    (data/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args();audit(a.data)
