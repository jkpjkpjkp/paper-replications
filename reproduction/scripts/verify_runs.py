"""Independent checks of forecast labels, metrics and affine variance units."""
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
source=pd.read_csv(ROOT/'results/synthetic.dat',sep=' ',dtype={'gvkey':str})
source['date']=pd.to_datetime(source.date.astype(str),format='%Y%m')
report={}
for mode in ['point','uq']:
    directory=ROOT/'results'/('smoke-'+mode)
    df=pd.read_csv(directory/'pred/preds.dat',sep=' ',dtype={'gvkey':str})
    print(mode,df.columns.tolist())
    assert len(df)==20*36
    assert not df.duplicated(['date','gvkey']).any()
    assert np.isfinite(df.select_dtypes('number').to_numpy()).all()
    truth=source[['date','gvkey','oiadpq_ttm']].copy()
    truth['date']=(truth.date-pd.DateOffset(years=1)).dt.strftime('%Y%m%d').astype(int)
    joined=df.merge(truth,on=['date','gvkey'],validate='one_to_one')
    assert len(joined)==len(df)
    np.testing.assert_allclose(joined.targets_1,joined.oiadpq_ttm,rtol=1e-5)
    np.testing.assert_allclose(df.norm_squared_diff_1,(df.norm_targets_1-df.norm_preds_1)**2,rtol=1e-4,atol=1e-7)
    scales=pickle.load(open(directory/'scales.dat','rb'))
    np.testing.assert_allclose(df.variance_1,df.norm_variance_1*(scales['scale'][1]*df.seq_norm)**2,rtol=1e-5)
    assert (df.norm_variance_1>0).all() if mode=='uq' else (df.variance_1==0).all()
    history=pd.read_csv(directory/'train_log'/('smoke-'+mode+'-train-logs-epoch.csv'))
    assert len(history)==2
    assert np.isfinite(history.valid_mse).all()
    report[mode]={'epochs':len(history),'forecast_rows':len(df),'scaled_mse':float(np.mean((df.norm_targets_1-df.norm_preds_1)**2)), 'targets_match_fixture':True,'variance_units_verified':True}
report['scope']='Synthetic integration checks only; no financial or paper-performance claim.'
(ROOT/'results/verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
