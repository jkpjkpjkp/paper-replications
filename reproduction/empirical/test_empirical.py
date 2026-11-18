import os
import numpy as np
import pandas as pd
from build_panel import quarterly_flow, DEFAULT, FIN
from pathlib import Path
DEFAULT=Path(os.environ.get('LFM_EMPIRICAL_OUTPUT',str(DEFAULT)))


def test_ytd_conversion_does_not_cross_year_or_missing_quarter():
    values=pd.Series([10.,25.,44.,60.,12.,50.])
    years=pd.Series([2000,2000,2000,2000,2001,2001])
    quarters=pd.Series([1,2,3,4,1,3])
    actual=quarterly_flow(values,years,quarters)
    np.testing.assert_allclose(actual.iloc[:5],[10,15,19,16,12])
    assert pd.isna(actual.iloc[5])


def test_built_sample_dates_and_scale():
    if not (DEFAULT/'samples.parquet').exists():
        import pytest
        pytest.skip('Run after building the empirical panel')
    panel=pd.read_parquet(DEFAULT/'monthly_panel.parquet',columns=['gvkey','month_id','date','available','ebit'])
    meta=pd.read_parquet(DEFAULT/'samples.parquet')
    rows=np.load(DEFAULT/'sample_rows.npy')
    x=np.load(DEFAULT/'x.npy',mmap_mode='r');y=np.load(DEFAULT/'y.npy',mmap_mode='r')
    import json
    scaling=json.loads((DEFAULT/'scaling.json').read_text());t=FIN.index('ebit')
    assert meta.loc[meta.split.isin(['train','valid']),'target_month_id'].max()<=1999*12+12
    train_keys=set(meta.loc[meta.split.eq('train'),'gvkey'])
    assert not train_keys.intersection(meta.loc[meta.split.eq('valid'),'gvkey'])
    for i in np.linspace(0,len(meta)-1,1000,dtype=int):
        current=panel.iloc[rows[i,:5]]
        np.testing.assert_array_equal(current.month_id,meta.month_id.iloc[i]+np.array([-48,-36,-24,-12,0]))
        assert current.gvkey.eq(meta.gvkey.iloc[i]).all()
        assert current.available.le(current.date).all()
        np.testing.assert_allclose(x[i,:,t],(current.ebit/meta.market_cap.iloc[i]-scaling['mean'][t])/scaling['std'][t],rtol=2e-5,atol=2e-6)
        if rows[i,5]>=0:
            target=panel.iloc[rows[i,5]]
            assert target.month_id==meta.month_id.iloc[i]+12
            assert target.gvkey==meta.gvkey.iloc[i]
            np.testing.assert_allclose(y[i,-1,t],(target.ebit/meta.market_cap.iloc[i]-scaling['mean'][t])/scaling['std'][t],rtol=2e-5,atol=2e-6)
        else:assert np.isnan(y[i,-1]).all()


def test_masked_loss_ignores_missing_targets():
    import tensorflow as tf
    from train import loss_value
    truth=tf.constant([[[1.,float('nan')],[3.,4.]]])
    mean=tf.constant([[[2.,100.],[5.,7.]]])
    actual=float(loss_value(truth,mean,None,True,0))
    expected=.5*4+.5*(.7*((4+9)/2)+.3*((1+4+9)/3))
    np.testing.assert_allclose(actual,expected,rtol=1e-6)


def test_uq_inference_changes_dropout_but_preserves_batch_statistics():
    import tensorflow as tf
    from train import network
    for kind in ['uq_mlp','uq_lstm']:
        tf.keras.backend.clear_session()
        model=network(kind,6,3)
        a=tf.random.normal((16,5,6))
        model(a,training=True)
        bn=next(layer for layer in model.layers if isinstance(layer,tf.keras.layers.BatchNormalization))
        before=[v.numpy().copy() for v in bn.non_trainable_weights]
        mean1,var1=model(a,training=False);mean2,_=model(a,training=False)
        assert np.isfinite(var1.numpy()).all() and (var1.numpy()>0).all()
        assert not np.allclose(mean1.numpy(),mean2.numpy())
        for old,new in zip(before,bn.non_trainable_weights):np.testing.assert_array_equal(old,new.numpy())
