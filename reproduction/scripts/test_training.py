from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import tensorflow as tf
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime/scripts'))
from train import Train


def test_batchnorm_updates_during_training():
    # Exercise the release's actual gradient-step method with a small model.
    obj = object.__new__(Train)
    obj.config = SimpleNamespace(UQ=False,forecast_steps=1,max_grad_norm=1)
    obj.model = tf.keras.Sequential([tf.keras.layers.Input((5,2)),tf.keras.layers.BatchNormalization(),tf.keras.layers.Dense(2)])
    obj.optimizer = tf.keras.optimizers.SGD(.01)
    class Loss:
        def weight_adjusted_mse(self, ys, ps):
            value=tf.reduce_mean(tf.square(ys[0]-ps[0]))
            return value,value
    obj.losses=Loss()
    bn=obj.model.layers[0]
    before=bn.moving_mean.numpy().copy()
    obj._train_step_point(tf.ones((4,5,2))*3,tf.ones((4,5,2)))
    assert not np.array_equal(before,bn.moving_mean.numpy())


def test_early_stop_counts_completed_nonimproving_epochs():
    obj=object.__new__(Train);obj.config=SimpleNamespace(early_stop=25)
    assert not obj._stop_criteria({'valid_mse':[1]+[2]*24})
    assert obj._stop_criteria({'valid_mse':[1]+[2]*25})
    assert not obj._stop_criteria({'valid_mse':[1]+[2]*25+[.5]})


def test_ensemble_total_variance_uses_mean_predictions_not_variances():
    import pandas as pd
    from utils.post_processing import PredictionsPostProcessing
    a=pd.DataFrame(dict(date=[20000101],gvkey=['000001'],norm_preds_1=[1.],preds_1=[1.],norm_variance_1=[4.],variance_1=[4.],norm_targets_1=[2.],targets_1=[2.],seq_norm=[1.]))
    b=a.copy(); b[['norm_preds_1','preds_1']]=3.
    before=a.copy(deep=True)
    processor=PredictionsPostProcessing([a,b],SimpleNamespace(UQ=True,forecast_steps=1))
    result=processor._get_mean_df()
    assert result.preds_1.iloc[0]==2
    assert result.variance_1.iloc[0]==5 # E[conditional variance] + Var[conditional mean]
    pd.testing.assert_frame_equal(a,before)
