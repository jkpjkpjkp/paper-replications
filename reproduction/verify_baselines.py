"""Independent baseline integration checks on synthetic data; no financial training run."""
import json
from pathlib import Path
import tempfile
import types

from common import UPSTREAM
from train_xgboost import build_features, normalize_features, normalize_train_labels
import numpy as np
import pandas as pd
import torch
import xgboost as xgb


def synthetic_data(values):
    return types.SimpleNamespace(data=torch.tensor(values),max_backtrack_days=100,max_future_days=30,
                                 _dates=pd.bdate_range('2010-01-01',periods=len(values)))


def test_features_and_normalization(tmp):
    rng=np.random.default_rng(987)
    values=np.cumprod(1+rng.normal(0,.01,(180,6,4)),axis=0).astype(np.float32)
    values[85,1,1]=np.nan
    data=synthetic_data(values)
    target=rng.normal(size=(50,4))
    low,high=str(data._dates[100].date()),str(data._dates[-31].date())
    x,labels,rows,columns,dates=build_features(data,tmp/'features.npy',target,low,high)
    raw=values[:,:5,:].transpose(0,2,1)
    frame=pd.DataFrame(raw.reshape(-1,5),index=pd.MultiIndex.from_product([data._dates,range(4)],names=['datetime','instrument']))
    frame.columns=['open','close','high','low','volume']
    feature=frame.unstack()
    feature=feature/feature.shift(1)-1
    blocks=[]
    for lag in range(1,60):
        cur=feature.shift(lag).stack(future_stack=True)
        cur.columns=[f'{column}{lag}' for column in cur.columns]
        blocks.append(cur)
    expected=pd.concat(blocks,axis=1).reindex(pd.MultiIndex.from_arrays([dates[rows],columns],names=['datetime','instrument'])).to_numpy()
    np.testing.assert_allclose(x,expected,equal_nan=True,rtol=1e-5,atol=1e-7)
    np.testing.assert_array_equal(labels,target[rows,columns])
    # Changing observations at and after a forecast date cannot affect that date's lagged features.
    changed=values.copy()
    changed[115:]*=100.
    changed_x,_,changed_rows,changed_columns,_=build_features(synthetic_data(changed),tmp/'changed.npy',target,low,high)
    before=rows<=15
    np.testing.assert_array_equal(rows,changed_rows)
    np.testing.assert_array_equal(columns,changed_columns)
    np.testing.assert_allclose(x[before],changed_x[before],equal_nan=True)
    expected_mean=pd.DataFrame(x).mean().to_numpy()
    expected_std=pd.DataFrame(x).std().to_numpy()
    expected_normalized=np.clip(np.nan_to_num((expected-expected_mean)/expected_std,nan=0),-4,4)
    mean,std=normalize_features(x)
    np.testing.assert_allclose(x,expected_normalized,equal_nan=True,rtol=1e-5,atol=1e-6)
    # Validation uses supplied training statistics and cannot change them.
    original_mean,original_std=mean.copy(),std.copy()
    normalize_features(changed_x,mean,std)
    np.testing.assert_array_equal(mean,original_mean)
    np.testing.assert_array_equal(std,original_std)
    label_series=pd.Series(labels).fillna(0)
    expected_labels=label_series.groupby(rows).transform(lambda s:(s-s.mean())/s.std()).clip(-4,4).to_numpy()
    np.testing.assert_allclose(normalize_train_labels(labels,rows),expected_labels,rtol=1e-6,atol=1e-6)
    return dict(feature_columns=295,feature_rows=len(x),lag_features_match_notebook=True,
                contemporaneous_and_future_observations_do_not_affect_features=True,
                normalization_matches_notebook=True,validation_cannot_change_training_normalization=True,
                training_label_standardization_matches_notebook=True)


def test_gp_compatibility():
    module=types.ModuleType('gp_verification')
    source=(UPSTREAM/'train_GP.py').read_text()
    exec(compile(source,str(UPSTREAM/'train_GP.py'),'exec'),module.__dict__)
    functions=[module.make_function(**function._asdict()) for function in module.generic_funcs]
    def metric(y_true,expressions,weight):
        return float(len(expressions[0]))/10000.
    estimator=module.SymbolicRegressor(population_size=8,generations=2,init_depth=(1,2),
        tournament_size=4,stopping_criteria=1.,p_crossover=.3,p_subtree_mutation=.1,
        p_hoist_mutation=.01,p_point_mutation=.1,p_point_replace=.6,max_samples=.9,
        parsimony_coefficient=0.,random_state=0,function_set=functions,
        metric=module.make_fitness(function=metric,greater_is_better=True),const_range=None,n_jobs=1)
    terminals=np.array([['open_','close','high','low','volume','Constant(1.)']])
    callbacks=[]
    estimator.fit(terminals,np.array([[1]]),callback=lambda:callbacks.append(True))
    assert len(callbacks)==2
    result=estimator._program.execute(terminals)[0]
    expression=eval(result,module.__dict__)
    assert expression is not None
    return dict(upstream_gp_functions=len(functions),string_terminal_gp_evolution_compatible=True,
                generation_callback_count=len(callbacks),generated_expression_parses=True)


def test_nonfinite_normalization(tmp):
    values=np.array([[1,np.inf,2,1,np.nan,-np.inf],
                     [3,4,2,3,np.nan,2],
                     [np.nan,6,2,5,np.nan,np.inf]],dtype=np.float32)
    validation=np.array([[5,7,3,np.inf,1,3]],dtype=np.float32)
    train=np.lib.format.open_memmap(tmp/'nonfinite.npy',mode='w+',dtype=np.float32,shape=values.shape)
    train[:]=values
    valid=np.lib.format.open_memmap(tmp/'nonfinite-valid.npy',mode='w+',dtype=np.float32,shape=validation.shape)
    valid[:]=validation
    frame=pd.DataFrame(values)
    with np.errstate(invalid='ignore',divide='ignore'):
        mean_ref,std_ref=frame.mean().to_numpy(),frame.std().to_numpy()
        expected=((frame-mean_ref)/std_ref).fillna(0).clip(-4,4).to_numpy()
        expected_valid=((pd.DataFrame(validation)-mean_ref)/std_ref).fillna(0).clip(-4,4).to_numpy()
    mean,std=normalize_features(train)
    normalize_features(valid,mean,std)
    np.testing.assert_allclose(mean,mean_ref,equal_nan=True)
    np.testing.assert_allclose(std,std_ref,equal_nan=True,rtol=1e-6)
    np.testing.assert_allclose(train,expected,equal_nan=True,rtol=1e-6)
    np.testing.assert_allclose(valid,expected_valid,equal_nan=True,rtol=1e-6)
    return dict(training_infinities_match_pandas=True,zero_variance_matches_pandas=True,
                all_missing_training_column_matches_pandas=True,validation_infinities_clip_as_notebook=True)


def test_xgboost_prefix(tmp):
    rng=np.random.default_rng(19)
    x=rng.normal(size=(80,7)).astype(np.float32)
    y=(x[:,0]-.3*x[:,2]).astype(np.float32)
    matrix=xgb.DMatrix(x,label=y,nthread=1)
    params=dict(objective='reg:squarederror',max_depth=2,seed=0,nthread=1,tree_method='hist',subsample=.8789,colsample_bytree=.8879)
    complete=xgb.train(params,matrix,num_boost_round=12)
    literal=xgb.train(params,matrix,num_boost_round=10)
    prefix=complete.predict(matrix,iteration_range=(0,10))
    np.testing.assert_array_equal(prefix,literal.predict(matrix))
    complete[:10].save_model(tmp/'prefix.ubj')
    restored=xgb.Booster()
    restored.load_model(tmp/'prefix.ubj')
    np.testing.assert_array_equal(restored.predict(matrix),prefix)
    return dict(first_ten_rounds_match_independent_literal_notebook_training=True,
                saved_ten_round_model_roundtrip_matches=True,xgboost_version=xgb.__version__)


def test_missing_labels():
    values = normalize_train_labels(np.array([1., 3., np.nan, np.inf]), np.zeros(4))
    np.testing.assert_allclose(values[:2], [-1/np.sqrt(2), 1/np.sqrt(2)], rtol=1e-6)
    assert np.isnan(values[2:]).all()
    return dict(missing_outcomes_not_zero_filled=True, missing_outcomes_excluded_from_daily_label_scaling=True)


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        tmp=Path(directory)
        result=dict(status='passed',xgboost_preprocessing=test_features_and_normalization(tmp),
                    nonfinite_preprocessing=test_nonfinite_normalization(tmp),
                    genetic_programming=test_gp_compatibility(),xgboost_rounds=test_xgboost_prefix(tmp),
                    missing_labels=test_missing_labels())
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
