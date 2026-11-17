
import xgboost as xgb
import pandas as pd

def train_xgboost_model(df_train, df_valid, df_test):
    # Fill NaN values with 0
    df_train_filled = df_train.fillna(0)
    df_valid_filled = df_valid.fillna(0)
    df_test_filled = df_test.fillna(0)

    # Separate features and labels
    X_train = df_train_filled.drop(columns=['label'])
    y_train = df_train_filled['label']
    X_valid = df_valid_filled.drop(columns=['label'])
    y_valid = df_valid_filled['label']
    X_test = df_test_filled.drop(columns=['label'])
    y_test = df_test_filled['label']

    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Set hyperparameters for XGBoost model
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'colsample_bytree': 0.8879,
        'learning_rate': 0.01,
        'subsample': 0.8789,
        'lambda': 205.6999,
        'alpha': 580.9768,
        'max_depth': 4,
        'num_boost_round': 100,
        'early_stopping_rounds': 100,
        'verbose_eval': 100
    }

    # Train the XGBoost model
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dvalid, 'valid')], early_stopping_rounds=params['early_stopping_rounds'], verbose_eval=params['verbose_eval'])

    # Convert test data to DMatrix format
    dtest = xgb.DMatrix(X_test)

    # Make predictions on the test set
    y_pred = model.predict(dtest)

    # Combine the predictions with the actual labels
    # pred = pd.concat([df_test['label'], pd.Series(y_pred, index=df_test.index)], axis=1)
    pred = pd.concat([pd.Series(y_pred,index=df_test.index),df_test['label']],axis=1)

    return model, pred



for instruments in ['sp500']:
    for train_end in range(2016,2017):
        returned = get_data_by_year(
            train_start = 2010,train_end=train_end,valid_year=train_end+1,test_year =train_end+2,
            instruments=instruments, target=target,freq='day',
            qlib_path='/your_path/data/qlib_data/us_data'
            )
        data_all, data,data_valid,data_valid_withhead,data_test,data_test_withhead,_ = returned
        df_train = get_ml_data(data)
        df_valid = get_ml_data(data_valid)
        df_test = get_ml_data(data_test)
        df_train, df_valid, df_test = normalize_data(df_train, df_valid, df_test)
        
        model_name = 'xgb'
        name = f"{instruments}_{model_name}_{train_end}"
        os.makedirs(f"out_ml/{name}",exist_ok=True)
        model,pred = train_xgboost_model(df_train, df_valid, df_test)
        model.save_model(f"out_ml/{name}/{model_name}.pt")
        pred.to_pickle(f"out_ml/{name}/pred.pkl")