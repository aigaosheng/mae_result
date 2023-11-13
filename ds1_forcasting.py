#*-* coding: utf-8 *-*
#!/usr/bin/env python3
import numpy as np
import math
import pickle
import xgboost as xgb
import datetime
import pandas as pd
from hyperopt import tpe, hp, fmin, Trials
import copy
import math

class learnerRegressor():
    def __init__(self):
        pass


    def error_metric(self, y1, y2):
        '''
        Calculate accuracy
        '''
        res = list(filter(lambda x: math.isfinite(x[0]) and math.isfinite(x[1]), zip(y1, y2)))
        correct = 0
        total = 0
        for v in res:
            correct += (v[0]-v[1]) * (v[0]-v[1])

        return math.sqrt(correct/float(len(res)))


    def fit(self, train_data, dev_data, save_model):
        '''
        Desc: tune ML model

        train_dev_eval_set: input train/dev/test data set

        '''
        tuning_report = {}
        target_price = "target"

        #step-1: model training
        label_cols = ["target"]
        feat_cols = list(filter(lambda x: x not in (target_price,), train_data.columns))

        x_train = train_data[feat_cols].to_numpy()
        y_train = train_data[label_cols].to_numpy()#.astype(int)

        x_dev = dev_data[feat_cols].to_numpy()
        y_dev = dev_data[label_cols].to_numpy()#.astype(int)

        eval_metric_me = 'mae' #eval_metric_me
        base_xgb_cfg = {
            'n_estimators': 20,
            'objective' : 'reg:squarederror',#'reg:squarederror',
            # 'booster' : 'gbtree', #'gbtree'
            'eta': 0.02,
            'reg_lambda' : 10,
        #     reg_alpha = 10,
            'max_depth': 6,
            'verbosity':0,
            'eval_metric' : eval_metric_me,# eval_metric_me, #'mlogloss',
            'nthread': 3,
            'subsample': 1.0,
            'base_score': 0,
            'tree_method': 'hist',

        }

        def model_cost(model_cfg):
            xgb_cfg = copy.deepcopy(base_xgb_cfg)
            xgb_cfg.update(
                {
                'n_estimators': model_cfg['n_estimators'],
                'eta': model_cfg['eta'],
                'reg_lambda' : model_cfg['reg_lambda'],
                'max_depth': model_cfg['max_depth'],
            }
            )
            # print(xgb_cfg)
            model_now = xgb.XGBRFRegressor(**xgb_cfg)
            model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)], verbose=True)
            dev_result = model_now.evals_result()
            v = dev_result['validation_0'][eval_metric_me][0]
            return v

        search_space = {
            'n_estimators': hp.randint('n_estimators', 2, 50),
            'eta': hp.uniform('eta', 0.1, 1),
            'max_depth': hp.randint('max_depth', 2, 8),
            'reg_lambda': hp.randint('reg_lambda', 1, 5),
        }
        best = fmin(
            fn = model_cost,
            space = search_space,
            algo = tpe.suggest,
            max_evals = 10
        )
        print(best)
        # best = base_xgb_cfg
        
        xgb_cfg = copy.deepcopy(base_xgb_cfg)
        xgb_cfg.update(
            {
            'n_estimators': best['n_estimators'],
            'eta': best['eta'],
            'reg_lambda' : best['reg_lambda'],
            'max_depth': best['max_depth'],
        }
        )
        print(xgb_cfg)

        model_now = xgb.XGBRFRegressor(**xgb_cfg)
        model_now.fit(x_train, y_train, eval_set = [(x_dev, y_dev)], verbose=True)
        dev_result=model_now.evals_result()
        a = dev_result['validation_0'][eval_metric_me][0]
        # print(a)
        model_now.save_model(save_model)
        
        # #report
        tuning_report['model'] = 'xgb'
        tuning_report['param'] = xgb_cfg

        #evaluate model
        model_now.load_model(save_model)
        y_train_pred = model_now.predict(x_train)
        #map to price
        lst_train = list(train_data[target_price])
        lst_train_true = list(train_data[target_price].apply(math.exp)) #next day
        y_train_pred2 = list(map(lambda x: math.exp(x), y_train_pred))
        #
        acc = self.error_metric(y_train_pred2, lst_train_true) #predicted price error
        # acc = self.error_metric(y_train_pred, lst_train) #predicted price error
        print(f'train = {acc}')
        tuning_report['metric'] = {'train': acc}   

        y_dev_pred = model_now.predict(x_dev)
        lst_dev_true = list(dev_data[target_price].apply(math.exp))
        lst_dev = list(dev_data[target_price])
        y_dev_pred2 = list(map(lambda x: math.exp(x), y_dev_pred))

        dev_acc = self.error_metric(y_dev_pred2, lst_dev_true)
        # dev_acc = self.error_metric(y_dev_pred, lst_dev)
        print(f'dev = {dev_acc},')
        tuning_report['metric'].update({'dev': dev_acc})
        
        dev1 = list(zip(y_dev_pred, y_dev))
        
        return tuning_report,dev1,list(zip(y_train_pred2, lst_train)),list(zip(y_dev_pred2, lst_dev))      


    def predict(self, eval_data, o_prediction_file, save_model = './xgb_model_forcast.json'):
        '''
        Desc: tune ML model

        '''

        target_price = 'target'
        feat_cols = list(filter(lambda x: x not in (target_price,), eval_data.columns))

        x_eval = eval_data[feat_cols].to_numpy()
        y_eval = eval_data[[target_price]].to_numpy()#.astype(int)

        #evaluate model
        model_now = xgb.XGBRFRegressor()
        model_now.load_model(save_model)
        
        y_eval_pred = model_now.predict(x_eval)
        
        lst_eval = list(eval_data[target_price])  #today price 

        #evaluate set
        y_eval_pred_df = pd.DataFrame(columns=["predict", "target"])
        y_eval_pred_df['target'] = eval_data[target_price].apply(math.exp).round(2)
        y_eval_pred_df['predict'] = y_eval_pred

        y_eval_pred_df['predict'] = y_eval_pred_df['predict'].apply(math.exp).round(2)
        #Save prediction 
        y_eval_pred_df.to_csv(o_prediction_file, index = False)
        
 
def dataLoader(i_data_file, target_bond_name, non_data_id, extra_signal = {}):
    i_df = pd.read_excel(i_data_file, sheet_name=None)
    data_used = i_df[target_bond_name].drop(non_data_id).dropna()
    
    return data_used

def price_checker(v):
    try:
        return float(v)
    except:
        return None
    
def prepareFeature(i_sq, lookback = 3):
    #list(map(lambda x: f"fr{x}", range(lookback))) + 
    feat_cols =  list(map(lambda x: f"fr{x}", range(lookback))) + list(map(lambda x: f"f{x}", range(lookback + 1))) + ["target"]
    # feat_cols =   list(map(lambda x: f"f{x}", range(lookback + 1))) + ["target"]
    feat = pd.DataFrame(columns=feat_cols)

    feat["target"] = i_sq.apply(math.log)

    col_now = f"f0"
    feat[col_now] = feat["target"].shift(1)
    for k in range(1, lookback + 1):
        col_now = f"f{k}"
        feat_dif = feat[f"f{k-1}"] - feat[f"f{k-1}"].shift(1)
        feat[col_now] = feat_dif

        feat[f"fr{k-1}"] = feat["target"].shift(k + 1)

    feat = feat.dropna()
    return feat


if __name__ == '__main__':

    #Load raw excel data, set forcasting target bond
    data_file_name = "./DS1-assessment-RMD-UST-Yield-Data.xlsx"
    target_bond_name = "DGS1"
    non_data_id = range(10)
    #extra signal data as feature which may help improve forecasting accuracy. but not used now
    extra_signal = {}
    try:
        with open('./data.pkl', 'rb') as i_pt:
            i_df = pickle.load(i_pt)
    except:
        i_df = dataLoader(data_file_name, target_bond_name, non_data_id, extra_signal)

    with open("./data.pkl", "wb") as o_pt:
        pickle.dump(i_df, o_pt)

    print(i_df.columns)
    #extract signal feature
    target_price = i_df.columns[1]
    df_feat = i_df[target_price].apply(price_checker).dropna()
    df_feat = prepareFeature(df_feat, lookback=5)

    #data split for train-dev-test
    n_sample = df_feat.shape[0]
    split_sz = (int(0.8 * n_sample), int(0.1 * n_sample))
    df_train, df_dev, df_test = df_feat.iloc[:split_sz[0]], df_feat.iloc[split_sz[0]: split_sz[0] + split_sz[1]], df_feat[split_sz[0] + split_sz[1]:]


    predictor = learnerRegressor()

    save_model = f'bond_{target_bond_name}_model_forcast.json'
    #
    predictor.fit(df_train, df_dev, save_model)

    #Predict on evaluation set
    o_pred_file = f'bond_{target_bond_name}_pred_result.csv'
    predictor.predict(df_test, o_pred_file, save_model)
    
