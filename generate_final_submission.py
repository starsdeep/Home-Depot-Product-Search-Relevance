import pandas as pd
import numpy as np
from feature import get_feature
from model_factory import ModelFactory
import sys, os
import json
from load_data import load_data    
from trials_helper import *
from base_model import fmean_squared_error_

feature_set_dict = dict()
num_train = 74067
num_test = 166693

def train_model(trial, trial_number):
    # get model
    with open(os.path.join(os.path.abspath(trial['path']), 'config.json')) as infile:
        config = json.load(infile)
    mf = ModelFactory()
    model = mf.create_model(config) 
    params = trial['params']
    for k, v in params.items():
        model = model.set_params(**{k:v})
    model_name = trial['model']
    # load feature    
    features_str = ' '.join(set(config['features']))
    if features_str not in feature_set_dict:
        df_all, dummy1, dummy2 = get_feature(config)
        feature_set_dict[features_str] = df_all
    else:
        df_all = feature_set_dict[features_str]
    df_train, df_test = df_all[:num_train], df_all[-num_test:]
    id_test = df_test['relevance'].values
    X_all, column_names = feature_union(df_all, model_name)
    X_train = X_all[:num_train]
    X_test = X_all[-num_test:]
    # fit and predict
    model.model.fit(X_train, y_train)
    y_pred = model.model.predict()
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(output_path,'submission%d.csv' % trial_number),index=False)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("<output_path><num_of_enseble>")
        sys.exit()
   
    output_path = sys.argv[1]
    num_of_ensemble = int(sys.argv[2])
    
    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)
    
    with open(os.path.join(output_path, 'ensemble_avg_index_list.json')) as outfile:
        index_list = json.load(outfile)
    
    
    df_all, num_train, num_test = load_data(-1)
    df_train, df_test = df_all[:num_train], df_all[-num_test:]    
    id_test = df_test['id']
    y_train = df_train['relevance']
    
    num_of_ensemble = min(num_of_ensemble, len(index_list))
    index_list = index_list[:num_of_ensemble] 
    
    test_pred_list = []
    train_pred_list = []
    for dir_path in config['model_library_path_list']:
        tmp_test_pred = load_test_pred_list(dir_path)
        tmp_train_pred = load_train_pred_list(dir_path)
        test_pred_list += tmp_test_pred
        train_pred_list += tmp_train_pred
        print("loaded dir %s done, model library size %d" % (dir_path, len(tmp_test_pred)))
        
    final_submission = np.zeros(num_test)
    avg_train_pred = np.zeros(num_train)
    
    i = 0
    for idx in index_list:
        avg_train_pred = avg_train_pred * (i/(i+1.0)) +  train_pred_list[idx] / (i + 1.0)
        final_submission += test_pred_list[idx] 
        print("idx %d, offline rmse %f" % (idx, fmean_squared_error_(y_train, avg_train_pred)))
        i += 1
    
    final_submission /= len(index_list)
    pd.DataFrame({"id": id_test, "relevance": final_submission}).to_csv(os.path.join(output_path,'final_submission.csv'),index=False)   
    


    


    
