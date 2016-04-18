import pandas as pd
import numpy as np
from feature import get_feature
from model_factory import ModelFactory
import sys, os
import json
from load_data import load_data
    

feature_set_dict = dict()
num_train = 74067
num_test = 166693

def train_single_model(trial, trial_number, y_train):
    # get model
    with open(os.path.join(os.path.abspath(trial['model_dir']), 'config.json')) as infile:
        config = json.load(infile)
    mf = ModelFactory()
    model = mf.create_model(config) 
    params = trial['params']
    for k, v in params.items():
        model.model = model.model.set_params(**{k:v})
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
    X_all, column_names = model.feature_union(df_all)
    X_train = X_all[:num_train]
    X_test = X_all[-num_test:]
    
    # random feature selection, make sure X_train here is the same as the X_train when building model library
    if 'feature_index' not in trial or trial['feature_index'] == 'all':
        new_X_train = X_train
        new_X_test = X_test
    else:
        feature_index = [int(idx) for idx in trial['feature_index'].split(" ")]
        new_X_train = X_train[:, feature_index]
        new_X_test = X_test[:, feature_index]
    # fit and predict
    model.model.fit(new_X_train, y_train)
    y_pred = model.model.predict(new_X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(output_path,'submission%d.csv' % trial_number),index=False)


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("<output_path><num_of_enseble>")
        sys.exit()
    
    output_path = sys.argv[1]
    num_of_ensemble = int(sys.argv[2])
    

    with open(os.path.join(output_path, 'ensemble_avg_model_list.json')) as outfile:
        trial_result_list = json.load(outfile)
    with open(os.path.join(output_path, 'ensemble_avg_rmse_list.json')) as outfile:
        rmse_list = json.load(outfile)

    min_index, min_value = min(enumerate(rmse_list), key=lambda k:k[1])
    print("min value index is: %d, your choice is %d" % (min_index, num_of_ensemble))
    #sys.exit(1)
    
    df_all, dummpy1, dummy2 = load_data(-1)
    df_train, df_test = df_all[:num_train], df_all[-num_test:]
    y_train = df_train['relevance'].values
    id_test = df_test['id']
    """
    for idx in range(0,num_of_ensemble): 
        trial = trial_result_list[idx]
        print("============== trial %d, model %s, model_loss %f ==============" % (idx, trial['model'], trial['loss']))
        train_single_model(trial, idx, y_train)
    """
    final_submission = np.zeros(num_test)
    for idx in range(num_of_ensemble):
        final_submission += pd.read_csv(os.path.join(output_path, 'submission%d.csv' % idx), encoding="ISO-8859-1")['relevance'].values
    final_submission /= num_of_ensemble
    pd.DataFrame({"id": id_test, "relevance": final_submission}).to_csv(os.path.join(output_path,'final_submission.csv'),index=False)   
    


    


    
