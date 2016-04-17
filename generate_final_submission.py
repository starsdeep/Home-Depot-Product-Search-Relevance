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
    # fit and predict
    model.model.fit(X_train, y_train)
    y_pred = model.model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(output_path,'submission%d.csv' % trial_number),index=False)



if __name__ == '__main__':

   # if len(sys.argv) != 3:
   #     print("<output_path><num_of_enseble>")
   #     sys.exit()
   # 
   # output_path = sys.argv[1]
   # num_of_ensemble = sys.argv[2]
    output_path = './output/ensemble_selection_avg'
    num_of_ensemble = 3
    
    with open(os.path.join(output_path, 'ensemble_avg_model_list.json')) as outfile:
        trial_result_list = json.load(outfile)
    
    df_all, dummpy1, dummy2 = load_data(-1)
    df_train, df_test = df_all[:num_train], df_all[-num_test:]
    y_train = df_train['relevance'].values
    id_test = df_test['id']

    min_index, min_value = min(enumerate(trial_result_list), key=lambda k:k[1]['loss'])
    print("min value index is: %f, your choice is %f" % (min_index, num_of_ensemble))
    trial_result_list = trial_result_list[:num_of_ensemble]
    
    """
    for idx, trial in enumerate(trial_result_list):
        print("============== trial %d ==============" % idx)
        train_single_model(trial, idx, y_train)
    """
    final_submission = np.zeros(num_test)
    for idx in range(len(trial_result_list)):
        final_submission += pd.read_csv(os.path.join(output_path, 'submission%d.csv' % idx), encoding="ISO-8859-1")['relevance'].values
    final_submission /= len(trial_result_list)
    pd.DataFrame({"id": id_test, "relevance": final_submission}).to_csv(os.path.join(output_path,'final_submission.csv'),index=False)   
    


    


    
