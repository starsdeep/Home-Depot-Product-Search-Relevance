#from sklearn import pipeline, model_selection
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
#import enchant
import pandas as pd
import json
import random
random.seed(2016)
from load_data import load_data
from trials_helper import load_trials
from feature import load_feature
import sys
import os
import time
from model_factory import ModelFactory
from param_space import param_space_dict 
from sklearn.base import clone
from sklearn import cross_validation
from base_model import fmean_squared_error_
import operator
num_train = 74067
num_test = 166693

def load_train_pred(train_pred_files):
    df = pd.concat(frames, axis=1)
    return df


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()

    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)
    
    #load y data
    row_df, dummy1, dummy2 = load_data(-1)
    y_train = row_df['relevance'][:num_train].values
    y_test = row_df['relevance'][-num_test:].values
    id_test = row_df['id'][-num_test:].values

    # feature extraction
    df_features = load_feature(config['features'])
    
    # load customize train preds
    model_names = []
    train_preds = []
    test_preds = []

    train_preds += [pd.read_csv(file, encoding="ISO-8859-1", index_col=0).values for file in config['train_pred_files']]
    test_preds += [pd.read_csv(file, encoding="ISO-8859-1", index_col=0).values for file in config['test_pred_files']]
    model_names += config['model_names']
    
    # load model library:
    for dir_path in config['model_library_path_list']:
        tmp_trial_results, tmp_train_preds, tmp_test_preds = load_trials(dir_path)
        train_preds += tmp_train_preds 
        test_preds += tmp_test_preds 
        model_names += ["%s_%d" % (trial['model'],idx)  for idx, trial in enumerate(tmp_trial_results)]
        rmses = [trial['loss'] for trial in tmp_trial_results]
        min_index, min_value = min(enumerate(rmses), key=operator.itemgetter(1))
        print("load dir %s done., model library size %d, best rmse %f" % (dir_path, len(tmp_trial_results), min_value))
        print("test rmse %f", fmean_squared_error_(y_train, train_preds[min_index]))
        break

    print("total train_preds %d, total test_preds %d, total model_names %d" % (len(train_preds), len(test_preds), len(model_names)))
    df_train_preds = pd.DataFrame()
    df_test_preds = pd.DataFrame()
    for i in range(len(model_names)):
        df_train_preds[model_names[i]] = train_preds[i]
        df_test_preds[model_names[i]] = test_preds[i]
    df_preds = pd.concat([df_train_preds, df_test_preds], axis=0)
    df = pd.concat([df_features, df_preds], axis=1)
    column_names = df.columns.values 
    
    X_train = df_train_preds[:num_train]
    X_test = df[-num_test:]

    # fit model
    start_time = time.time()
    mf = ModelFactory()
    model = mf.create_model(config)
    if config['param_space_name'] in param_space_dict:
        print("found %s in param_space_dict, use it." % config['param_space_name'])
        model.param_space = param_space_dict[config['param_space_name']]
    else:
        print("not found %s in param_space_dict, use default" % config['param_space_name'])
    model.column_names = column_names
    model.hyperopt_max_evals = config['hyperopt_max_evals']
    model.set_hyper_params_(X_train, y_train, X_test, save_result=False)
    tmp_model = clone(model.model)
    train_pred = cross_validation.cross_val_predict(tmp_model, X_train, y_train, cv=2)
    rmse = fmean_squared_error_(y_train, train_pred)
    print("\n[info]: offline rmse: %f\n" % rmse)
    model.model.fit(X_train, y_train)
    imps = model.get_column_importance_()
    model.print_importance_(imps, column_names)
    y_pred = model.predict(X_test) 
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[1],'submission.csv'),index=False)
