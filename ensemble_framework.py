#from sklearn import pipeline, model_selection
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
#import enchant
import hashlib
import pandas as pd
import random
random.seed(2016)
from load_data import load_data
from feature import get_feature
from feature import load_feature
import sys
import os
import json
import time
from model_factory import ModelFactory

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

    # feature extraction
    df_features = load_feature(config['features'])
    
    # load customize train preds
    model_names = []
    train_preds = []

    if 'train_pred_files':
        train_preds += [pd.read_csv(file, encoding="ISO-8859-1", index_col=0).values for file in config['train_pred_files']
        model_names += config['model_names']
    # load model library:
    Trials = TrialsList()
    for dir_path in config['model_library_path_list']:
        print("loading dir " + dir_path)
        trial_result_list, train_pred_list = load_trials(dir_path)
        train_preds.append(train_pred_list) 
        model_names += ["%s_%d" % (trial['model'],idx)  for idx, trial in enumerate(trial_result_list)]
        
    df_train_preds = pd.DataFrame(train_preds, columns=model_names)
    
    



    X_train = df[:num_train]
    y_train = X_train['relevance'].values
    X_test = df[-num_test:]
    id_test = X_test['id']

    #model
    start_time = time.time()
    mf = ModelFactory()
    y_pred = mf.create_model(config).predict(X_train, y_train, X_test)
    print("--- Fit Model: %s minutes ---" % round(((time.time() - start_time)/60), 2))
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[1],'submission.csv'),index=False)
