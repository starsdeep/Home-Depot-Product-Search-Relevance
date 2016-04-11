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
    frames = [pd.read_csv(file, encoding="ISO-8859-1", index_col=0) for file in train_pred_files]
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
    df_train_pred = load_train_pred(config['train_pred_files'])
    df = pd.concat((df_features, df_train_pred), axis=1)

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
