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

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("<output directory>")
        sys.exit()

    with open(os.path.join(sys.argv[1], 'config.json')) as infile:
        config = json.load(infile)

    # feature extraction
    df_train, df_test = get_feature(config)

    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train = df_train[:]
    X_test = df_test[:]

    #model
    start_time = time.time()
    mf = ModelFactory()
    y_pred = mf.create_model(config).predict(X_train, y_train, X_test)
    print("--- Fit Model: %s minutes ---" % round(((time.time() - start_time)/60), 2))
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[1],'submission.csv'),index=False)
