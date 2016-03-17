#from sklearn import pipeline, model_selection
#from sklearn.feature_extraction import DictVectorizer
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
#import enchant
import pandas as pd
import random
random.seed(2016)
from load_data import load_data
from feature import build_feature
from model import model_predict
import sys, os
import json
import time

if __name__ =='__main__':
    if len(sys.argv) != 3:
        print("<train_size><output directory>")
        sys.exit()

    train_size = int(sys.argv[1])
    with open(os.path.join(sys.argv[2], 'config.json')) as infile:
        config = json.load(infile)

    #data
    df, num_train, num_test = load_data(train_size)

    #feature extraction
    start_time = time.time()
    df_with_feature = build_feature(df, config['features'])
    print("--- Build Features: %s minutes ---" % round(((time.time() - start_time)/60),2))
    if config['save_feature']:
        df.to_csv(os.path.join(sys.argv[2], 'df.csv'), encoding="utf8")
    if config['load_exist_feature']:
        df = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
    df_train = df.iloc[:num_train]
    df_test = df.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train =df_train[:]
    X_test = df_test[:]

    #model
    start_time = time.time()
    y_pred = model_predict(config, X_train, y_train, X_test)
    print("--- Fit Model: %s minutes ---" % round(((time.time() - start_time)/60),2))
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(os.path.join(sys.argv[2],'submission.csv'),index=False)
