import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(2016)
from load_data import load_data
from feature import build_feature
import sys, os
import json
import time

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)

class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','search_term_fuzzy_match','product_title','title','product_description','description','tmp_compound_field','brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)



if __name__ =='__main__':
    if len(sys.argv) != 3:
        print("<train_size><output directory>")
        sys.exit()

    train_size = int(sys.argv[1])
    with open(os.path.join(sys.argv[2], 'config.json')) as infile:
        config = json.load(infile)
    features = config['features']

    #data
    df, num_train, num_test = load_data(train_size)

    #feature extraction
    start_time = time.time()
    df_with_feature = build_feature(df, features)
    print("--- Build Features: %s minutes ---" % round(((time.time() - start_time)/60),2))
    df.to_csv(file_name=os.path.join(sys.argv[2], 'df.csv'), encoding="ISO-8859-1")
    #df = pd.read_csv('df_all.csv', encoding="ISO-8859-1", index_col=0)
    df_train = df.iloc[:num_train]
    df_test = df.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train =df_train[:]
    X_test = df_test[:]

    #model
    start_time = time.time()
    rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term_fuzzy_match')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.0,
                            'txt4': 0.5
                            },
                    n_jobs = -1
                    )),
            ('rfr', rfr)])
    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(X_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    print("--- Fit Model: %s minutes ---" % round(((time.time() - start_time)/60),2))

    y_pred = model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)