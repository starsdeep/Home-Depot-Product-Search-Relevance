import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion

def model_predict(model, x_train, y_train, x_test):
    predict_func = ModelDict[model]
    prediction = predict_func(x_train, y_train, x_test)
    return prediction

def fmean_squared_error_(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

class cust_regression_vals_(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','search_term_fuzzy_match','product_title','title','product_description','description','tmp_compound_field','brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches

class cust_txt_col_(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

def random_forest_regression_(x_train, y_train, x_test):
    rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals_()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col_(key='search_term_fuzzy_match')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col_(key='title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col_(key='description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col_(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)]))
                            ],
                        transformer_weights = {
                            'cst': 1.0,
                            'txt1': 0.5,
                            'txt2': 0.25,
                            'txt3': 0.0,
                            'txt4': 0.5
                            },
                    #n_jobs = -1
                    )),
            ('rfr', rfr)])
    param_grid = {'rfr__max_features': (3, 5, 10, 20), 'rfr__max_depth': (15, 20, 30)}
    RMSE = make_scorer(fmean_squared_error_, greater_is_better=False)
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(x_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    y_pred = model.predict(x_test)
    return y_pred

def xgboost_regression_(x_train, y_train):
    pass

ModelDict = {
    'rfr': random_forest_regression_,
    'xgbr': xgboost_regression_,
}

