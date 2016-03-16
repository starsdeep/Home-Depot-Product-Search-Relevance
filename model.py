import os, sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
import xgboost as xgb



def model_predict(model, x_train, y_train, x_test):
    predict_func = ModelDict[model]
def model_predict(config, x_train, y_train, x_test):
    predict_func = ModelDict[config['model']]
    prediction = predict_func(x_train, y_train, x_test)
    return prediction

def fmean_squared_error_(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

def print_badcase_(x_train, y_train, model):
    train_pred = cross_validation.cross_val_predict(model, x_train, y_train, cv=3)
    output = x_train.copy(deep=True)
    output.insert(3, 'pred', pd.Series(train_pred, index=x_train.index))
    output.insert(3, 'diff', pd.Series(abs(train_pred-y_train), index=x_train.index))
    output = output.sort_values(by=['diff'], ascending=False)
    output[:1000].to_csv(os.path.join(os.path.abspath(sys.argv[2]),'badcase.csv'), encoding="utf-8")

RMSE = make_scorer(fmean_squared_error_, greater_is_better=False)

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

    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    RMSE = make_scorer(fmean_squared_error_, greater_is_better=False)

    # grid search cv is done in fitting, so set a param to print badcase
    print_badcase_(x_train, y_train, clf.set_params(rfr__max_features=10, rfr__max_depth=20))
    print("Badcase printing done.")

    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(x_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    y_pred = model.predict(x_test)
    return y_pred

def transform_labels_(y_pred):
    labels=[1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3]
    for i in range(len(y_pred)):
        for j in range(len(labels)):
            if y_pred[i]==labels[j]:
                y_pred[i] = j
                break
    return y_pred

def recover_labels_(y_pred):
    labels=[1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3]
    for i in range(len(y_pred)):
        y_pred[i] = labels[int(y_pred[i])]
    return y_pred

def random_forest_classification_(x_train, y_train, x_test):
    y_train = transform_labels_(y_train)
    rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
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
            ('rfc', rfc)])
    # grid search cv is done in fitting, so set a param to print badcase
    print_badcase_(x_train, y_train, clf.set_params(rfc__max_features=10, rfc__max_depth=20))
    print("Badcase printing done.")

    param_grid = {'rfc__max_features': [10], 'rfc__max_depth': [20]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(x_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    y_pred = recover_labels_(model.predict(x_test))
    return y_pred

def xgboost_regression_(x_train, y_train, x_test):
    xgb_model = xgb.XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state = 2016)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list = [
                            ('cst',  cust_regression_vals_()),  
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col_(key='search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col_(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col_(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
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
            ('xgb_model', xgb_model)])
    # grid search cv is done in fitting, so set a param to print badcase
    print_badcase_(x_train, y_train, clf.set_params(xgb_model__max_depth=5, rfr__xgb_model__n_estimators=10))
    print("Badcase printing done.")

    param_grid = {'xgb_model__max_depth': [5], 'xgb_model__n_estimators': [10]}
    model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -1, cv = 2, verbose = 20, scoring=RMSE)
    model.fit(x_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)
    #print(model.best_score_ + 0.47003199274) # why original script do that?

    y_pred = model.predict(x_test)
    for i in range(len(y_pred)):
        if y_pred[i]<1.0:
            y_pred[i] = 1.0
        if y_pred[i]>3.0:
            y_pred[i] = 3.0
    return y_pred


def gbdt_regression_(x_train, y_train, x_test):
    #rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)

    gbdtr = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=2016)

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

    param_grid = {'gbdt__n_estimators': (100,), 'gbdt__learning_rate': (0.1, 0.5), 'gbdt__max_features': (3, 10, 20), 'gbdt__max_depth': (5,15,30)}
    RMSE = make_scorer(fmean_squared_error_, greater_is_better=False)

    model = grid_search.GridSearchCV(estimator = gbdtr, param_grid = param_grid, n_jobs = 1, cv = 3, verbose = 20, scoring=RMSE)
    model.fit(x_train, y_train)

    print("Best parameters found by grid search:")
    print(model.best_params_)
    print("Best CV score:")
    print(model.best_score_)

    y_pred = model.predict(x_test)
    return y_pred




ModelDict = {
    'rfr': random_forest_regression_,
    'rfc': random_forest_classification_,
    'xgbr': xgboost_regression_,
    'gbdtr': gbdt_regression_,
}

