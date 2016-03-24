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

class CustRegressionVals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['id','relevance','search_term','search_term_fuzzy_match','product_title','title','main_title','product_description','description','brand']
        hd_searches = hd_searches.drop(d_col_drops, axis=1, errors='ignore').values
        return hd_searches

class CustTxtCol(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)

class Model(object):

    def __init__(self):
        self.RMSE = make_scorer(self.fmean_squared_error_, greater_is_better=False)
        self.config = dict()

    def set_config(self, config):
        self.config = config

    def predict(self, x_train, y_train, x_test):
        print('abstract method')

    def fmean_squared_error_(self, ground_truth, predictions):
        fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
        return fmean_squared_error_

    def print_badcase_(self, x_train, y_train, model, default_output_line=1000):
        train_pred = cross_validation.cross_val_predict(model, x_train, y_train, cv=3)
        output = x_train.copy(deep=True)
        #output.drop('product_description', axis=1, inplace=True)
        #output.drop('description', axis=1, inplace=True)
        #output.drop('product_title', axis=1, inplace=True)

        output.insert(3, 'pred', pd.Series(train_pred, index=x_train.index))
        output.insert(3, 'diff', pd.Series(abs(train_pred-y_train), index=x_train.index))
        output = output.sort_values(by=['diff', 'id'], ascending=False)
        len_output = len(output)
        output[:min(default_output_line, len_output)].to_csv(os.path.join(os.path.abspath(sys.argv[1]),'badcase.csv'), encoding="utf-8")

    def grid_search_fit_(self, clf, param_grid, x_train, y_train):
        model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=self.RMSE)
        model.fit(x_train, y_train)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        print()
        if self.config['save_badcase']:
            cvmodel = clf
            for k, v in model.best_params_.items():
                cvmodel = cvmodel.set_params(**{k: v})
            self.print_badcase_(x_train, y_train, cvmodel, 2000)
            print("Badcase printing done.\n")
        return model

    def make_pipeline_(self, model_name, model):
        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        tsvd = TruncatedSVD(n_components=10, random_state = 2016)
        clf = pipeline.Pipeline([
                ('union', FeatureUnion(
                            transformer_list = [
                                ('cst',  CustRegressionVals()),
                                ('txt1', pipeline.Pipeline([('s1', CustTxtCol(key='search_term_fuzzy_match')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                                ('txt2', pipeline.Pipeline([('s2', CustTxtCol(key='title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                                ('txt3', pipeline.Pipeline([('s3', CustTxtCol(key='description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                                ('txt4', pipeline.Pipeline([('s4', CustTxtCol(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                                ('txt5', pipeline.Pipeline([('s5', CustTxtCol(key='main_title')), ('tfidf5', tfidf), ('tsvd5', tsvd)]))
                                ],
                            transformer_weights = {
                                'cst': 1.0,
                                'txt1': 0.5,
                                'txt2': 0.0,
                                'txt3': 0.0,
                                'txt4': 0.5,
                                'txt5': 0.25 # split the 0.25 of txt2 get worse result
                                },
                        #n_jobs = -1
                        )),
                (model_name, model)])
        return clf

class RandomForestRegression(Model):

    def predict(self, x_train, y_train, x_test):
        rfr = RandomForestRegressor(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfr', rfr)
        param_grid = {'rfr__n_estimators': [900], 'rfr__max_features': [10], 'rfr__max_depth': [30]}
        model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
        return model.predict(x_test)

class RandomForestClassification(Model):

    def transform_labels_(self, y_pred):
        labels=[1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3]
        for i in range(len(y_pred)):
            for j in range(len(labels)):
                if y_pred[i]==labels[j]:
                    y_pred[i] = j
                    break
        return y_pred

    def recover_labels_(self, y_pred):
        labels=[1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3]
        for i in range(len(y_pred)):
            y_pred[i] = labels[int(y_pred[i])]
        return y_pred

    def predict(self, x_train, y_train, x_test, need_transform_label=False):
        if need_transform_label:
            y_train = self.transform_labels_(y_train)
        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfc', rfc)
        param_grid = {'rfc__max_features': [5], 'rfc__max_depth': [30]}
        model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
        result = model.predict(x_test)
        if need_transform_label:
            result = self.recover_labels_(result)
        return result

class XgboostRegression(Model):

    def predict(self, x_train, y_train, x_test):
        xgbr = xgb.XGBRegressor(learning_rate=0.25, silent=False, objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None)
        clf = self.make_pipeline_('xgbr', xgbr)
        param_grid = {'xgbr__max_depth': [5], 'xgbr__n_estimators': [10]}
        model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
        y_pred = model.predict(x_test)
        for i in range(len(y_pred)):
            if y_pred[i]<1.0:
                y_pred[i] = 1.0
            if y_pred[i]>3.0:
                y_pred[i] = 3.0
        return y_pred

class GbdtRegression(Model):

    def predict(self, x_train, y_train, x_test):
        gbdtr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=2016)
        clf = self.make_pipeline_('gbdtr', gbdtr)
        param_grid = {'gbdtr__n_estimators': (100,), 'gbdtr__learning_rate': (0.1, 0.5), 'gbdtr__max_features': (3, 10, 20), 'gbdtr__max_depth': (5,15,30)}
        model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
        return model.predict(x_test)
