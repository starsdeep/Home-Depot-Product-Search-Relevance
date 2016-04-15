import os, sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_predict
import xgboost as xgb
import pickle
import config.project
from operator import itemgetter
import json
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.base import clone
from trials_helper import TrialsHelper
# from unique_tfidf_vectorizer import UniqueTfidfVectorizer

# high dimension columns to drops. used for tree based model
hd_col_drops=['id','relevance','search_term','origin_search_term','ori_stem_search_term','search_term_fuzzy_match','product_title','title','main_title','product_description','description','brand','typeid','numsize_of_query','numsize_of_title','numsize_of_main_title','numsize_of_description']

# columns to drops for linear regression model
linear_model_col_drops = hd_col_drops+['len_of_main_title', 'len_of_title', 'len_of_description', 'len_of_brand', "len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description","search_term_fuzzy_match","len_of_search_term_fuzzy_match","noun_of_query", "noun_of_title", "noun_of_main_title", "noun_of_description","len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description",]

train_pred_filename_tpl = 'train_pred_trial_%d.csv'
trails_filename = 'hyperopt_trials.json'

def fmean_squared_error_(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

class CustRegressionVals(BaseEstimator, TransformerMixin):
    def __init__(self, col_drops):
        self.col_drops = col_drops
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        hd_searches = hd_searches.drop(self.col_drops, axis=1, errors='ignore').values
        return hd_searches

class CustTxtCol(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)


class UniqueTfidfVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    def fit(self, x, y=None):
        unique_x = list(set(x))
        self.tfidf.fit(unique_x)
        return self
    def transform(self, raw_documents):
        return self.tfidf.transform(raw_documents)


class CustArrayCol(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]


class Discretizer(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.discreate_points = [1.0, 1.33, 1.67, 2.0, 2.33, 2.67, 3.0]
        self.max_threshold = 0.165
        if threshold>=0.165:
            print("threshold should not >= 0.165")
            sys.exit()
        # self.discretize_func = np.vectorize(lambda x: self._discretize(x, threshold))

    def _discretize(self, x):
        for point in self.discreate_points:
            if x<point-self.threshold:
                break
            if x>=point-self.threshold and x<=point+self.threshold:
                return point
        return x

    def fit(self, x, y=None):
        return self

    def predict(self, values):
        return np.array([self._discretize(v) for v in values])


class Model(object):

    def __init__(self):
        self.RMSE = make_scorer(fmean_squared_error_, greater_is_better=False)
        self.config = dict()
        self.model = None
        self.hyperopt_max_evals = 3


    def make_in_range(self, y_pred):
        return [x if 1.0<=x<=3.0 else 1.0 if x<1.0 else 3.0 for x in y_pred]

    def set_config(self, config):
        self.config = config

    def predict(self, x_train, y_train, x_test):
        print('abstract method')


    # def get_column_importance_(self):
    #     print("abstract method")

    def get_best_cvmodel(self, cvmodel):
        for k, v in self.model.best_params_.items():
            cvmodel = cvmodel.set_params(**{k: v})
        return cvmodel

    def print_badcase_(self, x_train, y_train, train_pred, default_output_line=2000):

        output = x_train.copy(deep=True)
        #output.drop('product_description', axis=1, inplace=True)
        #output.drop('description', axis=1, inplace=True)
        #output.drop('product_title', axis=1, inplace=True)

        output.insert(3, 'pred', pd.Series(train_pred, index=x_train.index))
        output.insert(3, 'diff', pd.Series(train_pred-y_train, index=x_train.index))
        output = output.sort_values(by=['diff', 'id'], ascending=False)
        output_len = min(default_output_line, len(output))

        output[:output_len].to_csv(os.path.join(os.path.abspath(sys.argv[1]),'pos_badcase.csv'), encoding="utf-8")
        output = output[-output_len:]
        output = output.iloc[::-1] #reverse order
        output[:output_len].to_csv(os.path.join(os.path.abspath(sys.argv[1]),'neg_badcase.csv'), encoding="utf-8")

    def save_train_pred(self, df_train, train_pred):
        """
        for model ensemble
        :param x_train:
        :param train_pred:
        :return:
        """
        df_train_pred = pd.DataFrame({'train_pred': train_pred,}, index=df_train.index)
        df_train_pred.to_csv(os.path.join(os.path.abspath(sys.argv[1]), 'train_pred.csv'), encoding="utf8")

    def print_importance_(self, imps, column_names):
        print("======== Printing feature importance ========")
        ranked_imp = sorted(list(zip(column_names,imps)), key=lambda x: x[1], reverse=True)
        for k, v in ranked_imp:
            print(k, v)


    def feature_union_low_(self, X):
        X_drop = X.drop(linear_model_col_drops, axis=1, errors='ignore')
        X_features = X_drop.values
        column_names = X_drop.columns.values
        return X_features, column_names


    def feature_union_normal_(self, X, svdcomp=10):
        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        unique_tfidf = UniqueTfidfVectorizer()
        tsvd = TruncatedSVD(n_components=svdcomp, random_state = 2016)

        union_feature = FeatureUnion(
            transformer_list=[
                ('cst', CustRegressionVals(hd_col_drops)),
                ('txt1', pipeline.Pipeline([('s1', CustTxtCol(key='search_term_fuzzy_match')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                # ('txt2', pipeline.Pipeline([('s2', CustTxtCol(key='title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                #('txt3', pipeline.Pipeline([('s3', CustTxtCol(key='description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                ('txt4', pipeline.Pipeline([('s4', CustTxtCol(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                ('txt5', pipeline.Pipeline([('s5', CustTxtCol(key='main_title')), ('tfidf5', tfidf), ('tsvd5', tsvd)]))
            ],
            transformer_weights={
                'cst': 1.0,
                'txt1': 0.5,
                'txt4': 0.5,
                'txt5': 0.25  # split the 0.25 of txt2 get worse result
            },
            # n_jobs = -1
        )
        X_features = union_feature.fit(X).transform(X)
        column_names = list(X.drop(hd_col_drops, axis=1, errors='ignore').columns.values)
        i = 1
        while len(column_names) < X_features.shape[1]:
            column_names += ['svd'+str(i) for j in range(svdcomp)]
            i += 1
        return X_features, column_names


    def feature_union(self, X):
        print("start feature union ... ")
        low_dim_model = {'ridger', 'lassor'}
        if self.config['model'] in low_dim_model:
            print("model is " + self.config['model'] + ", using feature_union_low_ ...")
            return self.feature_union_low_(X)
        else:
            print("model is " + self.config['model'] + ", using feature_union_normal ...")
            return self.feature_union_normal_(X)

    def hyperopt_optimize_(self, X_train, y_train):
        trials = Trials()
        # variable that will be used in hyperopt_score
        clf = self.model
        best_rmse = 100
        trial_counter = 0
        best_trial_counter = 0

        def hyperopt_score(params):
            #create a new model with parameters equals to params
            nonlocal clf
            nonlocal best_rmse
            nonlocal trial_counter
            nonlocal best_trial_counter

            if 0 < best_rmse:
                pass
            trial_clf = clone(clf)
            for k,v in params.items():
                trial_clf = trial_clf.set_params(**{k: v})

            #compute score, rmse
            train_pred = cross_val_predict(trial_clf, X_train, y_train, cv=3)
            rmse = fmean_squared_error_(y_train, train_pred)
            if rmse < best_rmse:
                best_rmse = rmse
                best_trial_counter = trial_counter
                print('trial %d, new best %s, %s' % (trial_counter, str(best_rmse), str(params)))
            if trial_counter % 10 ==0:
                print('trial %d' % trial_counter)

            #save train_pred for model selection
            df_train_pred = pd.DataFrame({'train_pred': train_pred})
            df_train_pred.to_csv(os.path.join(os.path.abspath(sys.argv[1]), train_pred_filename_tpl % trial_counter), encoding="utf8")

            trial_counter += 1
            return {'loss': rmse, 'status': STATUS_OK, 'params': params}

        best_params = fmin(hyperopt_score, self.param_space, algo=tpe.suggest, trials=trials, max_evals=self.hyperopt_max_evals)

        # save tirals result
        result_list = [{'loss': trials.results[idx]['loss'], 'status': trials.results[idx]['status'], 'params': trials.results[idx]['params']} for idx in range(len(trials.trials))]
        #result_list = sorted(result_list, key=itemgetter('loss'), reverse=True)
        file_path = os.path.join(os.path.abspath(sys.argv[1]), trails_filename)
        with open(file_path, 'w') as outfile:
            json.dump(result_list, outfile)
        print(best_params)
        return best_params

    def grid_search_fit_(self, clf, param_grid, x_train, y_train):
        model = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 1, cv = 2, verbose = 20, scoring=self.RMSE)
        model.fit(x_train, y_train)
        print("Best parameters found by grid search:")
        print(model.best_params_)
        print("Best CV score:")
        print(model.best_score_)
        # f = "%s/%s/model.dump.pickle" % (config.project.project_path, sys.argv[1])
        # pickle.dump(model, f)
        return model

    def make_pipeline_(self, model_name, model):
        tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
        tsvd = TruncatedSVD(n_components=10, random_state = 2016)
        clf = pipeline.Pipeline([
                ('union', FeatureUnion(
                            transformer_list = [
                                ('cst',  CustRegressionVals(hd_col_drops)),
                                ('txt1', pipeline.Pipeline([('s1', CustTxtCol(key='search_term_fuzzy_match')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                                #('txt2', pipeline.Pipeline([('s2', CustTxtCol(key='title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                                #('txt3', pipeline.Pipeline([('s3', CustTxtCol(key='description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                                ('txt4', pipeline.Pipeline([('s4', CustTxtCol(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                                ('txt5', pipeline.Pipeline([('s5', CustTxtCol(key='main_title')), ('tfidf5', tfidf), ('tsvd5', tsvd)]))
                                ],
                            transformer_weights = {
                                'cst': 1.0,
                                'txt1': 0.5,
                                'txt4': 0.5,
                                'txt5': 0.25 # split the 0.25 of txt2 get worse result
                                },
                        #n_jobs = -1
                        )),
                (model_name, model)])
        return clf

    def make_pipeline_low_dim_(self, model_name, model):
        clf = pipeline.Pipeline([
            ('cst', CustRegressionVals(linear_model_col_drops)),
            (model_name, model)
        ])
        return clf

    def set_hyper_params_(self, X_train, y_train):
        if self.config['hyperopt_fit']:
            print("start hyperopt_fit ...")
            if 'hyperopt_max_evals' in self.config:
                self.hyperopt_max_evals = self.config['hyperopt_max_evals']
            best_params = self.hyperopt_optimize_(X_train, y_train)
            print("hyperopt done.")
            print("best_params is: %s" % str(best_params))
            for k, v in best_params.items():
                self.model = self.model.set_params(**{k: v})
        else:
            pass # do nothing, using default params

    def fit(self, X_train, y_train, df_train, column_names):
        self.set_hyper_params_(X_train, y_train)
        # see offline result
        tmp_model = clone(self.model)
        train_pred = cross_validation.cross_val_predict(tmp_model, X_train, y_train, cv=3)
        rmse = fmean_squared_error_(y_train, train_pred)
        self.save_train_pred(df_train, train_pred)
        print("\n======= offline rmse: %f =========" % rmse)
        self.print_badcase_(df_train, y_train, train_pred, 2000)
        # fit
        self.model.fit(X_train, y_train)
        imps = self.get_column_importance_()
        self.print_importance_(imps, column_names)

    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
