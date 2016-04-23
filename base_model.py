import random
import os, sys
import time
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
# from unique_tfidf_vectorizer import UniqueTfidfVectorizer

# high dimension columns to drops. used for tree based model
hd_col_drops=['id','relevance','search_term','origin_search_term','ori_stem_search_term','search_term_fuzzy_match','product_title','title','main_title','product_description','description','brand','typeid','numsize_of_query','numsize_of_title','numsize_of_main_title','numsize_of_description','query_title_co_occur_11gram','query_title_co_occur_22gram','query_title_co_occur_12gram','query_title_co_occur_21gram','query_description_co_occur_11gram','query_description_co_occur_12gram','query_description_co_occur_21gram','query_description_co_occur_22gram']

# columns to drops for linear regression model
linear_model_col_drops = hd_col_drops+['len_of_main_title', 'len_of_title', 'len_of_description', 'len_of_brand', "len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description","search_term_fuzzy_match","len_of_search_term_fuzzy_match","noun_of_query", "noun_of_title", "noun_of_main_title", "noun_of_description","len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description",]

train_pred_filename_tpl = 'train_pred_trial_%d.csv'
test_pred_filename_tpl = 'test_pred_trial_%d.csv'
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
        cutter = CustRegressionVals(hd_col_drops)
        X_features = cutter.transform(X)
        column_names = list(X.drop(hd_col_drops, axis=1, errors='ignore').columns.values)
        i = 1
        while len(column_names) < X_features.shape[1]:
            column_names += ['svd'+str(i) for j in range(svdcomp)]
            i += 1
        return X_features, np.array(column_names)


    def feature_union(self, X):
        print("start feature union ... ")
        low_dim_model = {'ridger', 'lassor'}
        if self.config['model'] in low_dim_model:
            print("model is " + self.config['model'] + ", using feature_union_low_ ...")
            return self.feature_union_low_(X)
        else:
            print("model is " + self.config['model'] + ", using feature_union_normal ...")
            return self.feature_union_normal_(X)

    def hyperopt_optimize_(self, X_train, y_train, X_test, save_result=True):
        trials = Trials()
        # variable that will be used in hyperopt_score
        clf = self.model
        best_rmse = 100
        trial_counter = 0
        best_trial_counter = 0
        model_name = self.config['model']
        model_dir = os.path.abspath(sys.argv[1])
        feature_random_selection = False
        if 'feature_random_selection' in self.config and self.config['feature_random_selection'] == True:
            feature_random_selection = True
            print("[info]: feature_random_selection: %r" % feature_random_selection)
        column_names = np.array(self.column_names)
        result_list = []
        file_path = os.path.join(os.path.abspath(sys.argv[1]), trails_filename)
        def hyperopt_score(params):
            #create a new model with parameters equals to params
            nonlocal clf
            nonlocal best_rmse
            nonlocal trial_counter
            nonlocal best_trial_counter
            nonlocal model_name
            nonlocal model_dir
            nonlocal X_train
            nonlocal X_test
            nonlocal y_train
            nonlocal feature_random_selection
            nonlocal column_names
            nonlocal save_result
            nonlocal result_list
            nonlocal file_path
            # randomly select K features from X_train
            new_X_train = X_train
            new_X_test = X_test
            features = "all"
            feature_index = "all"
            num_total_features = X_train.shape[1]
            K = num_total_features
            if feature_random_selection:
                ratio_list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                ratio = random.choice(ratio_list)
                K = int(num_total_features * ratio)
                K_index = sorted(random.sample(range(num_total_features), K))
                new_X_train = X_train[:,K_index]
                new_X_test = X_test[:,K_index]
                if not K==num_total_features:
                    features = " ".join(column_names[K_index]) 
                    feature_index = " ".join(map(str, K_index))

            if 0 < best_rmse:
                pass
            trial_clf = clone(clf)
            for k,v in params.items():
                trial_clf = trial_clf.set_params(**{k: v})

            #compute score, rmse
            train_pred = cross_val_predict(trial_clf, new_X_train, y_train, cv=2)
            rmse = fmean_squared_error_(y_train, train_pred)
            if rmse < best_rmse:
                best_rmse = rmse
                best_trial_counter = trial_counter
                print('trial %d, new best %s, %s' % (trial_counter, str(best_rmse), str(params)))
            if trial_counter % 20 ==0:
                print('current trial %d' % trial_counter)
            if save_result:
                #save train_pre10d for model selection
                df_train_pred = pd.DataFrame({'train_pred': train_pred})
                df_train_pred.to_csv(os.path.join(os.path.abspath(sys.argv[1]), train_pred_filename_tpl % trial_counter), encoding="utf8")
                #fit whole data and save test_pred
                trial_clf.fit(new_X_train, y_train)
                test_pred = trial_clf.predict(new_X_test)
                df_test_pred = pd.DataFrame({'test_pred': test_pred})
                df_test_pred.to_csv(os.path.join(os.path.abspath(sys.argv[1]), test_pred_filename_tpl % trial_counter), encoding="utf8")
                trial_counter += 1

            trial_result = {'loss': rmse, 'status': STATUS_OK, 'model': model_name, 'model_dir': model_dir, 'params': params, 'features': features, 'feature_index': feature_index, 'num_features':K}
            result_list.append(trial_result)
            with open(file_path, 'w') as outfile:
                json.dump(result_list, outfile)
            return trial_result

        #返回的并不是最优的结果，非常奇怪
        best_params = fmin(hyperopt_score, self.param_space, algo=tpe.suggest, trials=trials, max_evals=self.hyperopt_max_evals)
        
        result_list = trials.results
        #with open(file_path, 'w') as outfile:
        #    json.dump(result_list, outfile)
        index, trial_result = min(enumerate(result_list), key=lambda k: k[1]['loss']) # shallow copy
        print("[info]: best trial results")
        print(trial_result)
        return trial_result['params']

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

    def set_hyper_params_(self, X_train, y_train, X_test, save_result=True):
        if 'hyperopt_fit' in self.config and  self.config['hyperopt_fit']:
            print("[step]: start hyperopt_fit ...")
            if 'hyperopt_max_evals' in self.config:
                self.hyperopt_max_evals = self.config['hyperopt_max_evals']
            print("[info]: hyperopt evals %d" % self.hyperopt_max_evals)
            start_time = time.time()
            best_params = self.hyperopt_optimize_(X_train, y_train, X_test)
            print("[step]: hyperopt done. takes %s mins." % round(((time.time() - start_time)/60),2))
            for k, v in best_params.items():
                self.model = self.model.set_params(**{k: v})
        else:
            pass # do nothing, using default params

    def fit(self, X_train, y_train, df_train, column_names, X_test, save_result=True):
        self.column_names = column_names
        self.set_hyper_params_(X_train, y_train, X_test)
        # see offline result
        tmp_model = clone(self.model)
        train_pred = cross_validation.cross_val_predict(tmp_model, X_train, y_train, cv=3)
        rmse = fmean_squared_error_(y_train, train_pred)
        #self.save_train_pred(df_train, train_pred)
        print("\n[info]: offline rmse: %f\n" % rmse)
        self.print_badcase_(df_train, y_train, train_pred, 2000)
        # fit
        self.model.fit(X_train, y_train)
        imps = self.get_column_importance_()
        self.print_importance_(imps, column_names)


    def predict(self, X_test):
        y_pred = self.model.predict(X_test)
        return y_pred
