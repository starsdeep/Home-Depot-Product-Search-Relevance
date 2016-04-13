import os, sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn import cross_validation
import xgboost as xgb
import pickle
import config.project

# high dimension columns to drops. used for tree based model
hd_col_drops=['id','relevance','search_term','origin_search_term','ori_stem_search_term','search_term_fuzzy_match','product_title','title','main_title','product_description','description','brand','typeid','numsize_of_query','numsize_of_title','numsize_of_main_title','numsize_of_description']

# columns to drops for linear regression model
linear_model_col_drops = hd_col_drops+['len_of_main_title', 'len_of_title', 'len_of_description', 'len_of_brand', "len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description","search_term_fuzzy_match","len_of_search_term_fuzzy_match","noun_of_query", "noun_of_title", "noun_of_main_title", "noun_of_description","len_of_numsize_query","len_of_numsize_main_title","len_of_numsize_title","len_of_numsize_description",]


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
        self.RMSE = make_scorer(self.fmean_squared_error_, greater_is_better=False)
        self.config = dict()
        self.model = None


    def make_in_range(self, y_pred):
        return [x if 1.0<=x<=3.0 else 1.0 if x<1.0 else 3.0 for x in y_pred]

    def set_config(self, config):
        self.config = config

    def predict(self, x_train, y_train, x_test):
        print('abstract method')

    def fmean_squared_error_(self, ground_truth, predictions):
        fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
        return fmean_squared_error_

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

    def save_train_pred(self, x_train, train_pred):
        """
        for model ensemble
        :param x_train:
        :param train_pred:
        :return:
        """
        df_train_pred = pd.DataFrame({'train_pred': train_pred,}, index=x_train.index)
        df_train_pred.to_csv(os.path.join(os.path.abspath(sys.argv[1]), 'train_pred.csv'), encoding="utf8")

    def print_importance_(self, x_train, model, modelname='rfr', svdcomp=10):
        print("======== Printing feature importance ========")
        names = list(x_train.drop(hd_col_drops, axis=1, errors='ignore').columns.values)
        if modelname.find('xgb')>=0:
            imps = model.best_estimator_.named_steps[modelname]._Booster.get_fscore().items()
            imps = sorted(imps, key=lambda x: int(x[0][1:]))
            imps = [x[1] for x in list(imps)]
        else:
            imps = model.best_estimator_.named_steps[modelname].feature_importances_
        i = 1
        while len(names) < len(imps):
            names += ['svd'+str(i) for j in range(svdcomp)]
            i += 1
        ranked_imp = sorted(list(zip(names,imps)), key=lambda x: x[1], reverse=True)
        for k, v in ranked_imp:
            print(k,v)

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


class RandomForestRegression(Model):

    def fit(self, x_train, y_train):
        rfr = RandomForestRegressor(n_estimators = 2000, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfr', rfr)
        param_grid = {'rfr__n_estimators': [2000], 'rfr__max_features': [12], 'rfr__max_depth': [42]}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)

        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)

        self.print_importance_(x_train, self.model, 'rfr')

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        if 'try_discretize' in self.config and self.config['try_discretize']:
            print("\ntry discretize ...\n")
            x_train_predict = model.predict(x_train)
            discretizer = Discretizer(threshold=0.05)
            param_grid = {'threshold': [0.0, 0.02, 0.05, 0.08, 0.1, 0.13, 0.16]}
            discretize_model = self.grid_search_fit_(discretizer, param_grid, x_train_predict, y_train)
            return discretize_model.predict(y_pred)
        return y_pred

class ThreePartRandomForestClassification(Model):

    def get_low_score_(self, x):
        return int(x / 3)

    def get_mid_score_(self, x):
        low = self.get_low_score_(x)
        if int(x) % 3 >= 2:
            return low + 1
        else:
            return low

    def get_high_score_(self, x):
        low = self.get_low_score_(x)
        if x % 3 >= 1:
            return low + 1
        else:
            return low

    def predict(self, x_train, y_train, x_test, need_transform_label=False):
        y_train = list(y_train)
        y_train = [int(3*x + 0.5) for x in y_train]

        y_prefer_low = [self.get_low_score_(x) for x in y_train]
        y_prefer_mid = [self.get_mid_score_(x) for x in y_train]
        y_prefer_high = [self.get_high_score_(x) for x in y_train]


        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfc', rfc)
        param_grid = {'rfc__max_features': [1, 2, 5], 'rfc__max_depth': [2, 5, 10]}

        model_low = self.grid_search_fit_(clf, param_grid, x_train, y_prefer_low)
        result_low = model_low.predict(x_test)

        model_mid = self.grid_search_fit_(clf, param_grid, x_train, y_prefer_mid)
        result_mid = model_mid.predict(x_test)

        model_high = self.grid_search_fit_(clf, param_grid, x_train, y_prefer_high)
        result_high = model_high.predict(x_test)

        result = result_low
        for i in range(len(result)):
            result[i] = (result_low[i] + result_mid[i] + result_high[i]) / 3.0
        return result

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

    def fit(self, x_train, y_train, need_transform_label=False):
        if need_transform_label:
            y_train = self.transform_labels_(y_train)
        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfc', rfc)
        param_grid = {'rfc__max_features': [5], 'rfc__max_depth': [30]}

        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)

        self.print_importance_(x_train, self.model, 'rfc')

    def predict(self, x_test, need_transform_label=False):
        result = self.model.predict(x_test)
        if need_transform_label:
            result = self.recover_labels_(result)
        return result

class XgboostRegression(Model):

    def fit(self, x_train, y_train):
        xgbr = xgb.XGBRegressor(learning_rate=0.25, silent=True, objective="reg:linear", nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0,
subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=2, seed=2016, missing=None)
        clf = self.make_pipeline_('xgbr', xgbr)
        param_grid = {'xgbr__learning_rate': [0.01], 'xgbr__max_depth': [11], 'xgbr__n_estimators': [800], 'xgbr__min_child_weight': [3], 'xgbr__subsample': [0.7], 'xgbr__colsample_bytree': [0.48]}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        train_pred = self.make_in_range(train_pred,1.0,3.0)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)

        self.print_importance_(x_train, self.model, 'xgbr')

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = self.make_in_range(y_pred) 
        return y_pred

class GbdtRegression(Model):

    def fit(self, x_train, y_train):
        gbdtr = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=2016)
        clf = self.make_pipeline_('gbdtr', gbdtr)
        param_grid = {'gbdtr__n_estimators': (100,), 'gbdtr__learning_rate': (0.1, 0.5), 'gbdtr__max_features': (3, 10, 20), 'gbdtr__max_depth': (5,15,30)}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)

        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)
        self.print_importance_(x_train, self.model, 'gbdtr')

    def predict(self,  x_test):
        return self.model.predict(x_test)



class RidgeRegression(Model):

    def fit(self, x_train, y_train):
        ridger = linear_model.Ridge (alpha = .5)
        clf = self.make_pipeline_low_dim_('ridger', ridger)
        param_grid = {'ridger__alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0,3, 0.6, 1.0, 1.5, 3, 8, 15]}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)

        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)

    def predict(self, x_test):
        return self.model.predict(x_test)


class LassoRegression(Model):
    def fit(self, x_train, y_train):
        lassor = linear_model.Lasso (alpha = .1)
        clf = self.make_pipeline_low_dim_('lassor', lassor)
        param_grid = {'lassor__alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0,3, 0.6, 1.0, 1.5, 3, 8, 15]}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)

        best_cvmodel = self.get_best_cvmodel(clf)
        train_pred = cross_validation.cross_val_predict(best_cvmodel, x_train, y_train, cv=3)
        self.save_train_pred(x_train, train_pred)
        if self.config['save_badcase']:
            self.print_badcase_(x_train, y_train, train_pred, 2000)

    def predict(self, x_test):
        return self.model.predict(x_test)

class LessThan(Model):
    '''
        <= labels[label_index] 
        True: 1
        False: 0
    ''' 
    def __init__(self, label_index):
        super(LessThan, self).__init__()
        self.label_index = label_index
        self.labels=np.asarray([1., 1.25, 1.33, 1.5, 1.67, 1.75, 2., 2.25, 2.33, 2.5, 2.67, 2.75, 3.])
        self.total = len(self.labels)

    def predict(self, x_train, y_train, x_test):
        ''' return  [0 or 1]*13 '''
        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfc', rfc)
        y_lessthan = self.transform_labels_(y_train)
        param_grid = {'rfc__max_features': [5], 'rfc__max_depth': [30]}
        model = self.grid_search_fit_(clf, param_grid, x_train, y_lessthan)
        classes_order = model.best_estimator_.named_steps['rfc'].classes_
        binary_judge = model.predict_proba(x_test)[:,classes_order[1]]
        print('binary_judge')
        print(binary_judge)
        y_pred = self.recover_labels_(binary_judge)
        print(y_pred)
        return y_pred

    def transform_labels_(self, y_pred):
        y_each = np.asarray([int(i<=self.labels[self.label_index]) for i in y_pred])
        return y_each

    def recover_labels_(self, y_pred):
        '''
            y_pred = predict_proba : 0~1
            recover y
        '''
        res = []
        for i in y_pred:
            lh = self.label_index+1
            rh = self.total-self.label_index-1
            tmp = [i/lh]*lh + [(1-i)/rh]*rh
            res.append(tmp)
        return np.asarray(res)



class MultiClassifier(Model):
    ''' 13 clf for 1~3 '''
    def predict( self, x_train, y_train, x_test):
        #base _ is a number : 2 classifier base on this number
        #rfc = RandomForestClassifier(n_estimators = 500, n_jobs = -1, random_state = 2016, verbose = 1)
        #clf = self.make_pipeline_('rfc', rfc)
        # param_grid = {'rfc__max_features': [5], 'rfc__max_depth': [30]}
        model = [None]*13
        #result = [None]*len(self.labels)
        for i in range(13):
            model[i] = LessThan(i)
            res = model[i].predict(x_train, y_train, x_test)
            #print(res)
        # for i in range(len(self.labels)):
        #     y_each = self.transform_labels_(y_train, i)
        #     model[i] = self.grid_search_fit_(clf, param_grid, x_train, y_each)
        #     classes_order = model[i].best_estimator_.named_steps['rfc'].classes_
        #     result[i] = model[i].predict_proba(x_test)[:,classes_order[1]]

        # result = np.asarray(result).argmax(axis=0)
        # for i in range( len(result)):
        #     result[i] = self.labels[result[i]]

        return None
