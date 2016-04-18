#encoding=utf8

import os, sys
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor

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
from base_model import Model, fmean_squared_error_
class RandomForestRegression(Model):

    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 300
        self.param_space = {
            'max_depth': hp.choice('max_depth', range(3,60)),
            'max_features': hp.choice('max_features', range(2,30)),
            'n_estimators': hp.choice('n_estimators', range(20,30)),
            # 'criterion': hp.choice('criterion', ["gini", "entropy"]),
        }
        self.model = RandomForestRegressor(n_estimators = 2000, max_depth=42, max_features=12, n_jobs = 3, random_state = 2016, verbose = 1)

    def get_column_importance_(self):
        return self.model.feature_importances_

class ExtraTreesRegression(Model):
    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 300
        self.param_space = {
            'max_depth': hp.choice('max_depth', [None, 20, 30, 35, 40, 45, 50]),
            'max_features': hp.choice('max_features', [20, 30, 35, 40, 45, 50]),
            'n_estimators': hp.choice('n_estimators', [20, 30, 60, 90, 100]),
            'min_samples_split': hp.choice('min_samples_split', [1, 2, 3]),
        }
        self.model = ExtraTreesRegressor(n_estimators=2000, max_depth=42, max_features=12, n_jobs=3, random_state=2016, verbose=1)

    def get_column_importance_(self):
        return self.model.feature_importances_

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


        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = 3, random_state = 2016, verbose = 1)
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
        rfc = RandomForestClassifier(n_estimators = 500, n_jobs = 3, random_state = 2016, verbose = 1)
        clf = self.make_pipeline_('rfc', rfc)
        param_grid = {'rfc__max_features': [5], 'rfc__max_depth': [30]}
        self.model = self.grid_search_fit_(clf, param_grid, x_train, y_train)
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

    def get_column_importance_(self):
        return self.model.best_estimator_.named_steps[self.config['model']].feature_importances_


class XgboostRegression(Model):

    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 300
        self.param_space = {
            'n_estimators': hp.uniform('n_estimators', 500, 1000),
            'learning_rate': hp.choice('learning_rate', [0.001,0.003,0.01,0.03,0.1]),
            'objective': hp.choice('objective', ["reg:linear",]),
            'gamma': hp.choice('gamma', [0,]),
            'min_child_weight': hp.choice('min_child_weight', [3,]),
            'max_delta_step': hp.choice('max_delta_step', [0,]),
            'subsample': hp.choice('subsample', [1,]),
            'colsample_bytree': hp.choice('colsample_bytree', [1,]),
            'colsample_bylevel': hp.choice('colsample_bylevel', [1,]),
            'reg_alpha': hp.choice('reg_alpha', [0,]),
            'reg_lambda': hp.choice('reg_lambda', [1,]),
            'scale_pos_weight': hp.choice('scale_pos_weight', [1,]),
            'base_score': hp.choice('base_score', [2,]),
        }
        self.model = xgb.XGBRegressor(learning_rate=0.01, n_estimators=800, max_depth=11, silent=True, objective="reg:linear", nthread=3, gamma=0, min_child_weight=3, max_delta_step=0,
subsample=0.7, colsample_bytree=0.48, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=2, seed=2016, missing=None)
        #xgbr = xgb.XGBRegressor(learning_rate=0.01, max_deepth=11, n_estimators=800, ,silent=True, objective="reg:linear", nthread=3, gamma=0, min_child_weight=3, max_delta_step=0, subsample=0.7, colsample_bytree=0.48, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=2, seed=2016, missing=None)
        #self.model = self.make_pipeline_('xgbr', xgbr)
        

    def fit(self, X_train, y_train, df_train, column_names):
        """
        相比去其他模型，xgboost 的fit函数里面多有一个make_in_range的过程，因此重载
        :param X_train:
        :param y_train:
        :param df_train:
        :param column_names:
        :return:
        """

        self.set_hyper_params_(X_train, y_train)
        # see offline result
        tmp_model = clone(self.model)
        train_pred = cross_validation.cross_val_predict(tmp_model, X_train, y_train, cv=2)
        train_pred = self.make_in_range(train_pred)
        rmse = fmean_squared_error_(y_train, train_pred)
        print("\n======= offline rmse: %f =========" % rmse)
        self.save_train_pred(df_train, train_pred)
        self.print_badcase_(df_train, y_train, train_pred, 2000)
        # fit
        self.model.fit(X_train, y_train)
        imps = self.get_column_importance_()
        self.print_importance_(imps, column_names)

    def predict(self, x_test):
        y_pred = self.model.predict(x_test)
        y_pred = self.make_in_range(y_pred) 
        return y_pred

    def get_column_importance_(self):
        imps = self.model._Booster.get_fscore().items()
        imps = sorted(imps, key=lambda x: int(x[0][1:]))
        imps = [x[1] for x in list(imps)]
        return imps


class GbdtRegression(Model):

    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 300
        self.param_space = {
            'n_estimators': hp.choice('max_depth', [300,1000,2000,2400]),
            'learning_rate': hp.choice('learning_rates', [0.3, 1.0, 3.0]),
            'max_depth': hp.choice('max_depths', [3,5,10,20]),
        }
        self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=2016)

    def get_column_importance_(self):
        return self.model.feature_importances_


class RidgeRegression(Model):

    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 5
        self.param_space = {
            'alpha': hp.uniform('alpha', 0.0, 10),
            'normalize': hp.choice('normalize', [True, False]),
        }
        self.model = linear_model.Ridge(alpha = .5)

    def get_column_importance_(self):
        return self.model.coef_

class LassoRegression(Model):

    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 3
        self.param_space = {
            'alpha': hp.uniform('alpha', 0.0, 10),
            'normalize': hp.choice('normalize', [True, False]),
        }

        self.model = linear_model.Lasso(alpha = .5)

    def get_column_importance_(self):
        return self.model.coef_

class SVR(Model):
    def __init__(self):
        Model.__init__(self)
        self.hyperopt_max_evals = 5
        self.param_space = {
            'C':hp.choice('C',[0.01,0.1,0.5,1,5,10]),
            'epsilon': hp.choice('epsilon',[0.01,0.1,0.5,1]),
            'kernel': hp.choice('kernel',['rbf', 'sigmoid', 'linear', 'poly']),
            'gamma' : hp.choice('gamma',['auto',0.1,0.01,0.001])
        }
        self.model = svm.SVR()
    
    def get_column_importance_(self):
        return self.model.coef_

class LessThan():
    ''' 7 clf  for 1~3 '''
    CLF_DICT = {
        'rfc': RandomForestClassifier(n_estimators = 500, n_jobs = 3, random_state = 2016, verbose = 1),
        'lr':  linear_model.LogisticRegression(),
        'svm': svm.SVC()
    }
    PARAM = {
        'rfc':{'n_estimators':[50,200,500,1000], 'max_features': [5,10,20,30,40], 'max_depth': [4,8,16,32,64,128]},
        'lr': {'C':[0.0001,0.001,0.01,0.1,0.5,1]},
        'svm': {'C':[0.01,0.1,0.5,1,5,10],'kernel':['rbf','sigmoid'],'gamma':['auto',0.1,0.01,0.001]}

    }
    
    def __init__(self):
        self.clf_type = 'svm' # classifier type: rfc, logistic regression, svm_classifier
        self.labels=np.asarray([1., 1.33 , 1.67, 2., 2.33, 2.67, 3.])
        self.margin=np.asarray([1.2, 1.5 , 1.8, 2.2, 2.5, 2.8])
        self.label_num = len(self.margin)
        self.sub_clf = [None]*self.label_num
        self.ACCURACY = make_scorer(self.precision_, greater_is_better=False)
 
    def precision_(self, ground_truth, predictions):
        return accuracy_score(ground_truth, predictions)


    def fit( self, x_train, y_train):
        print(' LessThan __Fit__ was called ')

        for i in range(self.label_num):
            print('fit ',i,'th clf')
            y_train_binary = self.transform_labels_(y_train, i)
            clf = self.CLF_DICT[self.clf_type]
            param_grid = self.PARAM[self.clf_type]
            self.sub_clf[i] = grid_search.GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = 3, cv = 2, verbose = 20, scoring=self.ACCURACY)
            self.sub_clf[i].fit(x_train, y_train_binary)
            print("Best parameters found by grid search:")
            print(self.sub_clf[i].best_params_)
            print("Best CV score:")
            print(self.sub_clf[i].best_score_)
            
    
    def predict(self, x_test):
        print('predict is called')
        result = []
        for i in range(self.label_num):
            tmp = self.sub_clf[i].predict(x_test) 
            recovered_res = self.recover_labels_(tmp, i)
            result.append( recovered_res )
        y_pred_sum = np.sum(result, axis=0)
        y_pred_index = np.argmax(y_pred_sum, axis=1)
        y_pred = self.labels[y_pred_index]
        return y_pred

    def transform(self):
        print(' LessThan __tranform__was called ')
        result = []
        for i in range(self.label_num):
            tmp = self.sub_clf[i].predict(x) 
            recovered_res = self.recover_labels_(tmp, i)
            result.append( recovered_res )
        y_pred_sum = np.sum(result, axis=0)
        y_pred_index = np.argmax(y_pred_sum, axis=1)
        y_pred = self.labels[y_pred_index]
        return y_pred

    def transform_labels_(self, y_train, base):
        y_each = np.asarray([int(i<=self.margin[base]) for i in y_train])
        return y_each

    def recover_labels_(self, y_pred, base):
        '''
            y_pred = predict_proba : 0~1 (of mark<= base)
            return [.]*7  [ _prob /base_ | _1-prob /base ]

        '''
        res = []
        for i in y_pred:
            lh = base+1
            rh = self.label_num-base;
            tmp = [i/lh]*lh + [(1-i)/rh]*rh
            res.append(tmp)
        return np.asarray(res)

class MultiClassifier(Model):
    def fit(self, X_train, y_train, df_train, column_names):
        print( 'Start Multi FIT')
        base_clf = LessThan()
        clf = self.make_pipeline_('lessthan', base_clf)
        param_grid = {}
        self.model = self.grid_search_fit_(clf, param_grid, df_train, y_train)

    def predict(self, x_test):
        print( 'Start Multi Predict...')
        return self.model.predict(x_test)



