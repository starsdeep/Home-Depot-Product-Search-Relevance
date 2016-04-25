__author__ = 'starsdeep'
import os, sys
import json
import operator
import pandas as pd
import copy
import numpy as np

train_pred_filename_tpl = 'train_pred_trial_%d.csv'
test_pred_filename_tpl = 'test_pred_trial_%d.csv'
trails_filename = 'hyperopt_trials.json'



def load_trial_result_list(dir_path):
    trial_result_list = []
    with open(os.path.join(dir_path, trails_filename)) as infile:
        trial_result_list = json.load(infile)
    return trial_result_list

def load_train_pred_list(dir_path):
    trial_result_list = load_trial_result_list(dir_path)
    N = len(trial_result_list)
    files = [os.path.join(dir_path, train_pred_filename_tpl % i) for i in range(N)]
    train_pred_list = [pd.read_csv(file, encoding="ISO-8859-1", index_col=0)['train_pred'].values for file in files]
    return train_pred_list

def load_test_pred_list(dir_path):
    trial_result_list = load_trial_result_list(dir_path)
    N = len(trial_result_list)
    files = [os.path.join(dir_path, test_pred_filename_tpl % i) for i in range(N)]
    test_pred_list = [pd.read_csv(file, encoding="ISO-8859-1", index_col=0)['test_pred'].values for file in files]
    return test_pred_list 

def load_trials(dir_path):
    trial_result_list = load_trial_result_list(dir_path)
    train_pred_list = load_train_pred_list(dir_path)
    test_pred_list = load_test_pred_list(dir_path)
    return trial_result_list, train_pred_list, test_pred_list



class TrialsList():

    def __init__(self):
        self.trial_result_list = []
        self.train_pred_list = []
        self.test_pred_list = []

    def append(self, trial_result_list, train_pred_list, test_pred_list):
        if len(trial_result_list)!=len(train_pred_list):
            print("size must equal")
            sys.exit()
        self.trial_result_list += trial_result_list
        self.train_pred_list += train_pred_list
        self.test_pred_list += test_pred_list
    def clear_n_trial(self, n):
        if n<0 or n>=len(self.trial_result_list):
            print("%d is invalid, current size is %d" % (n, ))
            sys.exit()
        #del self.trial_result_list[n]
        #del self.train_pred_list[n]
        self.train_pred_list[n] = np.zeros(len(self.train_pred_list[n]))

    def plus_with_index(self, random_model_index, num=10):
        for idx in random_model_index:
            self.train_pred_list[idx] = self.train_pred_list[idx] + num

    def minus_with_index(self, random_model_index, num=10):
        for idx in random_model_index:
            self.train_pred_list[idx] = self.train_pred_list[idx] - num
    
    def best_trial(self, valid_models=set(), verbose=False):
        valid_models = set(valid_models)
        min_index = -1
        min_value=1000
        for idx,value in enumerate(self.trial_result_list):
            if idx not in valid_models:
                continue
            if self.trial_result_list[idx]['loss'] < min_value:
                min_value = self.trial_result_list[idx]['loss']
                min_index = idx
        if verbose:
            print("\nbest trials index is: %d" % min_index)
            print(min_value)
        r1 = copy.deepcopy(self.trial_result_list[min_index])
        r2 = copy.deepcopy(self.train_pred_list[min_index])
        r3 = copy.deepcopy(self.test_pred_list[min_index])
        return min_index, r1, r2, r3  # return does deep copy


if __name__ == '__main__':

    dir_path_list = ["./output/rfr_liaoyikang/", "./output/ridge/"]
    Trials = TrialsList()
    for dir_path in dir_path_list:
        print("loading dir " + dir_path)
        trial_result_list, train_pred_list, test_pred_list = load_trials(dir_path)
        Trials.append(trial_result_list, train_pred_list)

    print("number of trials %d" % len(Trials.trial_result_list))
    print("best trials index %d \n %s " % (Trials.get_best_trial()))



